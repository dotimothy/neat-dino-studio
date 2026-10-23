"""The Swin backbone (Swin-B for grounding-dino-base, Swin-T for -tiny) as fused MLA programs.

Swin-B makes 9 programs (stage 3's 18 blocks in groups of 6), Swin-T 7 (stage 3 is 6 blocks).

Every program works on tokens in plain window layout; the plain <-> shifted moves, Swin patch
merging and the return to image order are all Slice/Concat (and spatial-only reshapes) on the
MLA (model/swin_shift_mla.py). Programs:

  bb_stage1          patches (48 ch)        -> stage 1 (2 blocks)
  bb_stage2          stage 1                -> merge + stage 2 (2 blocks)
  bb_proj0           stage 2                -> level 0 tokens, image order  (into the encoder input)
  bb_stage3a/b/c     stage 2 / stage 3      -> merge + stage 3 in 3 groups of 6 blocks
  bb_proj1           stage 3                -> level 1 tokens
  bb_stage4          stage 3                -> merge + stage 4 (2 blocks; 576: padded 18 -> 24)
  bb_level23         stage 4                -> levels 2 and 3 tokens
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import dino_mla as M
import swin_shift_mla as SS
from backbone_plan import Step, LevelProj, Level23, layout, inverse

STAGE3_GROUP = 6          # Swin blocks per stage-3 program


class SwinGroup(nn.Module):
    """Consecutive Swin blocks on plain-layout tokens; shifted blocks move to the shifted layout and back."""

    def __init__(self, blocks, H, ws, shift, embed=None, merge=None, nh_prev=None):
        super().__init__()
        self.embed, self.merge, self.nh_prev = embed, merge, nh_prev
        self.ws, self.shift = ws, shift
        self.blocks = nn.ModuleList(M.SwinBlockW(b, H, H, ws) for b in blocks)
        self.n = -(-H // ws)

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        if self.merge is not None:
            if self.nh_prev is not None:          # merge on the MLA (miscompiles in 2.1.3: strided slices)
                x = SS.merge_windows(x, self.nh_prev, self.nh_prev)
            x = self.merge.red(self.merge.ln(x))
        for b in self.blocks:
            if b.shift:
                x = SS.shifted_to_plain(b(SS.plain_to_shifted(x, self.n, self.n, self.ws, b.shift)),
                                        self.n, self.n, self.ws, b.shift)
            else:
                x = b(x)
        return x


class ProjSpatial(nn.Module):
    """Stage output (plain window layout) -> encoder level tokens in image (row-major) order."""

    def __init__(self, lp: LevelProj, n, H):
        super().__init__()
        self.lp, self.n, self.H = lp, n, H

    def forward(self, x):
        return SS.unpartition(self.lp(x), self.n, self.n, self.H, self.H)      # (1, 256, H, H)


class Level23Spatial(nn.Module):
    def __init__(self, l23: Level23, n, H):
        super().__init__()
        self.l23, self.n, self.H = l23, n, H

    def forward(self, x):
        return self.l23(SS.unpartition(x, self.n, self.n, self.H, self.H))


def build(hf, S, stage3_groups=None):
    sw = hf.model.backbone.conv_encoder.model
    L = sw.encoder.layers
    ref = M.BackboneStage(hf, S)
    WS, SH, chans, depths = M.swin_cfg(hf)
    stage3_groups = stage3_groups or max(1, depths[2] // STAGE3_GROUP)
    dims = [S // 4, S // 8, S // 16, S // 32]
    pads = [-(-d // WS) * WS for d in dims]
    ns = [p // WS for p in pads]
    shapes = M.level_shapes(S)
    off = np.cumsum([0] + [h * w for h, w in shapes])
    nW = [n * n for n in ns]
    steps = [Step("patchify", "patchify", table=layout(pads[0], pads[0], 0, WS, SH), c=48, extra={"grid": pads[0]})]
    tok = lambda s, c: (1, c, nW[s], WS * WS)                                   # noqa: E731
    bufs = iter(["A", "B"] * 8)
    cur = next(bufs)

    def run(name, mod, in_shape, out_shape, dst=None, dst_off=0):
        nonlocal cur
        src = cur
        d = dst or ("B" if src == "A" else "A")
        steps.append(Step("run", name, mod.eval(), in_shape, out_shape, src=src, dst=d, dst_off=dst_off))
        if dst is None:
            cur = d

    run("bb_stage1", SwinGroup(L[0].blocks, dims[0], WS, SH, embed=M.PatchEmbed(sw)), tok(0, 48), tok(0, 128))
    run("bb_stage2", SwinGroup(L[1].blocks, dims[1], WS, SH, merge=M.PatchMerge(L[0].downsample), nh_prev=ns[0]),
        tok(0, 128), tok(1, 256))
    run("bb_proj0", ProjSpatial(LevelProj(sw.hidden_states_norms["stage2"], ref.proj[0]), ns[1], dims[1]),
        tok(1, 256), (1, M.D, dims[1], dims[1]), dst="X", dst_off=int(off[0]))
    blocks3 = list(L[2].blocks)
    k = len(blocks3) // stage3_groups
    for g in range(stage3_groups):
        first = g == 0
        grp = SwinGroup(blocks3[g * k:(g + 1) * k], dims[2], WS, SH, merge=M.PatchMerge(L[1].downsample) if first else None,
                        nh_prev=ns[1] if first else None)
        run(f"bb_stage3{'abcdefgh'[g]}", grp, tok(1, 256) if first else tok(2, 512), tok(2, 512))
    run("bb_proj1", ProjSpatial(LevelProj(sw.hidden_states_norms["stage3"], ref.proj[1]), ns[2], dims[2]),
        tok(2, 512), (1, M.D, dims[2], dims[2]), dst="X", dst_off=int(off[1]))
    run("bb_stage4", SwinGroup(L[3].blocks, dims[3], WS, SH, merge=M.PatchMerge(L[2].downsample), nh_prev=ns[2]),
        tok(2, 512), tok(3, 1024))
    n23 = shapes[2][0] * shapes[2][1] + shapes[3][0] * shapes[3][1]
    run("bb_level23", Level23Spatial(Level23(sw.hidden_states_norms["stage4"], ref.proj[2], ref.proj[3]),
                                     ns[3], dims[3]), tok(3, 1024), (1, M.D, 1, n23), dst="X", dst_off=int(off[2]))
    return steps


def build_cpu_merge(hf, S, stage3_groups=None):
    """Swin stages fused on the MLA (window shifts inside), with patch merging and the return to
    image order done by CPU gathers: on the board, the MLA computes the strided slices of
    merge_windows and the reshape of unpartition wrongly (sima-frontend 2.1.3), although both
    compile and pass the compiler's own check. 9 MLA programs, 6 CPU gathers."""
    sw = hf.model.backbone.conv_encoder.model
    L = sw.encoder.layers
    ref = M.BackboneStage(hf, S)
    WS, SH, chans, depths = M.swin_cfg(hf)
    stage3_groups = stage3_groups or max(1, depths[2] // STAGE3_GROUP)
    dims = [S // 4, S // 8, S // 16, S // 32]
    pads = [-(-d // WS) * WS for d in dims]
    ns = [p // WS for p in pads]
    shapes = M.level_shapes(S)
    off = np.cumsum([0] + [h * w for h, w in shapes])
    nW = [n * n for n in ns]
    plain = [layout(pads[i], pads[i], 0, WS, SH) for i in range(4)]
    inv = [inverse(plain[i], pads[i] * pads[i]) for i in range(4)]
    tok = lambda s, c: (1, c, nW[s], WS * WS)                                   # noqa: E731
    steps = [Step("patchify", "patchify", table=plain[0], c=48, extra={"grid": pads[0]})]

    def run(name, mod, src, dst, i_shape, o_shape, dst_off=0):
        steps.append(Step("run", name, mod.eval(), i_shape, o_shape, src=src, dst=dst, dst_off=dst_off))

    def merge(s, src, dst, c):
        H2, Hp2, Hp = dims[s + 1], pads[s + 1], pads[s]
        tab = -np.ones((Hp2 * Hp2, 4), np.int64)
        for t, sp in enumerate(plain[s + 1]):
            y, x = divmod(int(sp), Hp2)
            if y < H2 and x < H2:
                for j, (dy, dx) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                    tab[t, j] = inv[s][(2 * y + dy) * Hp + 2 * x + dx]
        steps.append(Step("gather", f"merge_s{s + 1}_to_s{s + 2}", src=src, dst=dst, table=tab.reshape(-1), k=4, c=c))

    def rowmajor(s, src, dst, c, dst_off=0, name=""):
        H, Hp = dims[s], pads[s]
        rows = (np.arange(H)[:, None] * Hp + np.arange(H)[None, :]).reshape(-1)
        steps.append(Step("gather", name, src=src, dst=dst, dst_off=dst_off, table=inv[s][rows], k=1, c=c))

    def level(s, lv, src):
        """Stage output -> encoder level lv. The projection ends in a GroupNorm, whose statistics
        must not see padding tokens: a padded stage (Swin-T's 7-px windows at 576) goes to image
        order first, and the projection runs on that."""
        lp = LevelProj(sw.hidden_states_norms[f"stage{s + 1}"], ref.proj[lv])
        H = dims[s]
        if pads[s] == H:
            run(f"bbm_proj{lv}", lp, src, "P", tok(s, chans[s]), tok(s, M.D))
            rowmajor(s, "P", "X", M.D, int(off[lv]), f"level{lv}_rowmajor")
        else:
            rowmajor(s, src, "P", chans[s], 0, f"stage{s + 1}_rowmajor")
            run(f"bbm_proj{lv}", lp, "P", "X", (1, chans[s], H, H), (1, M.D, H, H), dst_off=int(off[lv]))

    run("bb_stage1", SwinGroup(L[0].blocks, dims[0], WS, SH, embed=M.PatchEmbed(sw)), "A", "B", tok(0, 48), tok(0, chans[0]))
    merge(0, "B", "A", chans[0])
    run("bbm_stage2", SwinGroup(L[1].blocks, dims[1], WS, SH, merge=M.PatchMerge(L[0].downsample)), "A", "B",
        tok(1, 4 * chans[0]), tok(1, chans[1]))
    level(1, 0, "B")
    merge(1, "B", "A", chans[1])
    blocks3 = list(L[2].blocks)
    k = len(blocks3) // stage3_groups
    cur = "A"
    for g in range(stage3_groups):
        first = g == 0
        grp = SwinGroup(blocks3[g * k:(g + 1) * k], dims[2], WS, SH, merge=M.PatchMerge(L[1].downsample) if first else None)
        nxt = "B" if cur == "A" else "A"
        run(("bbm_stage3a" if first else f"bb_stage3{'abcdefgh'[g]}"), grp, cur, nxt,
            tok(2, 4 * chans[1]) if first else tok(2, chans[2]), tok(2, chans[2]))
        cur = nxt
    level(2, 1, cur)
    other = "B" if cur == "A" else "A"
    merge(2, cur, other, chans[2])
    fin = "B" if other == "A" else "A"
    run("bbm_stage4", SwinGroup(L[3].blocks, dims[3], WS, SH, merge=M.PatchMerge(L[2].downsample)), other, fin,
        tok(3, 4 * chans[2]), tok(3, chans[3]))
    rowmajor(3, fin, other, chans[3], 0, "stage4_spatial")
    n23 = shapes[2][0] * shapes[2][1] + shapes[3][0] * shapes[3][1]
    run("bbm_level23", Level23(sw.hidden_states_norms["stage4"], ref.proj[2], ref.proj[3]), other, "X",
        (1, chans[3], dims[3], dims[3]), (1, M.D, 1, n23), dst_off=int(off[2]))
    return steps
