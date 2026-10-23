"""The Swin-B backbone as a program of MLA programs and CPU token gathers.

Every Swin block runs as its own MLA program on tokens that are already in window
layout (window-major, then row-major inside the 12x12 window). The window layout
alternates between the plain grid (L0) and the grid rolled by 6 (L1, shifted
windows). Between blocks the CPU moves tokens from one layout to the other; at a
stage boundary it also gathers each 2x2 neighbourhood into one token (Swin patch
merging), and at the stage outputs it puts tokens back in row-major order for the
encoder. All of these are fixed gathers, precomputed as int32 tables.

Buffers (token-major, C channels each):
  A  block inputs        B  block outputs        P  projection outputs
  X  the encoder's token region (enc0 input), row-major levels 0..3

The same plan drives `run_torch` (reference check against HF) and the runtime
(manifest "backbone" steps + tables written by tools/make_bundle.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dino_mla as M

WS = 12          # Swin-B defaults; build() reads the model's own window size and shift
SHIFT = 6


def layout(Hp, Wp, shifted, ws=WS, shift=SHIFT):
    """Token order of a window layout: entry t = spatial index (y * Wp + x) of token t."""
    idx = torch.arange(Hp * Wp, dtype=torch.float64).view(1, 1, Hp, Wp)
    if shifted:
        idx = M.roll_hw(idx, -shift)
    return M.window_partition(idx, ws).reshape(-1).long().numpy()


def inverse(order, n):
    inv = -np.ones(n, np.int64)
    inv[order] = np.arange(len(order))
    return inv


@dataclass
class Step:
    op: str                         # "patchify" | "run" | "gather"
    name: str = ""
    module: nn.Module | None = None
    in_shape: tuple = ()            # 4-D NCHW shape the ELF sees
    out_shape: tuple = ()
    src: str = "B"
    dst: str = "A"
    dst_off: int = 0                # token offset in dst
    k: int = 1                      # tokens gathered per destination token (4 for patch merging)
    c: int = 0                      # channels per gathered token
    table: np.ndarray | None = None  # (T_dst * k,) source token indices, -1 = zero
    extra: dict = field(default_factory=dict)


class LevelProj(nn.Module):
    """Stage output -> encoder level: out LayerNorm, input projection (1x1 conv), GroupNorm."""

    def __init__(self, out_ln, proj):
        super().__init__()
        self.ln, self.conv, self.gn = M.LN(out_ln), proj[0], proj[1]

    def forward(self, x):
        return self.gn(self.conv(self.ln(x)))


class Level23(nn.Module):
    """Stage-4 output (spatial) -> levels 2 and 3 as row-major tokens [L2 | L3]."""

    def __init__(self, out_ln, proj2, proj3):
        super().__init__()
        self.ln, self.p2, self.p3 = M.LN(out_ln), proj2, proj3

    def forward(self, x):
        y = self.ln(x)
        a, b = self.p2(y), self.p3(y)
        return torch.cat([a.reshape(1, M.D, 1, -1), b.reshape(1, M.D, 1, -1)], 3)


def build(hf, S):
    sw = hf.model.backbone.conv_encoder.model
    L = sw.encoder.layers
    ref = M.BackboneStage(hf, S)        # for the GN-converted input projections
    WS, SHIFT, chans, _ = M.swin_cfg(hf)
    dims = [S // 4, S // 8, S // 16, S // 32]
    pads = [-(-d // WS) * WS for d in dims]
    shapes = M.level_shapes(S)
    level_off = np.cumsum([0] + [h * w for h, w in shapes])
    steps: list[Step] = []
    lay = {}
    for s in range(4):
        for sh in (0, 1):
            lay[s, sh] = layout(pads[s], pads[s], sh, WS, SHIFT)

    # stage 1 input: patch pixels in L0 order
    steps.append(Step("patchify", "patchify", table=lay[0, 0], c=48, extra={"grid": pads[0]}))
    for s in range(4):
        H, Hp, C = dims[s], pads[s], chans[s]
        nW = (Hp // WS) ** 2
        for bi, blk in enumerate(L[s].blocks):
            cur = int(blk.shift_size) > 0
            embed = M.PatchEmbed(sw) if (s == 0 and bi == 0) else None
            merge = M.PatchMerge(L[s - 1].downsample) if (s > 0 and bi == 0) else None
            cin = 48 if embed else (4 * chans[s - 1] if merge else C)
            mod = M.SwinBlockW(blk, H, H, WS, merge=merge, embed=embed).eval()
            steps.append(Step("run", f"bb_s{s + 1}_b{bi}", mod, (1, cin, nW, WS * WS), (1, C, nW, WS * WS),
                              src="A", dst="B"))
            if bi + 1 < len(L[s].blocks):           # into the next block's layout
                nxt = int(L[s].blocks[bi + 1].shift_size) > 0
                inv = inverse(lay[s, cur], Hp * Hp)
                steps.append(Step("gather", f"s{s + 1}_b{bi}_to_b{bi + 1}", table=inv[lay[s, nxt]], k=1, c=C))
        last = int(L[s].blocks[-1].shift_size) > 0
        inv_last = inverse(lay[s, last], Hp * Hp)
        # stage outputs -> encoder levels
        if s in (1, 2):
            lv = s - 1
            pm = LevelProj(sw.hidden_states_norms[f"stage{s + 1}"], ref.proj[lv]).eval()
            steps.append(Step("run", f"bb_proj{lv}", pm, (1, C, nW, WS * WS), (1, M.D, nW, WS * WS),
                              src="B", dst="P"))
            rows = (np.arange(H)[:, None] * Hp + np.arange(H)[None, :]).reshape(-1)
            steps.append(Step("gather", f"level{lv}_rowmajor", src="P", dst="X", dst_off=int(level_off[lv]),
                              table=inv_last[rows], k=1, c=M.D))
        if s == 3:
            rows = (np.arange(H)[:, None] * Hp + np.arange(H)[None, :]).reshape(-1)
            steps.append(Step("gather", "stage4_spatial", table=inv_last[rows], k=1, c=C))
            l23 = Level23(sw.hidden_states_norms["stage4"], ref.proj[2], ref.proj[3]).eval()
            n23 = shapes[2][0] * shapes[2][1] + shapes[3][0] * shapes[3][1]
            steps.append(Step("run", "bb_level23", l23, (1, C, H, H), (1, M.D, 1, n23), src="A", dst="X",
                              dst_off=int(level_off[2])))
        if s < 3:                                    # patch merging into the next stage's L0 layout
            H2, Hp2 = dims[s + 1], pads[s + 1]
            tab = -np.ones((Hp2 * Hp2, 4), np.int64)
            for t, sp in enumerate(lay[s + 1, 0]):
                y, x = divmod(int(sp), Hp2)
                if y < H2 and x < H2:
                    for j, (dy, dx) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                        tab[t, j] = inv_last[(2 * y + dy) * Hp + 2 * x + dx]
            steps.append(Step("gather", f"merge_s{s + 1}_to_s{s + 2}", table=tab.reshape(-1), k=4, c=C))
    return steps


def patchify_rows(pixels48: torch.Tensor, table, grid):
    """(1, 48, G, G) space-to-depth image -> (T, 48) tokens in the given layout (pad tokens zero)."""
    G = pixels48.shape[-1]
    flat = pixels48[0].permute(1, 2, 0).reshape(-1, 48)
    ys, xs = np.divmod(table, grid)
    ok = (ys < G) & (xs < G)
    out = torch.zeros(len(table), 48)
    out[ok] = flat[torch.from_numpy(ys[ok] * G + xs[ok])]
    return out


def run_torch(steps, pixels48, N):
    """Execute the plan in torch: returns the encoder tokens (1, 256, 1, N)."""
    bufs = {"A": None, "B": None, "P": None, "X": torch.zeros(N, M.D)}

    def to4d(rows, shape):
        _, C, H, W = shape
        return rows[: H * W].reshape(H, W, C).permute(2, 0, 1)[None]

    def to_rows(t):
        return t[0].permute(1, 2, 0).reshape(-1, t.shape[1])

    with torch.no_grad():
        for st in steps:
            if st.op == "patchify":
                bufs["A"] = patchify_rows(pixels48, st.table, st.extra["grid"])
            elif st.op == "run":
                y = to_rows(st.module(to4d(bufs[st.src], st.in_shape)))
                if st.dst == "X":
                    bufs["X"][st.dst_off: st.dst_off + len(y)] = y
                else:
                    bufs[st.dst] = y
            else:
                src = bufs[st.src]
                idx = torch.from_numpy(st.table)
                g = torch.where((idx >= 0)[:, None], src[idx.clamp(min=0)], torch.zeros(1, src.shape[1]))
                g = g.reshape(-1, st.k * src.shape[1])
                if st.dst == "X":
                    bufs["X"][st.dst_off: st.dst_off + len(g)] = g
                else:
                    bufs[st.dst] = g
    return M.rows_to_tok(bufs["X"])
