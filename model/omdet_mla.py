"""OmDet-Turbo (omlab/omdet-turbo-swin-tiny-hf, Apache-2.0) rebuilt in MLA-friendly pieces.

OmDet-Turbo is a real-time open-vocabulary detector: a timm Swin-T backbone, an RT-DETR hybrid
encoder (one plain attention layer on the stride-32 level, then a conv FPN/PAN), CLIP text
(one pooled embedding per class, plus the tokens of a task prompt "Detect a, b."), a top-900
query selection by cosine class similarity, and a 6-layer decoder whose self-attention also
sees the task tokens and whose cross-attention is multi-scale deformable (3 levels x 4 points).

Board split (same pattern as the Grounding DINO port, model/dino_mla.py):
  Swin-T      one MLA program per block on tokens in window layout (1, C, nW, 49); the CPU moves
              tokens between window layouts (timm rolls BEFORE padding, so the tables differ from
              HF Swin's) and gathers 2x2 neighbourhoods for patch merging. The last block of
              stages 2-4 also applies the level LayerNorm + input projection (extra 256 channels).
  aifi        the hybrid encoder's attention layer on the stride-32 level (1, 256, 1, 400)
  ccfm        FPN + PAN convs on the three levels in image layout
  dec_in      decoder channel projections, the encoder head (class features, box) and the
              6 decoder layers' value projections, on tokens (1, 256, 1, 8400)
  dec_pre_k   query position MLP, self-attention over [queries | task tokens], sampling
              offsets and attention weights
  dec_post_k  deformable output projection, FFN, box head
  text        CLIP text encoder (12 layers, width 512), several sequences at once on H
The CPU does the deformable sampling, the cosine similarities, the top-900 and the box updates.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import sam3_mla as S3  # noqa: E402

HF_ID = "omlab/omdet-turbo-swin-tiny-hf"
S = 640
MEAN = (123.675, 116.28, 103.53)      # on 0..255 pixels (the processor does not rescale)
STD = (58.395, 57.12, 57.375)
WS, SHIFT = 7, 3
NQ = 900
D = 256


def load_hf():
    from transformers import OmDetTurboForObjectDetection
    return OmDetTurboForObjectDetection.from_pretrained(HF_ID).eval()


# --------------------------------------------------------------------------- Swin-T (timm) in window layout
def window_partition(x, ws):
    """(1, C, Hp, Wp) -> (1, C, nW, ws*ws), windows row-major, tokens row-major inside."""
    _, C, Hp, Wp = x.shape
    return x.view(1, C, Hp // ws, ws, Wp // ws, ws).permute(0, 1, 2, 4, 3, 5).reshape(1, C, -1, ws * ws)


def layout(H, W, shift, ws=WS):
    """timm window layout of an H x W grid: entry t = original index (y * W + x) of the token at
    window-layout position t, or -1 for padding. timm rolls the unpadded grid by -shift, then pads
    to a multiple of the window, then partitions."""
    Hp, Wp = -(-H // ws) * ws, -(-W // ws) * ws
    yy, xx = torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing="ij")
    orig = ((yy + shift) % H) * W + (xx + shift) % W
    orig = torch.where((yy < H) & (xx < W), orig, torch.full_like(orig, -1))
    return window_partition(orig[None, None].double(), ws).reshape(-1).long().numpy()


def inverse(order, n):
    inv = -np.ones(n, np.int64)
    ok = order >= 0
    inv[order[ok]] = np.nonzero(ok)[0]
    return inv


class SwinBlockW(nn.Module):
    """One timm SwinTransformerBlock on tokens in its window layout (1, C, nW, 49).

    Optional at the front: the patch embedding (1x1 conv on the host's 4x4 space-to-depth, with the
    pixel normalisation folded in, + LayerNorm) or patch merging (LayerNorm + reduction on the 4
    gathered neighbours). Optional at the end (`level`): the backbone's level LayerNorm and the
    hybrid encoder's input projection (conv + BN), appended as 256 extra output channels."""

    def __init__(self, blk, H, W, embed=None, merge=None, level=None):
        super().__init__()
        a = blk.attn
        C = a.qkv.in_features
        self.heads = a.num_heads
        self.hd = C // self.heads
        self.shift = int(blk.shift_size[0]) if isinstance(blk.shift_size, (tuple, list)) else int(blk.shift_size)
        ws = blk.window_size[0] if isinstance(blk.window_size, (tuple, list)) else blk.window_size
        assert ws == WS
        w, b = a.qkv.weight.detach(), a.qkv.bias.detach()
        mk = lambda i, sc=1.0: S3.conv_from_linear(  # noqa: E731
            nn.Linear(C, C), weight=w[i * C:(i + 1) * C], bias=b[i * C:(i + 1) * C], scale=sc)
        self.q, self.k, self.v = mk(0, a.scale), mk(1), mk(2)
        self.o = S3.conv_from_linear(a.proj)
        self.ln1, self.ln2 = S3.LN(blk.norm1), S3.LN(blk.norm2)
        self.fc1, self.fc2 = S3.conv_from_linear(blk.mlp.fc1), S3.conv_from_linear(blk.mlp.fc2)
        rpb = a._get_rel_pos_bias()[0].detach()                         # (h, q, k)
        self.register_buffer("bias", rpb.permute(0, 2, 1)[:, None, :, None, :].contiguous())  # (h, 1, k, 1, q)
        lay = layout(H, W, self.shift)
        nW = len(lay) // (ws * ws)
        if self.shift:
            m = blk.attn_mask.detach()                                  # (nW, q, k), 0 / -100
            assert m.shape[0] == nW, (m.shape, nW)
            self.register_buffer("mask", m.permute(2, 0, 1)[None].contiguous())   # (1, k, nW, q)
        else:
            self.mask = None
        valid = torch.from_numpy(lay >= 0).float().view(1, 1, nW, ws * ws)
        self.register_buffer("pad", valid.expand(1, C, -1, -1).contiguous() if (valid < 1).any() else None)
        self.embed, self.merge, self.level = embed, merge, level
        self.H, self.W, self.lay = H, W, lay

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        if self.merge is not None:
            x = self.merge(x)
        y = self.ln1(x)
        if self.pad is not None:
            y = y * self.pad
        q, k, v = self.q(y), self.k(y), self.v(y)
        outs = []
        for h in range(self.heads):
            sl = slice(h * self.hd, (h + 1) * self.hd)
            s_ = torch.einsum("nchw,nchq->nqhw", q[:, sl], k[:, sl]) + self.bias[h]
            if self.mask is not None:
                s_ = s_ + self.mask
            outs.append(torch.einsum("nchw,nqhc->nqhw", torch.softmax(s_, dim=1), v[:, sl]))
        x = x + self.o(torch.cat(outs, 1))
        x = x + self.fc2(S3.gelu(self.fc1(self.ln2(x))))
        if self.level is not None:
            return torch.cat([x, self.level(x)], 1)
        return x


class PatchEmbed(nn.Module):
    """4x4 patch conv as a 1x1 conv on space-to-depth pixels in [0, 1] (normalisation folded in) + LN."""

    def __init__(self, pe):
        super().__init__()
        w, b = pe.proj.weight.detach(), pe.proj.bias.detach()           # (96, 3, 4, 4)
        std = torch.tensor(STD).view(1, 3, 1, 1) / 255.0
        mean = torch.tensor(MEAN).view(1, 3, 1, 1) / 255.0
        wn = w / std
        self.patch = nn.Conv2d(48, w.shape[0], 1)
        self.patch.weight.data.copy_(wn.reshape(w.shape[0], 48)[:, :, None, None])
        self.patch.bias.data.copy_(b - (wn * mean).sum((1, 2, 3)))
        self.ln = S3.LN(pe.norm)

    def forward(self, x):
        return self.ln(self.patch(x))


class Merge(nn.Module):
    def __init__(self, pm):
        super().__init__()
        self.ln = S3.LN(pm.norm)
        w = pm.reduction.weight.detach()
        self.red = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
        self.red.weight.data.copy_(w[:, :, None, None])

    def forward(self, x):
        return self.red(self.ln(x))


def conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold an eval-mode BatchNorm into the conv before it."""
    s = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    c = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
    c.weight.data.copy_(conv.weight.detach() * s[:, None, None, None])
    b0 = conv.bias.detach() if conv.bias is not None else torch.zeros_like(s)
    c.bias.data.copy_((b0 - bn.running_mean) * s + bn.bias.detach())
    return c


class LevelProj(nn.Module):
    """Backbone level LayerNorm + hybrid encoder input projection (1x1 conv + BN)."""

    def __init__(self, ln, proj):
        super().__init__()
        self.ln = S3.LN(ln)
        self.proj = conv_bn(proj[0], proj[1])

    def forward(self, x):
        return self.proj(self.ln(x))


def space_to_depth4(px01):
    """(3, S, S) pixels in [0, 1] -> (1, 48, S/4, S/4), channel order (c, py, px)."""
    return S3.space_to_depth(px01, 4)


class Backbone:
    """The block programs and the CPU gather tables between them."""

    def __init__(self, hf, S_=S):
        bb = hf.vision_backbone.vision_backbone
        tm = bb._backbone if hasattr(bb, "_backbone") else bb
        enc = hf.encoder
        lns = hf.vision_backbone.layer_norms
        self.blocks, self.steps = [], []
        H = W = S_ // 4
        prev = None                               # (layout, H, W, C, channels per token in the buffer)
        stages = list(tm.layers) if hasattr(tm, "layers") else [getattr(tm, f"layers_{i}") for i in range(4)]
        for si, stage in enumerate(stages):
            if si > 0:
                H, W = H // 2, W // 2
            C = stage.blocks[0].attn.qkv.in_features
            for bi, blk in enumerate(stage.blocks):
                last = bi == len(stage.blocks) - 1
                embed = PatchEmbed(tm.patch_embed) if (si == 0 and bi == 0) else None
                merge = Merge(stage.downsample) if (si > 0 and bi == 0) else None
                level = LevelProj(lns[si - 1], enc.channel_projection_layers[si - 1]) if (last and si > 0) else None
                m = SwinBlockW(blk, H, W, embed=embed, merge=merge, level=level).eval()
                name = f"bb{si + 1}_{bi}"
                lay = m.lay
                if embed is not None:
                    self.steps.append({"op": "patchify", "table": lay, "grid": W})
                elif merge is not None:
                    pl, pH, pW, pC, pstride = prev
                    inv = inverse(pl, pH * pW)
                    tab = np.full((len(lay), 4), -1, np.int64)
                    for t, o in enumerate(lay):
                        if o < 0:
                            continue
                        y, x = divmod(int(o), W)
                        for j, (dy, dx) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                            tab[t, j] = inv[(2 * y + dy) * pW + 2 * x + dx]
                    self.steps.append({"op": "gather", "k": 4, "c": pC, "src_stride": pstride, "src_off": 0,
                                       "table": tab.reshape(-1)})
                else:
                    pl, pH, pW, pC, pstride = prev
                    inv = inverse(pl, pH * pW)
                    tab = np.where(lay >= 0, inv[np.maximum(lay, 0)], -1)
                    self.steps.append({"op": "gather", "k": 1, "c": pC, "src_stride": pstride, "src_off": 0,
                                       "table": tab})
                nW = len(lay) // (WS * WS)
                cin = 48 if embed is not None else (4 * C // 2 if merge is not None else C)
                cout = C + (D if level is not None else 0)
                self.steps.append({"op": "run", "name": name, "in_shape": (1, cin, nW, WS * WS),
                                   "out_shape": (1, cout, nW, WS * WS)})
                self.blocks.append((name, m))
                prev = (lay, H, W, C, cout)
                if level is not None:            # level output: the projected channels in row-major order
                    inv = inverse(lay, H * W)
                    self.steps.append({"op": "level", "level": si - 1, "c": D, "src_stride": cout, "src_off": C,
                                       "table": inv, "hw": (H, W)})
        self.modules = dict(self.blocks)

    def run_torch(self, px01):
        """Reference run with torch gathers: returns the 3 projected levels (1, 256, H, W)."""
        buf = None
        levels = [None] * 3
        x = None
        for st in self.steps:
            if st["op"] == "patchify":
                g = space_to_depth4(px01).reshape(48, -1)            # row-major tokens
                tab = torch.from_numpy(st["table"])
                x = torch.where(tab[None] >= 0, g[:, tab.clamp(min=0)], torch.zeros(()))
                nW = len(tab) // (WS * WS)
                x = x.view(1, 48, nW, WS * WS)
            elif st["op"] == "gather":
                src = buf[0].reshape(buf.shape[1], -1)[st["src_off"]:st["src_off"] + st["c"]]
                tab = torch.from_numpy(st["table"]).view(-1, st["k"])
                g = torch.where(tab[None] >= 0, src[:, tab.clamp(min=0)], torch.zeros(()))   # (c, T, k)
                g = g.permute(2, 0, 1).reshape(st["k"] * st["c"], -1)                     # neighbour-major
                x = g.view(1, st["k"] * st["c"], -1, WS * WS)
            elif st["op"] == "run":
                with torch.no_grad():
                    buf = self.modules[st["name"]](x)
            elif st["op"] == "level":
                src = buf[0].reshape(buf.shape[1], -1)[st["src_off"]:st["src_off"] + st["c"]]
                Hh, Ww = st["hw"]
                levels[st["level"]] = src[:, torch.from_numpy(st["table"])].view(1, st["c"], Hh, Ww)
        return levels


# --------------------------------------------------------------------------- hybrid encoder
TT = 64          # task tokens (CLIP tokens of "Detect a, b, c.") the decoder programs take
TEXT_T = 64      # CLIP tokens per sequence in the text program
TEXT_B = 8       # sequences per text-program run
NEG = -1e4


def sincos_2d(w, h, dim=256, temp=10000.0):
    """build_2d_sincos_position_embedding as the hybrid encoder uses it -> (N, dim)."""
    gw, gh = torch.meshgrid(torch.arange(w, dtype=torch.float32), torch.arange(h, dtype=torch.float32), indexing="ij")
    pd = dim // 4
    om = 1.0 / temp ** (torch.arange(pd, dtype=torch.float32) / pd)
    ow, oh = gw.flatten()[:, None] @ om[None], gh.flatten()[:, None] @ om[None]
    return torch.cat([ow.sin(), ow.cos(), oh.sin(), oh.cos()], 1)


class MHA(nn.Module):
    """OmDetTurboMultiheadAttention on token tensors (1, C, 1, T), one einsum pair per head."""

    def __init__(self, a):
        super().__init__()
        self.heads = a.num_attention_heads
        self.hd = a.attention_head_size
        self.q = S3.conv_from_linear(a.query, scale=self.hd ** -0.5)
        self.k, self.v, self.o = (S3.conv_from_linear(l) for l in (a.key, a.value, a.out_proj))

    def forward(self, q, k, v, k2=None, v2=None, mask2=None):
        """Keys/values k, v plus an optional second key/value set (k2, v2) with an additive mask on
        its keys (1, T2, 1, 1); the softmax runs over both sets (concatenated on C)."""
        qp, kp, vp = self.q(q), self.k(k), self.v(v)
        if k2 is not None:
            kp2, vp2 = self.k(k2), self.v(v2)
        outs = []
        for h in range(self.heads):
            sl = slice(h * self.hd, (h + 1) * self.hd)
            s_ = torch.einsum("nchw,nchq->nqhw", qp[:, sl], kp[:, sl])
            if k2 is None:
                outs.append(torch.einsum("nchw,nqhc->nqhw", torch.softmax(s_, dim=1), vp[:, sl]))
                continue
            s2 = torch.einsum("nchw,nchq->nqhw", qp[:, sl], kp2[:, sl]) + mask2
            p = torch.softmax(torch.cat([s_, s2], 1), dim=1)
            n1 = s_.shape[1]
            outs.append(torch.einsum("nchw,nqhc->nqhw", p[:, :n1], vp[:, sl]) +
                        torch.einsum("nchw,nqhc->nqhw", p[:, n1:], vp2[:, sl]))
        return self.o(torch.cat(outs, 1))


class Aifi(nn.Module):
    """The hybrid encoder's transformer layer on the stride-32 level: (1, 256, 1, 400) -> same."""

    def __init__(self, hf, hw=(S // 32, S // 32)):
        super().__init__()
        L = hf.encoder.encoder[0].layers[0]
        self.sa = MHA(L.self_attn)
        self.ln1, self.ln2 = S3.LN(L.self_attn_layer_norm), S3.LN(L.final_layer_norm)
        self.fc1, self.fc2 = S3.conv_from_linear(L.fc1), S3.conv_from_linear(L.fc2)
        h, w = hw
        self.register_buffer("pos", S3.tok(sincos_2d(w, h, D)))

    def forward(self, x):
        q = x + self.pos
        x = self.ln1(x + self.sa(q, q, x))
        return self.ln2(x + self.fc2(F.relu(self.fc1(x))))


def _act(name):
    return {"gelu": S3.gelu, "silu": lambda x: x * torch.sigmoid(x), None: lambda x: x}[name]


class ConvNorm(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.c = conv_bn(m.conv, m.norm)
        self.act = {"GELUActivation": "gelu", "GELU": "gelu", "SiLU": "silu", "SiLUActivation": "silu",
                    "Identity": None}[type(m.activation).__name__]

    def forward(self, x):
        return _act(self.act)(self.c(x))


class CSPRep(nn.Module):
    def __init__(self, m, act="silu"):
        super().__init__()
        self.c1, self.c2 = ConvNorm(m.conv1), ConvNorm(m.conv2)
        self.blocks = nn.ModuleList(nn.ModuleList([conv_bn(b.conv1.conv, b.conv1.norm), conv_bn(b.conv2.conv, b.conv2.norm)])
                                    for b in m.bottlenecks)
        self.act = _act(act)
        self.c3 = None if isinstance(m.conv3, nn.Identity) else ConvNorm(m.conv3)

    def forward(self, x):
        h1 = self.c1(x)
        for a, b in self.blocks:
            h1 = self.act(a(h1) + b(h1))
        y = h1 + self.c2(x)
        return y if self.c3 is None else self.c3(y)


def upsample2(x):
    return F.interpolate(x, scale_factor=2.0, mode="nearest")


class Ccfm(nn.Module):
    """FPN + PAN of the hybrid encoder, then the decoder's channel projections (conv + BN).
    in  l0 (1, 256, 80, 80), l1 (1, 256, 40, 40), l2 (1, 256, 20, 20) (AIFI output, same bytes)
    out the three decoder levels, each (1, 256, h, w); `which` picks one (single-output builds)."""

    def __init__(self, hf, which=None):
        super().__init__()
        e = hf.encoder
        self.lat = nn.ModuleList(ConvNorm(m) for m in e.lateral_convs)
        self.fpn = nn.ModuleList(CSPRep(m) for m in e.fpn_blocks)
        self.down = nn.ModuleList(ConvNorm(m) for m in e.downsample_convs)
        self.pan = nn.ModuleList(CSPRep(m) for m in e.pan_blocks)
        self.proj = nn.ModuleList(conv_bn(p[0], p[1]) for p in hf.decoder.channel_projection_layers)
        self.which = which

    def forward(self, l0, l1, l2):
        feats = [l0, l1, l2]
        maps = [l2]
        for i, idx in enumerate((2, 1)):
            hi = self.lat[i](maps[0])
            maps[0] = hi
            maps.insert(0, self.fpn[i](torch.cat([upsample2(hi), feats[idx - 1]], 1)))
        outs = [maps[0]]
        for i in range(2):
            outs.append(self.pan[i](torch.cat([self.down[i](outs[-1]), maps[i + 1]], 1)))
        outs = [p(o) for p, o in zip(self.proj, outs)]
        return outs if self.which is None else outs[self.which]


def level_sizes(S_=S):
    return [(S_ // 8, S_ // 8), (S_ // 16, S_ // 16), (S_ // 32, S_ // 32)]


def anchors(S_=S, grid=0.05, eps=1e-2):
    """generate_anchors: (N, 4) logit-space anchors and the (N,) valid mask."""
    out = []
    for lvl, (h, w) in enumerate(level_sizes(S_)):
        gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
        xy = (torch.stack([gx, gy], -1) + 0.5) / torch.tensor([w, h], dtype=torch.float32)
        wh = torch.ones_like(xy) * grid * 2.0 ** lvl
        out.append(torch.cat([xy, wh], -1).reshape(-1, 4))
    a = torch.cat(out)
    valid = ((a > eps) & (a < 1 - eps)).all(-1)
    return torch.log(a / (1 - a)), valid


class DecIn(nn.Module):
    """in  l0 (1, 256, 1, 6400), l1 (1, 256, 1, 1600), l2 (1, 256, 1, 400): the decoder levels as tokens
    out [V0 .. V5 | CF | BOX]: each decoder layer's value projection, the encoder head's class features
    (Linear + LN) and its box MLP (4 of 256 channels; the host adds the anchors)."""

    def __init__(self, hf):
        super().__init__()
        d = hf.decoder
        self.values = nn.ModuleList(S3.conv_from_linear(L.cross_attn.value_proj) for L in d.layers)
        self.cf = S3.conv_from_linear(d.encoder_vision_features[0])
        self.cf_ln = S3.LN(d.encoder_vision_features[1])
        n = len(d.encoder_bbox_head.layers)
        self.box = nn.ModuleList(conv_from_linear_pad(l, D if i == n - 1 else None)
                                 for i, l in enumerate(d.encoder_bbox_head.layers))

    def forward(self, l0, l1, l2):
        v = torch.cat([l0, l1, l2], 3)
        cf = self.cf_ln(self.cf(v))
        b = cf
        for i, l in enumerate(self.box):
            b = l(b)
            if i < len(self.box) - 1:
                b = F.relu(b)
        return torch.cat([f(v) for f in self.values] + [cf, b], 3)


def conv_from_linear_pad(lin, out=None):
    w, b = lin.weight.detach(), lin.bias.detach()
    if out and out > w.shape[0]:
        w = torch.cat([w, w.new_zeros(out - w.shape[0], w.shape[1])])
        b = torch.cat([b, b.new_zeros(out - b.shape[0])])
    return S3.conv_from_linear(lin, weight=w, bias=b)


# --------------------------------------------------------------------------- decoder
class DecPre(nn.Module):
    """in  [H (900) | REF (900; ch 0-3 = sigmoid box) | TASK (TT) | KM (1; ch 0..TT-1 additive task mask)]
    out [H2 (900) | TASK2 (TT) | OFF (900; 192 of 256 ch) | AW (900; 96 of 256 ch)]"""

    def __init__(self, hf, k):
        super().__init__()
        d = hf.decoder
        qp = d.query_position_head.layers
        w0 = torch.cat([qp[0].weight.detach(), qp[0].weight.new_zeros(qp[0].out_features, D - 4)], 1)
        self.qp1 = S3.conv_from_linear(qp[0], weight=w0)
        self.qp2 = S3.conv_from_linear(qp[1])
        L = d.layers[k]
        self.sa = MHA(L.self_attn)
        self.ln1 = S3.LN(L.norm1)
        ca = L.cross_attn
        self.off = conv_from_linear_pad(ca.sampling_offsets, D)
        self.aw = conv_from_linear_pad(ca.attention_weights, D)

    def forward(self, x):
        H, REF, TASK, KM = torch.split(x, [NQ, NQ, TT, 1], 3)
        km = KM[:, :TT]                                                      # (1, TT, 1, 1)
        pos = self.qp2(F.relu(self.qp1(REF)))
        q = H + pos
        # self-attention over [queries | task tokens]: the task keys/values are the second set
        Hq = H + self.sa(q, q, H, TASK, TASK, km)
        Tq = TASK + self.sa(TASK, q, H, TASK, TASK, km)
        H2, T2 = self.ln1(Hq), self.ln1(Tq)
        h = H2 + pos
        return torch.cat([H2, T2, self.off(h), self.aw(h)], 3)


class DecPost(nn.Module):
    """in [H2 (900) | S (900) sampled]  ->  out [H' (900) | BOX (900; ch 0-3 box delta)]"""

    def __init__(self, hf, k):
        super().__init__()
        d = hf.decoder
        L = d.layers[k]
        self.out = S3.conv_from_linear(L.cross_attn.output_proj)
        self.ln2, self.ln3 = S3.LN(L.norm2), S3.LN(L.norm3)
        self.fc1, self.fc2 = S3.conv_from_linear(L.linear1), S3.conv_from_linear(L.linear2)
        n = len(d.decoder_bbox_head[k].layers)
        self.box = nn.ModuleList(conv_from_linear_pad(l, D if i == n - 1 else None)
                                 for i, l in enumerate(d.decoder_bbox_head[k].layers))

    def forward(self, x):
        H2, Sx = torch.split(x, [NQ, NQ], 3)
        h = self.ln2(H2 + self.out(Sx))
        h = self.ln3(h + self.fc2(F.relu(self.fc1(h))))
        b = h
        for i, l in enumerate(self.box):
            b = l(b)
            if i < len(self.box) - 1:
                b = F.relu(b)
        return torch.cat([h, b], 3)


# --------------------------------------------------------------------------- text (CLIP)
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class TextEncoder(nn.Module):
    """CLIP text transformer on TEXT_B sequences at once: in (1, 512, B, T) embeddings (host: token +
    position), out (1, 512, B, T) last hidden state (after the final LayerNorm). Causal mask only:
    tokens after the end token never reach the ones the host reads."""

    def __init__(self, hf, T=TEXT_T):
        super().__init__()
        tm = getattr(hf.language_backbone.model, "text_model", hf.language_backbone.model)
        self.layers = nn.ModuleList()
        for L in tm.encoder.layers:
            m = nn.Module()
            a = L.self_attn
            m.heads, m.hd = a.num_heads, a.head_dim
            m.q = S3.conv_from_linear(a.q_proj, scale=a.head_dim ** -0.5)
            m.k, m.v, m.o = (S3.conv_from_linear(l) for l in (a.k_proj, a.v_proj, a.out_proj))
            m.ln1, m.ln2 = S3.LN(L.layer_norm1), S3.LN(L.layer_norm2)
            m.fc1, m.fc2 = S3.conv_from_linear(L.mlp.fc1), S3.conv_from_linear(L.mlp.fc2)
            self.layers.append(m)
        self.lnf = S3.LN(tm.final_layer_norm)
        k = torch.arange(T)[:, None]
        q = torch.arange(T)[None, :]
        self.register_buffer("mask", torch.where(k <= q, 0.0, NEG).view(1, T, 1, T))

    def forward(self, x):
        for m in self.layers:
            y = m.ln1(x)
            q, k, v = m.q(y), m.k(y), m.v(y)
            outs = []
            for h in range(m.heads):
                sl = slice(h * m.hd, (h + 1) * m.hd)
                s_ = torch.einsum("nchw,nchq->nqhw", q[:, sl], k[:, sl]) + self.mask
                outs.append(torch.einsum("nchw,nqhc->nqhw", torch.softmax(s_, dim=1), v[:, sl]))
            x = x + m.o(torch.cat(outs, 1))
            x = x + m.fc2(quick_gelu(m.fc1(m.ln2(x))))
        return self.lnf(x)


class TaskHead(nn.Module):
    """Task encoder (MLP + residual LN) and projection to 256: (1, 512, 1, TT) -> (1, 256, 1, TT)."""

    def __init__(self, hf):
        super().__init__()
        te = hf.decoder.task_encoder
        self.l1, self.l2 = S3.conv_from_linear(te.mlp.linear1), S3.conv_from_linear(te.mlp.linear2)
        self.ln = S3.LN(te.res1.norm1)
        self.proj = S3.conv_from_linear(hf.decoder.task_project)

    def forward(self, x):
        return self.proj(self.ln(x + self.l2(F.relu(self.l1(x)))))


# --------------------------------------------------------------------------- host side + reference pipeline
def msda(value, shapes, loc, aw):
    """Deformable sampling (HF pytorch path): value (N, 8, 32); loc (Q, 8, L, P, 2); aw (Q, 8, L, P)."""
    from transformers.models.omdet_turbo.modeling_omdet_turbo import MultiScaleDeformableAttention
    ss = torch.tensor(shapes)
    lsi = torch.cat([ss.new_zeros(1), ss.prod(1).cumsum(0)[:-1]])
    return MultiScaleDeformableAttention()(value[None], ss, shapes, lsi, loc[None], aw[None], 64)[0]


def rows(x):
    """(1, C, 1, T) -> (T, C)"""
    return x.reshape(x.shape[1], -1).T


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def cosine_logits(feats, cls):
    """feats (Q, 256), cls (C, 256) -> (Q, C) = cos / 0.07 (OmDet's class_distance_type 'cosine')."""
    a = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = cls / cls.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (a @ b.T) / 0.07


class Text:
    """CLIP tokens -> text-program inputs, and the class / task embeddings from its output."""

    def __init__(self, hf, tokenizer=None):
        from transformers import AutoTokenizer
        self.tok = tokenizer or AutoTokenizer.from_pretrained(HF_ID)
        e = getattr(hf.language_backbone.model, "text_model", hf.language_backbone.model).embeddings
        self.token = e.token_embedding.weight.detach()
        self.position = e.position_embedding.weight.detach()
        self.proj = hf.language_backbone.text_projection.detach()     # (512, 512)
        self.eot = self.tok.eos_token_id

    def ids(self, text):
        ids = self.tok(text)["input_ids"][:TEXT_T]
        if ids[-1] != self.eot:
            ids[-1] = self.eot
        return ids

    def embed(self, seqs):
        """list of id lists -> (1, 512, TEXT_B, TEXT_T) (padding rows/tokens are EOT embeddings)."""
        out = torch.zeros(TEXT_B, TEXT_T, 512)
        for i, ids in enumerate(seqs):
            full = ids + [self.eot] * (TEXT_T - len(ids))
            out[i] = self.token[full] + self.position[:TEXT_T]
        return out.permute(2, 0, 1)[None]

    def class_embedding(self, hidden_row, ids):
        """hidden (TEXT_T, 512) of one class sequence -> (512,) (EOT token @ text_projection)."""
        return hidden_row[ids.index(self.eot)] @ self.proj


class Pipeline(nn.Module):
    """The whole detector from the MLA-form pieces, with the host steps in torch (for checks)."""

    def __init__(self, hf):
        super().__init__()
        self.hf = hf
        self.bb = Backbone(hf)
        self.aifi = Aifi(hf).eval()
        self.ccfm = Ccfm(hf).eval()
        self.decin = DecIn(hf).eval()
        self.pre = nn.ModuleList(DecPre(hf, k) for k in range(len(hf.decoder.layers))).eval()
        self.post = nn.ModuleList(DecPost(hf, k) for k in range(len(hf.decoder.layers))).eval()
        self.text_enc = TextEncoder(hf).eval()
        self.task_head = TaskHead(hf).eval()
        self.text = Text(hf)
        self.anchor, self.valid = anchors()
        d = hf.decoder
        # encoder class features of the tokens with invalid anchors: Linear(0) = bias, then LN
        b = d.encoder_vision_features[0].bias.detach()
        self.cf_invalid = d.encoder_vision_features[1](b[None])[0].detach()

    @torch.no_grad()
    def prompt(self, classes):
        """class names -> class embeddings (C, 512), task tokens (1, 256, 1, TT), task mask (TT,)."""
        task = f"Detect {', '.join(classes)}."
        seqs = [self.text.ids(c) for c in classes] + [self.text.ids(task)]
        hid = []
        for i in range(0, len(seqs), TEXT_B):
            chunk = seqs[i:i + TEXT_B]
            o = self.text_enc(self.text.embed(chunk))                   # (1, 512, B, T)
            hid += [o[0, :, j].T for j in range(len(chunk))]
        cls = torch.stack([self.text.class_embedding(h, ids) for h, ids in zip(hid[:-1], seqs[:-1])])
        n = min(len(seqs[-1]), TT)
        th = torch.zeros(1, 512, 1, TT)
        th[0, :, 0, :n] = hid[-1][:n].T
        task_feats = self.task_head(th)
        km = torch.full((TT,), NEG)
        km[:n] = 0
        return cls, task_feats, km

    @torch.no_grad()
    def forward(self, rgb_u8, classes):
        """rgb (S, S, 3) uint8 (squashed to 640) -> class logits (900, C), boxes (900, 4) cx cy w h."""
        d = self.hf.decoder
        cls, task, km = self.prompt(classes)
        px = torch.from_numpy(np.asarray(rgb_u8, np.float32) / 255).permute(2, 0, 1)
        l0, l1, l2 = self.bb.run_torch(px)
        a = self.aifi(l2.reshape(1, D, 1, -1)).reshape(l2.shape)
        o0, o1, o2 = self.ccfm(l0, l1, a)
        tok = [o.reshape(1, D, 1, -1) for o in (o0, o1, o2)]
        out = self.decin(*tok)
        N = sum(h * w for h, w in level_sizes())
        V = [rows(out[:, :, :, i * N:(i + 1) * N]) for i in range(6)]
        cf = rows(out[:, :, :, 6 * N:7 * N]).clone()
        box = rows(out[:, :, :, 7 * N:8 * N])[:, :4]
        cf[~self.valid] = self.cf_invalid
        # selection
        sim = cosine_logits(cf, d.encoder_class_head(cls))
        top = torch.topk(sim.max(1).values, NQ).indices
        ref_logit = box[top] + self.anchor[top]
        ref_logit[~self.valid[top]] = float("inf")
        ref = ref_logit.sigmoid()
        H = S3.tok(cf[top])
        TASK = task
        shapes = level_sizes()
        for k in range(len(d.layers)):
            REF = torch.zeros(1, D, 1, NQ)
            REF[0, :4, 0] = ref.T
            KM = torch.zeros(1, D, 1, 1)
            KM[0, :TT, 0, 0] = km
            o = self.pre[k](torch.cat([H, REF, TASK, KM], 3))
            H2, T2, OFF, AW = torch.split(o, [NQ, TT, NQ, NQ], 3)
            off = rows(OFF)[:, :192].view(NQ, 8, 3, 4, 2)
            aw = torch.softmax(rows(AW)[:, :96].view(NQ, 8, 12), -1).view(NQ, 8, 3, 4)
            loc = ref[:, None, None, None, :2] + off / 4 * ref[:, None, None, None, 2:] * 0.5
            sampled = msda(V[k].view(N, 8, 32), shapes, loc, aw)       # (Q, 256)
            o = self.post[k](torch.cat([H2, S3.tok(sampled)], 3))
            H, B = torch.split(o, [NQ, NQ], 3)
            ref = torch.sigmoid(rows(B)[:, :4] + inverse_sigmoid(ref))
            TASK = T2
        logits = cosine_logits(rows(H), d.decoder_class_head[-1](cls))
        return logits, ref
