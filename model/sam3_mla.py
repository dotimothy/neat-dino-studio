"""SAM 3 (facebook/sam3) rebuilt in MLA-friendly ops, starting with the Perception Encoder ViT.

Tokens use the layout of the Grounding DINO port: (1, C, H, W) with channels on C, a window
per row of H and the 24x24 tokens of a window along W (row-major). Windowed layers attend
within each row; the four global layers see the same buffer as (1, C, 1, H*W): in the MLA's
channel-last memory those are the same bytes, so the runtime just hands the next program the
other shape (no copy, no reshape on the MLA).

Rotary position embedding (axial 2-D, pairwise rotation) is folded into the weights: the
pairwise rotation is a fixed signed permutation P of each head's channels, so rot(Wq x) =
(P Wq) x is just another 1x1 conv, and q_rope = q * cos + q_rot * sin with full-shape
constant tables (no broadcast multiplies).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_from_linear(lin: nn.Linear, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None,
                     scale: float = 1.0) -> nn.Conv2d:
    w = (lin.weight if weight is None else weight).detach() * scale
    b = lin.bias if bias is None else bias
    c = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=b is not None)
    c.weight.data.copy_(w[:, :, None, None])
    if b is not None:
        c.bias.data.copy_(b.detach() * scale)
    return c


class LN(nn.Module):
    """LayerNorm over C of (1, C, H, W)."""

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.eps = ln.eps
        self.register_buffer("w", ln.weight.detach().view(1, -1, 1, 1).clone())
        self.register_buffer("b", ln.bias.detach().view(1, -1, 1, 1).clone())

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        d = x - mu
        var = (d * d).mean(1, keepdim=True)
        return d * torch.pow(var + self.eps, -0.5) * self.w + self.b


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


def pair_rotation(dim: int) -> torch.Tensor:
    """(dim, dim) matrix R with R @ x == rotate_pairwise(x): out[2i] = -x[2i+1], out[2i+1] = x[2i]."""
    r = torch.zeros(dim, dim)
    for i in range(0, dim, 2):
        r[i, i + 1] = -1.0
        r[i + 1, i] = 1.0
    return r


def rope_tables(layer, grid_hw: tuple[int, int], heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin per token for one HF Sam3ViTLayer at a token grid, as (heads*hd, tokens), row-major."""
    cfg = layer.config
    gh, gw = grid_hw
    hpos, wpos = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    pos = torch.stack([wpos.flatten(), hpos.flatten()], dim=-1).float()
    pos = pos * (cfg.window_size / gw)
    cos, sin = layer.rotary_emb(torch.zeros(1), pos)          # (1, tokens, hd)
    cos, sin = cos[0].T, sin[0].T                              # (hd, tokens)
    return cos.repeat(heads, 1), sin.repeat(heads, 1)          # same for every head


class ViTLayerW(nn.Module):
    """One Sam3ViTLayer on (1, C, nW, T) tokens. Windowed: nW windows of T = ws*ws tokens.
    Global: nW = 1 and T = all tokens (row-major over the window layout, see rope)."""

    def __init__(self, layer, n_windows: int, grid_hw: tuple[int, int], token_order=None):
        super().__init__()
        att = layer.attention
        self.heads, self.hd = att.num_attention_heads, att.head_dim
        C = att.hidden_size
        self.ln1, self.ln2 = LN(layer.layer_norm1), LN(layer.layer_norm2)
        rot = torch.block_diag(*[pair_rotation(self.hd)] * self.heads)
        s = att.scaling
        self.q = conv_from_linear(att.q_proj, scale=s)
        self.qr = conv_from_linear(att.q_proj, rot @ att.q_proj.weight, rot @ att.q_proj.bias, scale=s)
        self.k = conv_from_linear(att.k_proj)
        self.kr = conv_from_linear(att.k_proj, rot @ att.k_proj.weight, rot @ att.k_proj.bias)
        self.v = conv_from_linear(att.v_proj)
        self.o = conv_from_linear(att.o_proj)
        self.fc1 = conv_from_linear(layer.mlp.fc1)
        self.fc2 = conv_from_linear(layer.mlp.fc2)
        cos, sin = rope_tables(layer, grid_hw, self.heads)     # (C, T)
        if token_order is not None:                            # global layer on window-layout tokens
            cos, sin = cos[:, token_order], sin[:, token_order]
        T = cos.shape[1]
        self.register_buffer("cos", cos.view(1, C, 1, T).expand(1, C, n_windows, T).contiguous())
        self.register_buffer("sin", sin.view(1, C, 1, T).expand(1, C, n_windows, T).contiguous())

    def forward(self, x):
        y = self.ln1(x)
        q = self.q(y) * self.cos + self.qr(y) * self.sin
        k = self.k(y) * self.cos + self.kr(y) * self.sin
        v = self.v(y)
        outs = []
        for h in range(self.heads):
            sl = slice(h * self.hd, (h + 1) * self.hd)
            s = torch.einsum("nchw,nchq->nqhw", q[:, sl], k[:, sl])          # (1, Tk, nW, Tq)
            outs.append(torch.einsum("nchw,nqhc->nqhw", torch.softmax(s, dim=1), v[:, sl]))
        x = x + self.o(torch.cat(outs, 1))
        return x + self.fc2(gelu(self.fc1(self.ln2(x))))


def window_order(gh: int, gw: int, ws: int) -> torch.Tensor:
    """Spatial index (row-major) of each token in window layout (window-major, row-major inside)."""
    idx = torch.arange(gh * gw).view(gh // ws, ws, gw // ws, ws).permute(0, 2, 1, 3)
    return idx.reshape(-1)
