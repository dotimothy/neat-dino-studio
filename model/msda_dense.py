"""Deformable-attention sampling for the small (coarse) feature levels, done densely on the MLA.

For a level of Hl x Wl positions, bilinear sampling with zero padding is exactly
    sample(x, y) = sum_ij hat(y - i) * hat(x - j) * V[i, j],  hat(d) = relu(1 - |d|)
               = sum_ij [relu(d+1) - 2 relu(d) + relu(d-1)]...
so the weighted sum over points becomes an attention matrix A[q, (i, j)] built from ReLUs and an
outer product, applied with one Einsum per head. Pixel coordinates come from a 1x1 conv over
[q | R]: R holds each token's reference point in level pixels, so the conv computes
offset + ref - j directly (no broadcast). Cost grows with Hl * Wl, so it is only for levels 2-3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import dino_mla as M

NH, NL, NP = 8, 4, 4


def hat(d):
    return F.relu(d + 1) - 2 * F.relu(d) + F.relu(d - 1)


class DenseLevels(nn.Module):
    """Levels `levels` of one deformable attention (encoder: 2-d refs) for all queries.

    forward(q (1,256,1,N), R (1,256,1,N), V (1,256,1,N), probs (1,128,1,N)) -> (1,256,1,N)
    R channels: 4*k + 0/1 = ref_x * W - 0.5 / ref_y * H - 0.5 for the k-th entry of `levels`.
    """

    def __init__(self, off_lin: nn.Linear, shapes, starts, levels=(2, 3)):
        super().__init__()
        self.levels, self.shapes, self.starts = levels, shapes, starts
        W, b = off_lin.weight.detach(), off_lin.bias.detach()        # (256 out: h,l,p,xy) x 256
        self.cx, self.cy = nn.ModuleList(), nn.ModuleList()
        for k, l in enumerate(levels):
            Hl, Wl = shapes[l]
            for axis, n, mods in ((0, Wl, self.cx), (1, Hl, self.cy)):
                conv = nn.Conv2d(512, NH * NP * n, 1)
                w = torch.zeros(NH * NP * n, 512)
                bb = torch.zeros(NH * NP * n)
                for h in range(NH):
                    for p in range(NP):
                        row = ((h * NL + l) * NP + p) * 2 + axis
                        for j in range(n):
                            o = (h * NP + p) * n + j
                            w[o, :256] = W[row]
                            w[o, 256 + 4 * k + axis] = 1.0
                            bb[o] = b[row] - j
                conv.weight.data.copy_(w[:, :, None, None])
                conv.bias.data.copy_(bb)
                mods.append(conv)

    def forward(self, q, R, V, probs):
        qr = torch.cat([q, R], 1)
        out = None
        for k, l in enumerate(self.levels):
            Hl, Wl = self.shapes[l]
            K = Hl * Wl
            Vl = V[:, :, :, self.starts[l]:self.starts[l] + K]          # (1, 256, 1, K)
            hx, hy = hat(self.cx[k](qr)), hat(self.cy[k](qr))           # (1, 32*Wl, 1, N), (1, 32*Hl, 1, N)
            heads = []
            for h in range(NH):
                A = None
                for p in range(NP):
                    c = h * NP + p
                    a = probs[:, h * 16 + l * NP + p: h * 16 + l * NP + p + 1]          # (1, 1, 1, N)
                    wx = hx[:, c * Wl:(c + 1) * Wl] * a.expand(1, Wl, 1, a.shape[3])    # weight folded in
                    wy = hy[:, c * Hl:(c + 1) * Hl]
                    # outer product (i, j) -> channel i*Wl + j via repeat / tile
                    o = torch.repeat_interleave(wy, Wl, dim=1) * wx.repeat(1, Hl, 1, 1)
                    A = o if A is None else A + o
                heads.append(torch.einsum("nchw,nqhc->nqhw", A, Vl[:, h * 32:(h + 1) * 32]))
            s = torch.cat(heads, 1)
            out = s if out is None else out + s
        return out


def _ones_conv(cin, cout, fn):
    """1x1 conv with 0/1 weights: output channel o copies input channel fn(o)."""
    c = nn.Conv2d(cin, cout, 1, bias=False)
    w = torch.zeros(cout, cin)
    for o in range(cout):
        w[o, fn(o)] = 1.0
    c.weight.data.copy_(w[:, :, None, None])
    return c


class DenseLevelsMLA(DenseLevels):
    """Same math as DenseLevels with MLA ops only: replication by 0/1 convs, no Tile/Expand."""

    def __init__(self, off_lin, shapes, starts, levels=(2, 3)):
        super().__init__(off_lin, shapes, starts, levels)
        self.rep_y, self.rep_x, self.rep_a = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for l in levels:
            Hl, Wl = shapes[l]
            self.rep_y.append(_ones_conv(Hl, Hl * Wl, lambda o, W=Wl: o // W))
            self.rep_x.append(_ones_conv(Wl, Hl * Wl, lambda o, W=Wl: o % W))
            self.rep_a.append(_ones_conv(1, Wl, lambda o: 0))

    def forward(self, q, R, V, probs):
        qr = torch.cat([q, R], 1)
        out = None
        for k, l in enumerate(self.levels):
            Hl, Wl = self.shapes[l]
            K = Hl * Wl
            Vl = V[:, :, :, self.starts[l]:self.starts[l] + K]
            hx, hy = hat(self.cx[k](qr)), hat(self.cy[k](qr))
            heads = []
            for h in range(NH):
                A = None
                for p in range(NP):
                    c = h * NP + p
                    a = probs[:, h * 16 + l * NP + p: h * 16 + l * NP + p + 1]
                    wx = hx[:, c * Wl:(c + 1) * Wl] * self.rep_a[k](a)
                    o = self.rep_y[k](hy[:, c * Hl:(c + 1) * Hl]) * self.rep_x[k](wx)
                    A = o if A is None else A + o
                heads.append(torch.einsum("nchw,nqhc->nqhw", A, Vl[:, h * 32:(h + 1) * 32]))
            s = torch.cat(heads, 1)
            out = s if out is None else out + s
        return out
