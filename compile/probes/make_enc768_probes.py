#!/usr/bin/env python3
"""Which part of encoder stage 0 fails the compiler's check at 768 (N = 12240)?"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "compile" / "probes"))
import dino_mla as M  # noqa: E402
from make_probes import export  # noqa: E402
from transformers import AutoModelForZeroShotObjectDetection  # noqa: E402

hf = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval()
N = 12240
pre = M.EncLayerPre(hf.model.encoder.layers[0]).eval()
x = np.load(ROOT / "build/probes/s768_enc0_in.npy")          # [X | P | TP]


class Part(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.pre = pre

    def forward(self, x):
        pre = self.pre
        X, P, TP = M.wsplit(x, [N, N, M.TP_W])
        t, tpos, sm, km = M.unpack_tp(TP)
        b = pre.bi
        Xn, tn = pre.lnv(X), pre.lnt(t)
        qv = M.heads_to_h(b.vq(Xn), b.hd)
        kt = M.heads_to_h(b.tk(tn), b.hd)
        if self.mode == "vt":       # vision -> text only
            vt = M.heads_to_h(b.tv(tn), b.hd)
            s = torch.einsum("nchw,nchq->nqhw", qv, kt) + km
            return b.vo(M.h_to_heads(torch.einsum("nchw,nqhc->nqhw", torch.softmax(s, 1), vt)))
        if self.mode == "tv":       # text -> vision only
            vv = M.heads_to_h(b.vv(Xn), b.hd)
            s = torch.einsum("nchw,nchq->nqhw", kt, qv)
            return b.tout(M.h_to_heads(torch.einsum("nchw,nqhc->nqhw", torch.softmax(s, 1), vv)))
        if self.mode == "tv_scores":  # text -> vision softmax only (output (1, N, 4, 64) -> sum over heads)
            s = torch.einsum("nchw,nchq->nqhw", kt, qv)
            return torch.softmax(s, 1)
        if self.mode == "proj":     # deformable projections only
            q = Xn + P
            return torch.cat([pre.value(Xn), pre.off(q), pre.aw(q)], 3)


for mode in sys.argv[1:] or ["vt", "tv", "tv_scores", "proj"]:
    export(f"e768_{mode}", Part(mode).eval(), x)
