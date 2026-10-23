#!/usr/bin/env python3
"""Bisect the enc6 compile failure with variants of the stage."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "compile" / "probes"))
import dino_mla as M  # noqa: E402
from make_probes import export  # noqa: E402
from transformers import AutoModelForZeroShotObjectDetection  # noqa: E402

hf = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval()
S = 576
N = M.geometry(hf, S).N
base = M.EncStage(hf, 6, N)
x = np.load(ROOT / "build/probes/s576_enc6_in.npy")


class V(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.b, self.mode = base, mode

    def forward(self, x):
        b = self.b
        X, S_, P, TP = M.wsplit(x, [N, N, N, M.TP_W])
        X = b.post(X, S_)
        t = M.unpack_tp(TP)[0]
        if self.mode == "post_only":
            return torch.cat([X, t], 3)
        oq = b.enc_ln(b.enc_out(X))
        if self.mode.startswith("chunk"):
            k = int(self.mode[5:])
            logits = torch.einsum("nchw,nchq->nqhw", oq, t)
            outs = []
            for c in torch.chunk(oq, k, dim=3):
                h = c
                for i, l in enumerate(b.bbox):
                    h = l(h)
                    if i < len(b.bbox) - 1:
                        h = F.relu(h)
                outs.append(h)
            return torch.cat([X, t, torch.cat([logits, torch.cat(outs, 3)], 1)], 3)
        if self.mode in ("wcat", "headfirst"):
            t256 = torch.cat([t, torch.zeros(1, M.D, 1, M.D - M.T_TXT)], 3)   # pad text to 256 tokens
            L = torch.einsum("nchw,nchq->nqhw", oq, t256)                   # (1, 256, 1, N)
            h = oq
            for i, l in enumerate(b.bbox):
                h = l(h)
                if i < len(b.bbox) - 1:
                    h = F.relu(h)
            B = torch.cat([h, torch.zeros(1, M.D - h.shape[1], 1, N)], 1) if h.shape[1] < M.D else h
            if self.mode == "wcat":
                return torch.cat([X, t, L, B], 3)
            return torch.cat([torch.cat([L[:, :M.T_TXT], h], 1), X, t], 3)
        if self.mode == "bb_only":
            h = oq
            for i, l in enumerate(b.bbox):
                h = l(h)
                if i < len(b.bbox) - 1:
                    h = F.relu(h)
            return torch.cat([X, t, h[:, :192].repeat(1, 1, 1, 1) if False else torch.cat([h, torch.zeros(1, 64, 1, N)], 1)], 3)
        parts = []
        if self.mode in ("no_bbox", "full"):
            parts.append(torch.einsum("nchw,nchq->nqhw", oq, t))
        if self.mode in ("no_einsum", "full"):
            h = oq
            for i, l in enumerate(b.bbox):
                h = l(h)
                if i < len(b.bbox) - 1:
                    h = F.relu(h)
            parts.append(h)
        used = sum(p.shape[1] for p in parts)
        head = torch.cat(parts + [torch.zeros(1, M.D - used, 1, N)], 1)
        return torch.cat([X, t, head], 3)


for mode in sys.argv[1:] or ["post_only", "no_einsum", "no_bbox"]:
    export(f"e6_{mode}", V(mode), x)
