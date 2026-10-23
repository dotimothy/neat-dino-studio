#!/usr/bin/env python3
"""Probe: coarse-level (2-3) deformable sampling of one encoder layer, dense on the MLA."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "compile" / "probes"))
import dino_mla as M  # noqa: E402
import msda_dense as MD  # noqa: E402
from make_probes import export  # noqa: E402
from transformers import AutoModelForZeroShotObjectDetection  # noqa: E402

hf = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval()


class Probe(nn.Module):
    """[X | P | R] -> sampled levels 2-3 (1, 256, 1, N)"""

    def __init__(self, S):
        super().__init__()
        g = M.geometry(hf, S)
        self.N = g.N
        starts = [0]
        for h, w in g.shapes[:-1]:
            starts.append(starts[-1] + h * w)
        da = hf.model.encoder.layers[1].deformable_layer.self_attn
        self.value = M.conv_from_linear(da.value_proj)
        self.aw = M.conv_from_linear(da.attention_weights)
        self.dense = MD.DenseLevelsMLA(da.sampling_offsets, g.shapes, starts)

    def forward(self, x):
        X, P, R = M.wsplit(x, [self.N] * 3)
        q = X + P
        probs = M.h_to_heads(torch.softmax(M.heads_to_h(self.aw(q), 16), 1))
        return self.dense(q, R, self.value(X), probs)


for S in [int(s) for s in (sys.argv[1:] or ["576"])]:
    g = M.geometry(hf, S)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1, 256, 1, g.N)).astype(np.float32)
    R = np.zeros_like(X)
    for k, l in enumerate((2, 3)):
        Hl, Wl = g.shapes[l]
        R[0, 4 * k, 0] = g.ref[:, 0].numpy() * Wl - 0.5
        R[0, 4 * k + 1, 0] = g.ref[:, 1].numpy() * Hl - 0.5
    x = np.concatenate([X, g.pos.numpy(), R], 3)
    export(f"dense23_{S}", Probe(S).eval(), x)
