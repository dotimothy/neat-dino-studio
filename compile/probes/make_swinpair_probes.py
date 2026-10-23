#!/usr/bin/env python3
"""Two Swin blocks in one MLA program, with the plain -> shifted window move done by Slice+Concat."""
import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "compile" / "probes"))
import dino_mla as M  # noqa: E402
import swin_shift_mla as SS  # noqa: E402
from make_probes import export  # noqa: E402
from transformers import AutoModelForZeroShotObjectDetection  # noqa: E402

hf = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval()
L = hf.model.backbone.conv_encoder.model.encoder.layers


class Pair(nn.Module):
    def __init__(self, s, b, H):
        super().__init__()
        self.b0 = M.SwinBlockW(L[s].blocks[b], H, H)
        self.b1 = M.SwinBlockW(L[s].blocks[b + 1], H, H)
        self.n = -(-H // 12)

    def forward(self, x):                       # x in plain layout, out in plain layout
        y = self.b1(SS.plain_to_shifted(self.b0(x), self.n, self.n))
        return SS.shifted_to_plain(y, self.n, self.n)


rng = np.random.default_rng(0)
cases = {"pair_s3_576": (Pair(2, 2, 36), (1, 512, 9, 144)), "pair_s1_576": (Pair(0, 0, 144), (1, 128, 144, 144))}
for name in sys.argv[1:] or list(cases):
    m, shape = cases[name]
    export(name, m.eval(), rng.standard_normal(shape).astype(np.float32))
