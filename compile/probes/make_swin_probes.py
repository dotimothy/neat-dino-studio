#!/usr/bin/env python3
"""Real-weight Swin-B probes (single blocks / patch merge) for MLA compile checks."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "compile" / "probes"))
import dino_mla as M  # noqa: E402
from make_probes import export  # noqa: E402

from transformers import AutoModelForZeroShotObjectDetection  # noqa: E402

hf = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval()
sw = hf.model.backbone.conv_encoder.model.encoder.layers
rng = np.random.default_rng(0)
class BareWindowAttn(torch.nn.Module):
    """q/k/v straight from convs on a (1, C, nW, 144) input: the Swin attention alone."""

    def __init__(self, blk):
        super().__init__()
        self.b = blk

    def forward(self, x):
        b = self.b
        q, k, v = b.q(x), b.k(x), b.v(x)
        outs = []
        for h in range(b.heads):
            sl = slice(h * b.hd, (h + 1) * b.hd)
            s = torch.einsum("nchw,nchq->nqhw", q[:, sl], k[:, sl]) + b.bias[h]
            if b.mask is not None:
                s = s + b.mask
            outs.append(torch.einsum("nchw,nqhc->nqhw", torch.softmax(s, 1), v[:, sl]))
        return b.o(torch.cat(outs, 1))


cases = {
    "wattn_s3": (BareWindowAttn(M.SwinBlock(sw[2].blocks[1], 36, 36)), (1, 512, 9, 144)),
    "swin_s1_shift": (M.SwinBlock(sw[0].blocks[1], 144, 144), (1, 128, 144, 144)),
    "swin_s1_plain": (M.SwinBlock(sw[0].blocks[0], 144, 144), (1, 128, 144, 144)),
    "swin_s3_shift": (M.SwinBlock(sw[2].blocks[1], 36, 36), (1, 512, 36, 36)),
    "swin_s4_pad": (M.SwinBlock(sw[3].blocks[1], 18, 18), (1, 1024, 18, 18)),
    "pmerge_s1": (M.PatchMerge(sw[0].downsample), (1, 128, 144, 144)),
}
want = sys.argv[1:] or list(cases)
for name in want:
    mod, shape = cases[name]
    export(name, mod, rng.standard_normal(shape).astype(np.float32))
