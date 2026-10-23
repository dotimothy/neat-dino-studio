#!/usr/bin/env python3
"""Window-layout Swin block probes (input already partitioned) at 576."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "compile" / "probes"))
import dino_mla as M  # noqa: E402
from make_probes import export  # noqa: E402
from transformers import AutoModelForZeroShotObjectDetection  # noqa: E402

hf = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval()
sw = hf.model.backbone.conv_encoder.model
L = sw.encoder.layers
rng = np.random.default_rng(0)
cases = {
    "sw_s1_b0": (M.SwinBlockW(L[0].blocks[0], 144, 144, embed=M.PatchEmbed(sw)), (1, 48, 144, 144)),
    "sw_s1_b1": (M.SwinBlockW(L[0].blocks[1], 144, 144), (1, 128, 144, 144)),
    "sw_s2_b0": (M.SwinBlockW(L[1].blocks[0], 72, 72, merge=M.PatchMerge(L[0].downsample)), (1, 512, 36, 144)),
    "sw_s3_b1": (M.SwinBlockW(L[2].blocks[1], 36, 36), (1, 512, 9, 144)),
    "sw_s4_b1": (M.SwinBlockW(L[3].blocks[1], 18, 18), (1, 1024, 4, 144)),
}
for name in sys.argv[1:] or list(cases):
    mod, shape = cases[name]
    export(name, mod, rng.standard_normal(shape).astype(np.float32))
