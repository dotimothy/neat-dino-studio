#!/usr/bin/env python3
"""Wall-clock vs runtime-reported time of dino_rt calls (protocol + Python overhead)."""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from dino.runtime_client import DinoRuntime, TextEncoderHost  # noqa: E402

bundle, text, rt_bin = sys.argv[1:4]
te = TextEncoderHost(text)
rt = DinoRuntime(rt_bin, bundle, text)
S = rt.size
rt.set_text(*te.tensors(te.build("person . car . dog .")))
rgb = (np.random.default_rng(0).random((S, S, 3)) * 255).astype(np.uint8)
for mode in ("full", "cached"):
    walls, totals = [], []
    for i in range(8):
        t = time.perf_counter()
        d = rt.detect(rgb, S, S) if mode == "full" or i == 0 else rt.detect_cached()
        walls.append((time.perf_counter() - t) * 1e3)
        totals.append(d.timing["total"])
    print(f"{mode:6s} wall median {np.median(walls[1:]):6.1f} ms   runtime total {np.median(totals[1:]):6.1f} ms   "
          + " ".join(f"{k}={v:.1f}" for k, v in d.timing.items() if v))
rt.close()
