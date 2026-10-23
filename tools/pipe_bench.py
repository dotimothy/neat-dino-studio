#!/usr/bin/env python3
"""Pipelined vs one-at-a-time throughput of dino_rt, and a check that both give the same answers.

    python3 tools/pipe_bench.py <detector bundle> <text bundle> <dino_rt> [image ...]

Frames alternate between the given images (two COCO images by default), so a slot mix-up
would show as a mismatch against the one-at-a-time results.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dino.runtime_client import DinoRuntime, TextEncoderHost  # noqa: E402

bundle, text, rt_bin = sys.argv[1:4]
paths = sys.argv[4:] or ["/media/nvme/dino-dev/coco/val2017/000000000139.jpg",
                         "/media/nvme/dino-dev/coco/val2017/000000000285.jpg"]
te = TextEncoderHost(text)
import os  # noqa: E402
rt = DinoRuntime(rt_bin, bundle, text, env={k: v for k, v in os.environ.items() if k.startswith("DINO_")})
S = rt.size
print(f"S={S} slots={rt.slots} load {rt.load_seconds:.1f} s")
rt.set_text(*te.tensors(te.build("person . chair . tv . bear . potted plant .")))
frames = []
for p in paths:
    img = Image.open(p).convert("RGB")
    sc = S / max(img.size)
    nw, nh = round(img.width * sc), round(img.height * sc)
    sq = np.zeros((S, S, 3), np.uint8)
    sq[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR))
    frames.append((sq, nw, nh))

ref = [rt.detect(*f) for f in frames]                       # one at a time, also warms up
n = 12
t = time.perf_counter()
sres = [rt.detect(*frames[i % len(frames)]) for i in range(n)]
serial = (time.perf_counter() - t) * 1e3 / n


def phases(ds):
    keys = [k for k in ds[0].timing if k != "slot"]
    return " ".join(f"{k}={np.median([d.timing[k] for d in ds]):.0f}" for k in keys)


print("one at a time:", phases(sres))

for depth in (2, 3):
    t = time.perf_counter()
    futs, res = [], []
    for i in range(n):
        futs.append((i, rt.detect_async(*frames[i % len(frames)])))
        while len([f for _, f in futs if not f.done()]) >= depth:
            time.sleep(0.002)
    for i, f in futs:
        res.append((i, f.result()))
    pipe = (time.perf_counter() - t) * 1e3 / n
    bad = 0
    for i, d in res:
        r = ref[i % len(frames)]
        same = np.array_equal(d.logits, r.logits) and np.array_equal(d.boxes, r.boxes)
        bad += not same
    slots = sorted({int(d.timing["slot"]) for _, d in res})
    tot = np.median([d.timing["total"] for _, d in res])
    print(f"depth {depth}: {pipe:6.1f} ms/frame ({1000 / pipe:.2f} FPS) vs one at a time {serial:6.1f} ms "
          f"({1000 / serial:.2f} FPS); per-frame runtime latency {tot:.0f} ms; slots used {slots}; "
          f"mismatches {bad}/{n}")
    print("   ", phases([d for _, d in res]))
rt.close()
