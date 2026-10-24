#!/usr/bin/env python3
"""End-to-end check of omdet_rt on the board against HF OmDet-Turbo in fp32.

  host:  .venv-sam3/bin/python tools/omdet_e2e.py prepare --out build/e2e/omdet.npz
  board: python3 tools/omdet_e2e.py run --case omdet.npz --bundle B --rt omdet_rt
  host:  .venv/bin/python tools/omdet_e2e.py compare --case build/e2e/omdet.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [("data/images/cats.jpg", ["cat", "remote control", "couch"]), ("webapp/web/samples/cows.jpg", ["cow", "person"]),
         ("webapp/web/samples/pizza.jpg", ["pizza", "bottle", "cup"]),
         ("webapp/web/samples/boat-dog.jpg", ["dog", "person", "boat", "sunglasses"]),
         ("webapp/web/samples/beach-horses.jpg", ["horse", "person"]), ("webapp/web/samples/bird.jpg", ["yellow bird"])]


def prepare(a):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, OmDetTurboForObjectDetection
    proc = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
    m = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf").eval()
    out = {"n": len(CASES)}
    for i, (path, classes) in enumerate(CASES):
        img = Image.open(ROOT / path).convert("RGB")
        inp = proc(img, text=classes, return_tensors="pt", max_length=64)
        with torch.no_grad():
            o = m(**inp)
        out[f"rgb{i}"] = np.asarray(img.resize((640, 640), Image.BILINEAR), np.uint8)
        out[f"classes{i}"] = np.array("|".join(classes))
        out[f"logits{i}"] = o.decoder_class_logits[0].numpy()
        out[f"boxes{i}"] = o.decoder_coord_logits[0].numpy()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out)


def run(a):
    from dino.omdet_client import OmdetRuntime
    z = np.load(a.case)
    t0 = time.time()
    rt = OmdetRuntime(a.rt, a.bundle)
    print(f"loaded in {time.time() - t0:.1f} s; {rt.info()}")
    res = {}
    for i in range(int(z["n"])):
        classes = str(z[f"classes{i}"]).split("|")
        tms = rt.set_classes(classes)
        for it in range(a.iters):
            r = rt.detect(z[f"rgb{i}"])
        c = rt.detect_cached()
        print(f"case {i}: text {tms:.1f} ms  " + " ".join(f"{k}={v:.1f}" for k, v in r.timing.items() if v) +
              f"  | cached total={c.timing['total']:.1f}")
        res[f"logits{i}"], res[f"boxes{i}"] = r.logits, r.boxes
    rt.close()
    out = a.case.replace(".npz", "_board.npz")
    np.savez_compressed(out, **res)
    print("saved", out)


def dets(logits, boxes, thr=0.3):
    sc = 1 / (1 + np.exp(-logits))
    q, c = np.nonzero(sc > thr)
    o = np.argsort(-sc[q, c])
    return [(int(q[j]), int(c[j]), float(sc[q[j], c[j]]), boxes[q[j]]) for j in o]


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    return iw * ih / max(a[2] * a[3] + b[2] * b[3] - iw * ih, 1e-9)


def compare(a):
    z, b = np.load(a.case), np.load(a.case.replace(".npz", "_board.npz"))
    for i in range(int(z["n"])):
        rd, bd = dets(z[f"logits{i}"], z[f"boxes{i}"]), dets(b[f"logits{i}"], b[f"boxes{i}"])
        # match each reference detection to the best board box of the same class
        lines = []
        for _, c, s, bx in rd[:8]:
            cand = [(iou(bx, x[3]), x[2]) for x in bd if x[1] == c]
            best = max(cand) if cand else (0.0, 0.0)
            lines.append(f"{s:.2f}/{best[1]:.2f}@{best[0]:.2f}")
        print(f"case {i} ({str(z[f'classes{i}'])}): found ref {len(rd)} board {len(bd)}   " + "  ".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out")
    r = sub.add_parser("run")
    r.add_argument("--case")
    r.add_argument("--bundle")
    r.add_argument("--rt")
    r.add_argument("--iters", type=int, default=3)
    c = sub.add_parser("compare")
    c.add_argument("--case")
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
