#!/usr/bin/env python3
"""COCO box AP of Grounding DINO on a fixed val2017 subset: board (dino_rt) vs PyTorch reference.

The 80 category names are split into prompts that fit 64 text tokens. For each image the
first prompt runs the whole network; the rest rerun only what depends on the text (the
board keeps the backbone output). Every query gets one detection per category whose phrase
score passes 0.05; the 300 best per image are kept.

  host (reference):  .venv/bin/python eval/coco_eval.py run --backend torch --size 576 --out build/eval/torch576.json
  board:             python3 eval/coco_eval.py run --backend mla --bundle B --text-bundle TB --rt dino_rt \\
                         --images val2017 --ids subset500.txt --out mla576.json
  host (score):      .venv/bin/python eval/coco_eval.py score build/eval/torch576.json build/eval/mla576.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dino import postprocess as PP  # noqa: E402
from dino import prompt as P  # noqa: E402
from dino.letterbox import Letterbox  # noqa: E402

COCO = ROOT / "data" / "coco"
CATS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]
COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def chunks(tok):
    """Split the category list into prompts that each fit 64 tokens."""
    out, rest = [], list(CATS)
    while rest:
        pr = P.build(" . ".join(rest), tok)
        out.append((pr, [CATS.index(p) for p in pr.phrases]))
        rest = rest[len(pr.phrases):]
    return out


def letterbox_rgb(img, S):
    from PIL import Image
    w, h = img.size
    s = S / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    sq = np.zeros((S, S, 3), np.uint8)
    sq[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR))
    return sq, Letterbox(S, s, w, h, nw, nh)


def run(a):
    from PIL import Image
    ids = [int(x) for x in Path(a.ids).read_text().split()][: a.n]
    if a.backend == "torch":
        sys.path.insert(0, str(ROOT / "app"))
        from app.backends import TorchBackend
        be = TorchBackend(a.size)
        tok = be.tokenizer
        set_prompt = be.set_prompt

        def det(rgb, lb, first):
            return be.detect(rgb, lb.new_w, lb.new_h, same_image=not first)
    else:
        from dino.runtime_client import DinoRuntime, TextEncoderHost
        te = TextEncoderHost(a.text_bundle)
        rt = DinoRuntime(a.rt, a.bundle, a.text_bundle)
        tok = te.tok

        def set_prompt(pr):
            rt.set_text(*te.tensors(pr))

        def det(rgb, lb, first):
            return rt.detect(rgb, lb.new_w, lb.new_h) if first else rt.detect_cached()
    prompts = chunks(tok)
    print(f"{len(ids)} images, {len(prompts)} prompts per image", flush=True)
    results, t0, times = [], time.time(), []
    for i, iid in enumerate(ids):
        img = Image.open(Path(a.images) / f"{iid:012d}.jpg").convert("RGB")
        rgb, lb = letterbox_rgb(img, a.size)
        dets = []
        for j, (pr, cats) in enumerate(prompts):
            set_prompt(pr)
            t = time.time()
            d = det(rgb, lb, j == 0)
            times.append((j == 0, time.time() - t))
            for c in PP.candidates(d.logits, d.boxes, pr, lb, floor=0.05):
                for p, ps in enumerate(c["phrase_scores"]):
                    if ps >= 0.05:
                        x0, y0, x1, y1 = c["box"]
                        dets.append({"image_id": iid, "category_id": COCO_IDS[cats[p]], "score": float(ps),
                                     "bbox": [x0, y0, x1 - x0, y1 - y0]})
        dets.sort(key=lambda r: -r["score"])
        results += dets[:300]
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            full = [t for f, t in times if f]
            part = [t for f, t in times if not f]
            print(f"{i + 1}/{len(ids)}  {el:.0f}s  full {np.median(full) * 1e3:.0f} ms  "
                  f"text-only {np.median(part) * 1e3 if part else 0:.0f} ms", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"ids": ids, "size": a.size, "backend": a.backend, "results": results}))
    print("wrote", a.out)
    if a.backend != "torch":
        rt.close()


def score(a):
    from pycocotools.coco import COCO as C
    from pycocotools.cocoeval import COCOeval
    import contextlib
    import io
    gt = C(str(COCO / "annotations" / "instances_val2017.json"))
    for f in a.files:
        d = json.loads(Path(f).read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(d["results"]) if d["results"] else None
            ev = COCOeval(gt, dt, "bbox")
            ev.params.imgIds = d["ids"]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = ev.stats
        print(f"{f}: {d['backend']} {d['size']}px  AP {s[0] * 100:.1f}  AP50 {s[1] * 100:.1f}  AP75 {s[2] * 100:.1f}  "
              f"APs {s[3] * 100:.1f}  APm {s[4] * 100:.1f}  APl {s[5] * 100:.1f}  ({len(d['ids'])} images)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=["torch", "mla"], required=True)
    r.add_argument("--size", type=int, default=576)
    r.add_argument("--images", default=str(COCO / "val2017"))
    r.add_argument("--ids", default=str(COCO / "subset500.txt"))
    r.add_argument("--n", type=int, default=500)
    r.add_argument("--bundle")
    r.add_argument("--text-bundle")
    r.add_argument("--rt")
    r.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)


if __name__ == "__main__":
    main()
