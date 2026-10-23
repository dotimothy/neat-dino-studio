#!/usr/bin/env python3
"""End-to-end check of dino_rt on the board against the torch reference pipeline.

  host:  .venv/bin/python tools/e2e_check.py prepare --size 576 --image X --prompt "..." --out build/e2e/case.npz
  board: python3 tools/e2e_check.py run --case case.npz --bundle B --text-bundle TB --rt dino_rt [--tokens]
  host:  .venv/bin/python tools/e2e_check.py compare --case build/e2e/case.npz --board build/e2e/case_board.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(a):
    import torch
    sys.path.insert(0, str(ROOT / "model"))
    import dino_mla as M
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection
    from dino import prompt as P
    from dino.letterbox import letterbox
    hf = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval()
    pipe = M.Pipeline(hf, a.size)
    pr = P.build(a.prompt, P.Tokenizer())
    img = Image.open(a.image).convert("RGB")
    px, lb = letterbox(img, a.size)
    with torch.no_grad():
        logits, boxes = pipe(M.space_to_depth4(px), pr)
    tokens = M.tok_to_rows(pipe.trace["backbone"]).numpy()
    sq = np.zeros((a.size, a.size, 3), np.uint8)
    sq[: lb.new_h, : lb.new_w] = np.asarray(img.resize((lb.new_w, lb.new_h), Image.BILINEAR))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, tokens=tokens, logits=logits.numpy(), boxes=boxes.numpy(), rgb=sq,
             valid=np.array([lb.new_w, lb.new_h]), prompt=np.array(a.prompt), size=np.array(a.size))
    print(f"wrote {a.out}: tokens {tokens.shape}")


def run(a):
    from dino.runtime_client import DinoRuntime, TextEncoderHost
    z = np.load(a.case)
    te = TextEncoderHost(a.text_bundle)
    pr = te.build(str(z["prompt"]))
    t0 = time.time()
    rt = DinoRuntime(a.rt, a.bundle, a.text_bundle, log=lambda s: print("  rt:", s))
    print(f"runtime ready in {time.time() - t0:.1f} s; {rt.info()}")
    print(f"text {rt.set_text(*te.tensors(pr)):.1f} ms")
    res = []
    for i in range(a.iters):
        if a.tokens:
            d = rt.detect_tokens(z["tokens"])
        else:
            vw, vh = (int(v) for v in z["valid"])
            d = rt.detect(z["rgb"], vw, vh)
        res.append(d)
        print(f"  run {i}: " + " ".join(f"{k}={v:.1f}" for k, v in d.timing.items() if v))
    rt.close()
    out = a.case.replace(".npz", "_board.npz")
    np.savez(out, logits=res[-1].logits, boxes=res[-1].boxes)
    print("saved", out)


def compare(a):
    z, b = np.load(a.case), np.load(a.board)
    sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -60, 60)))
    ref_s, brd_s = sig(z["logits"]).max(1), sig(b["logits"]).max(1)
    print("top ref  ", np.sort(ref_s)[::-1][:8].round(4))
    print("top board", np.sort(brd_s)[::-1][:8].round(4))

    def xyxy(x):
        return np.concatenate([x[:, :2] - x[:, 2:] / 2, x[:, :2] + x[:, 2:] / 2], 1)
    rb, bb = xyxy(z["boxes"]), xyxy(b["boxes"])
    order = np.argsort(-ref_s)[: a.top]
    worst_iou, worst_ds = 1.0, 0.0
    for q in order:
        lt = np.maximum(rb[q, :2], bb[:, :2]); r_ = np.minimum(rb[q, 2:], bb[:, 2:])
        inter = np.clip(r_ - lt, 0, None).prod(1)
        u = (rb[q, 2:] - rb[q, :2]).prod() + (bb[:, 2:] - bb[:, :2]).prod(1) - inter
        iou = inter / np.maximum(u, 1e-12)
        j = int(iou.argmax())
        worst_iou = min(worst_iou, float(iou[j]))
        worst_ds = max(worst_ds, abs(float(ref_s[q] - brd_s[j])))
    print(f"top-{a.top} reference detections: worst matched IoU {worst_iou:.4f}, worst score diff {worst_ds:.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--size", type=int, default=576); p.add_argument("--image")
    p.add_argument("--prompt"); p.add_argument("--out")
    r = sub.add_parser("run"); r.add_argument("--case"); r.add_argument("--bundle"); r.add_argument("--text-bundle")
    r.add_argument("--rt"); r.add_argument("--tokens", action="store_true"); r.add_argument("--iters", type=int, default=3)
    c = sub.add_parser("compare"); c.add_argument("--case"); c.add_argument("--board"); c.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
