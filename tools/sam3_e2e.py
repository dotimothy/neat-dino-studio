#!/usr/bin/env python3
"""End-to-end check of sam3_rt on the board against the fp32 PyTorch pipeline (model/sam3_mla.py).

  host:  .venv-sam3/bin/python tools/sam3_e2e.py prepare --size 672 --image X --prompt "cat" --out build/e2e/sam3_cat.npz
  board: python3 tools/sam3_e2e.py run --case sam3_cat.npz --bundle B --rt sam3_rt
  host:  .venv/bin/python tools/sam3_e2e.py compare --case build/e2e/sam3_cat.npz --board build/e2e/sam3_cat_board.npz
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
    from PIL import Image
    sys.path.insert(0, str(ROOT / "model"))
    import sam3_mla as S
    from transformers import AutoTokenizer, Sam3Config, Sam3Model
    cfg = Sam3Config.from_pretrained("facebook/sam3")
    cfg.vision_config.backbone_config.image_size = a.size
    hf = Sam3Model.from_pretrained("facebook/sam3", config=cfg).eval()
    img = Image.open(a.image).convert("RGB").resize((a.size, a.size), Image.BILINEAR)
    rgb = np.asarray(img, np.uint8)
    px = torch.from_numpy((rgb.astype(np.float32) / 255 - 0.5) / 0.5).permute(2, 0, 1)
    t = AutoTokenizer.from_pretrained("facebook/sam3")(a.prompt, padding="max_length", max_length=32,
                                                       truncation=True, return_tensors="pt")
    with torch.no_grad():
        o = S.Pipeline(hf, a.size).eval()(px, t.input_ids[0], int(t.attention_mask.sum()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, rgb=rgb, prompt=np.array(a.prompt), logits=o["logits"].numpy(), presence=float(o["presence"]),
             boxes=o["boxes"].numpy(), masks=(o["masks"] > 0).numpy())
    print(f"wrote {a.out}")


def run(a):
    from dino.sam3_client import Sam3Runtime
    z = np.load(a.case)
    t0 = time.time()
    rt = Sam3Runtime(a.rt, a.bundle, log=lambda s: None)
    print(f"loaded in {time.time() - t0:.1f} s; {rt.info()}")
    print(f"text {rt.set_text(str(z['prompt'])):.1f} ms")
    res = None
    for i in range(a.iters):
        res = rt.detect(z["rgb"], 0.3)
        print(f"  run {i}: " + " ".join(f"{k}={v:.1f}" for k, v in res.timing.items() if v))
    c = rt.detect_cached(0.3)
    print("  cached: " + " ".join(f"{k}={v:.1f}" for k, v in c.timing.items() if v))
    rt.close()
    q = sorted(res.masks)
    out = a.case.replace(".npz", "_board.npz")
    np.savez(out, logits=res.logits, presence=res.presence, boxes=res.boxes, mask_q=np.array(q, int),
             masks=np.stack([res.masks[k] for k in q]) if q else np.zeros((0, 1, 1), bool))
    print("saved", out)


def compare(a):
    z, b = np.load(a.case), np.load(a.board)
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    rs = sig(z["logits"]) * sig(z["presence"])
    bs = sig(b["logits"]) * sig(b["presence"])
    print(f"presence ref {float(z['presence']):.3f} board {float(b['presence']):.3f}")
    print("top ref  ", np.round(np.sort(rs)[::-1][:8], 3))
    print("top board", np.round(np.sort(bs)[::-1][:8], 3))
    keep = np.nonzero(rs > 0.3)[0]
    print(f"found: ref {len(keep)}, board {int((bs > 0.3).sum())}")
    for qi in keep:
        rb, bb = z["boxes"][qi], b["boxes"][qi]
        x = lambda v: np.r_[v[:2] - v[2:] / 2, v[:2] + v[2:] / 2]  # noqa: E731
        r_, b_ = x(rb), x(bb)
        inter = np.clip(np.minimum(r_[2:], b_[2:]) - np.maximum(r_[:2], b_[:2]), 0, None).prod()
        iou = inter / ((r_[2:] - r_[:2]).prod() + (b_[2:] - b_[:2]).prod() - inter)
        mq = list(b["mask_q"])
        miou = float("nan")
        if qi in mq:
            bm, rm = b["masks"][mq.index(qi)], z["masks"][qi]
            miou = (bm & rm).sum() / max(1, (bm | rm).sum())
        print(f"  query {qi:3d}: score ref {rs[qi]:.3f} board {bs[qi]:.3f}  box IoU {iou:.3f}  mask IoU {miou:.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--size", type=int, default=672)
    p.add_argument("--image")
    p.add_argument("--prompt")
    p.add_argument("--out")
    r = sub.add_parser("run")
    r.add_argument("--case")
    r.add_argument("--bundle")
    r.add_argument("--rt")
    r.add_argument("--iters", type=int, default=3)
    c = sub.add_parser("compare")
    c.add_argument("--case")
    c.add_argument("--board")
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
