#!/usr/bin/env python3
"""End-to-end check of esam_rt on the board against upstream EfficientSAM in fp32 (PyTorch).

  host:  .venv-sam3/bin/python tools/esam_e2e.py prepare --variant ti --out build/e2e/esam_ti.npz
         (boxes from Grounding DINO Tiny (transformers) on the sample images)
  board: python3 tools/esam_e2e.py run --case esam_ti.npz --bundle B --rt esam_rt
  host:  .venv/bin/python tools/esam_e2e.py compare --case build/e2e/esam_ti.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [("data/images/cats.jpg", "cat . remote control ."), ("webapp/web/samples/cows.jpg", "cow ."),
         ("webapp/web/samples/pizza.jpg", "pizza . bottle . cup ."),
         ("webapp/web/samples/boat-dog.jpg", "dog . person . boat . sunglasses ."),
         ("webapp/web/samples/beach-horses.jpg", "horse . person ."), ("webapp/web/samples/bird.jpg", "yellow bird .")]


def prepare(a):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    sys.path.insert(0, str(ROOT / "model"))
    import esam_mla as E
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(dev).eval()
    sam = E.load_esam(a.variant, a.size).to(dev)
    S = a.size
    out = {}
    for i, (path, text) in enumerate(CASES):
        img = Image.open(ROOT / path).convert("RGB")
        W, H = img.size
        inp = proc(images=img, text=text, return_tensors="pt").to(dev)
        with torch.no_grad():
            r = proc.post_process_grounded_object_detection(gd(**inp), inp.input_ids, threshold=0.35, text_threshold=0.25,
                                                            target_sizes=[(H, W)])[0]
        boxes = r["boxes"].cpu() * torch.tensor([S / W, S / H, S / W, S / H])           # in the S x S image
        rgb = np.asarray(img.resize((S, S), Image.BILINEAR), np.uint8)
        x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            emb = sam.image_encoder((x - sam.pixel_mean) / sam.pixel_std)
            lab = torch.tensor([2, 3], device=dev).view(1, 1, 2).expand(1, len(boxes), 2)
            lg, iou = sam.predict_masks(emb, boxes.view(1, -1, 2, 2).to(dev), lab, True, S, S)  # (1, Q, 3, M, M)
            best = iou[0].argmax(1)
            q = torch.arange(len(boxes))
            m = lg[0, q, best][:, None]
            m = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)[:, 0] > 0
        out[f"rgb{i}"], out[f"boxes{i}"] = rgb, boxes.numpy().astype(np.float32)
        out[f"mask{i}"], out[f"iou{i}"] = m.cpu().numpy(), iou[0, q, best].cpu().numpy()
        print(f"{path}: {len(boxes)} boxes")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, n=len(CASES), **out)
    print("wrote", a.out)


def run(a):
    from dino.esam_client import EsamRuntime
    z = np.load(a.case)
    t0 = time.time()
    rt = EsamRuntime(a.rt, a.bundle)
    print(f"loaded in {time.time() - t0:.1f} s; {rt.info()}")
    res = {}
    for i in range(int(z["n"])):
        rgb, boxes = z[f"rgb{i}"], z[f"boxes{i}"]
        for it in range(a.iters):
            te = rt.encode(rgb)
            m, iou, tm = rt.masks(boxes)
        print(f"case {i}: {len(boxes)} boxes  encode " + " ".join(f"{k}={v:.1f}" for k, v in te.items()) +
              "  masks " + " ".join(f"{k}={v:.1f}" for k, v in tm.items()))
        res[f"mask{i}"], res[f"iou{i}"] = m, iou
    for n in (1, 2, 4, 8, 12, 16):                                  # decoder cost by box count
        b = np.tile(z["boxes1"][:1], (n, 1))
        ts = [rt.masks(b)[2]["total"] for _ in range(5)]
        print(f"  {n:2d} boxes: masks {np.median(ts):.1f} ms")
    rt.close()
    out = a.case.replace(".npz", "_board.npz")
    np.savez_compressed(out, **res)
    print("saved", out)


def compare(a):
    z, b = np.load(a.case), np.load(a.case.replace(".npz", "_board.npz"))
    alls = []
    for i in range(int(z["n"])):
        rm, bm = z[f"mask{i}"], b[f"mask{i}"]
        inter = (rm & bm).reshape(len(rm), -1).sum(1)
        uni = (rm | bm).reshape(len(rm), -1).sum(1)
        iou = inter / np.maximum(uni, 1)
        alls.append(iou)
        print(f"case {i}: mask IoU " + " ".join(f"{v:.3f}" for v in iou) +
              f"   score ref {np.round(z[f'iou{i}'], 2).tolist()} board {np.round(b[f'iou{i}'], 2).tolist()}")
    al = np.concatenate(alls)
    print(f"all {len(al)} boxes: mean {al.mean():.4f} median {np.median(al):.4f} worst {al.min():.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--variant", default="ti")
    p.add_argument("--size", type=int, default=768)
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
