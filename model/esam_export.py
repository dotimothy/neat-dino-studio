#!/usr/bin/env python3
"""Export EfficientSAM (Ti / S) as MLA programs for Grounded EfficientSAM.

    .venv-sam3/bin/python model/esam_export.py --variant ti --size 768 [--boxes 1 8]

Programs (S = 768: 48 x 48 tokens):
  esam_<v><S>_enc     patch tokens (1, 768, 1, 2304) -> ViT + neck 1x1 + LN -> (1, 256, 1, 2304)
  esam_<v><S>_neck    (1, 256, 48, 48) (the same bytes) -> neck 3x3 conv + LN
  esam_<v><S>_dec<B>  image embedding (1, 256, 1, 2304) + per-box tokens (1, 256, B, 11)
                      -> (1, 48, B, 2305): 3 masks at 192 x 192 as sub-pixel channels + IoU column
Calibration (build/calib/<name>.npz) uses the sample images and boxes drawn around their content.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import esam_mla as E  # noqa: E402
from sam3_export import export  # noqa: E402

SAMPLES = sorted(p for p in (ROOT / "webapp" / "web" / "samples").glob("*.jpg") if "thumb" not in p.name)


def random_boxes(rng, W, H, n):
    out = []
    for _ in range(n):
        w, h = rng.uniform(0.08, 0.9) * W, rng.uniform(0.08, 0.9) * H
        x, y = rng.uniform(0, W - w), rng.uniform(0, H - h)
        out.append([x, y, x + w, y + h])
    return torch.tensor(out, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ti", choices=list(E.VARIANTS))
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--boxes", type=int, nargs="*", default=[1, 8])
    ap.add_argument("--only", nargs="*", help="enc neck dec")
    a = ap.parse_args()
    sam = E.load_esam(a.variant, a.size)
    G = a.size // 16
    tag = f"esam_{a.variant}{a.size}"
    enc, neck = E.EncoderMLA(sam.image_encoder).eval(), E.NeckMLA(sam.image_encoder).eval()
    rng = np.random.default_rng(0)
    toks, mids, embs, prompts = [], [], [], {B: [] for B in a.boxes}
    with torch.no_grad():
        for p in SAMPLES:
            img = Image.open(p).convert("RGB")
            W, H = img.size
            x = torch.from_numpy(np.asarray(img.resize((a.size, a.size), Image.BILINEAR), np.float32) / 255).permute(2, 0, 1)
            t = E.patch_tokens(x)
            mid = enc(t)
            emb = neck(mid.reshape(1, 256, G, G))
            toks.append(t)
            mids.append(mid.reshape(1, 256, G, G))
            embs.append(emb.reshape(1, 256, 1, G * G))
            for B in a.boxes:
                prompts[B].append((random_boxes(rng, W, H, B), W, H))
    print(f"{tag}: {len(SAMPLES)} calibration images")

    def go(name, mod, inputs, calib):
        export(name, mod, inputs)
        np.savez(ROOT / "build" / "calib" / f"{name}.npz", **{k: torch.cat(v).numpy() for k, v in calib.items()})

    want = lambda k: not a.only or k in a.only  # noqa: E731
    if want("enc"):
        go(f"{tag}_enc", enc, {"x": toks[0]}, {"x": toks})
    if want("neck"):
        go(f"{tag}_neck", neck, {"x": mids[0]}, {"x": mids})
    if want("dec"):
        for B in a.boxes:
            dec = E.DecoderMLA(sam, B).eval()
            toks_b = [E.box_tokens(dec, sam, *p) for p in prompts[B]]
            go(f"{tag}_dec{B}", dec, {"img": embs[0], "tok": toks_b[0]}, {"img": embs, "tok": toks_b})


if __name__ == "__main__":
    main()
