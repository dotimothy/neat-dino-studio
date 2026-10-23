#!/usr/bin/env python3
"""How far do Grounding DINO's deformable sampling points reach?

GridSample does not compile for the MLA, so deformable attention has to be
rebuilt from fixed (constant-index) ops. Encoder queries sit on fixed grids, so a
window of +-R target-level pixels around each query's reference point can be
gathered with constant slices. This measures how much attention mass falls
outside such a window, per encoder layer and target level, on COCO images.

Offsets are in target-level pixels (HF: loc = ref + offset / (W_l, H_l)).
A bilinear sample at offset o touches pixels up to ceil(|o|) + 1 away (the
reference point itself can sit between pixels for cross-level pairs), so the
"needed R" of a sample is ceil(max(|ox|, |oy|) + 0.5).

    .venv/bin/python model/offsets_study.py --size 768 --n 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dino.letterbox import letterbox  # noqa: E402

COCO = ROOT / "data" / "coco"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--prompt", default="person . car . dog . cat . chair . bottle . cup . bird . "
                                        "bicycle . tv . laptop . dining table . book . clock . bus .")
    a = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    mid = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval().cuda()

    caps = {"enc": [], "dec": []}

    # Patch the two linears: sampling_offsets and attention_weights outputs.
    for kind, layers in (("enc", [l.deformable_layer.self_attn for l in model.model.encoder.layers]),
                         ("dec", [l.encoder_attn for l in model.model.decoder.layers])):
        for li, m in enumerate(layers):
            store = {}

            def off_hook(mod, inp, out, store=store):
                store["off"] = out.detach()

            def aw_hook(mod, inp, out, store=store, m=m, kind=kind, li=li):
                nH, nL, nP = m.n_heads, m.n_levels, m.n_points
                off = store["off"].view(1, -1, nH, nL, nP, 2)
                aw = out.detach().view(1, -1, nH, nL * nP).softmax(-1).view(1, -1, nH, nL, nP)
                caps[kind].append((li, off.float().cpu(), aw.float().cpu()))

            m.sampling_offsets.register_forward_hook(off_hook)
            m.attention_weights.register_forward_hook(aw_hook)

    ids = [int(x) for x in (COCO / "subset500.txt").read_text().split()][: a.n]
    Rs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
    # mass[layer][level][R] accumulators
    nl, nlev = 6, 4
    mass = np.zeros((nl, nlev, len(Rs)))
    total = np.zeros((nl, nlev))
    maxoff = np.zeros((nl, nlev))
    for k, iid in enumerate(ids):
        img = Image.open(COCO / "val2017" / f"{iid:012d}.jpg").convert("RGB")
        px, _ = letterbox(img, a.size)
        txt = proc.tokenizer(a.prompt, return_tensors="pt")
        caps["enc"].clear()
        caps["dec"].clear()
        with torch.no_grad():
            model(pixel_values=torch.from_numpy(px)[None].cuda(),
                  pixel_mask=torch.ones(1, a.size, a.size, dtype=torch.long).cuda(),
                  input_ids=txt.input_ids.cuda(), attention_mask=txt.attention_mask.cuda(),
                  token_type_ids=txt.token_type_ids.cuda())
        for li, off, aw in caps["enc"]:
            need = torch.ceil(off.abs().amax(-1) + 0.5)  # (1,Q,H,L,P)
            for lev in range(nlev):
                n = need[..., lev, :]
                w = aw[..., lev, :]
                total[li, lev] += w.sum().item()
                maxoff[li, lev] = max(maxoff[li, lev], off[..., lev, :, :].abs().max().item())
                for ri, R in enumerate(Rs):
                    mass[li, lev, ri] += w[n > R].sum().item()
        print(f"\r{k + 1}/{len(ids)}", end="", flush=True)
    print()
    frac = mass / total[..., None]
    print(f"Attention mass OUTSIDE a +-R window (target-level pixels), size {a.size}, {len(ids)} images")
    print("layer lvl  " + " ".join(f"R={R:<5d}" for R in Rs) + "  max|off|  mass/lvl")
    for li in range(nl):
        for lev in range(nlev):
            print(f"  {li}    {lev}   " + " ".join(f"{frac[li, lev, ri]*100:6.3f}%" for ri in range(len(Rs)))
                  + f"  {maxoff[li, lev]:7.1f}  {total[li, lev] / total[li].sum() * 100:5.1f}%")
    allf = mass.sum((0, 1)) / total.sum()
    print("  all       " + " ".join(f"{f*100:6.3f}%" for f in allf))
    np.savez(ROOT / "build" / f"offsets_{a.size}.npz", frac=frac, Rs=Rs, total=total, maxoff=maxoff)


if __name__ == "__main__":
    main()
