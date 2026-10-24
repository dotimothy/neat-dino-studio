#!/usr/bin/env python3
"""Package the compiled EfficientSAM programs into a runtime bundle for esam_rt.

    .venv-sam3/bin/python tools/make_esam_bundle.py --variant ti --size 768 [--boxes 1 8]

build/bundles/esam-<v><size>-bf16/
    manifest.json        size, decoder box counts, host weight index, ELF input order, sha256 of every file
    elf/enc.elf neck.elf dec<B>.elf
    host.f32             prompt encoder: random-Fourier matrix (2 x 128), box corner embeddings,
                         the decoder's constant tokens (IoU + mask tokens, padding points)
    LICENSE              EfficientSAM (Apache-2.0)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))
from make_bundle import extract_elf, finish  # noqa: E402
from make_sam3_bundle import elf_input_order  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ti")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--boxes", type=int, nargs="*", default=[1, 8])
    a = ap.parse_args()
    import esam_mla as E
    sam = E.load_esam(a.variant, a.size)
    tag = f"esam_{a.variant}{a.size}"
    name = f"esam-{a.variant}{a.size}-bf16"
    out = ROOT / "build" / "bundles" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "elf").mkdir(parents=True)
    progs = ["enc", "neck"] + [f"dec{b}" for b in a.boxes]
    inputs = {}
    for p in progs:
        sd = ROOT / "build" / "stages" / "bf16" / f"{tag}_{p}"
        extract_elf(sd, out / "elf" / f"{p}.elf")
        inputs[p] = elf_input_order(sd)
    pe = sam.prompt_encoder
    host, index = [], {}

    def put(k, t):
        t = t.detach().float().contiguous().numpy().ravel()
        index[k] = {"offset": int(sum(x.size for x in host)), "shape": list(t.shape)}
        host.append(t)

    put("pe_gauss", pe.pe_layer.positional_encoding_gaussian_matrix)
    put("corner_tl", pe.bbox_top_left_embeddings.weight[0])
    put("corner_br", pe.bbox_bottom_right_embeddings.weight[0])
    dec = E.DecoderMLA(sam, 1)
    put("tok_head", dec.tok_head)                 # (5, 256) IoU + mask tokens
    put("tok_tail", dec.tok_tail)                 # (4, 256) padding-point embeddings
    np.concatenate(host).astype(np.float32).tofile(out / "host.f32")
    shutil.copy(ROOT / "third_party" / "efficient_sam" / "LICENSE", out / "LICENSE")
    dim, heads, _ = E.VARIANTS[a.variant]
    finish(out, {
        "format": 1, "model": "yformer/EfficientSAM", "variant": f"efficientsam-{a.variant}", "kind": "esam",
        "name": name, "size": a.size, "grid": a.size // 16, "mask_size": a.size // 4, "mask_upsample": 2,
        "embed_dim": dim, "heads": heads, "precision": "bf16", "decoders": a.boxes, "programs": progs,
        "pixel_mean": list(E.PIXEL_MEAN), "pixel_std": list(E.PIXEL_STD), "preprocess": "squash",
        "host": index, "inputs": inputs, "license": "Apache-2.0 (see LICENSE)",
    })


if __name__ == "__main__":
    main()
