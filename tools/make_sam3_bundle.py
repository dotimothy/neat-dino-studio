#!/usr/bin/env python3
"""Package the compiled SAM 3 programs into a runtime bundle for sam3_rt.

    .venv-sam3/bin/python tools/make_sam3_bundle.py --size 672

build/bundles/sam3-<size>-bf16/
    manifest.json        sizes, program order, host weight index, LayerNorm eps, sha256 of every file
    elf/<program>.elf    vit*, neck4/2/1, text, enc, dec0..5, maskpre, pixdec
    host.f32             decoder query embeddings, reference boxes, box / presence / ref-point heads,
                         box bias MLPs, scoring and mask-embedding weights (see sam3_rt.cpp)
    pos.bf16             encoder position embedding (N x 256)
    embeddings.npz       CLIP token (49408 x 1024) and position (32 x 1024) embeddings, fp16
    vocab.json, merges.txt   CLIP BPE tokenizer (dino/clip_bpe.py), tokenizer.json
    LICENSE              the SAM License (required when redistributing)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))
from make_bundle import bf16, extract_elf, finish  # noqa: E402

PROGRAMS_VIT = ["vit0", "vit7", "vit8", "vit15", "vit16", "vit23", "vit24", "vit31"]
PROGRAMS = PROGRAMS_VIT + ["neck4", "neck2", "neck1", "text", "enc"] + [f"dec{k}" for k in range(6)] + \
           ["maskpre", "pixdec"]          # query . pixel runs on the CPU in sam3_rt


def elf_input_order(stage_dir: Path) -> list[str]:
    """ONNX input names in the order the compiled MLA program takes them. The compiler may
    reorder the inputs of a multi-input graph; the MPK's EV74 cast plugins map each MLA input
    back to its ONNX name."""
    import tarfile
    mpk = next(stage_dir.glob("*_mpk.tar.gz"))
    with tarfile.open(mpk) as t:
        name = next(n for n in t.getnames() if n.endswith("_mpk.json"))
        d = json.load(t.extractfile(name))
    src = {}
    for p in d["plugins"]:
        outs = p.get("output_nodes", [])
        ins = p.get("input_nodes", [])
        if p.get("processor") != "MLA" and len(ins) == 1 and len(outs) == 1:
            src[outs[0]["name"]] = ins[0]["name"]
    mla = next(p for p in d["plugins"] if p.get("processor") == "MLA")
    order = []
    for i in mla["input_nodes"]:
        n = i["name"]
        while n in src:                        # follow cast plugins back to the model input
            n = src[n]
        order.append(n)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=672)
    ap.add_argument("--skip-elf", action="store_true", help="host files only (for checks before compiling)")
    a = ap.parse_args()
    import sam3_mla as S
    from huggingface_hub import hf_hub_download
    from transformers import Sam3Config, Sam3Model
    cfg = Sam3Config.from_pretrained("facebook/sam3")
    cfg.vision_config.backbone_config.image_size = a.size
    hf = Sam3Model.from_pretrained("facebook/sam3", config=cfg).eval()
    vc = cfg.vision_config.backbone_config
    G, ws = a.size // vc.patch_size, vc.window_size
    name = f"sam3-{a.size}-bf16"
    out = ROOT / "build" / "bundles" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "elf").mkdir(parents=True)
    inputs = {}
    if not a.skip_elf:
        for p in PROGRAMS:
            sd = ROOT / "build" / "stages" / "bf16" / f"sam3_{a.size}_{p}"
            extract_elf(sd, out / "elf" / f"{p}.elf")
            inputs[p] = elf_input_order(sd)

    d, md, sc = hf.detr_decoder, hf.mask_decoder, hf.dot_product_scoring
    host, index = [], {}

    def put(k, t):
        t = t.detach().float().contiguous().numpy().ravel()
        index[k] = {"offset": int(sum(x.size for x in host)), "shape": list(t.shape)}
        host.append(t)

    def lin(k, l):
        put(f"{k}.w", l.weight)
        put(f"{k}.b", l.bias)

    def mlp(k, m):
        for i, l in enumerate((m.layer1, m.layer2, m.layer3), 1):
            if l is not None:
                lin(f"{k}.l{i}", l)

    put("query_embed", d.query_embed.weight)
    put("presence_token", d.presence_token.weight[0])
    put("ref0", d.reference_points.weight.sigmoid())
    mlp("ref_head", d.ref_point_head)
    mlp("box_head", d.box_head)
    mlp("pres_head", d.presence_head)
    mlp("rpb_x", d.box_rpb_embed_x)
    mlp("rpb_y", d.box_rpb_embed_y)
    for k, ln in (("out_ln", d.output_layer_norm), ("pres_ln", d.presence_layer_norm), ("sc_ln", sc.text_mlp_out_norm)):
        put(f"{k}.w", ln.weight)
        put(f"{k}.b", ln.bias)
    mlp("sc_mlp", sc.text_mlp)
    lin("sc_text", sc.text_proj)
    lin("sc_query", sc.query_proj)
    for i, l in enumerate(md.mask_embedder.layers):
        lin(f"mask_embed.{i}", l)
    np.concatenate(host).astype(np.float32).tofile(out / "host.f32")
    bf16(S.rows(S.tok(S.sine_pos_2d(G, G).reshape(256, -1).T)).numpy()).tofile(out / "pos.bf16")

    e = hf.text_encoder.text_model.embeddings
    np.savez(out / "embeddings.npz", token=e.token_embedding.weight.detach().numpy().astype(np.float16),
             position=e.position_embedding.weight.detach().numpy().astype(np.float16))
    for f in ("tokenizer.json", "vocab.json", "merges.txt", "LICENSE"):
        shutil.copy(hf_hub_download("facebook/sam3", f), out / f)
    finish(out, {
        "format": 1, "model": "facebook/sam3", "variant": "sam3", "kind": "sam3", "name": name, "size": a.size,
        "grid": G, "window": ws, "queries": d.config.num_queries, "text_tokens": 32,
        "heads": d.config.num_attention_heads, "mask_size": 4 * G, "score_scale": float(sc.scale),
        "precision": "bf16", "vit": PROGRAMS_VIT, "programs": PROGRAMS,
        "eps": {"out_ln": d.output_layer_norm.eps, "pres_ln": d.presence_layer_norm.eps,
                "sc_ln": sc.text_mlp_out_norm.eps},
        "text": {"bos": 49406, "eos": 49407, "pad": 49407, "max_tokens": 32},
        "host": index,
        "inputs": inputs,                      # per program: ONNX input names in the ELF's order
        "license": "SAM License (see LICENSE)",
    })


if __name__ == "__main__":
    main()
