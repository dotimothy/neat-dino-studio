#!/usr/bin/env python3
"""Package the compiled OmDet-Turbo programs into a runtime bundle for omdet_rt.

    .venv-sam3/bin/python tools/make_omdet_bundle.py

build/bundles/omdet-turbo-640-bf16/
    manifest.json        sizes, backbone steps, ELF input/output order, host weight index, sha256s
    elf/*.elf            bb1_0 .. bb4_1, aifi, ccfm, decin, dec_pre0-5, dec_post0-5, text, task
    tables/*.i32         backbone gather tables
    host.f32             anchors, valid mask, encoder class feature of invalid tokens, class heads,
                         CLIP text projection
    embeddings.npz       CLIP token (49408 x 512, fp16) and position (77 x 512) embeddings
    vocab.json, merges.txt   CLIP BPE (dino/clip_bpe.py)
"""
from __future__ import annotations

import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))
from make_bundle import extract_elf, finish  # noqa: E402
from make_sam3_bundle import elf_input_order  # noqa: E402


def elf_output_order(stage_dir: Path) -> list[str]:
    """ONNX output names in the order the MLA program writes them (follow its outputs through the
    EV74 cast plugins; a cast's output is named '<plugin>/<onnx output>')."""
    mpk = next(stage_dir.glob("*_mpk.tar.gz"))
    with tarfile.open(mpk) as t:
        d = json.load(t.extractfile(next(n for n in t.getnames() if n.endswith("_mpk.json"))))
    nxt = {}
    for p in d["plugins"]:
        ins, outs = p.get("input_nodes", []), p.get("output_nodes", [])
        if p.get("processor") != "MLA" and len(ins) == 1 and len(outs) == 1:
            nxt[ins[0]["name"]] = outs[0]["name"]
    mla = next(p for p in d["plugins"] if p.get("processor") == "MLA")
    order = []
    for o in mla["output_nodes"]:
        n = o["name"]
        while n in nxt:
            n = nxt[n]
        order.append(n.split("/")[-1])
    return order


def main():
    import omdet_mla as O
    from huggingface_hub import hf_hub_download
    hf = O.load_hf()
    bb = O.Backbone(hf)
    name = f"omdet-turbo-{O.S}-bf16"
    out = ROOT / "build" / "bundles" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "elf").mkdir(parents=True)
    (out / "tables").mkdir()
    progs = [n for n, _ in bb.blocks] + ["aifi", "ccfm", "decin"] + \
        [f"dec_{w}{k}" for k in range(6) for w in ("pre", "post")] + ["text", "task"]
    inputs, outputs = {}, {}
    for p in progs:
        sd = ROOT / "build" / "stages" / "bf16" / f"omdet_{p}"
        extract_elf(sd, out / "elf" / f"{p}.elf")
        inputs[p] = elf_input_order(sd)
        outputs[p] = elf_output_order(sd)
    steps = []
    for i, st in enumerate(bb.steps):
        s = {k: v for k, v in st.items() if k not in ("table", "in_shape", "out_shape", "hw")}
        if "table" in st:
            f = f"tables/{i:02d}_{st['op']}.i32"
            np.asarray(st["table"], np.int32).tofile(out / f)
            s["table"] = f
        if st["op"] == "run":
            s["in_elems"] = int(np.prod(st["in_shape"]))
            s["out_elems"] = int(np.prod(st["out_shape"]))
        steps.append(s)

    P = O.Pipeline(hf)
    d = hf.decoder
    host, index = [], {}

    def put(k, t):
        t = torch.as_tensor(t).detach().float().contiguous().numpy().ravel()
        index[k] = {"offset": int(sum(x.size for x in host)), "shape": list(t.shape)}
        host.append(t)

    a = P.anchor.clone()
    a[~P.valid] = 0
    put("anchors", a)
    put("valid", P.valid.float())
    put("cf_invalid", P.cf_invalid)
    put("enc_cls.w", d.encoder_class_head.weight)
    put("enc_cls.b", d.encoder_class_head.bias)
    put("dec_cls.w", d.decoder_class_head[-1].weight)
    put("dec_cls.b", d.decoder_class_head[-1].bias)
    put("text_proj", hf.language_backbone.text_projection)
    np.concatenate(host).astype(np.float32).tofile(out / "host.f32")
    np.savez(out / "embeddings.npz", token=P.text.token.numpy().astype(np.float16),
             position=P.text.position.numpy().astype(np.float32))
    for f in ("vocab.json", "merges.txt"):
        shutil.copy(hf_hub_download(O.HF_ID, f), out / f)
    (out / "LICENSE").write_text("OmDet-Turbo (omlab/omdet-turbo-swin-tiny-hf, om-ai-lab/OmDet): Apache License 2.0\n"
                                 "https://www.apache.org/licenses/LICENSE-2.0\n")
    levels, st_ = [], 0
    for h, w in O.level_sizes():
        levels.append([h, w, st_])
        st_ += h * w
    finish(out, {
        "format": 1, "model": O.HF_ID, "variant": "omdet-turbo-tiny", "kind": "omdet", "name": name, "size": O.S,
        "precision": "bf16", "levels": levels, "tokens": st_, "queries": O.NQ, "task_tokens": O.TT,
        "text_tokens": O.TEXT_T, "text_batch": O.TEXT_B, "mean": list(O.MEAN), "std": list(O.STD),
        "preprocess": "squash", "text": {"bos": 49406, "eos": 49407, "max_tokens": O.TEXT_T},
        "backbone": steps, "programs": progs, "inputs": inputs, "outputs": outputs, "host": index,
        "license": "Apache-2.0",
    })


if __name__ == "__main__":
    main()
