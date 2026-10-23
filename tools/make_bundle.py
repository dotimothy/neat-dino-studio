#!/usr/bin/env python3
"""Package compiled Grounding DINO stages into runtime bundles for dino_rt.

    .venv/bin/python tools/make_bundle.py --size 576 [--prec bf16]
    .venv/bin/python tools/make_bundle.py --text

Detector bundle  build/bundles/dino-<size>-<prec>/
    manifest.json        size, tokens, levels, stages, precision, sha256 of every file
    elf/<stage>.elf      backbone, enc0..6, dec_values, dec_pre0..5, dec_post0..5
    pos.bf16             encoder position embeddings (N x 256)
    ref.f32              encoder reference points (N x 2)
    proposals.f32        two-stage proposals in logit space, inf where invalid (N x 4)
    valid.u8             proposal validity (N)
    invalid_query.f32    object query of an invalid token, LN(enc_output(0)) (256)
    query_embed.bf16     decoder query embeddings (900 x 256)

Text bundle  build/bundles/dino-text-t64-<prec>/
    text_t64.elf, tokenizer.json, vocab.txt, embeddings.npz (word / position / type, fp16), manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

STAGES = [f"enc{k}" for k in range(7)] + ["dec_values"] + \
         [f"dec_pre{k}" for k in range(6)] + [f"dec_post{k}" for k in range(6)]
FORMAT = 1


def bf16(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_elf(stage_dir: Path, dst: Path):
    mpk = next(stage_dir.glob("*_mpk.tar.gz"))
    with tarfile.open(mpk) as t:
        names = [n for n in t.getnames() if n.endswith("_mla.elf")]
        if len(names) != 1:
            raise SystemExit(f"{mpk}: expected one MLA ELF, found {names}")
        with t.extractfile(names[0]) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)
    summ = json.loads((stage_dir / "compile_summary.json").read_text())
    if not summ.get("single_mla_elf"):
        raise SystemExit(f"{stage_dir}: not a single-MLA-ELF compile")


def finish(out: Path, manifest: dict):
    manifest["files"] = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"{out}: {len(manifest['files'])} files, {total / 2**20:.0f} MB")


def detector(S: int, prec: str, skip=(), fused=True):
    import dino_mla as M
    from transformers import AutoModelForZeroShotObjectDetection
    hf = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval()
    out = ROOT / "build" / "bundles" / f"dino-{S}-{prec}"
    if out.exists():
        shutil.rmtree(out)
    (out / "elf").mkdir(parents=True)
    import backbone_plan as BP
    if fused:
        import backbone_fused as BF
        plan = BF.build_cpu_merge(hf, S)
    else:
        plan = BP.build(hf, S)
    (out / "tables").mkdir()
    program = []
    for i, st in enumerate(plan):
        e = {"op": st.op, "name": st.name}
        if st.op == "patchify":
            e.update(grid=int(st.extra["grid"]))
        if st.op == "run":
            e.update(src=st.src, dst=st.dst, dst_off=st.dst_off,
                     in_elems=int(np.prod(st.in_shape)), out_elems=int(np.prod(st.out_shape)))
            if "backbone" not in skip:
                extract_elf(ROOT / "build" / "stages" / prec / f"s{S}_{st.name}", out / "elf" / f"{st.name}.elf")
        if st.op == "gather":
            e.update(src=st.src, dst=st.dst, dst_off=st.dst_off, k=st.k, c=st.c)
        if st.table is not None:
            rel = f"tables/{i:02d}_{st.name}.i32"
            np.asarray(st.table, np.int32).tofile(out / rel)
            e["table"] = rel
        program.append(e)
    for st in STAGES:
        if st in skip:
            continue
        extract_elf(ROOT / "build" / "stages" / prec / f"s{S}_{st}", out / "elf" / f"{st}.elf")
    g = M.geometry(hf, S)
    bf16(M.tok_to_rows(g.pos).numpy()).tofile(out / "pos.bf16")
    g.ref.numpy().astype(np.float32).tofile(out / "ref.f32")
    g.proposals.numpy().astype(np.float32).tofile(out / "proposals.f32")
    g.valid.numpy().astype(np.uint8).tofile(out / "valid.u8")
    with torch.no_grad():
        iq = hf.model.enc_output_norm(hf.model.enc_output(torch.zeros(1, 256)))[0]
    iq.numpy().astype(np.float32).tofile(out / "invalid_query.f32")
    bf16(hf.model.query_position_embeddings.weight.detach().numpy()).tofile(out / "query_embed.bf16")
    starts = np.cumsum([0] + [h * w for h, w in g.shapes])[:-1]
    finish(out, {"format": FORMAT, "model": "IDEA-Research/grounding-dino-base", "kind": "detector",
                 "name": f"dino-{S}-{prec}", "requires": "dino-text-t64-bf16", "size": S, "precision": prec, "tokens": g.N, "queries": M.NQ, "text_tokens": M.T_TXT,
                 "levels": [[h, w, int(s)] for (h, w), s in zip(g.shapes, starts)], "stages": STAGES,
                 "backbone": program})


def text(prec: str):
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForZeroShotObjectDetection
    hf = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval()
    out = ROOT / "build" / "bundles" / f"dino-text-t64-{prec}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    extract_elf(ROOT / "build" / "stages" / prec / "text_t64", out / "text_t64.elf")
    for f in ("tokenizer.json", "vocab.txt"):
        shutil.copy(hf_hub_download("IDEA-Research/grounding-dino-base", f), out / f)
    e = hf.model.text_backbone.embeddings
    np.savez(out / "embeddings.npz",
             word=e.word_embeddings.weight.detach().numpy().astype(np.float16),
             position=e.position_embeddings.weight.detach().numpy().astype(np.float16),
             token_type=e.token_type_embeddings.weight.detach().numpy()[0].astype(np.float16))
    finish(out, {"format": FORMAT, "model": "IDEA-Research/grounding-dino-base", "kind": "text",
                 "name": f"dino-text-t64-{prec}",
                 "precision": prec, "text_tokens": 64})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int)
    ap.add_argument("--prec", default="bf16")
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="stages to leave out (debug bundles)")
    ap.add_argument("--unfused", action="store_true", help="27-program backbone with CPU gathers (old layout)")
    a = ap.parse_args()
    if a.text:
        text(a.prec)
    if a.size:
        detector(a.size, a.prec, set(a.skip), fused=not a.unfused)


if __name__ == "__main__":
    main()
