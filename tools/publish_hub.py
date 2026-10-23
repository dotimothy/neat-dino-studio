#!/usr/bin/env python3
"""Publish compiled Grounding DINO bundles to Hugging Face for the studio's model manager.

Each bundle (build/bundles/<name>/, from tools/make_bundle.py) is packed as one
<name>.tar.gz and uploaded into a folder of its name. models.json at the top of the
repository lists them with sizes, SHA-256 sums and what each detector needs (the text
encoder); webapp/server.py's Hub reads it. The repository also gets a model card.

    python3 tools/publish_hub.py --dry-run                 # pack + write build/hub/ only
    python3 tools/publish_hub.py                           # pack, upload, list, card
    python3 tools/publish_hub.py --only dino-768-bf16      # one bundle (the list keeps the others)
    python3 tools/publish_hub.py --card-only               # just the model card

Needs huggingface_hub and a login with write access (`hf auth login`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "dotimothy/sima-grounding-dino"
COLLECTION = "SiMa.ai Open-Vocabulary Detection Models"
COLLECTION_ABOUT = "Open-vocabulary detectors compiled for SiMa.ai Modalix: type what to find, get boxes. Runs on the MLA."
UPSTREAM = "IDEA-Research/grounding-dino-base"

ABOUT = {
    "dino-text-t64-bf16": ("Text Encoder · BERT (64 Tokens)", "BERT-base + projection, used by every Base detector."),
    "dino-576-bf16": ("Fast · 576 px", "Grounding DINO base at 576 px: best for the webcam."),
    "dino-768-bf16": ("Accurate · 768 px", "Grounding DINO base at 768 px: finds smaller things."),
    "dino-tiny-text-t64-bf16": ("Tiny Text Encoder · BERT (64 Tokens)", "Grounding DINO Tiny's BERT-base + projection."),
    "dino-tiny-576-bf16": ("Tiny · 576 px", "Grounding DINO tiny (Swin-T, 172.2 M) at 576 px: a bit quicker, smaller."),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def pack(bundle: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    tgz = out / f"{bundle.name}.tar.gz"
    if not tgz.exists() or tgz.stat().st_mtime < max(p.stat().st_mtime for p in bundle.rglob("*")):
        with tarfile.open(tgz, "w:gz", compresslevel=6) as t:
            t.add(bundle, arcname=bundle.name)
    return tgz


def card(repo: str, models: dict) -> str:
    rows = "\n".join(f"| `{n}` | {e['title']} | {e['files'][0]['bytes'] / 2**20:.0f} MB | "
                     f"{e.get('latency_ms') and str(round(e['latency_ms'])) + ' ms' or '—'} | {e['about']} |"
                     for n, e in models.items())
    return f"""---
license: apache-2.0
base_model: {UPSTREAM}
pipeline_tag: zero-shot-object-detection
tags: [sima-ai, modalix, mla, grounding-dino, open-vocabulary-detection, edge]
---

# Grounding DINO (base and tiny) for SiMa.ai Modalix

[{UPSTREAM}](https://huggingface.co/{UPSTREAM}) compiled for the SiMa.ai Modalix MLA, for
[Neat DINO Studio](https://github.com/dotimothy/neat-dino-studio): type what to find
("person . red car . dog ."), get boxes, from a photo or a live webcam.

**Weights:** [{UPSTREAM}](https://huggingface.co/{UPSTREAM}), the **Base** size of Grounding DINO: **232.3 M parameters** in total (Swin-B image backbone 86.9 M, BERT-base text encoder 108.9 M, feature enhancer, deformable decoder and heads 36.5 M), unchanged from the original and run in bf16. Apache-2.0.

The `dino-tiny-*` bundles are [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) instead: **172.2 M parameters** (Swin-T backbone 27.5 M, its own BERT-base weights 108.9 M, the rest 35.8 M), also unchanged and in bf16. A Tiny detector needs the Tiny text encoder.

| Bundle | What | Size | Per image (board) | About |
|---|---|---|---|---|
{rows}

## How it runs

The network is cut into 30 MLA ELFs, all bf16: the text encoder (BERT +
projection); the Swin-B backbone as 9 ELFs (whole stages, with the window shifts inside);
7 encoder ELFs; decoder value maps and 12 decoder ELFs. Between them the board's Arm
cores do what the MLA cannot: multi-scale deformable-attention sampling (NEON), the top-900
query selection, the reference-box updates, and Swin patch merging. The runtime (`dino_rt`)
runs the ELFs with SimaLMM's `MLAModelWithBuffer`, reads and writes their bf16 buffers in
place, and keeps two frames in flight in Live mode.

| Measured on a Modalix DevKit | Base · Fast · 576 px | Base · Accurate · 768 px | Tiny · 576 px |
|---|---|---|---|
| Per new image | 0.33 s | 0.54 s | 0.29 s |
| Live, two frames in flight | 3.9 FPS | 2.3 FPS | 4.3 FPS |
| COCO AP, 500-image val2017 subset (square letterbox) | 42.0 | 46.4 | 41.8 |
| Same, FP32 PyTorch on GPU, same inputs | 41.7 | 45.8 | 41.4 |

Tiny's ELFs: text encoder 1, Swin-T backbone 7, encoder 7, decoder 13.

Use the bundles with the studio (Settings → Models downloads them) or `webapp/run.sh --fetch fast`.

## License

Apache-2.0, as Grounding DINO. Grounding DINO © IDEA Research.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-collection", action="store_true")
    ap.add_argument("--latency", nargs="*", default=[], help="name=ms pairs measured on the board")
    ap.add_argument("--card-only", action="store_true", help="only rewrite the model card from the published models.json")
    a = ap.parse_args()
    if a.card_only:
        import urllib.request
        from huggingface_hub import HfApi
        with urllib.request.urlopen(f"https://huggingface.co/{a.repo}/resolve/main/models.json", timeout=20) as r:
            published = json.loads(r.read())["models"]
        HfApi().upload_file(path_or_fileobj=card(a.repo, published).encode(), path_in_repo="README.md",
                            repo_id=a.repo, commit_message="Update model card")
        print(f"https://huggingface.co/{a.repo}")
        return
    lat = dict(x.split("=") for x in a.latency)
    out = ROOT / "build" / "hub"
    bundles = sorted(p for p in (ROOT / "build" / "bundles").iterdir() if (p / "manifest.json").is_file())
    if a.only:
        bundles = [b for b in bundles if b.name in a.only]
    models, files = {}, {}
    for b in bundles:
        m = json.loads((b / "manifest.json").read_text())
        tgz = pack(b, out)
        title, about = ABOUT.get(b.name, (b.name, ""))
        e = {"title": title, "about": about, "kind": m["kind"], "precision": m.get("precision"), "path": b.name,
             "files": [{"name": tgz.name, "bytes": tgz.stat().st_size, "sha256": sha256(tgz)}]}
        if m["kind"] == "detector":
            e.update(size=m["size"], requires=m.get("requires", "dino-text-t64-bf16"))
            if b.name in lat:
                e["latency_ms"] = float(lat[b.name])
        models[b.name] = e
        files[b.name] = tgz
        print(f"{b.name:22s} {tgz.stat().st_size / 2**20:7.0f} MB  {e['files'][0]['sha256'][:12]}")

    if a.dry_run:
        doc = {"format": 1, "models": models}
        (out / "models.json").write_text(json.dumps(doc, indent=1))
        (out / "README.md").write_text(card(a.repo, models))
        print(f"dry run: wrote {out}/models.json and README.md")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=False)
    try:
        import urllib.request
        with urllib.request.urlopen(f"https://huggingface.co/{a.repo}/resolve/main/models.json", timeout=20) as r:
            old = json.loads(r.read()).get("models", {})
    except Exception:  # noqa: BLE001
        old = {}
    for name, tgz in files.items():
        print(f"uploading {tgz.name} …")
        api.upload_file(path_or_fileobj=str(tgz), path_in_repo=f"{name}/{tgz.name}", repo_id=a.repo,
                        commit_message=f"Add {name}")
    merged = {**old, **models}
    api.upload_file(path_or_fileobj=json.dumps({"format": 1, "models": merged}, indent=1).encode(),
                    path_in_repo="models.json", repo_id=a.repo, commit_message="Update models.json")
    api.upload_file(path_or_fileobj=card(a.repo, merged).encode(), path_in_repo="README.md", repo_id=a.repo,
                    commit_message="Update model card")
    print(f"https://huggingface.co/{a.repo}")
    if not a.no_collection:
        owner = a.repo.split("/")[0]
        found = next((c for c in api.list_collections(owner=owner, limit=100) if c.title == COLLECTION), None)
        slug = found.slug if found else api.create_collection(COLLECTION, description=COLLECTION_ABOUT,
                                                              private=False).slug
        if not any(i.item_id == a.repo for i in api.get_collection(slug).items):
            api.add_collection_item(slug, a.repo, "model", note="Grounding DINO base, 576 + 768 px, bf16. Apache-2.0.")
        print(f"collection: https://huggingface.co/collections/{slug}")


if __name__ == "__main__":
    main()
