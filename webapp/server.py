#!/usr/bin/env python3
"""Neat DINO Studio: open-vocabulary detection with Grounding DINO on the Modalix MLA.

Upload a photo or use the webcam, type what to find ("person . red car . dog ."),
and Grounding DINO boxes it. On the board the model runs as compiled MLA stages
under dino_rt (runtime/), with deformable-attention sampling on the CPU.

    python3 webapp/server.py [--port 8092] [--model dino-576-bf16] [--backend mla|torch|fake]

Models live in the shared models directory (DINO_MODELS_DIR, .models-dir, or
/media/nvme/models/dino) and are downloaded from Hugging Face in Settings -> Models.
Only the model manager loads and unloads: one detector on the MLA at a time.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import io
import json
import mimetypes
import os
import re
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "web"
sys.path.insert(0, str(ROOT))
from app import backends  # noqa: E402
from app.assist import NullAssist  # noqa: E402,F401  (the no-LLM assist, kept for tests)
from app.voice import VoiceError, VoiceService  # noqa: E402
from dino import postprocess  # noqa: E402
from dino.letterbox import Letterbox  # noqa: E402

mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/manifest+json", ".webmanifest")

APP_ID = "dino-studio"
VERSION = "0.1.0"
HUB_REPO = "dotimothy/sima-grounding-dino"
DEFAULT_PORT = 8092
MAX_IMAGE_BYTES = 40 << 20
PALETTE = ["#16c8a6", "#3a86ec", "#f2801d", "#a9c81c", "#e5484d", "#b36bff", "#ffc21a", "#1fc3e8",
           "#ff40a0", "#4bb54a", "#8f9bff", "#ff8a65"]


class StudioError(Exception):
    pass


UPSTREAM = "IDEA-Research/grounding-dino-base"
FAMILY = "Grounding DINO Base"
SPEED = {576: "Fast", 768: "Accurate"}
# Each size of Grounding DINO this studio runs: upstream weights, name, parameters, backbone.
VARIANTS = {
    "base": {"upstream": UPSTREAM, "family": FAMILY, "short": "DINO-B", "params": "232.3 M", "swin": "Swin-B"},
    "tiny": {"upstream": "IDEA-Research/grounding-dino-tiny", "family": "Grounding DINO Tiny", "short": "DINO-T",
             "params": "172.2 M", "swin": "Swin-T"},
}


def variant_of(name: str, manifest: dict | None = None) -> str:
    v = (manifest or {}).get("variant")
    return v if v in VARIANTS else ("tiny" if name.startswith("dino-tiny") else "base")
PREC_LABEL = {"bf16": "bf16", "w8a16": "int8 weights"}


ESAM = {"ti": {"key": "ti", "name": "Ti", "params": "10.2 M"}, "s": {"key": "s", "name": "S", "params": "26.4 M"}}
MAX_MASKS = 16          # masks per frame (the highest-scoring boxes above the thresholds)


def model_info(name: str | None, manifest: dict | None = None) -> dict | None:
    """What exactly a model is, for the page: names, sizes, how it runs."""
    if not name:
        return None
    m = manifest or {}
    size = m.get("size")
    if size is None:
        g = re.search(r"-(\d{3})-", name)
        size = int(g.group(1)) if g else None
    prec = m.get("precision") or (name.rsplit("-", 1)[-1] if "-" in name else "bf16")
    if m.get("kind") == "esam" or name.startswith("esam"):
        v = ESAM.get(str(m.get("variant", "")).rsplit("-", 1)[-1], ESAM["ti"])     # manifest: efficientsam-ti
        g = re.search(r"esam-(ti|s)(\d{3})", name)
        if g:
            v, size = ESAM[g.group(1)], size or int(g.group(2))
        n = len(m.get("programs", []))
        return {"name": name, "kind": "esam", "size": size, "precision": prec, "upstream": "yformer/EfficientSAM",
                "variant": v["key"], "title": f"EfficientSAM {v['name']} · Masks · {size} px · {prec}",
                "short": f"ESAM {v['name']}",
                "details": f"yformer/EfficientSAM · EfficientSAM-{v['name']}, {v['params']} parameters (ViT-{v['name']}) · "
                           f"{size}×{size} input · {prec} · masks for the detector's boxes" + (f" · {n} ELFs" if n else "")}
    if m.get("kind") == "sam3" or name.startswith("sam3"):
        n = len(m.get("programs", []))
        return {"name": name, "kind": "sam3", "size": size, "precision": prec, "upstream": "facebook/sam3",
                "variant": "sam3", "title": f"SAM 3 · {size} px · {prec}", "short": f"SAM3 {size}",
                "details": f"facebook/sam3 · SAM 3, 840 M parameters (Perception Encoder ViT + CLIP text) · "
                           f"{size}×{size} input · {prec} · boxes and masks" + (f" · {n} ELFs" if n else "")}
    v = VARIANTS[variant_of(name, m)]
    if m.get("kind") == "text" or "text" in name:
        return {"name": name, "kind": "text", "title": f"{v['family']} · Text Encoder (BERT Base)",
                "short": "Text Encoder", "precision": prec, "upstream": v["upstream"], "variant": variant_of(name, m),
                "details": f"BERT Base (110 M parameters) + projection · 64 text tokens · {prec} · 1 ELF"}
    elfs = len(m.get("stages", [])) + sum(1 for st in m.get("backbone", []) if st.get("op") == "run")
    speed = SPEED.get(size, "") if v is VARIANTS["base"] else ""
    pl = PREC_LABEL.get(prec, prec)
    how = {"w8a16": "int8 weights for the backbone and encoder, bf16 activations"}.get(prec, prec)
    return {"name": name, "kind": "detector", "size": size, "precision": prec, "upstream": v["upstream"],
            "variant": variant_of(name, m),
            "title": f"{v['family']} · {speed} {size} px · {pl}".replace("  ", " "),
            "short": f"{v['short']} {size}" + (" int8" if prec == "w8a16" else ""),
            "details": f"{v['upstream']} · {v['family']}, {v['params']} parameters ({v['swin']} + BERT Base) · "
                       f"{size}×{size} input · {how}"
                       + (f" · {elfs} ELFs + text encoder" if elfs else "")}


def stage_label(stage: str) -> str:
    """dino_rt stage name -> words for the loading bar."""
    m = re.fullmatch(r"bb_s(\d)_b(\d+)", stage)
    if m:
        return f"Swin backbone · stage {m.group(1)} block {int(m.group(2)) + 1}"
    m = re.fullmatch(r"bbm?_stage(\d)([abc]?)", stage)
    if m:
        part = {"a": " (blocks 1–6)", "b": " (blocks 7–12)", "c": " (blocks 13–18)"}.get(m.group(2), "")
        return f"Swin backbone · stage {m.group(1)}{part}"
    m = re.fullmatch(r"bbm?_proj(\d)", stage)
    if m:
        return f"Backbone → feature level {m.group(1)}"
    if stage in ("bb_level23", "bbm_level23"):
        return "Backbone → feature levels 2–3"
    m = re.fullmatch(r"enc(\d)", stage)
    if m:
        k = int(m.group(1))
        return f"Encoder layer {k + 1}" if k < 6 else "Encoder output + query selection head"
    if stage == "dec_values":
        return "Decoder value maps"
    m = re.fullmatch(r"dec_(pre|post)(\d)", stage)
    if m:
        return f"Decoder layer {int(m.group(2)) + 1} · " + ("attention" if m.group(1) == "pre" else "feed-forward + box head")
    if stage == "text":
        return "Text encoder (BERT Base)"
    return stage


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def default_models_dir() -> Path:
    """DINO_MODELS_DIR, else what setup.sh --models-dir remembered, else /media/nvme/models/dino, else models/."""
    env = os.environ.get("DINO_MODELS_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    remembered = ROOT / ".models-dir"
    if remembered.is_file() and remembered.read_text().strip():
        return Path(remembered.read_text().strip()).expanduser()
    if Path("/media/nvme").is_dir():
        return Path("/media/nvme/models/dino")
    return ROOT / "models"


def _free_bytes(path: Path) -> int:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- local bundles
def read_bundle(d: Path) -> dict | None:
    try:
        m = json.loads((d / "manifest.json").read_text())
    except (OSError, ValueError):
        return None
    if m.get("kind") not in ("detector", "text", "sam3", "esam"):
        return None
    return m


def local_bundles(models_dir: Path) -> dict[str, dict]:
    out = {}
    if Path(models_dir).is_dir():
        for d in sorted(Path(models_dir).iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                m = read_bundle(d)
                if m:
                    out[d.name] = m
    return out


def verify_bundle(d: Path) -> tuple[bool, str]:
    m = read_bundle(d)
    if not m:
        return False, "no readable manifest.json"
    for rel, f in m.get("files", {}).items():
        p = d / rel
        if not p.is_file() or p.stat().st_size != f["bytes"]:
            return False, f"{rel} is missing or the wrong size"
        if _sha256(p) != f["sha256"]:
            return False, f"{rel} does not match its checksum"
    return True, "all files match"


def bundle_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- Hugging Face hub
class Hub:
    """Model bundles published on Hugging Face (tools/publish_hub.py) and fetching them.

    models.json lists each bundle as one .tar.gz with its size and SHA-256. A download goes
    to <models>/<name>.tar.gz.part (a stopped download continues), is checked, unpacked into a
    hidden directory, checked file by file against the bundle manifest, and only then moved
    to <models>/<name>, so the model list never sees half a model.
    """

    REFRESH = 300
    NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,48}")
    FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\.tar\.gz")
    SPARE = 1 << 30

    def __init__(self, repo: str, models_dir: Path):
        self.repo, self.dir = repo, Path(models_dir)
        self._lock = threading.Lock()
        self._models: dict | None = None
        self._fetched = 0.0
        self._fetching = False
        self._error: str | None = None
        self._job: dict | None = None
        self._failed: dict[str, str] = {}
        self.quiet = False

    def url(self, path: str) -> str:
        base = os.environ.get("DINO_HUB_URL")              # a stand-in server, for tests
        return f"{base.rstrip('/')}/{path}" if base else f"https://huggingface.co/{self.repo}/resolve/main/{path}"

    def _refresh(self) -> None:
        try:
            req = urllib.request.Request(self.url("models.json"), headers={"User-Agent": "neat-dino-studio"})
            with urllib.request.urlopen(req, timeout=15) as r:
                doc = json.loads(r.read(1 << 20))
            models = {}
            for name, e in doc["models"].items():
                files = e["files"]
                if not (self.NAME.fullmatch(name) and re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,48}", e["path"])
                        and len(files) == 1 and self.FILE.fullmatch(files[0]["name"])
                        and isinstance(files[0]["bytes"], int) and re.fullmatch(r"[0-9a-f]{64}", files[0]["sha256"])
                        and e.get("kind") in ("detector", "text", "sam3", "esam")):
                    raise ValueError(f"the entry for {name!r} is not one this app understands")
                models[name] = e
            self._models, self._error = models, None
        except (OSError, ValueError, KeyError, TypeError) as e:
            self._error = f"Could not read the model list from Hugging Face: {e}"
        finally:
            self._fetched, self._fetching = time.monotonic(), False

    def models(self, wait: bool = False) -> dict:
        if not self.repo:
            return {}
        with self._lock:
            stale = time.monotonic() - self._fetched > (self.REFRESH if self._models is not None else 15)
            if stale and not self._fetching:
                self._fetching = True
                t = threading.Thread(target=self._refresh, name="hub-list", daemon=True)
                t.start()
                if wait:
                    t.join(20)
        return self._models or {}

    def _part(self, name: str) -> Path:
        return self.dir / f".{name}.tar.gz.part"

    def job(self) -> dict | None:
        j = self._job
        if j is None or not j["thread"].is_alive():
            return None
        secs = time.monotonic() - j["started"]
        rate = j["fetched"] / secs if secs > 1 else 0
        total = j["total"]
        return {"name": j["name"], "stage": j["stage"], "fraction": round(j["done"] / total, 4) if total else 0,
                "done_bytes": j["done"], "total_bytes": total, "bytes_per_second": int(rate),
                "seconds_left": int((total - j["done"]) / rate) if rate else None}

    def failed(self, name: str) -> str | None:
        return self._failed.get(name)

    def download(self, name: str, replace: bool = False) -> None:
        e = self.models(wait=True).get(name)
        if e is None:
            raise StudioError(f"Hugging Face lists no model named {name!r}")
        f = e["files"][0]
        with self._lock:
            if self._job is not None and self._job["thread"].is_alive():
                raise StudioError(f"{self._job['name']} is downloading; one download at a time")
            if not replace and (self.dir / name / "manifest.json").is_file():
                raise StudioError(f"{name} is already on this board")
            part = self._part(name)
            need = f["bytes"] - (part.stat().st_size if part.is_file() else 0)
            free = _free_bytes(self.dir)
            if free < need + f["bytes"] + self.SPARE:       # the archive and the unpacked bundle
                raise StudioError(f"Not enough disk for {name}: {(need + f['bytes']) >> 20} MB needed, "
                                  f"{free >> 20} MB free")
            self._failed.pop(name, None)
            job = {"name": name, "done": 0, "fetched": 0, "total": f["bytes"], "stage": "download",
                   "started": time.monotonic(), "cancel": threading.Event()}
            job["thread"] = threading.Thread(target=self._run, args=(job, e), name="hub-download", daemon=True)
            self._job = job
            job["thread"].start()

    def cancel(self, name: str) -> None:
        j = self._job
        if j is not None and j["name"] == name:
            j["cancel"].set()

    def discard(self, name: str) -> None:
        j = self._job
        if j is not None and j["name"] == name and j["thread"].is_alive():
            raise StudioError(f"{name} is downloading; cancel it first")
        self._part(name).unlink(missing_ok=True)
        self._failed.pop(name, None)

    def partial_bytes(self, name: str) -> int:
        p = self._part(name)
        j = self._job
        if j is not None and j["name"] == name and j["thread"].is_alive():
            return 0
        return p.stat().st_size if p.is_file() else 0

    def _run(self, job: dict, e: dict) -> None:
        name, f = job["name"], e["files"][0]
        part = self._part(name)
        tmp = self.dir / f".{name}.unpack"
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            have = part.stat().st_size if part.is_file() else 0
            if have > f["bytes"]:
                part.unlink()
                have = 0
            digest = hashlib.sha256()
            if have:
                with part.open("rb") as old:
                    while chunk := old.read(1 << 22):
                        digest.update(chunk)
            job["done"] = have
            headers = {"User-Agent": "neat-dino-studio", **({"Range": f"bytes={have}-"} if have else {})}
            req = urllib.request.Request(self.url(f"{e['path']}/{f['name']}"), headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have, digest, job["done"] = 0, hashlib.sha256(), 0
                with part.open("ab" if have else "wb") as out:
                    while chunk := r.read(1 << 20):
                        if job["cancel"].is_set():
                            raise InterruptedError()
                        out.write(chunk)
                        digest.update(chunk)
                        job["done"] += len(chunk)
                        job["fetched"] += len(chunk)
                    out.flush()
                    os.fsync(out.fileno())
            size = part.stat().st_size
            if size != f["bytes"] or digest.hexdigest() != f["sha256"]:
                part.unlink(missing_ok=True)
                raise ValueError(f"{f['name']} arrived damaged ({size} of {f['bytes']} bytes); try again")
            job["stage"] = "unpack"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            with tarfile.open(part) as t:
                for m in t.getmembers():
                    if not (m.isfile() or m.isdir()) or m.name.startswith(("/", "..")) or ".." in Path(m.name).parts:
                        raise ValueError(f"unexpected entry {m.name!r} in {f['name']}")
                t.extractall(tmp, filter="data") if sys.version_info >= (3, 12) else t.extractall(tmp)
            inner = [p for p in tmp.iterdir() if p.is_dir()]
            src = inner[0] if len(inner) == 1 and not (tmp / "manifest.json").exists() else tmp
            job["stage"] = "verify"
            ok, why = verify_bundle(src)
            if not ok:
                raise ValueError(f"the unpacked model is not right: {why}")
            dst = self.dir / name
            old = self.dir / f".{name}.old"
            if dst.exists():
                os.replace(dst, old)
            os.replace(src, dst)
            shutil.rmtree(old, ignore_errors=True)
            shutil.rmtree(tmp, ignore_errors=True)
            part.unlink(missing_ok=True)
            if not self.quiet:
                log(f"downloaded {name} from {self.repo} ({f['bytes'] >> 20} MB)")
        except InterruptedError:
            self._failed[name] = "Download cancelled. What arrived is kept; Download continues from there."
        except (OSError, ValueError, tarfile.TarError) as err:
            self._failed[name] = f"The download stopped: {err}"
            log(f"download of {name} failed: {err}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- the studio
class Studio:
    """Model manager + detection. Every load/unload and every detection holds `lock`."""

    def __init__(self, args):
        self.args = args
        self.models_dir = Path(args.models)
        self.hub = Hub(args.hub_repo, self.models_dir)
        self.lock = threading.RLock()
        self.backend: backends.Backend | None = None
        self.model: str | None = None
        self.state = "empty"
        self.stage: str | None = None
        self.error: str | None = None
        self.load_started = 0.0
        self.load_seconds: float | None = None
        self.progress = {"done": 0, "total": 0, "stage": ""}
        self.last_image: str | None = None
        self.last_ms: float | None = None
        # Voice prompts (Whisper + an LLM on the MLA, in a sidecar); its LLM is also the prompt assist.
        # one MLA client connects and loads at a time (detector or voice)
        self.mla_load_lock = threading.Lock()
        self.voice = VoiceService(models_dir=Path(getattr(args, "llima_models", "/media/nvme/llima/models")),
                                  port=getattr(args, "voice_port", 9987), log=log, load_lock=self.mla_load_lock)
        self.assist = self.voice
        self.prefs_path = self.models_dir / ".studio.json"
        self.events: list[dict] = []
        self._verified: dict[str, tuple] = {}
        # Masks for the detector's boxes (EfficientSAM, its own MLA runtime next to the detector)
        self.masker: backends.EsamBackend | None = None
        self.mask_model: str | None = None
        self.mask_state = "off"           # off | loading | ready | error
        self.mask_error: str | None = None

    # ---- catalog
    def catalog(self) -> dict:
        local = local_bundles(self.models_dir)
        hub = self.hub.models()
        names = sorted(set(local) | set(hub), key=lambda n: (0 if (local.get(n) or hub.get(n, {})).get("kind") in
                                                              ("detector", "sam3") else 1, n))
        job = self.hub.job()
        out = []
        for n in names:
            m, e = local.get(n), hub.get(n)
            meta = m or e
            kind = meta.get("kind")
            requires = (m or {}).get("requires") or (e or {}).get("requires")
            state = "on_board" if m else "available"
            if job and job["name"] == n:
                state = "downloading"
            mark = self._verified.get(n)
            info = model_info(n, m or e)
            out.append({
                "name": n, "kind": kind, "title": info["title"], "short": info["short"], "details": info["details"],
                "variant": info.get("variant", "base"),
                "about": (e or {}).get("about", ""), "size": meta.get("size"), "precision": meta.get("precision"),
                "latency_ms": (e or {}).get("latency_ms"), "requires": requires,
                "requires_present": (requires in local) if requires else True,
                "state": state, "bytes": bundle_bytes(self.models_dir / n) if m else (e or {}).get("files", [{}])[0].get("bytes"),
                "download_bytes": (e or {}).get("files", [{}])[0].get("bytes") if e else None,
                "partial_bytes": self.hub.partial_bytes(n) if e else 0,
                "loaded": (n == self.mask_model and self.mask_state in ("ready", "loading")) if kind == "esam"
                else n == self.model and self.state in ("ready", "loading"),
                "loading": (n == self.mask_model and self.mask_state == "loading") if kind == "esam"
                else n == self.model and self.state == "loading",
                "verified": mark[1] if mark else None, "verify_note": mark[2] if mark else None,
                "progress": job if job and job["name"] == n else None, "error": self.hub.failed(n),
                "on_hub": e is not None,
            })
        return {"models": out, "hub": {"repo": self.hub.repo, "page": f"https://huggingface.co/{self.hub.repo}" if
                                       self.hub.repo else None, "error": self.hub._error,
                                       "listed": self.hub._models is not None},
                "directory": str(self.models_dir), "free_bytes": _free_bytes(self.models_dir),
                "used_bytes": sum(bundle_bytes(self.models_dir / n) for n in local)}

    @staticmethod
    def _title(meta, name):
        tiny = variant_of(name, meta) == "tiny"
        if meta.get("kind") == "text":
            return "Tiny Text Encoder · BERT (64 Tokens)" if tiny else "Text Encoder · BERT (64 Tokens)"
        size = meta.get("size")
        if tiny:
            return f"Tiny · {size} px" if size else name
        return {768: "Accurate · 768 px", 576: "Fast · 576 px"}.get(size, name)

    def status(self) -> dict:
        b = self.backend
        prog = None
        if self.state in ("loading", "unloading"):
            d, t, st = self.progress["done"], self.progress["total"], self.progress["stage"]
            labels = {"release": "Unloading the previous model", "unload": "Unloading",
                      "load": "Starting the MLA runtime", "warm": "Warming up (first detection)"}
            label = labels.get(self.stage, self.stage or "")
            if self.stage == "load" and t:
                label = f"Loading ELF {min(d + 1, t)} of {t}"
            # load is ~90 % of the time; release and warm-up take the ends of the bar
            frac = {"release": 0.02, "unload": None, "warm": 0.95}.get(self.stage)
            if self.stage == "load":
                frac = 0.05 + 0.88 * (d / t) if t else 0.05
            prog = {"step": self.stage, "label": label, "detail": stage_label(st) if st and st != "done" else "",
                    "done": d, "total": t, "fraction": frac}
        info = None
        if self.model:
            info = model_info(self.model, read_bundle(self.models_dir / self.model) or {})
        return {"app": APP_ID, "version": VERSION, "state": self.state, "model": self.model, "stage": self.stage,
                "model_info": info, "progress": prog,
                "error": self.error, "backend": self.args.backend, "size": b.size if b else None,
                "loading_seconds": round(time.time() - self.load_started, 1) if self.state == "loading" else None,
                "load_seconds": self.load_seconds, "last_ms": self.last_ms,
                "https": bool(getattr(self.args, "tls", False)), "port": self.args.port,
                "assist": self.assist.describe(), "max_tokens": 64,
                "pipeline": getattr(b, "slots", 1) if b else 1, "voice": self.voice.status(),
                "masks": self.masks_status()}

    # ---- masks (EfficientSAM)
    def masks_status(self) -> dict:
        local = local_bundles(self.models_dir)
        choices = [{"name": n, "short": model_info(n, m)["short"], "title": model_info(n, m)["title"]}
                   for n, m in sorted(local.items(), key=lambda x: ("-s" in x[0], x[0])) if m.get("kind") == "esam"]
        if self.args.backend == "fake":
            choices = choices or [{"name": "esam-ti768-bf16", "short": "ESAM Ti", "title": "EfficientSAM Ti (fake)"}]
        mk = self.masker
        return {"model": self.mask_model, "state": self.mask_state, "error": self.mask_error, "choices": choices,
                "mask_size": mk.mask_size if mk else None, "max": MAX_MASKS,
                "info": model_info(self.mask_model, local.get(self.mask_model, {})) if self.mask_model else None}

    def set_masks(self, name: str | None, who: str = "") -> None:
        """Load EfficientSAM `name` for masks (replacing the current one), or None to turn masks off."""
        if name and self.args.backend != "fake":
            m = local_bundles(self.models_dir).get(name)
            if m is None or m.get("kind") != "esam":
                raise StudioError(f"{name} is not a mask model on this board")
        with self.lock:
            if self.mask_state == "loading":
                raise StudioError("The mask model is still loading")
            if name == self.mask_model and self.mask_state == "ready":
                return
            old, self.masker = self.masker, None
            self.mask_model, self.mask_error = name, None
            self.mask_state = "loading" if name else "off"
            self._record(f"masks on ({name})" if name else "masks off", who)
        self._save_prefs(masks=name)
        threading.Thread(target=self._masks_load, args=(old, name), name="masks-load", daemon=True).start()

    def _masks_load(self, old, name) -> None:
        try:
            if old is not None:
                old.close()
            if not name:
                return
            if self.args.backend == "fake":
                b = backends.FakeEsamBackend()
            elif self.args.backend != "mla":
                raise StudioError("masks need the board (esam_rt)")
            else:
                with self.mla_load_lock:
                    b = backends.EsamBackend(Path(self.args.runtime).with_name("esam_rt"), self.models_dir / name,
                                             log=lambda s: log(f"  esam_rt: {s}"))
            with self.lock:
                if self.mask_model == name:
                    self.masker, self.mask_state = b, "ready"
                    log(f"masks ready: {name}")
                    return
            b.close()
        except Exception as e:  # noqa: BLE001
            with self.lock:
                if self.mask_model == name:
                    self.mask_state, self.mask_error = "error", f"Loading {name} failed: {e}"
            log(f"masks: loading {name} failed: {e}")

    # ---- preferences that survive a restart (voice on/off)
    def prefs(self) -> dict:
        try:
            return json.loads(self.prefs_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save_prefs(self, **kw) -> None:
        p = {**self.prefs(), **kw}
        try:
            self.prefs_path.write_text(json.dumps(p))
        except OSError as e:
            log(f"could not save preferences: {e}")

    # ---- voice
    def voice_start(self, llm: str | None = None, who: str = "") -> None:
        try:
            self.voice.start(llm=llm)
        except VoiceError as e:
            raise StudioError(str(e)) from e
        self._record(f"voice on ({self.voice.asr} + {self.voice.llm or 'no LLM'})", who)
        self._save_prefs(voice=True, voice_llm=self.voice.llm)

    def voice_stop(self, who: str = "") -> None:
        self._record("voice off", who)
        self._save_prefs(voice=False)
        threading.Thread(target=self.voice.stop, name="voice-stop", daemon=True).start()

    def voice_prompt(self, audio: bytes, filename: str, mime: str) -> dict:
        """Speech -> transcript (Whisper) -> prompt (LLM, or rules when there is none)."""
        try:
            tr = self.voice.transcribe(audio, filename, mime)
        except VoiceError as e:
            raise StudioError(str(e)) from e
        if not tr["text"] or (tr["no_speech"] > 0.6 and len(tr["text"]) < 4):
            log(f"voice: no speech in {len(audio)} bytes of {mime} (transcript {tr['text']!r}, "
                f"no_speech {tr['no_speech']:.2f})")
            raise StudioError("Didn’t catch that. Click Speak and say what to find.")
        pr = self.voice.to_prompt(tr["text"])
        return {"transcript": tr["text"], "prompt": pr["prompt"], "by": pr["by"],
                "timing": {"asr": tr["ms"], "llm": pr["ms"]}, "llm": self.voice.llm}

    # ---- loading
    def _record(self, what: str, who: str = "") -> None:
        self.events.append({"t": time.time(), "what": what, "who": who})
        del self.events[:-50]
        log(f"{what}" + (f" (from {who})" if who else ""))

    def load(self, name: str, who: str = "") -> None:
        with self.lock:
            if self.state == "loading":
                raise StudioError(f"{self.model} is still loading")
            local = local_bundles(self.models_dir)
            if self.args.backend == "mla":
                m = local.get(name)
                if m is None or m.get("kind") not in ("detector", "sam3"):
                    raise StudioError(f"{name} is not a detector on this board")
                req = m.get("requires")
                if req and req not in local:
                    raise StudioError(f"{name} needs {req}; download it first")
            self.state, self.stage, self.error = "loading", "release", None
            self.progress = {"done": 0, "total": 0, "stage": ""}
            self.load_started = time.time()
            previous = self.model
            self.model = name
            self._record(f"load {name}" + (f" (replacing {previous})" if previous else ""), who)
        threading.Thread(target=self._load, args=(name,), name="model-load", daemon=True).start()

    def _load(self, name: str) -> None:
        try:
            with self.lock:
                if self.backend is not None:
                    self.backend.close()
                    self.backend = None
                    time.sleep(0.5)               # never two detectors on the MLA at once
                self.stage = "load"
                with self.mla_load_lock:
                    b = self._make_backend(name)
                self.stage = "warm"
                pr = b.build_prompt("object .")
                b.set_prompt(pr)
                gray = np.full((b.size, b.size, 3), 114, np.uint8)
                b.detect(gray, b.size, b.size)
                self.backend, self.state, self.stage = b, "ready", None
                self.load_seconds = round(time.time() - self.load_started, 1)
                self.last_image = None
                log(f"{name} ready in {self.load_seconds} s")
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.backend, self.state, self.stage = None, "error", None
                self.error = f"Loading {name} failed: {e}"
            log(self.error)
            traceback.print_exc()

    def _make_backend(self, name: str) -> backends.Backend:
        kind = self.args.backend
        if kind == "fake":
            # walk through the same progress steps as the board, quickly
            total = 49
            for i in range(total):
                self.progress = {"done": i, "total": total, "stage": f"bb_s{1 + i // 13}_b{i % 13}"}
                time.sleep(getattr(self.args, "fake_load_seconds", 0.0) / total)
            self.progress = {"done": total, "total": total, "stage": "done"}
            return backends.FakeBackend(int(re.search(r"(\d{3})", name).group(1)) if re.search(r"\d{3}", name) else 576)
        if kind == "torch":
            size = int(re.search(r"(\d{3})", name).group(1)) if re.search(r"\d{3}", name) else 576
            return backends.TorchBackend(size)
        m = read_bundle(self.models_dir / name)

        def progress(done, total, stage):
            self.progress = {"done": done, "total": total, "stage": stage}
        if m.get("kind") == "sam3":
            binary = Path(self.args.runtime).with_name("sam3_rt")
            return backends.Sam3Backend(binary, self.models_dir / name, log=lambda s: log(f"  sam3_rt: {s}"),
                                        progress=progress)
        text = self.models_dir / m.get("requires", "dino-text-t64-bf16")
        return backends.RuntimeBackend(self.args.runtime, self.models_dir / name, text,
                                       log=lambda s: log(f"  dino_rt: {s}"), progress=progress)

    def unload(self, who: str = "", wait: bool = False) -> None:
        with self.lock:
            if self.state in ("loading", "unloading"):
                raise StudioError("a model is loading or unloading; wait for it")
            self._record(f"unload {self.model}", who)
            self.state, self.stage = "unloading", "unload"
            self.progress = {"done": 0, "total": 0, "stage": ""}

        def work():
            with self.lock:
                if self.backend is not None:
                    self.backend.close()        # SIGTERM to dino_rt: it frees the MLA
                self.backend, self.model, self.state, self.stage, self.error = None, None, "empty", None, None
                self.last_image = None
        t = threading.Thread(target=work, name="model-unload", daemon=True)
        t.start()
        if wait:
            t.join()

    def delete(self, name: str, who: str = "") -> None:
        with self.lock:
            if (name == self.model and self.state in ("ready", "loading")) or \
                    (name == self.mask_model and self.mask_state in ("ready", "loading")):
                raise StudioError(f"{name} is loaded; unload it first")
            d = self.models_dir / name
            if not (d / "manifest.json").is_file():
                raise StudioError(f"{name} is not on this board")
            shutil.rmtree(d)
            self._verified.pop(name, None)
            self._record(f"deleted {name}", who)

    def verify(self, name: str) -> bool:
        d = self.models_dir / name
        if not d.is_dir():
            raise StudioError(f"{name} is not on this board")
        ok, why = verify_bundle(d)
        self._verified[name] = (time.time(), ok, why)
        return ok

    # ---- detection
    def parse(self, text: str) -> dict:
        b = self.backend
        if b is None:
            from dino import prompt as P
            tok = getattr(self, "_tok", None) or P.Tokenizer(self._any_text_bundle())
            self._tok = tok
            pr = P.build(text, tok, 64)
        else:
            pr = b.build_prompt(text)
        return {"caption": pr.caption, "phrases": [{"text": p, "color": PALETTE[i % len(PALETTE)]}
                                                   for i, p in enumerate(pr.phrases)],
                "tokens": pr.n_tokens, "max_tokens": 64, "truncated": pr.truncated}

    def _any_text_bundle(self):
        for n, m in local_bundles(self.models_dir).items():
            if m.get("kind") == "text":
                return self.models_dir / n
        return None

    def detect_sam3(self, image_bytes, text, image_id, nms):
        """SAM 3: per phrase boxes, scores and masks (masks cover the whole stretched image)."""
        import base64
        from dino.letterbox import Squash
        t0 = time.time()
        same = image_bytes is None
        b = self.backend
        rgb = geo = None
        if not same:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
            geo = Squash(b.size, img.width, img.height)
            rgb = np.asarray(img.resize((b.size, b.size), Image.BILINEAR), np.uint8)
        dec_ms = (time.time() - t0) * 1e3
        phrases = b.phrases(text)
        if not phrases:
            raise StudioError("Type at least one thing to find, e.g. “person . dog .”")
        with self.lock:
            if same:
                if image_id is None or image_id != self.last_image:
                    raise StudioError("The image is gone from the board; send it again")
                geo = self._last_lb
            res = b.detect_phrases(rgb, phrases)
            if not same:
                self.last_image = hashlib.sha1(image_bytes).hexdigest()[:16]
                self._last_lb = geo
            model, image_ref = self.model, self.last_image
        t_post = time.time()
        dets = []
        P = len(phrases)
        for p, (_, r, _) in enumerate(res):
            sc = r.scores
            keep = np.nonzero(sc > 0.05)[0]
            xyxy = np.round(geo.boxes_to_source(r.boxes[keep]), 1).tolist()
            for j, q in enumerate(keep):
                d = {"box": xyxy[j], "score": round(float(sc[q]), 4), "phrase": p,
                     "phrase_scores": [round(float(sc[q]), 4) if k == p else 0.0 for k in range(P)], "query": int(q)}
                if int(q) in r.masks:
                    d["mask"] = base64.b64encode(np.packbits(r.masks[int(q)].ravel())).decode()
                dets.append(d)
        dets.sort(key=lambda d: -d["score"])
        if nms:
            dets = postprocess.nms(dets, nms)
        tm = [r.timing for _, r, _ in res]
        timing = {"text": round(sum(t for _, _, t in res), 1),
                  "encoder_mla": round(sum(t.get("encoder", 0) for t in tm), 1),
                  "decoder_mla": round(sum(t.get("decoder_mla", 0) for t in tm), 1),
                  "decoder_cpu": round(sum(t.get("decoder_cpu", 0) for t in tm), 1)}
        if not same:
            t = tm[0]
            timing["backbone"] = round(t.get("vit", 0) + t.get("neck", 0), 1)
            timing["backbone_cpu"] = round(t.get("patchify", 0) + t.get("to_image", 0), 1)
            timing["decode"] = round(dec_ms, 1)
        total = (time.time() - t0) * 1e3
        timing["postprocess"] = round((time.time() - t_post) * 1e3, 1)
        timing["server_total"] = round(total, 1)
        self.last_ms = round(total, 1)
        return {"model": model, "size": b.size, "image_id": image_ref, "width": geo.width, "height": geo.height,
                "caption": " . ".join(phrases) + " .",
                "phrases": [{"text": p, "color": PALETTE[i % len(PALETTE)]} for i, p in enumerate(phrases)],
                "truncated": False, "tokens": len(phrases), "detections": dets[:300], "timing": timing,
                "same_image": same, "mask_size": b.mask_size}

    def detect(self, image_bytes: bytes | None, text: str, image_id: str | None, nms: float | None,
               masks: tuple[float, float] | None = None) -> dict:
        """One detection. Decoding and post-processing run outside the lock, and the runtime
        takes the next frame while this one is still running, so Live mode keeps two in flight.
        masks = (box score, phrase score) thresholds: EfficientSAM masks for the boxes above them."""
        if getattr(self.backend, "kind", "") == "sam3" and self.state == "ready":
            return self.detect_sam3(image_bytes, text, image_id, nms)
        t0 = time.time()
        same = image_bytes is None
        dec_ms = 0.0
        rgb = lb = None
        mk = self.masker if masks and self.mask_state == "ready" else None
        job = None
        if not same:
            b = self.backend
            if b is not None and self.state == "ready":
                rgb, lb, img = self._letterbox(image_bytes, b.size, keep=mk is not None)
                dec_ms = (time.time() - t0) * 1e3
                if mk is not None:      # the mask encoder runs while the detector does
                    sq = np.asarray(img.resize((mk.size, mk.size), Image.BILINEAR), np.uint8)
                    job = mk.start(sq, hashlib.sha1(image_bytes).hexdigest()[:16])
        try:
            return self._detect(image_bytes, text, image_id, nms, masks, t0, same, dec_ms, rgb, lb, mk, job)
        finally:
            if job is not None:
                job.boxes(None)

    def _detect(self, image_bytes, text, image_id, nms, masks, t0, same, dec_ms, rgb, lb, mk, job) -> dict:
        with self.lock:
            if self.backend is None or self.state != "ready":
                raise StudioError("No model is loaded. Open Settings → Models to load one.")
            b = self.backend
            pr = b.build_prompt(text)
            if not pr.phrases:
                raise StudioError("Type at least one thing to find, e.g. “person . dog .”")
            text_ms = b.set_prompt(pr)
            if same:
                # Only the prompt changed: the backbone output of this image is still on the board.
                if image_id is None or image_id != self.last_image:
                    raise StudioError("The image is gone from the board; send it again")
                lb = self._last_lb
                fut = b.detect_async(None, lb.new_w, lb.new_h, same_image=True)
            else:
                if rgb is None or rgb.shape[0] != b.size:          # model changed meanwhile
                    t_dec = time.time()
                    rgb, lb, _ = self._letterbox(image_bytes, b.size)
                    dec_ms = (time.time() - t_dec) * 1e3
                fut = b.detect_async(rgb, lb.new_w, lb.new_h)
                self.last_image = hashlib.sha1(image_bytes).hexdigest()[:16]
                self._last_lb = lb
            model, image_ref = self.model, self.last_image
            if mk is not None and job is None:                 # prompt change: the masker reuses its image
                job = mk.start(None, image_ref)
        det = fut.result()
        cands = postprocess.candidates(det.logits, det.boxes, pr, lb, floor=0.05, nms_iou=nms)
        t_det = time.time()
        extra, mtime = {}, {}
        if job is not None:
            extra, mtime = self._masks_for(mk, job, cands, masks, lb.width, lb.height)
        total = (time.time() - t0) * 1e3
        self.last_ms = round(total, 1)
        timing = {k: round(v, 1) for k, v in det.timing.items() if v and k != "slot"}
        timing["text"] = round(text_ms, 1)
        if not same:
            timing["decode"] = round(dec_ms, 1)
        timing["postprocess"] = round((t_det - t0) * 1e3 - sum(
            v for k, v in timing.items() if k in ("total", "text", "decode")), 1)
        timing.update(mtime)
        timing["server_total"] = round(total, 1)
        return {"model": model, "size": b.size, "image_id": image_ref,
                "width": lb.width, "height": lb.height, "caption": pr.caption,
                "phrases": [{"text": p, "color": PALETTE[i % len(PALETTE)]} for i, p in enumerate(pr.phrases)],
                "truncated": pr.truncated, "tokens": pr.n_tokens, "detections": cands, "timing": timing,
                "same_image": same, **extra}

    def _masks_for(self, mk, job, cands, thr, W, H) -> tuple[dict, dict]:
        """EfficientSAM masks for the best boxes above the thresholds; adds 'mask' to those detections."""
        import base64
        bt, tt = thr
        sel = [d for d in cands if d["score"] >= bt and d["phrase"] >= 0 and d["phrase_scores"][d["phrase"]] >= tt]
        sel = sel[:MAX_MASKS]
        S_ = mk.size
        boxes = np.array([[x0 * S_ / W, y0 * S_ / H, x1 * S_ / W, y1 * S_ / H] for x0, y0, x1, y1 in
                          (d["box"] for d in sel)], np.float32).reshape(-1, 4)
        job.boxes(boxes)
        try:
            res = job.result.result(timeout=60)
        except Exception as e:  # noqa: BLE001
            log(f"masks failed: {e}")
            return {"mask_error": str(e)}, {}
        tm = {}
        enc = job.encode_ms
        if enc:
            tm["masks_encoder"] = round(enc.get("encoder", 0) + enc.get("neck", 0), 1)
            tm["masks_cpu"] = round(enc.get("patchify", 0), 1)
        if res is not None:
            m, iou, t = res
            for d, mm, sc in zip(sel, m, iou):
                d["mask"] = base64.b64encode(np.packbits(mm.ravel())).decode()
                d["mask_score"] = round(float(sc), 3)
            tm["masks_decoder"] = round(t.get("decoder", 0), 1)
            tm["masks_cpu"] = round(tm.get("masks_cpu", 0) + t.get("masks_cpu", 0), 1)
        return {"mask_size": mk.mask_size, "masks_for": [bt, tt]}, tm

    def _letterbox(self, data: bytes, size: int, keep: bool = False) -> tuple[np.ndarray, Letterbox, Image.Image]:
        img = Image.open(io.BytesIO(data))
        if img.format == "JPEG":
            big = max(size, self.masker.size if keep and self.masker else 0)
            img.draft("RGB", (big, big))                  # decode at 1/2..1/8 scale when big
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        s = size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        sq = np.zeros((size, size, 3), np.uint8)
        sq[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR))
        return sq, Letterbox(size, s, w, h, nw, nh), img

    def reset_mla(self, who: str = "") -> str:
        """The explicit 'Reset MLA' button: unload, then the DevKit's recovery ladder."""
        with self.lock:
            if self.backend is not None:
                self.backend.close()
            self.backend, self.model, self.state, self.stage = None, None, "empty", None
            self._record("reset MLA", who)
            cmds = []
            if os.environ.get("DINO_RESET_COMMAND"):
                cmds.append(["sh", "-c", os.environ["DINO_RESET_COMMAND"]])
            if shutil.which("fix_devkit_runtime.sh"):
                cmds.append(["sudo", "-n", "fix_devkit_runtime.sh"])
            cmds.append(["sudo", "-n", "sh", "-c", "systemctl restart simaai-appcomplex.service && "
                         "( [ -x /usr/bin/init_mla_memory.sh ] && /usr/bin/init_mla_memory.sh || true )"])
            for c in cmds:
                try:
                    r = subprocess.run(c, capture_output=True, text=True, timeout=120)
                    if r.returncode == 0:
                        return f"MLA reset with {' '.join(c[:3])}"
                except (OSError, subprocess.TimeoutExpired):
                    continue
            raise StudioError("Could not reset the MLA (needs passwordless sudo or DINO_RESET_COMMAND)")

    def close(self) -> None:
        self.voice.close()
        with self.lock:
            if self.masker is not None:
                self.masker.close()
                self.masker = None
            if self.backend is not None:
                self.backend.close()
                self.backend = None


# --------------------------------------------------------------------------- web app
def create_app(studio: Studio) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES

    def who():
        return request.remote_addr or ""

    def err(e, code=400):
        return jsonify({"error": str(e)}), code

    @app.get("/")
    @app.get("/models")
    def index():
        return send_from_directory(WEB, "index.html", max_age=0)

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(WEB, name, max_age=0 if name.endswith((".js", ".css", ".html")) else 3600)

    @app.get("/api/status")
    def api_status():
        return jsonify(studio.status())

    @app.get("/api/models")
    def api_models():
        return jsonify(studio.catalog())

    @app.post("/api/models/<action>")
    def api_models_action(action):
        name = (request.get_json(silent=True) or {}).get("name", "")
        try:
            if action == "load":
                studio.load(name, who())
            elif action == "unload":
                studio.unload(who(), wait=bool((request.get_json(silent=True) or {}).get("wait")))
            elif action == "delete":
                studio.delete(name, who())
            elif action == "verify":
                return jsonify({"ok": studio.verify(name), "models": studio.catalog()})
            elif action == "download":
                studio.hub.download(name)
            elif action == "redownload":
                studio.hub.download(name, replace=True)
            elif action == "cancel":
                studio.hub.cancel(name)
            elif action == "discard":
                studio.hub.discard(name)
            else:
                return err(f"unknown action {action}", 404)
        except StudioError as e:
            return err(e, 409)
        return jsonify({"ok": True, "status": studio.status(), "models": studio.catalog()})

    @app.post("/api/prompt/parse")
    def api_parse():
        try:
            return jsonify(studio.parse((request.get_json(silent=True) or {}).get("text", "")))
        except Exception as e:  # noqa: BLE001  (no tokenizer yet: nothing downloaded)
            return err(e, 503)

    @app.post("/api/prompt/assist")
    def api_assist():
        d = studio.assist.describe()
        if not d["available"]:
            return err("Prompt assist needs Voice on (Settings → Voice): it uses the voice LLM", 503)
        r = studio.voice.to_prompt((request.get_json(silent=True) or {}).get("text", ""))
        return jsonify({"text": r["prompt"], "by": r["by"], "ms": r["ms"]})

    @app.post("/api/voice/<action>")
    def api_voice(action):
        try:
            if action == "start":
                studio.voice_start((request.get_json(silent=True) or {}).get("llm") or None, who())
            elif action == "stop":
                studio.voice_stop(who())
            elif action in ("download", "cancel", "discard"):
                name = str((request.get_json(silent=True) or {}).get("name") or "")
                try:
                    studio.voice.download(action, name)
                except VoiceError as e:
                    raise StudioError(str(e)) from e
                studio._record(f"voice model {action}: {name}", who())
            elif action == "prompt":
                f = request.files.get("audio")
                if f is None:
                    return err("Send the recording as 'audio'")
                return jsonify(studio.voice_prompt(f.read(), f.filename or "speech.webm",
                                                   f.mimetype or "audio/webm"))
            else:
                return err(f"unknown action {action}", 404)
        except StudioError as e:
            return err(e, 409)
        return jsonify({"ok": True, "voice": studio.voice.status()})

    @app.post("/api/masks")
    def api_masks():
        name = (request.get_json(silent=True) or {}).get("model") or None
        try:
            studio.set_masks(name, who())
        except StudioError as e:
            return err(e)
        return jsonify(studio.masks_status())

    @app.post("/api/detect")
    def api_detect():
        f = request.files.get("image")
        text = request.form.get("prompt", "")
        image_id = request.form.get("image_id") or None
        nms = request.form.get("nms")
        try:
            nms_v = float(nms) if nms not in (None, "", "0", "off") else None
            data = f.read() if f else None
            if data is None and image_id is None:
                raise StudioError("Send an image")
            mk = request.form.get("masks")              # "box,phrase" thresholds when masks are on
            mk_v = tuple(float(x) for x in mk.split(",")[:2]) if mk else None
            return jsonify(studio.detect(data, text, image_id if data is None else None, nms_v,
                                         masks=mk_v if mk_v and len(mk_v) == 2 else None))
        except StudioError as e:
            return err(e, 409)
        except (OSError, ValueError) as e:
            return err(f"Could not read that image: {e}")
        except RuntimeError as e:                       # the runtime failed (dino_rt error)
            log(f"detect failed: {e}")
            return err(e, 500)

    @app.post("/api/reset-mla")
    def api_reset():
        try:
            return jsonify({"ok": True, "message": studio.reset_mla(who())})
        except StudioError as e:
            return err(e, 500)

    return app


def serve_both_protocols(server, cert: str, key: str) -> None:
    """One port for HTTP and HTTPS: a TLS handshake starts with byte 0x16, an HTTP request never does."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    plain_finish = server.finish_request

    def finish_request(req, client_address):
        try:
            req.settimeout(60)
            first = req.recv(1, socket.MSG_PEEK)
        except OSError:
            return
        if first != b"\x16":
            req.settimeout(None)
            plain_finish(req, client_address)
            return
        try:
            tls = ctx.wrap_socket(req, server_side=True)
        except (ssl.SSLError, OSError):
            return
        tls.settimeout(None)
        try:
            plain_finish(tls, client_address)
        except ssl.SSLError:
            pass
        finally:
            try:
                tls.close()
            except OSError:
                pass

    server.finish_request = finish_request


def hub_command(args) -> int:
    hub = Hub(args.hub_repo, Path(args.models))
    hub.quiet = True
    models = hub.models(wait=True)
    if hub._error:
        print(hub._error, file=sys.stderr)
        return 1
    if args.list_hub:
        local = local_bundles(Path(args.models))
        for n, e in models.items():
            mark = "on board" if n in local else "available"
            print(f"{n:24s} {e['files'][0]['bytes'] >> 20:6d} MB  {mark:10s} {e.get('title', '')}")
        return 0
    names = list(models) if args.fetch == ["all"] else args.fetch
    for n in names:
        e = models.get(n)
        if e and e.get("requires") and e["requires"] not in names and not (Path(args.models) / e["requires"]).is_dir():
            names.append(e["requires"])
    rc = 0
    for n in names:
        if (Path(args.models) / n / "manifest.json").is_file():
            print(f"{n}: already here")
            continue
        try:
            hub.download(n)
        except StudioError as e:
            print(f"{n}: {e}", file=sys.stderr)
            rc = 1
            continue
        while (j := hub.job()) is not None:
            print(f"\r{n}: {j['stage']} {j['fraction'] * 100:5.1f}%  {j['bytes_per_second'] >> 20} MB/s   ",
                  end="", flush=True)
            time.sleep(1)
        print()
        if hub.failed(n):
            print(f"{n}: {hub.failed(n)}", file=sys.stderr)
            rc = 1
        else:
            print(f"{n}: done")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model", default="", help="model to load at start (default: none; load from Settings)")
    ap.add_argument("--models", default=str(default_models_dir()))
    ap.add_argument("--backend", default="mla", choices=["mla", "torch", "fake"],
                    help="mla on the board; torch (GPU) or fake to try the page on a PC")
    ap.add_argument("--fake-load-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--runtime", default=os.environ.get("DINO_RT") or str(ROOT / "runtime/build/dino_rt"))
    ap.add_argument("--hub-repo", default=HUB_REPO)
    ap.add_argument("--llima-models", default=os.environ.get("DINO_LLIMA_MODELS", "/media/nvme/llima/models"),
                    help="LLiMa model folders (Whisper and the LLM for voice prompts)")
    ap.add_argument("--voice-port", type=int, default=9987, help="local port of the voice sidecar")
    ap.add_argument("--voice", choices=["auto", "on", "off"], default="auto",
                    help="start voice at launch: auto = as last time (Settings → Voice)")
    ap.add_argument("--list-hub", action="store_true")
    ap.add_argument("--fetch", nargs="+", metavar="NAME")
    ap.add_argument("--no-https", dest="https", action="store_false")
    ap.add_argument("--cert", default=str(ROOT / "webapp/certs/cert.pem"))
    ap.add_argument("--key", default=str(ROOT / "webapp/certs/key.pem"))
    args = ap.parse_args()
    if args.list_hub or args.fetch:
        return hub_command(args)
    faulthandler.register(signal.SIGUSR1)
    studio = Studio(args)
    if args.model:
        studio.load(args.model, "startup")
    if studio.prefs().get("masks"):                 # masks were on when the studio last stopped
        try:
            studio.set_masks(studio.prefs()["masks"], "startup")
        except StudioError as e:
            log(f"masks: {e}")
    if args.backend == "mla" and (args.voice == "on" or (args.voice == "auto" and studio.prefs().get("voice"))):
        try:
            studio.voice.start(llm=studio.prefs().get("voice_llm"))
        except VoiceError as e:
            log(f"voice not started: {e}")
    server = make_server(args.host, args.port, create_app(studio), threaded=True)
    args.tls = False
    if args.https and Path(args.cert).exists() and Path(args.key).exists():
        serve_both_protocols(server, args.cert, args.key)
        args.tls = True

    def shutdown(signum, _frame):
        log(f"signal {signum}: shutting down")
        threading.Thread(target=server.shutdown, daemon=True).start()

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, shutdown)
    log(f"Neat DINO Studio on port {args.port} (backend {args.backend}, models {args.models})")
    if args.tls:
        log("the same port speaks HTTPS (self-signed) for the webcam")
    try:
        server.serve_forever()
    finally:
        faulthandler.dump_traceback_later(20, exit=True)
        studio.close()
        log("closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
