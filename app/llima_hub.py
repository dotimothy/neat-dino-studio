"""Downloads the LLiMa voice models (Whisper and the prompt LLM) from Hugging Face.

SiMa publishes each precompiled LLiMa model as a repo `simaai/<name>`; Neat GenAI Studio
downloads the whole repo into `<llima models>/<name>` and so does this, so a model fetched
by either app works in both.

A download pins the repo's current revision and goes into `<llima models>/.<name>.partial/`.
Files already there with the right size are kept, so a stopped download continues; large
(LFS) files are checked against their SHA-256. Only a complete download is moved to
`<llima models>/<name>`, so the voice service never sees half a model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ORG = "simaai"
REPO_MARKER = ".neat-hub-repo"           # what Neat GenAI Studio writes next to a downloaded model
SPARE = 1 << 30                          # keep 1 GB of disk free

# name -> (title, role, approximate size shown before the file list is fetched, note)
CATALOG = {
    "whisper-small-a16w8": ("Whisper Small", "asr", "1.3 GB", "Speech to text. Needed for voice."),
    "Gemma-4-E2B-it-TextOnly-GPTQ-a16w4": ("Gemma 4 E2B", "llm", "6.2 GB",
                                           "Turns requests into phrases. Understands indirect ones."),
    "LFM2.5-230M-a16w4": ("LFM2.5 230M", "llm", "0.6 GB", "Turns requests into phrases. Small and fast."),
}

_PATH = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.+@=-]*")


class HubError(RuntimeError):
    pass


def _safe_rel(path: str) -> bool:
    parts = path.split("/")
    return 0 < len(parts) <= 8 and all(_PATH.fullmatch(p) and p not in (".", "..") for p in parts)


def _free_bytes(path: Path) -> int:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


class LlimaHub:
    def __init__(self, models_dir: Path, log=None):
        self.dir = Path(models_dir)
        self._log = log or (lambda s: None)
        self._lock = threading.Lock()
        self._job: dict | None = None
        self._failed: dict[str, str] = {}

    @staticmethod
    def base() -> str:
        return os.environ.get("DINO_LLIMA_HUB_URL", "https://huggingface.co").rstrip("/")

    def _get_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "neat-dino-studio"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read(8 << 20))

    def files(self, name: str) -> tuple[str, list[dict]]:
        """(revision, [{'path', 'size', 'sha256' or None}]) of simaai/<name>."""
        d = self._get_json(f"{self.base()}/api/models/{ORG}/{name}?blobs=true")
        rev = d.get("sha") or "main"
        if not re.fullmatch(r"[0-9a-f]{40}|main", rev):
            raise HubError("Hugging Face returned an unexpected revision")
        out = []
        for s in d.get("siblings", []):
            path, size = s.get("rfilename", ""), s.get("size")
            if not _safe_rel(path) or not isinstance(size, int):
                raise HubError(f"unexpected file {path!r} in {ORG}/{name}")
            sha = (s.get("lfs") or {}).get("sha256")
            out.append({"path": path, "size": size, "sha256": sha if sha and re.fullmatch(r"[0-9a-f]{64}", sha) else None})
        if not out:
            raise HubError(f"{ORG}/{name} has no files")
        return rev, out

    # ---- state
    def present(self, name: str) -> bool:
        return (self.dir / name).is_dir()

    def _partial(self, name: str) -> Path:
        return self.dir / f".{name}.partial"

    def job(self) -> dict | None:
        j = self._job
        if j is None or not j["thread"].is_alive():
            return None
        secs = time.monotonic() - j["started"]
        rate = j["fetched"] / secs if secs > 1 else 0
        total = j["total"]
        return {"name": j["name"], "stage": j["stage"], "fraction": round(j["done"] / total, 4) if total else 0,
                "done_bytes": j["done"], "total_bytes": total, "bytes_per_second": int(rate),
                "seconds_left": int((total - j["done"]) / rate) if rate and total else None}

    def models(self) -> list[dict]:
        j = self.job()
        out = []
        for name, (title, role, size, note) in CATALOG.items():
            part = self._partial(name)
            out.append({"name": name, "title": title, "role": role, "size": size, "note": note,
                        "present": self.present(name), "error": self._failed.get(name),
                        "partial": part.is_dir() and not (j and j["name"] == name),
                        "job": j if j and j["name"] == name else None})
        return out

    # ---- actions
    def download(self, name: str) -> None:
        if name not in CATALOG:
            raise HubError(f"{name!r} is not a voice model this studio uses")
        with self._lock:
            if self._job is not None and self._job["thread"].is_alive():
                raise HubError(f"{CATALOG[self._job['name']][0]} is downloading; one download at a time")
            if self.present(name):
                raise HubError(f"{CATALOG[name][0]} is already on this board")
            self._failed.pop(name, None)
            job = {"name": name, "done": 0, "fetched": 0, "total": 0, "stage": "listing",
                   "started": time.monotonic(), "cancel": threading.Event()}
            job["thread"] = threading.Thread(target=self._run, args=(job,), name="llima-download", daemon=True)
            self._job = job
            job["thread"].start()

    def cancel(self, name: str) -> None:
        j = self._job
        if j is not None and j["name"] == name:
            j["cancel"].set()

    def discard(self, name: str) -> None:
        j = self._job
        if j is not None and j["name"] == name and j["thread"].is_alive():
            raise HubError("Cancel the download first")
        shutil.rmtree(self._partial(name), ignore_errors=True)
        self._failed.pop(name, None)

    def _fetch(self, job: dict, url: str, dst: Path, size: int, sha: str | None) -> None:
        part = dst.with_name(dst.name + ".part")
        have = part.stat().st_size if part.is_file() else 0
        if have > size:
            part.unlink()
            have = 0
        job["done"] += have
        headers = {"User-Agent": "neat-dino-studio", **({"Range": f"bytes={have}-"} if have else {})}
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
            if have and r.status != 206:
                job["done"] -= have
                have = 0
            with part.open("ab" if have else "wb") as out:
                while chunk := r.read(1 << 20):
                    if job["cancel"].is_set():
                        raise InterruptedError()
                    out.write(chunk)
                    job["done"] += len(chunk)
                    job["fetched"] += len(chunk)
        got = part.stat().st_size
        if got != size:
            raise HubError(f"{dst.name} arrived incomplete ({got} of {size} bytes)")
        if sha:
            h = hashlib.sha256()
            with part.open("rb") as f:
                while chunk := f.read(1 << 22):
                    if job["cancel"].is_set():
                        raise InterruptedError()
                    h.update(chunk)
            if h.hexdigest() != sha:
                part.unlink()
                raise HubError(f"{dst.name} arrived damaged; try again")
        os.replace(part, dst)

    def _run(self, job: dict) -> None:
        name = job["name"]
        title = CATALOG[name][0]
        tmp = self._partial(name)
        try:
            rev, files = self.files(name)
            total = sum(f["size"] for f in files)
            job["total"] = total
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp.mkdir(exist_ok=True)
            done_already = sum(f["size"] for f in files if (tmp / f["path"]).is_file()
                               and (tmp / f["path"]).stat().st_size == f["size"])
            if _free_bytes(self.dir) < total - done_already + SPARE:
                raise HubError(f"Not enough disk: {title} needs {(total - done_already) / 1e9:.1f} GB "
                               f"and {_free_bytes(self.dir) / 1e9:.1f} GB is free")
            job["stage"] = "download"
            job["done"] = done_already
            for f in files:
                dst = tmp / f["path"]
                if dst.is_file() and dst.stat().st_size == f["size"]:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                url = f"{self.base()}/{ORG}/{name}/resolve/{rev}/{urllib.parse.quote(f['path'])}"
                self._fetch(job, url, dst, f["size"], f["sha256"])
            (tmp / REPO_MARKER).write_text(f"{ORG}/{name}\n", encoding="utf-8")
            if self.present(name):
                raise HubError(f"{title} appeared on the board during the download")
            os.replace(tmp, self.dir / name)
            self._log(f"downloaded {ORG}/{name} ({total >> 20} MB)")
        except InterruptedError:
            self._failed[name] = "Download cancelled. What arrived is kept; Download continues from there."
        except (OSError, ValueError, HubError) as e:
            self._failed[name] = f"The download stopped: {e}"
            self._log(f"download of {ORG}/{name} failed: {e}")
