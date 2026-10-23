"""Voice model downloads (app/llima_hub.py) against a stand-in for Hugging Face."""
import hashlib
import io
import json
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import server
from app.llima_hub import REPO_MARKER, LlimaHub

NAME = "LFM2.5-230M-a16w4"
REV = "d" * 40
FILES = {"README.md": b"# model\n", "devkit/config.json": b'{"a": 1}',
         "elf_files/layer0.elf": bytes(range(256)) * 4000}       # ~1 MB, an "LFS" file with a SHA-256
LFS = {"elf_files/layer0.elf"}


class FakeHF(BaseHTTPRequestHandler):
    corrupt = False
    stall_after = None            # bytes to send of the big file before closing (an interrupted download)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith(f"/api/models/simaai/{NAME}"):
            sib = [{"rfilename": p, "size": len(b), **({"lfs": {"sha256": hashlib.sha256(b).hexdigest()}} if p in LFS else {})}
                   for p, b in FILES.items()]
            return self._send(json.dumps({"sha": REV, "siblings": sib}).encode())
        prefix = f"/simaai/{NAME}/resolve/{REV}/"
        if not self.path.startswith(prefix):
            self.send_response(404)
            self.end_headers()
            return
        data = FILES[self.path[len(prefix):]]
        if FakeHF.corrupt and self.path.endswith(".elf"):
            data = b"x" + data[1:]
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
        body = data[start:]
        if FakeHF.stall_after is not None and self.path.endswith(".elf") and not rng:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:FakeHF.stall_after])
            self.wfile.flush()
            self.connection.close()
            return
        self._send(body, 206 if rng else 200)

    def _send(self, body, code=200):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def hf(monkeypatch):
    FakeHF.corrupt, FakeHF.stall_after = False, None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeHF)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("DINO_LLIMA_HUB_URL", f"http://127.0.0.1:{httpd.server_address[1]}")
    yield httpd
    httpd.shutdown()


def wait(hub, name, timeout=20):
    t = time.time()
    while hub.job() and time.time() - t < timeout:
        time.sleep(0.05)
    assert hub.job() is None


def test_download(hf, tmp_path):
    hub = LlimaHub(tmp_path)
    assert not hub.present(NAME)
    hub.download(NAME)
    wait(hub, NAME)
    d = tmp_path / NAME
    assert hub.present(NAME), hub.models()
    for p, b in FILES.items():
        assert (d / p).read_bytes() == b
    assert (d / REPO_MARKER).read_text().strip() == f"simaai/{NAME}"
    assert not (tmp_path / f".{NAME}.partial").exists()
    with pytest.raises(Exception, match="already"):
        hub.download(NAME)


def test_interrupted_download_resumes(hf, tmp_path):
    hub = LlimaHub(tmp_path)
    FakeHF.stall_after = 300_000
    hub.download(NAME)
    wait(hub, NAME)
    m = {x["name"]: x for x in hub.models()}[NAME]
    assert not m["present"] and m["partial"] and "stopped" in m["error"]
    part = tmp_path / f".{NAME}.partial" / "elf_files" / "layer0.elf.part"
    assert part.stat().st_size == 300_000
    FakeHF.stall_after = None
    hub.download(NAME)                      # continues with a Range request
    wait(hub, NAME)
    assert hub.present(NAME)
    assert (tmp_path / NAME / "elf_files" / "layer0.elf").read_bytes() == FILES["elf_files/layer0.elf"]


def test_damaged_file_is_rejected(hf, tmp_path):
    hub = LlimaHub(tmp_path)
    FakeHF.corrupt = True
    hub.download(NAME)
    wait(hub, NAME)
    m = {x["name"]: x for x in hub.models()}[NAME]
    assert not m["present"] and "damaged" in m["error"]
    hub.discard(NAME)
    assert not (tmp_path / f".{NAME}.partial").exists()


def test_unknown_model_refused(tmp_path):
    with pytest.raises(Exception, match="not a voice model"):
        LlimaHub(tmp_path).download("../../etc")


def test_voice_api_download(hf, tmp_path):
    args = types.SimpleNamespace(models=str(tmp_path / "dino"), hub_repo="", backend="fake", port=8092, runtime="",
                                 tls=False, llima_models=str(tmp_path / "llima"), voice_port=1)
    st = server.Studio(args)
    c = server.create_app(st).test_client()
    v = c.get("/api/status").get_json()["voice"]
    assert not v["can_start"] and {m["name"] for m in v["models"]} >= {NAME, "whisper-small-a16w8"}
    r = c.post("/api/voice/download", json={"name": NAME})
    assert r.status_code == 200, r.get_json()
    wait(st.voice.hub, NAME)
    v = c.get("/api/status").get_json()["voice"]
    assert {m["name"]: m for m in v["models"]}[NAME]["present"]
    assert NAME in v["choices"]                      # now offered as the LLM
    assert c.post("/api/voice/download", json={"name": "nope"}).status_code == 409
