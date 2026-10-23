"""Voice prompts: request parsing, LLM answer cleanup, and the studio's voice API against a
stand-in for the GenAI server (no MLA, no models)."""
import io
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import server
from app.voice import clean_llm_answer, heuristic_prompt


@pytest.mark.parametrize("said,want", [
    ("can you find the red cars and any dogs", "red cars . dogs ."),
    ("Where are my keys?", "keys ."),
    ("Find all the traffic lights and stop signs.", "traffic lights . stop signs ."),
    ("um, the blue backpack I think", "blue backpack ."),
    ("detect the laptop, the coffee mug and the phone", "laptop . coffee mug . phone ."),
    ("look for a cat please", "cat ."),
    ("", ""),
])
def test_heuristic_prompt(said, want):
    assert heuristic_prompt(said) == want


@pytest.mark.parametrize("answer,want", [
    ("red car . dog .", "red car . dog ."),
    ("Objects: red car, dog", "red car . dog ."),
    ("laptop . coffee mug . phone", "laptop . coffee mug . phone ."),
    ("none", ""),
    ("cat .\nI found a cat for you.", "cat ."),
    ("", ""),
])
def test_clean_llm_answer(answer, want):
    assert clean_llm_answer(answer) == want


class FakeGenAI(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        body = self.rfile.read(n)
        if self.path == "/v1/audio/transcriptions":
            assert b'name="file"' in body and b"RIFF" in body
            out = {"text": " Can you find the red cars and any dogs?", "no_speech_prob": 0.02}
        elif self.path == "/v1/chat/completions":
            req = json.loads(body)
            assert req["messages"][0]["role"] == "system" and req["model"] == "llm"
            out = {"choices": [{"message": {"content": "red car . dog ."}}]}
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def client(tmp_path):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeGenAI)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    args = types.SimpleNamespace(models=str(tmp_path), hub_repo="", backend="fake", port=8092, runtime="", tls=False,
                                 llima_models=str(tmp_path / "llima"), voice_port=httpd.server_address[1])
    st = server.Studio(args)
    c = server.create_app(st).test_client()
    c.studio = st
    yield c
    httpd.shutdown()


def test_voice_off(client):
    v = client.get("/api/status").get_json()["voice"]
    assert v["state"] == "off" and not v["can_start"]
    r = client.post("/api/voice/start", json={})
    assert r.status_code == 409
    assert any(w in r.get_json()["error"] for w in ("Whisper", "pyneat"))
    r = client.post("/api/voice/prompt", data={"audio": (io.BytesIO(b"RIFF...."), "s.wav", "audio/wav")})
    assert r.status_code == 409
    assert client.post("/api/prompt/assist", json={"text": "x"}).status_code == 503


def test_voice_prompt(client):
    v = client.studio.voice
    v.state, v.asr, v.llm = "ready", "whisper-small-a16w8", "LFM2.5-230M-a16w4"   # as if the sidecar were up
    r = client.post("/api/voice/prompt", data={"audio": (io.BytesIO(b"RIFF0000WAVEfmt "), "s.wav", "audio/wav")},
                    content_type="multipart/form-data")
    d = r.get_json()
    assert r.status_code == 200, d
    assert d["transcript"] == "Can you find the red cars and any dogs?"
    assert d["prompt"] == "red car . dog ." and d["by"] == "llm"
    a = client.post("/api/prompt/assist", json={"text": "find the red cars and any dogs"}).get_json()
    assert a["text"] == "red car . dog ."
    assert client.get("/api/status").get_json()["assist"]["available"]
