"""The web app's API with the fake backend (no model, no MLA)."""
import io
import json
import time
import types

import numpy as np
import pytest
from PIL import Image

import server


@pytest.fixture()
def client(tmp_path):
    d = tmp_path / "dino-576-bf16"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"kind": "detector", "size": 576, "precision": "bf16", "files": {}}))
    args = types.SimpleNamespace(models=str(tmp_path), hub_repo="", backend="fake", port=8092, runtime="", tls=False)
    st = server.Studio(args)
    app = server.create_app(st)
    c = app.test_client()
    c.studio = st
    return c


def jpeg(w=320, h=240):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (120, 30, 200)).save(b, "JPEG")
    return b.getvalue()


def wait_ready(c):
    for _ in range(100):
        s = c.get("/api/status").get_json()
        if s["state"] in ("ready", "error"):
            return s
        time.sleep(0.05)
    return s


def test_status_and_catalog(client):
    s = client.get("/api/status").get_json()
    assert s["app"] == "dino-studio" and s["state"] == "empty"
    cat = client.get("/api/models").get_json()
    assert [m["name"] for m in cat["models"]] == ["dino-576-bf16"]


def test_detect_needs_model(client):
    r = client.post("/api/detect", data={"prompt": "cat .", "image": (io.BytesIO(jpeg()), "a.jpg")})
    assert r.status_code == 409


def test_load_detect_rerun_unload(client):
    assert client.post("/api/models/load", json={"name": "dino-576-bf16"}).status_code == 200
    assert wait_ready(client)["state"] == "ready"
    r = client.post("/api/detect", data={"prompt": "cat . dog .", "image": (io.BytesIO(jpeg()), "a.jpg")})
    d = r.get_json()
    assert r.status_code == 200, d
    assert d["width"] == 320 and [p["text"] for p in d["phrases"]] == ["cat", "dog"]
    assert len(d["detections"]) == 2
    r2 = client.post("/api/detect", data={"prompt": "bird .", "image_id": d["image_id"]}).get_json()
    assert r2["same_image"] and r2["phrases"][0]["text"] == "bird"
    assert client.post("/api/detect", data={"prompt": "bird .", "image_id": "nope"}).status_code == 409
    assert client.post("/api/models/unload", json={}).status_code == 200
    assert client.get("/api/status").get_json()["state"] == "empty"


def test_parse_and_assist(client):
    p = client.post("/api/prompt/parse", json={"text": "Cat, red car"}).get_json()
    assert p["caption"] == "cat . red car ." and p["tokens"] > 2
    assert client.post("/api/prompt/assist", json={"text": "x"}).status_code == 503


def test_static(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_model_names_per_variant():
    b = server.model_info("dino-576-bf16", {"kind": "detector", "size": 576, "precision": "bf16"})
    t = server.model_info("dino-tiny-576-bf16", {"kind": "detector", "size": 576, "precision": "bf16", "variant": "tiny"})
    assert b["title"].startswith("Grounding DINO Base · Fast 576") and b["short"] == "DINO-B 576"
    assert t["title"].startswith("Grounding DINO Tiny · 576") and t["short"] == "DINO-T 576"
    assert "172.2 M" in t["details"] and "grounding-dino-tiny" in t["upstream"]
    assert server.model_info("dino-tiny-text-t64-bf16")["title"].startswith("Grounding DINO Tiny · Text")
