"""Python side of omdet_rt (runtime/src/omdet_rt.cpp): OmDet-Turbo on the MLA.

    rt = OmdetRuntime("runtime/build/omdet_rt", "/media/nvme/models/dino/omdet-turbo-640-bf16")
    rt.set_classes(["red car", "dog"])        # CLIP on the board; task prompt "Detect red car, dog."
    r = rt.detect(rgb_square_uint8)            # 640 x 640 x 3, the image squashed to the model size
    r.scores (900, C) sigmoid scores, r.boxes (900, 4) cx, cy, w, h in [0, 1]
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dino.clip_bpe import ClipTokenizer

TIMING_KEYS = ["patchify", "backbone_mla", "backbone_cpu", "encoder_mla", "select_cpu", "decoder_mla", "decoder_cpu",
               "total"]


class OmdetError(RuntimeError):
    pass


@dataclass
class OmdetResult:
    logits: np.ndarray            # (900, C)
    boxes: np.ndarray             # (900, 4) cx, cy, w, h normalised
    timing: dict = field(default_factory=dict)

    @property
    def scores(self) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logits))


class OmdetText:
    """Class names -> the TEXT payload: CLIP input embeddings of every class and of the task prompt."""

    def __init__(self, bundle):
        b = Path(bundle)
        m = json.loads((b / "manifest.json").read_text())
        self.T = m["text_tokens"]
        self.tok = ClipTokenizer(b / "vocab.json", b / "merges.txt")
        z = np.load(b / "embeddings.npz")
        self.token, self.position = z["token"], z["position"][: self.T].astype(np.float32)

    @staticmethod
    def task(classes) -> str:
        return f"Detect {', '.join(classes)}."

    def payload(self, classes) -> tuple[bytes, int]:
        seqs = [self.tok.encode(c, self.T) for c in classes] + [self.tok.encode(self.task(classes), self.T)]
        emb = np.empty((len(seqs), self.T, 512), np.float32)
        eot = []
        for i, ids in enumerate(seqs):
            eot.append(len(ids) - 1)
            full = ids + [self.tok.eos] * (self.T - len(ids))
            emb[i] = self.token[full].astype(np.float32) + self.position
        head = struct.pack("<II", len(seqs), len(seqs[-1])) + np.asarray(eot, np.uint32).tobytes()
        return head + emb.tobytes(), len(seqs[-1])


class OmdetRuntime:
    def __init__(self, binary, bundle, log=None, progress=None, env=None):
        self.bundle = Path(bundle)
        m = json.loads((self.bundle / "manifest.json").read_text())
        self.manifest, self.size, self.TT = m, m["size"], m["task_tokens"]
        self.text = OmdetText(self.bundle)
        self._log = log or (lambda s: None)
        self.log_lines = deque(maxlen=200)
        t0 = time.time()
        self.p = subprocess.Popen([str(binary), str(self.bundle)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env={**os.environ, **(env or {})})
        total = len(m["programs"])
        while True:
            line = self.p.stderr.readline()
            if not line:
                raise OmdetError("omdet_rt exited while loading:\n" + "\n".join(list(self.log_lines)[-20:]))
            s = line.decode(errors="replace").rstrip()
            self.log_lines.append(s)
            if s.startswith("OMDTLOAD") and progress:
                parts = s.split()
                progress(int(parts[1]), total, parts[2])
            if s.startswith("OMDTREADY"):
                break
        self.load_seconds = time.time() - t0
        self._lock = threading.Lock()
        self.classes: list[str] = []
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in self.p.stderr:
            s = line.decode(errors="replace").rstrip()
            self.log_lines.append(s)
            self._log(s)

    def _call(self, cmd, payload=b""):
        with self._lock:
            self.p.stdin.write(b"OMDT" + struct.pack("<II", cmd, len(payload)) + payload)
            self.p.stdin.flush()
            hdr = self.p.stdout.read(12)
            if len(hdr) < 12:
                raise OmdetError("omdet_rt stopped:\n" + "\n".join(list(self.log_lines)[-10:]))
            status, n = struct.unpack("<II", hdr[4:])
            data = self.p.stdout.read(n)
        if status:
            raise OmdetError(data.decode(errors="replace"))
        return data

    def info(self):
        return json.loads(self._call(1))

    def set_classes(self, classes) -> float:
        """Run CLIP for the classes and the task prompt; 0 when they did not change."""
        classes = list(classes)
        if classes == self.classes:
            return 0.0
        payload, n_task = self.text.payload(classes)
        if n_task > self.TT:
            raise OmdetError(f"the prompt is too long ({n_task} task tokens, {self.TT} max)")
        ms = struct.unpack("<f", self._call(2, payload))[0]
        self.classes = classes
        return ms

    def _result(self, data):
        C = struct.unpack_from("<I", data, 0)[0]
        o = 4
        logits = np.frombuffer(data, np.float32, 900 * C, o).reshape(900, C)
        o += 900 * C * 4
        boxes = np.frombuffer(data, np.float32, 900 * 4, o).reshape(900, 4)
        o += 900 * 16
        tm = np.frombuffer(data, np.float32, 16, o)
        return OmdetResult(logits.copy(), boxes.copy(), dict(zip(TIMING_KEYS, map(float, tm))))

    def detect(self, rgb: np.ndarray) -> OmdetResult:
        assert rgb.shape == (self.size, self.size, 3) and rgb.dtype == np.uint8
        return self._result(self._call(3, np.ascontiguousarray(rgb).tobytes()))

    def detect_cached(self) -> OmdetResult:
        return self._result(self._call(6))

    def close(self):
        try:
            self._call(4)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.p.terminate()
