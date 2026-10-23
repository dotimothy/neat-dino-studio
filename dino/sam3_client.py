"""Python side of sam3_rt (runtime/src/sam3_rt.cpp): starts it, tokenizes prompts, sends images.

    rt = Sam3Runtime("runtime/build/sam3_rt", "/media/nvme/models/dino/sam3-672-bf16")
    rt.set_text("red car")
    r = rt.detect(rgb_square_uint8)          # S x S x 3, already resized (squashed) to the model size
    r.scores, r.boxes (cx, cy, w, h normalised), r.masks {query: (M, M) bool}
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

TIMING_KEYS = ["patchify", "vit", "to_image", "neck", "encoder", "decoder_mla", "decoder_cpu", "masks", "n_masks"]


class Sam3Error(RuntimeError):
    pass


@dataclass
class Sam3Result:
    logits: np.ndarray            # (Q,)
    presence: float               # logit
    boxes: np.ndarray             # (Q, 4) cx, cy, w, h in [0, 1]
    masks: dict = field(default_factory=dict)   # query -> (M, M) bool
    timing: dict = field(default_factory=dict)

    @property
    def scores(self) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logits)) / (1 + np.exp(-self.presence))


class Sam3Text:
    """CLIP tokens -> the text program's packed input: per token the 1024 input embeddings,
    then that token's column of the additive causal + padding mask (keys on channels)."""

    def __init__(self, bundle):
        b = Path(bundle)
        self.tok = ClipTokenizer(b / "vocab.json", b / "merges.txt")
        z = np.load(b / "embeddings.npz")
        self.token, self.position = z["token"], z["position"].astype(np.float32)
        self.T = self.position.shape[0]

    def packed(self, text: str):
        ids = self.tok.encode(text, self.T)
        n = len(ids)
        ids = ids + [self.tok.eos] * (self.T - n)
        emb = self.token[ids].astype(np.float32) + self.position
        k = np.arange(self.T)[:, None]
        q = np.arange(self.T)[None, :]
        mask = np.where((k <= q) & (k < n), 0.0, -1e4).astype(np.float32)     # (keys, queries)
        return np.concatenate([emb, mask.T], 1), n                            # (T, 1024 + T): row q


class Sam3Runtime:
    def __init__(self, binary, bundle, log=None, progress=None, env=None):
        self.bundle = Path(bundle)
        m = json.loads((self.bundle / "manifest.json").read_text())
        self.size, self.Q, self.M = m["size"], m["queries"], m["mask_size"]
        self.text = Sam3Text(self.bundle)
        self._log = log or (lambda s: None)
        self.log_lines = deque(maxlen=200)
        t0 = time.time()
        self.p = subprocess.Popen([str(binary), str(self.bundle)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env={**os.environ, **(env or {})})
        total = len(m["programs"])
        while True:
            line = self.p.stderr.readline()
            if not line:
                raise Sam3Error("sam3_rt exited while loading:\n" + "\n".join(list(self.log_lines)[-20:]))
            s = line.decode(errors="replace").rstrip()
            self.log_lines.append(s)
            if s.startswith("SAM3LOAD") and progress:
                parts = s.split()
                progress(int(parts[1]), total, parts[2])
            if s.startswith("SAM3READY"):
                break
        self.load_seconds = time.time() - t0
        self._lock = threading.Lock()
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in self.p.stderr:
            s = line.decode(errors="replace").rstrip()
            self.log_lines.append(s)
            self._log(s)

    def _call(self, cmd, payload=b""):
        with self._lock:
            self.p.stdin.write(b"SAM3" + struct.pack("<II", cmd, len(payload)) + payload)
            self.p.stdin.flush()
            hdr = self.p.stdout.read(12)
            if len(hdr) < 12:
                raise Sam3Error("sam3_rt stopped:\n" + "\n".join(list(self.log_lines)[-10:]))
            status, n = struct.unpack("<II", hdr[4:])
            data = self.p.stdout.read(n)
        if status:
            raise Sam3Error(data.decode(errors="replace"))
        return data

    def info(self):
        return json.loads(self._call(1))

    def set_text(self, text: str) -> float:
        packed, n = self.text.packed(text)
        return struct.unpack("<f", self._call(2, packed.astype(np.float32).tobytes() + struct.pack("<I", n)))[0]

    def _result(self, data):
        Q, MM = self.Q, self.M * self.M
        o = 0
        logits = np.frombuffer(data, np.float32, Q, o); o += 4 * Q
        presence = float(np.frombuffer(data, np.float32, 1, o)[0]); o += 4
        boxes = np.frombuffer(data, np.float32, Q * 4, o).reshape(Q, 4); o += 16 * Q
        n = struct.unpack_from("<I", data, o)[0]; o += 4
        masks = {}
        for _ in range(n):
            q = struct.unpack_from("<I", data, o)[0]; o += 4
            masks[q] = np.frombuffer(data, np.uint8, MM, o).reshape(self.M, self.M).astype(bool); o += MM
        tm = np.frombuffer(data, np.float32, 16, o)
        return Sam3Result(logits.copy(), presence, boxes.copy(), masks, dict(zip(TIMING_KEYS, map(float, tm))))

    def detect(self, rgb: np.ndarray, mask_min: float = 0.3) -> Sam3Result:
        assert rgb.shape == (self.size, self.size, 3) and rgb.dtype == np.uint8
        return self._result(self._call(3, struct.pack("<f", mask_min) + rgb.tobytes()))

    def detect_cached(self, mask_min: float = 0.3) -> Sam3Result:
        return self._result(self._call(6, struct.pack("<f", mask_min)))

    def close(self):
        try:
            self._call(4)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.p.terminate()
