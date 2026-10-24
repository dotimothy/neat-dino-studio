"""Python side of esam_rt (runtime/src/esam_rt.cpp): EfficientSAM masks for boxes (Grounded EfficientSAM).

    rt = EsamRuntime("runtime/build/esam_rt", "/media/nvme/models/dino/esam-ti768-bf16")
    rt.encode(rgb_square_uint8)              # S x S x 3, the image squashed to the model size
    masks, iou, timing = rt.masks(boxes)     # boxes (n, 4) x0, y0, x1, y1 in S x S pixels
                                             # -> masks (n, M, M) bool over the squashed image
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ENC_KEYS = ["patchify", "encoder", "neck"]
MASK_KEYS = ["", "", "", "decoder", "masks_cpu"]


class EsamError(RuntimeError):
    pass


def _timing(t, keys):
    d = {k: float(v) for k, v in zip(keys, t) if k}
    d["total"] = float(t[7])
    return d


class EsamRuntime:
    def __init__(self, binary, bundle, log=None, progress=None, env=None):
        self.bundle = Path(bundle)
        m = json.loads((self.bundle / "manifest.json").read_text())
        self.manifest = m
        self.size = m["size"]
        self._log = log or (lambda s: None)
        self.log_lines = deque(maxlen=200)
        t0 = time.time()
        self.p = subprocess.Popen([str(binary), str(self.bundle)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env={**os.environ, **(env or {})})
        total, done = len(m["programs"]), 0
        while True:
            line = self.p.stderr.readline()
            if not line:
                raise EsamError("esam_rt exited while loading:\n" + "\n".join(list(self.log_lines)[-20:]))
            s = line.decode(errors="replace").rstrip()
            self.log_lines.append(s)
            if s.startswith("ESAMLOAD") and progress:
                progress(done, total, s.split()[1])
                done += 1
            if s.startswith("ESAMREADY"):
                break
        self.load_seconds = time.time() - t0
        self._lock = threading.Lock()
        threading.Thread(target=self._drain, daemon=True).start()
        self.M = self.info()["mask_size"]

    def _drain(self):
        for line in self.p.stderr:
            s = line.decode(errors="replace").rstrip()
            self.log_lines.append(s)
            self._log(s)

    def _call(self, cmd, payload=b""):
        with self._lock:
            self.p.stdin.write(b"ESAM" + struct.pack("<II", cmd, len(payload)) + payload)
            self.p.stdin.flush()
            hdr = self.p.stdout.read(12)
            if len(hdr) < 12:
                raise EsamError("esam_rt stopped:\n" + "\n".join(list(self.log_lines)[-10:]))
            status, n = struct.unpack("<II", hdr[4:])
            data = self.p.stdout.read(n)
        if status:
            raise EsamError(data.decode(errors="replace"))
        return data

    def info(self):
        return json.loads(self._call(1))

    def encode(self, rgb: np.ndarray) -> dict:
        assert rgb.shape == (self.size, self.size, 3) and rgb.dtype == np.uint8
        return _timing(np.frombuffer(self._call(2, np.ascontiguousarray(rgb).tobytes()), np.float32), ENC_KEYS)

    def masks(self, boxes_xyxy) -> tuple[np.ndarray, np.ndarray, dict]:
        b = np.asarray(boxes_xyxy, np.float32).reshape(-1, 4)
        if not len(b):
            return np.zeros((0, self.M, self.M), bool), np.zeros(0, np.float32), {}
        data = self._call(3, struct.pack("<I", len(b)) + b.tobytes())
        M = struct.unpack_from("<I", data, 0)[0]
        o, masks, iou = 4, [], []
        for _ in range(len(b)):
            iou.append(np.frombuffer(data, np.float32, 1, o)[0]); o += 4
            masks.append(np.frombuffer(data, np.uint8, M * M, o).reshape(M, M).astype(bool)); o += M * M
        return np.stack(masks), np.array(iou, np.float32), _timing(np.frombuffer(data, np.float32, 8, o), MASK_KEYS)

    def close(self):
        try:
            self._call(4)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.p.terminate()
