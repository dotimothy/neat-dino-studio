"""Talk to dino_rt (the C++ MLA runtime) from Python.

dino_rt runs as a child process; requests and replies go over its stdin/stdout
(see runtime/src/dino_rt.cpp for the framing). Stopping it with SIGTERM frees
the MLA cleanly, which is how the app unloads a model.
"""
from __future__ import annotations

import json
import os
import signal
import struct
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dino import prompt as P

T, NQ, TPW, D = 64, 900, 193, 256
TIMING_KEYS = ["patchify", "backbone", "encoder_mla", "encoder_cpu", "dec_values", "topk", "decoder_mla",
               "decoder_cpu", "total", "backbone_cpu", "slot"]


class RuntimeError_(RuntimeError):
    pass


class TextEncoderHost:
    """Builds the text stage's input (BERT embeddings + phrase mask) and the text pack."""

    def __init__(self, text_bundle: str | Path, pure_python_tokenizer: bool | None = None):
        d = Path(text_bundle)
        z = np.load(d / "embeddings.npz")
        self.word, self.pos, self.ttype = (z[k].astype(np.float32) for k in ("word", "position", "token_type"))
        self.tok = P.Tokenizer(d, pure_python=bool(pure_python_tokenizer))

    def build(self, text: str) -> P.Prompt:
        return P.build(text, self.tok, T)

    def tensors(self, pr: P.Prompt) -> tuple[np.ndarray, np.ndarray]:
        emb = self.word[pr.input_ids] + self.pos[pr.position_ids] + self.ttype[None, :]
        text_in = np.zeros((T, 832), np.float32)
        text_in[:, :768] = emb
        text_in[:, 768:] = pr.additive_self_mask()                # [query][key]
        tp = np.zeros((TPW, D), np.float32)
        tp[64:128] = P.sine_embed_1d(pr.position_ids)            # text position embedding
        tp[128:192, :T] = pr.additive_self_mask()                 # row = query, channel = key
        tp[192, :T] = pr.additive_token_mask()
        return text_in, tp


@dataclass
class Detections:
    logits: np.ndarray      # (900, 64) raw logits, -inf for pad tokens
    boxes: np.ndarray       # (900, 4) normalised cx, cy, w, h in model space
    timing: dict


class DinoRuntime:
    def __init__(self, binary: str | Path, bundle: str | Path, text_bundle: str | Path,
                 env: dict | None = None, ready_timeout: float = 180.0, log=None, progress=None):
        self.bundle, self.text_bundle = Path(bundle), Path(text_bundle)
        self.manifest = json.loads((self.bundle / "manifest.json").read_text())
        self.size = int(self.manifest["size"])
        self.log_lines: deque[str] = deque(maxlen=400)
        self._log = log or (lambda s: None)
        self._progress = progress                  # progress(done, total, stage) while loading
        self.load_progress = (0, 0, "")
        self._lock = threading.Lock()
        e = dict(os.environ)
        e.setdefault("OMP_NUM_THREADS", "16")
        if env:
            e.update(env)
        self.proc = subprocess.Popen([str(binary), str(self.bundle), str(self.text_bundle)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=e, bufsize=0, start_new_session=True)
        self._ready = threading.Event()
        self._err_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._err_thread.start()
        t0 = time.time()
        while not self._ready.wait(0.2):
            if self.proc.poll() is not None:
                raise RuntimeError_("dino_rt exited while loading:\n" + "\n".join(list(self.log_lines)[-20:]))
            if time.time() - t0 > ready_timeout:
                self.close()
                raise RuntimeError_("dino_rt did not become ready in time")
        self.load_seconds = time.time() - t0
        self.prompt: P.Prompt | None = None
        # Replies come back in request order, so the reader resolves the oldest pending future.
        self._pending: deque[Future] = deque()
        self._out_thread = threading.Thread(target=self._read_replies, daemon=True)
        self._out_thread.start()
        self.slots = int(self.info().get("slots", 1))

    def _read_stderr(self):
        buf = b""
        while True:
            chunk = self.proc.stderr.read1(4096) if hasattr(self.proc.stderr, "read1") else self.proc.stderr.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                s = line.decode("utf-8", "replace")
                if s.strip() == "DINOREADY":
                    self._ready.set()
                    continue
                if s.startswith("DINOLOAD "):
                    _, done, total, name = s.split(" ", 3)
                    self.load_progress = (int(done), int(total), name)
                    if self._progress:
                        self._progress(int(done), int(total), name)
                    continue
                self.log_lines.append(s)
                self._log(s)

    def _send(self, cmd: int, payload: bytes = b"") -> Future:
        """Queue a request; the future resolves to the reply body (or raises RuntimeError_)."""
        fut: Future = Future()
        with self._lock:
            if self.proc.poll() is not None:
                raise RuntimeError_("dino_rt is not running")
            self._pending.append(fut)
            try:
                self.proc.stdin.write(b"DINO" + struct.pack("<II", cmd, len(payload)) + payload)
                self.proc.stdin.flush()
            except OSError as e:
                self._pending.remove(fut)
                raise RuntimeError_(f"dino_rt is not running ({e})") from e
        return fut

    def _call(self, cmd: int, payload: bytes = b"") -> bytes:
        return self._send(cmd, payload).result()

    def _read_replies(self):
        err = None
        try:
            while True:
                head = self._read(12)
                if head[:4] != b"DINO":
                    raise RuntimeError_("bad reply from dino_rt")
                status, n = struct.unpack("<II", head[4:])
                body = self._read(n)
                with self._lock:
                    fut = self._pending.popleft() if self._pending else None
                if fut is None:
                    continue
                if status:
                    fut.set_exception(RuntimeError_(body.decode("utf-8", "replace")))
                else:
                    fut.set_result(body)
        except Exception as e:  # noqa: BLE001  (process gone: fail whatever is still waiting)
            err = e
        with self._lock:
            while self._pending:
                self._pending.popleft().set_exception(RuntimeError_(str(err) or "dino_rt stopped"))

    def _read(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            c = self.proc.stdout.read(n - len(out))
            if not c:
                raise RuntimeError_("dino_rt closed its output:\n" + "\n".join(list(self.log_lines)[-20:]))
            out += c
        return bytes(out)

    def info(self) -> dict:
        return json.loads(self._call(1))

    def set_text(self, text_in: np.ndarray, tp: np.ndarray) -> float:
        payload = np.ascontiguousarray(text_in, np.float32).tobytes() + np.ascontiguousarray(tp, np.float32).tobytes()
        return struct.unpack("<f", self._call(2, payload))[0]

    def detect(self, rgb_square: np.ndarray, valid_w: int, valid_h: int) -> Detections:
        return self.detect_async(rgb_square, valid_w, valid_h).result()

    def detect_async(self, rgb_square: np.ndarray, valid_w: int, valid_h: int) -> Future:
        """Send a frame without waiting: with two runtime slots, two frames are in flight
        (one on the MLA while the other's sampling runs on the CPU). Resolves to Detections."""
        S = self.size
        if rgb_square.shape != (S, S, 3) or rgb_square.dtype != np.uint8:
            raise ValueError(f"expected ({S},{S},3) uint8, got {rgb_square.shape} {rgb_square.dtype}")
        raw = self._send(3, struct.pack("<II", valid_w, valid_h) + np.ascontiguousarray(rgb_square).tobytes())
        out: Future = Future()

        def done(f: Future):
            try:
                out.set_result(self._detections(f.result()))
            except Exception as e:  # noqa: BLE001
                out.set_exception(e)
        raw.add_done_callback(done)
        return out

    def detect_cached(self) -> Detections:
        """Rerun on the last image (its backbone output is still on the device)."""
        return self._detections(self._call(6))

    def _detections(self, body: bytes) -> Detections:
        f = np.frombuffer(body, np.float32)
        logits = f[: NQ * T].reshape(NQ, T).copy()
        boxes = f[NQ * T: NQ * T + NQ * 4].reshape(NQ, 4).copy()
        tm = f[NQ * T + NQ * 4:]
        return Detections(logits, boxes, {k: float(tm[i]) for i, k in enumerate(TIMING_KEYS)})

    def detect_tokens(self, tokens: np.ndarray) -> Detections:
        """Debug: run everything after the backbone on given (N, 256) backbone tokens."""
        return self._detections(self._call(5, np.ascontiguousarray(tokens, np.float32).tobytes()))

    def close(self, timeout: float = 20.0):
        if self.proc.poll() is None:
            try:
                self._send(4).result(timeout)        # after every frame still in flight
            except Exception:  # noqa: BLE001
                pass
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.send_signal(signal.SIGTERM)       # never SIGKILL: it leaks MLA memory
                try:
                    self.proc.wait(timeout)
                except subprocess.TimeoutExpired:
                    pass
