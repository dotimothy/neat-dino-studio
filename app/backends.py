"""Detection backends for Neat DINO Studio.

  RuntimeBackend  the board: dino_rt (C++, SimaLMM MLA runtime) as a child process
  TorchBackend    a PC: HF GroundingDinoForObjectDetection in PyTorch (GPU if present)
  FakeBackend     tests: deterministic boxes, no model

All three take the same inputs (a Prompt and a letterboxed RGB square) and return
raw logits (900 x 64) and boxes (900 x 4, normalised cx, cy, w, h), so the server
and post-processing never care which one runs.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from dino import prompt as P
from dino.runtime_client import DinoRuntime, TextEncoderHost, Detections, NQ, T

# One accelerator workload at a time across the whole process.
ACCELERATOR_LOCK = threading.RLock()


class Backend:
    name = "base"
    size = 576

    def set_prompt(self, pr: P.Prompt) -> float: ...
    def detect(self, rgb: np.ndarray, vw: int, vh: int, same_image: bool = False) -> Detections: ...
    def close(self) -> None: ...


class RuntimeBackend(Backend):
    name = "mla"

    def __init__(self, binary, bundle, text_bundle, log=None, progress=None):
        self.text = TextEncoderHost(text_bundle)
        with ACCELERATOR_LOCK:
            self.rt = DinoRuntime(binary, bundle, text_bundle, log=log, progress=progress)
        self.size = self.rt.size
        self.tokenizer = self.text.tok
        self._prompt_key = None

    def build_prompt(self, text: str) -> P.Prompt:
        return self.text.build(text)

    def set_prompt(self, pr):
        key = tuple(pr.input_ids.tolist())
        if key == self._prompt_key:
            return 0.0
        with ACCELERATOR_LOCK:
            ms = self.rt.set_text(*self.text.tensors(pr))
        self._prompt_key = key
        return ms

    def detect(self, rgb, vw, vh, same_image=False):
        with ACCELERATOR_LOCK:
            if same_image:
                return self.rt.detect_cached()
            return self.rt.detect(rgb, vw, vh)

    def close(self):
        with ACCELERATOR_LOCK:
            self.rt.close()


class TorchBackend(Backend):
    """HF reference on a PC (CUDA when available). Same letterbox and prompt handling as the board."""
    name = "torch"

    def __init__(self, size=576, device=None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base").eval().to(self.device)
        self.size = size
        self.tokenizer = P.Tokenizer()
        self.pr = None

    def build_prompt(self, text):
        return P.build(text, self.tokenizer, T)

    def set_prompt(self, pr):
        self.pr = pr
        return 0.0

    def detect(self, rgb, vw, vh, same_image=False):
        torch = self.torch
        from dino.letterbox import MEAN, STD
        t0 = time.time()
        if same_image:
            rgb, vw, vh = self._last
        self._last = (rgb, vw, vh)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        x[vh:, :] = 0
        x[:, vw:] = 0
        px = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.device)
        n = self.pr.n_tokens
        ids = torch.from_numpy(self.pr.input_ids[:n])[None].to(self.device)
        with torch.no_grad():
            out = self.model(pixel_values=px, pixel_mask=torch.ones(1, self.size, self.size, dtype=torch.long,
                                                                   device=self.device),
                             input_ids=ids, attention_mask=torch.ones_like(ids), token_type_ids=torch.zeros_like(ids))
        logits = np.full((NQ, T), -np.inf, np.float32)
        logits[:, :n] = out.logits[0, :, :n].float().cpu().numpy()
        boxes = out.pred_boxes[0].float().cpu().numpy()
        return Detections(logits, boxes, {"total": (time.time() - t0) * 1e3})

    def close(self):
        del self.model


class FakeBackend(Backend):
    """Deterministic stand-in: one box per phrase, positioned from a hash of the phrase."""
    name = "fake"

    def __init__(self, size=576):
        self.size = size
        self.tokenizer = None
        self.pr = None

    def build_prompt(self, text):
        if self.tokenizer is None:
            self.tokenizer = P.Tokenizer()
        return P.build(text, self.tokenizer, T)

    def set_prompt(self, pr):
        self.pr = pr
        return 0.0

    def detect(self, rgb, vw, vh, same_image=False):
        logits = np.full((NQ, T), -8.0, np.float32)
        boxes = np.tile(np.array([[0.5, 0.5, 0.1, 0.1]], np.float32), (NQ, 1))
        logits[:, ~self.pr.token_mask] = -np.inf
        for p, ph in enumerate(self.pr.phrases):
            h = abs(hash(ph)) % 1000 / 1000.0
            toks = np.nonzero(self.pr.phrase_of_token == p)[0]
            logits[p, toks] = 2.0
            boxes[p] = [0.2 + 0.6 * h * vw / self.size, 0.3 + 0.4 * h * vh / self.size, 0.2, 0.25]
        return Detections(logits, boxes, {"total": 1.0})

    def close(self):
        pass
