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
from concurrent.futures import Future
from pathlib import Path

import numpy as np

from dino import prompt as P
from dino.runtime_client import DinoRuntime, TextEncoderHost, Detections, NQ, T

# One accelerator workload at a time across the whole process.
ACCELERATOR_LOCK = threading.RLock()


class Backend:
    name = "base"
    size = 576
    slots = 1          # frames that can be in flight at once (Live mode keeps this many queued)

    def set_prompt(self, pr: P.Prompt) -> float: ...
    def detect(self, rgb: np.ndarray, vw: int, vh: int, same_image: bool = False) -> Detections: ...
    def close(self) -> None: ...

    def detect_async(self, rgb, vw, vh, same_image=False) -> Future:
        """Start a detection; the default runs it now and returns a finished future."""
        f: Future = Future()
        try:
            f.set_result(self.detect(rgb, vw, vh, same_image))
        except Exception as e:  # noqa: BLE001
            f.set_exception(e)
        return f


class RuntimeBackend(Backend):
    name = "mla"

    def __init__(self, binary, bundle, text_bundle, log=None, progress=None):
        self.text = TextEncoderHost(text_bundle)
        with ACCELERATOR_LOCK:
            self.rt = DinoRuntime(binary, bundle, text_bundle, log=log, progress=progress)
        self.size = self.rt.size
        self.slots = self.rt.slots
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
        return self.detect_async(rgb, vw, vh, same_image).result()

    def detect_async(self, rgb, vw, vh, same_image=False):
        # Only sending is serialised; dino_rt answers in order and runs two frames at once.
        with ACCELERATOR_LOCK:
            if same_image:
                f: Future = Future()
                raw = self.rt._send(6)
                raw.add_done_callback(lambda r: f.set_exception(r.exception()) if r.exception()
                                      else f.set_result(self.rt._detections(r.result())))
                return f
            return self.rt.detect_async(rgb, vw, vh)

    def close(self):
        with ACCELERATOR_LOCK:
            self.rt.close()


class Sam3Backend(Backend):
    """SAM 3 on the board (sam3_rt). One concept per pass: the first phrase runs the whole model,
    further phrases rerun only the detector on the image features still on the board."""
    name = "mla"
    kind = "sam3"
    MAX_PHRASES = 8

    def __init__(self, binary, bundle, log=None, progress=None):
        from dino.sam3_client import Sam3Runtime
        with ACCELERATOR_LOCK:
            self.rt = Sam3Runtime(binary, bundle, log=log, progress=progress)
        self.size = self.rt.size
        self.mask_size = self.rt.M
        self._text = None

    @staticmethod
    def phrases(text: str, cap: bool = True) -> list[str]:
        out = []
        for p in text.replace("\n", " ").split("."):
            p = " ".join(p.split()).strip(" ,;")
            if p and p.lower() not in [o.lower() for o in out]:
                out.append(p)
        return out[: Sam3Backend.MAX_PHRASES] if cap else out

    def build_prompt(self, text):
        from types import SimpleNamespace
        ph = self.phrases(text)
        return SimpleNamespace(phrases=ph, caption=" . ".join(ph) + (" ." if ph else ""), n_tokens=len(ph),
                               truncated=len(self.phrases(text, cap=False)) > self.MAX_PHRASES)

    def set_prompt(self, pr) -> float:
        return 0.0

    def detect(self, rgb, vw=None, vh=None, same_image=False):
        """Warm-up / simple use: the first phrase of the last prompt, or 'object'."""
        return self.detect_phrases(None if same_image else rgb, ["object"])

    def detect_phrases(self, rgb, phrases, mask_min=0.2):
        """rgb (S, S, 3) uint8 for a new image, or None to reuse the last one.
        Returns [(phrase, Sam3Result, text_ms)]."""
        out = []
        with ACCELERATOR_LOCK:
            for i, ph in enumerate(phrases):
                t_ms = 0.0
                if ph != self._text:
                    t_ms = self.rt.set_text(ph)
                    self._text = ph
                r = self.rt.detect(rgb, mask_min) if (i == 0 and rgb is not None) else self.rt.detect_cached(mask_min)
                out.append((ph, r, t_ms))
        return out

    def close(self):
        with ACCELERATOR_LOCK:
            self.rt.close()


class TorchBackend(Backend):
    """HF reference on a PC (CUDA when available). Same letterbox and prompt handling as the board."""
    name = "torch"

    def __init__(self, size=576, device=None, model="IDEA-Research/grounding-dino-base"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model).eval().to(self.device)
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


class MaskJob:
    """One frame's masks: the encoder starts as soon as the image arrives (while Grounding DINO runs),
    the decoder runs once the boxes are known. Jobs run one after another (the image embedding on
    the board belongs to one frame at a time)."""

    def __init__(self):
        self._boxes: Future = Future()
        self.result: Future = Future()
        self.encode_ms: dict = {}

    def boxes(self, boxes_xyxy):
        """(n, 4) boxes in the masker's S x S image, or None to cancel."""
        if not self._boxes.done():
            self._boxes.set_result(boxes_xyxy)


class EsamBackend:
    """EfficientSAM masks for detector boxes (esam_rt), alongside the detector."""
    kind = "esam"

    def __init__(self, binary, bundle, log=None, progress=None):
        from dino.esam_client import EsamRuntime
        with ACCELERATOR_LOCK:
            self.rt = EsamRuntime(binary, bundle, log=log, progress=progress)
        self.size, self.mask_size = self.rt.size, self.rt.M
        self._lock = threading.Lock()
        self._image = None          # id of the image whose embedding is on the board
        self._rgb = None            # (id, rgb) of the last image, to re-encode for a prompt-only run

    def start(self, rgb, image_id) -> MaskJob:
        """rgb (S, S, 3) uint8 squashed image, or None for the last image (prompt changed only)."""
        if rgb is not None:
            self._rgb = (image_id, rgb)
        elif self._rgb and self._rgb[0] == image_id:
            rgb = self._rgb[1]
        job = MaskJob()

        def work():
            try:
                with self._lock:
                    if self._image != image_id or image_id is None:
                        if rgb is None:
                            raise RuntimeError("the image for the masks is gone; send it again")
                        job.encode_ms = self.rt.encode(rgb)
                        self._image = image_id
                    boxes = job._boxes.result(timeout=60)
                    if boxes is None or not len(boxes):
                        job.result.set_result(None)
                        return
                    job.result.set_result(self.rt.masks(boxes))
            except Exception as e:  # noqa: BLE001
                if not job.result.done():
                    job.result.set_exception(e)
        threading.Thread(target=work, name="masks", daemon=True).start()
        return job

    def close(self):
        with ACCELERATOR_LOCK:
            self.rt.close()


class FakeEsamBackend(EsamBackend):
    """Tests: each box's mask is the box itself."""

    def __init__(self, size=768, upsample=2):
        self.size, self.mask_size = size, size // 4 * upsample
        self._lock = threading.Lock()
        self._image = self._rgb = None
        M, sc = self.mask_size, self.mask_size / size

        class _RT:
            def encode(self, rgb):
                return {"patchify": 0.1, "encoder": 1.0, "neck": 0.1, "total": 1.2}

            def masks(self, boxes):
                out = np.zeros((len(boxes), M, M), bool)
                for i, (x0, y0, x1, y1) in enumerate(np.asarray(boxes) * sc):
                    out[i, int(max(0, y0)):int(np.ceil(y1)), int(max(0, x0)):int(np.ceil(x1))] = True
                return out, np.full(len(boxes), 0.9, np.float32), {"decoder": 1.0, "masks_cpu": 0.1, "total": 1.1}

            def close(self):
                pass
        self.rt = _RT()
