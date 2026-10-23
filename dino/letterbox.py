"""Letterbox an RGB image into the detector's fixed square input.

The image is scaled so its long side is `size`, placed at the top-left, and the
rest is filled with the ImageNet mean colour, which is exactly 0 after
normalisation. Placing it top-left (instead of centring) matches how DETR-style
models pad batches, so box coordinates only need one scale factor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Letterbox:
    size: int          # model input side (square)
    scale: float       # model pixels per source pixel
    width: int         # source image size
    height: int
    new_w: int         # scaled image size inside the square
    new_h: int

    def boxes_to_source(self, boxes_cxcywh: np.ndarray) -> np.ndarray:
        """Normalised (cx, cy, w, h) in model space -> (x0, y0, x1, y1) source pixels."""
        b = np.asarray(boxes_cxcywh, np.float32) * self.size
        x0, y0 = b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2
        x1, y1 = b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2
        out = np.stack([x0, y0, x1, y1], -1) / self.scale
        out[..., 0::2] = out[..., 0::2].clip(0, self.width)
        out[..., 1::2] = out[..., 1::2].clip(0, self.height)
        return out


@dataclass
class Squash:
    """The whole image stretched to size x size (SAM 3's own preprocessing): box coordinates
    and masks cover the full image, each axis scaled on its own."""
    size: int
    width: int
    height: int

    @property
    def new_w(self) -> int:
        return self.size

    @property
    def new_h(self) -> int:
        return self.size

    def boxes_to_source(self, boxes_cxcywh: np.ndarray) -> np.ndarray:
        b = np.asarray(boxes_cxcywh, np.float32)
        x0, y0 = b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2
        x1, y1 = b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2
        out = np.stack([x0 * self.width, y0 * self.height, x1 * self.width, y1 * self.height], -1)
        out[..., 0::2] = out[..., 0::2].clip(0, self.width)
        out[..., 1::2] = out[..., 1::2].clip(0, self.height)
        return out


def letterbox(img: Image.Image | np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Return the normalised (3, size, size) float32 image and its geometry."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    rs = np.asarray(img.resize((nw, nh), Image.BILINEAR), np.float32) / 255.0
    out = np.zeros((size, size, 3), np.float32)
    out[:nh, :nw] = (rs - MEAN) / STD
    return np.ascontiguousarray(out.transpose(2, 0, 1)), Letterbox(size, s, w, h, nw, nh)
