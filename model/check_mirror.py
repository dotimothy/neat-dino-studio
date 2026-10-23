#!/usr/bin/env python3
"""Check the MLA-form stage pipeline against HF GroundingDinoForObjectDetection (fp32).

    .venv/bin/python model/check_mirror.py --size 768 --image data/images/cats.jpg --prompt "a cat. a remote control."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from dino.letterbox import letterbox  # noqa: E402
from dino import prompt as P  # noqa: E402
import dino_mla as M  # noqa: E402


def cos(a, b):
    a, b = a.double().ravel(), b.double().ravel()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--image", default=str(ROOT / "data/images/cats.jpg"))
    ap.add_argument("--prompt", default="a cat. a remote control.")
    ap.add_argument("--model", default="base", choices=sorted(M.MODELS))
    a = ap.parse_args()
    from PIL import Image
    hf = M.load_hf(a.model)
    tok = P.Tokenizer()
    pr = P.build(a.prompt, tok)
    img = Image.open(a.image).convert("RGB")
    px, lb = letterbox(img, a.size)
    n = pr.n_tokens
    ids = torch.from_numpy(pr.input_ids[:n])[None]
    with torch.no_grad():
        out = hf(pixel_values=torch.from_numpy(px)[None], pixel_mask=torch.ones(1, a.size, a.size, dtype=torch.long),
                 input_ids=ids, attention_mask=torch.ones_like(ids), token_type_ids=torch.zeros_like(ids),
                 output_hidden_states=True, return_dict=True)
    pipe = M.Pipeline(hf, a.size)
    with torch.no_grad():
        logits, boxes = pipe(M.space_to_depth4(px), pr)
    hl, hb = out.logits[0, :, :n], out.pred_boxes[0]
    # compare as sets: match each HF query to ours by box
    print(f"tokens {n}, N {pipe.g.N}")
    print(f"enc text  cos {cos(M.tok_to_rows(pipe.trace['enc_text'])[:n], out.encoder_last_hidden_state_text[0, :n]):.6f}")
    hs = torch.sigmoid(hl).max(-1).values
    ms = torch.sigmoid(logits[:, :n]).max(-1).values
    print(f"top scores HF   {hs.sort(descending=True).values[:6].numpy().round(4)}")
    print(f"top scores ours {ms.sort(descending=True).values[:6].numpy().round(4)}")
    def box_convert(b, *_):
        return torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)

    def box_iou(a, b):
        lt = torch.max(a[:, None, :2], b[None, :, :2])
        rb = torch.min(a[:, None, 2:], b[None, :, 2:])
        inter = (rb - lt).clamp(min=0).prod(-1)
        area = lambda x: (x[:, 2:] - x[:, :2]).prod(-1)
        return inter / (area(a)[:, None] + area(b)[None, :] - inter)
    hi = hs.argsort(descending=True)[:30]
    hbx = box_convert(hb[hi], "cxcywh", "xyxy")
    mbx = box_convert(boxes, "cxcywh", "xyxy")
    iou = box_iou(hbx, mbx)                         # (30, 900)
    worst_iou, worst_ds = 1.0, 0.0
    for r in range(len(hi)):
        j = int(iou[r].argmax())
        worst_iou = min(worst_iou, float(iou[r, j]))
        worst_ds = max(worst_ds, abs(float(hs[hi[r]] - ms[j])))
    print(f"top-30 HF detections: worst matched IoU {worst_iou:.4f}, worst score diff {worst_ds:.4f}")

if __name__ == "__main__":
    main()
