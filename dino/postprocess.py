"""Turn Grounding DINO outputs into labelled boxes (HF post_process_grounded_object_detection).

HF keeps a query when max_t sigmoid(logit) > box_threshold and labels it with
every token whose probability exceeds text_threshold (joined into one string).
`mode="phrase"` (the studio default) instead assigns each box to the phrase of
its best token, which gives one clean label per box.
"""
from __future__ import annotations

import numpy as np

from dino.letterbox import Letterbox
from dino.prompt import Prompt, Tokenizer


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def candidates(logits, boxes, pr: Prompt, lb: Letterbox, floor: float = 0.05, nms_iou: float | None = None):
    """All queries scoring above `floor`, with per-phrase scores, so thresholds can move client-side.

    Returns a list of dicts sorted by score:
      box [x0, y0, x1, y1] in source pixels, score, phrase (index), phrase_scores [P], token_probs [T]
    """
    prob = sigmoid(logits)
    prob[:, ~pr.token_mask] = 0
    score = prob.max(1)
    keep = np.nonzero(score > floor)[0]
    keep = keep[np.argsort(-score[keep])]
    xyxy = lb.boxes_to_source(boxes[keep])
    nph = len(pr.phrases)
    out = []
    for j, q in enumerate(keep):
        ps = np.zeros(nph, np.float32)
        for p in range(nph):
            m = pr.phrase_of_token == p
            if m.any():
                ps[p] = prob[q, m].max()
        out.append({"box": [round(float(v), 1) for v in xyxy[j]], "score": round(float(score[q]), 4),
                    "phrase": int(ps.argmax()) if nph else -1, "phrase_scores": [round(float(v), 4) for v in ps],
                    "query": int(q)})
    if nms_iou:
        out = nms(out, nms_iou)
    return out


def nms(dets, iou_thr):
    """Per-phrase greedy NMS (Grounding DINO itself is NMS-free; this is optional)."""
    kept = []
    for d in dets:
        if all(k["phrase"] != d["phrase"] or iou(k["box"], d["box"]) < iou_thr for k in kept):
            kept.append(d)
    return kept


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def hf_style(logits, boxes, pr: Prompt, lb: Letterbox, tok: Tokenizer, box_threshold=0.35, text_threshold=0.25):
    """Exactly HF's post_process_grounded_object_detection labels (for comparisons)."""
    prob = sigmoid(logits)
    prob[:, ~pr.token_mask] = 0
    score = prob.max(1)
    keep = score > box_threshold
    labels = []
    for q in np.nonzero(keep)[0]:
        pm = prob[q] > text_threshold
        pm[0] = False
        pm[pr.n_tokens - 1:] = False
        labels.append(tok.decode(pr.input_ids[pm]))
    return score[keep], lb.boxes_to_source(boxes[keep]), labels
