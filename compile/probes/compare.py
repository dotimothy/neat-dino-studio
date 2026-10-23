#!/usr/bin/env python3
"""Compare board outputs (<name>_board.npy, HWC or flat) with onnxruntime refs (NCHW)."""
import json
import sys
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[2] / "build" / "probes"


def align(board, ref):
    """Return the board output in the ref's NCHW layout."""
    if board.size != ref.size:
        raise ValueError(f"size {board.shape} vs ref {ref.shape}")
    n, c, h, w = ref.shape if ref.ndim == 4 else (1, *ref.shape[-3:])
    if board.ndim == 3 and board.shape == (h, w, c):
        return board.transpose(2, 0, 1)[None]
    if board.ndim == 4 and board.shape == (n, h, w, c):
        return board.transpose(0, 3, 1, 2)
    return board.reshape(h, w, c).transpose(2, 0, 1)[None]


def main(names):
    print(f"{'probe':14s} {'cos':>8s} {'max_abs':>9s} {'rel_rms':>8s} {'ms':>8s}  verdict")
    for n in names:
        try:
            ref = np.load(P / f"{n}_ref.npy")
            b = align(np.load(P / f"{n}_board.npy"), ref)
            t = json.loads((P / f"{n}_board.json").read_text())
        except Exception as e:  # noqa: BLE001
            print(f"{n:14s} no board result ({e})")
            continue
        r, bb = ref.ravel().astype(np.float64), b.ravel().astype(np.float64)
        cos = float(r @ bb / (np.linalg.norm(r) * np.linalg.norm(bb) + 1e-30))
        mx = float(np.abs(r - bb).max())
        rel = float(np.sqrt(np.mean((r - bb) ** 2)) / (np.sqrt(np.mean(r ** 2)) + 1e-30))
        ok = cos >= 0.999
        print(f"{n:14s} {cos:8.5f} {mx:9.4f} {rel:8.4f} {t.get('wall_ms_median') or 0:8.2f}  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1:])
