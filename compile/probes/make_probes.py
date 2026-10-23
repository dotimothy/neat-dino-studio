#!/usr/bin/env python3
"""Export small sub-graphs that exercise the risky MLA patterns of Grounding DINO.

Each probe is one single-input, single-output ONNX graph. Next to it this writes
<name>_in.npy (a test input) and <name>_ref.npy (the onnxruntime fp32 output), so
board_check.py can compare the MLA result on the DevKit numerically. That matters:
the host simulator runs reference math, so it cannot catch a kernel that ignores
part of its input (e.g. a W-only GridSample).

Usage (project venv):
    .venv/bin/python compile/probes/make_probes.py [name ...] [--list]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "probes"
sys.path.insert(0, str(ROOT))

PROBES: dict[str, callable] = {}


def probe(fn):
    PROBES[fn.__name__] = fn
    return fn


# ---------------------------------------------------------------- GridSample (G1)

class GridSamplePacked(nn.Module):
    """in = [value (C ch) | grid x,y (2 ch)] at one size; optional avg-pool of value.

    The sample coordinates come straight from the input, so the board test controls
    them all, including out-of-range ones (zeros padding).
    """

    def __init__(self, c, pool=1):
        super().__init__()
        self.c, self.pool = c, pool

    def forward(self, x):
        v = x[:, : self.c]
        if self.pool > 1:
            v = F.avg_pool2d(v, self.pool)
        g = x[:, self.c: self.c + 2].permute(0, 2, 3, 1)  # (1, Hg, Wg, 2)
        return F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)


def _gs_input(c, h, w, seed=0, oob=1.15):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((1, c, h, w)).astype(np.float32)
    g = rng.uniform(-oob, oob, (1, 2, h, w)).astype(np.float32)
    return np.concatenate([v, g], 1)


@probe
def gs24():
    """2D GridSample, value (1,32,24,24), 576 random points incl. out-of-range."""
    return GridSamplePacked(32), _gs_input(32, 24, 24)


@probe
def gs96():
    """2D GridSample at level-0 size (1,32,96,96), 9216 points."""
    return GridSamplePacked(32), _gs_input(32, 96, 96, seed=1)


@probe
def gs96to24():
    """Value pooled to 24x24, sampled at 96x96 points (few -> many)."""
    return GridSamplePacked(32, pool=4), _gs_input(32, 96, 96, seed=2)


@probe
def gs_rowonly():
    """Control: grid y constant per row (what a W-only kernel can do)."""
    x = _gs_input(32, 24, 24, seed=3)
    ys = np.linspace(-1, 1, 24, dtype=np.float32)[:, None]
    x[0, 33] = np.broadcast_to(ys, (24, 24))
    return GridSamplePacked(32), x


def _gs1d_input(c, w, seed=0, oob=1.1):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((1, c, 1, w)).astype(np.float32)
    gx = rng.uniform(-oob, oob, (1, 1, 1, w)).astype(np.float32)
    gy = np.zeros((1, 1, 1, w), np.float32)
    return np.concatenate([v, gx, gy], 1)


@probe
def gs1d_c32():
    """RAFT-style 1D GridSample: value (1,32,1,576), grid y=0."""
    return GridSamplePacked(32), _gs1d_input(32, 576)


@probe
def gs1d_c1():
    """RAFT-style 1D GridSample with one channel, value (1,1,1,576)."""
    return GridSamplePacked(1), _gs1d_input(1, 576, seed=1)


@probe
def gs1d_c16():
    """RAFT-style 1D GridSample, value (1,16,1,576)."""
    return GridSamplePacked(16), _gs1d_input(16, 576, seed=2)


def _gs_sweep(c, h, w, gh, gw, seed=7):
    rng = np.random.default_rng(seed)
    pool = h // 1
    return c, h, w, gh, gw


class GridSampleTwoIn(nn.Module):
    """Value and grid as separate graph inputs (value NCHW, grid packed NCHW 2ch)."""

    def forward(self, value, grid):
        return F.grid_sample(value, grid.permute(0, 2, 3, 1), mode="bilinear",
                             padding_mode="zeros", align_corners=False)


for _c, _h, _gh in [(1, 8, 8), (16, 8, 8), (32, 8, 8), (32, 12, 12), (32, 12, 48), (16, 16, 16), (32, 6, 32)]:
    def _mk(c=_c, h=_h, gh=_gh):
        """sweep: value (1,c,h,h) sampled at a (gh,gh) grid via avg-pool packing."""
        pool = gh // h if gh >= h else 1
        x = _gs_input(c, gh, gh, seed=c + h + gh)
        return GridSamplePacked(c, pool=pool), x
    _mk.__name__ = f"gsw_c{_c}_v{_h}_g{_gh}"
    _mk.__doc__ = f"sweep: value (1,{_c},{_h},{_h}) sampled at ({_gh},{_gh}) points"
    if _gh % _h == 0:
        PROBES[_mk.__name__] = _mk


class GridSampleQP(nn.Module):
    """Deformable-attention shape: value (1,16,H,W) from AvgPool of the packed input,
    grid (1, Q, P, 2) built from the input's two grid channels laid out as (Hg, Wg)."""

    def __init__(self, c, pool, q, p):
        super().__init__()
        self.c, self.pool, self.q, self.p = c, pool, q, p

    def forward(self, x):
        v = x[:, : self.c]
        if self.pool > 1:
            v = F.avg_pool2d(v, self.pool)
        g = x[:, self.c: self.c + 2].reshape(1, 2, self.q, self.p).permute(0, 2, 3, 1)
        return F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)


for _h, _pool in [(24, 1), (48, 1), (96, 1), (24, 4), (12, 8)]:
    def _mk(h=_h, pool=_pool):
        gh = h * pool
        x = _gs_input(16, gh, gh, seed=h + pool)
        return GridSamplePacked(16, pool=pool), x
    _mk.__name__ = f"gsw_c16_v{_h}_g{_h*_pool}"
    _mk.__doc__ = f"sweep: value (1,16,{_h},{_h}) sampled at ({_h*_pool},{_h*_pool}) points"
    PROBES[_mk.__name__] = _mk


@probe
def gsqp_l3_768():
    """Real level-3 decoder shape at 768: value (1,16,12,12), grid (1,900,4,2) (input 60x60)."""
    x = _gs_input(16, 60, 60, seed=11)
    return GridSampleQP(16, 5, 900, 4), x


@probe
def gsqp_l0_576():
    """Encoder level-0 at 576: value (1,16,72,72), grid (1,6885*4 -> 27540 as (1,6885,4,2))."""
    x = _gs_input(16, 72, 72, seed=12)[:, :, :, :]
    # 72*72 = 5184 grid points available; use Q=1296, P=4
    return GridSampleQP(16, 1, 1296, 4), x


for _c, _h, _w in [(16, 32, 32), (8, 32, 32), (8, 48, 48), (4, 48, 48), (1, 96, 96), (2, 72, 72), (16, 24, 48), (16, 12, 96), (4, 96, 96)]:
    def _mk(c=_c, h=_h, w=_w):
        rng = np.random.default_rng(c * 7 + h + w)
        v = rng.standard_normal((1, c, h, w)).astype(np.float32)
        g = rng.uniform(-1.15, 1.15, (1, 2, h, w)).astype(np.float32)
        return GridSamplePacked(c), np.concatenate([v, g], 1)
    _mk.__name__ = f"gsl_c{_c}_{_h}x{_w}"
    _mk.__doc__ = f"limit sweep: value (1,{_c},{_h},{_w}), {_h}x{_w} points"
    PROBES[_mk.__name__] = _mk


# ---------------------------------------------------------------- hop cost / multi-input (G2)

class HopConv(nn.Module):
    """Big single-input / single-output hop: conv1x1 on tokens, output k x wider."""

    def __init__(self, cin, cout, k):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(cin, cout, 1) for _ in range(k))

    def forward(self, x):
        return torch.cat([c(x) for c in self.convs], 3)


@probe
def hop768():
    """Encoder-layer-sized hop at 768: in (1,256,1,24544) -> out (1,256,1,49088)."""
    torch.manual_seed(0)
    x = np.random.default_rng(0).standard_normal((1, 256, 1, 24544)).astype(np.float32)
    return HopConv(256, 256, 2), x


@probe
def hop576():
    """Encoder-layer-sized hop at 576: in (1,256,1,13834) -> out (1,256,1,27668)."""
    torch.manual_seed(0)
    x = np.random.default_rng(1).standard_normal((1, 256, 1, 13834)).astype(np.float32)
    return HopConv(256, 256, 2), x


@probe
def hopsmall():
    """Small hop for fixed overhead: in (1,256,1,900) -> out (1,256,1,900)."""
    torch.manual_seed(0)
    x = np.random.default_rng(2).standard_normal((1, 256, 1, 900)).astype(np.float32)
    return HopConv(256, 256, 1), x


for _nm, _shape, _k in [("hopA", (1, 256, 1, 13834), 1), ("hopB", (1, 256, 96, 144), 1),
                        ("hopC", (1, 256, 1, 6917), 2), ("hopD", (1, 256, 96, 144), 2),
                        ("hopE", (1, 256, 1, 4096), 1), ("hopF", (1, 256, 1, 8192), 1)]:
    def _mk(shape=_shape, k=_k):
        torch.manual_seed(0)
        x = np.random.default_rng(sum(shape)).standard_normal(shape).astype(np.float32)
        return HopConv(256, 256, k), x
    _mk.__name__ = _nm
    _mk.__doc__ = f"hop shape {_shape} x{_k} along W"
    PROBES[_nm] = _mk


# ---------------------------------------------------------------- export helpers

def export(name, model, x, opset=17):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.onnx"
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, torch.from_numpy(x), str(path), input_names=["input"],
                          output_names=["output"], opset_version=opset, dynamo=False)
    m = onnx.load(str(path))
    try:
        import onnxsim
        m, ok = onnxsim.simplify(m)
        if ok:
            onnx.save(m, str(path))
    except Exception as e:  # noqa: BLE001
        print(f"  onnxsim skipped: {e}")
    ops = sorted({n.op_type for n in onnx.load(str(path)).graph.node})
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {"input": x})[0]
    with torch.no_grad():
        tref = model(torch.from_numpy(x)).numpy()
    err = float(np.abs(ref - tref).max())
    np.save(OUT / f"{name}_in.npy", x)
    np.save(OUT / f"{name}_ref.npy", ref)
    print(f"{name:14s} in {tuple(x.shape)} out {tuple(ref.shape)} ops {ops} ort-vs-torch {err:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, f in PROBES.items():
            print(f"{k:14s} {f.__doc__.strip().splitlines()[0]}")
        return
    for name in a.names or PROBES:
        model, x = PROBES[name]()
        export(name, model, x)


if __name__ == "__main__":
    main()
