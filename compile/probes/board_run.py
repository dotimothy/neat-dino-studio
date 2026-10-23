#!/usr/bin/env python3
"""Run MPKs on the DevKit with pyneat and save their outputs.

Runs on the board (pyneat venv). For each <name>: loads <dir>/<name>.tar.gz, feeds
<dir>/<name>_in.npy (1, C, H, W) as one HWC tensor (or, for several inputs, the files
<name>_in_<input>.npy in the order listed in <name>_inputs.txt), saves the raw output as
<dir>/<name>_board.npy and the wall-clock timing as <dir>/<name>_board.json.

    python3 board_run.py --dir /media/nvme/dino-dev/probes gs24 gs96
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import pyneat


def run_one(d, name, iters, timeout_ms):
    if os.path.exists(f"{d}/{name}_inputs.txt"):
        names = open(f"{d}/{name}_inputs.txt").read().split()
        xs = [np.load(f"{d}/{name}_in_{n}.npy").astype(np.float32) for n in names]
    else:
        xs = [np.load(f"{d}/{name}_in.npy").astype(np.float32)]
    hwcs = [np.ascontiguousarray(x[0].transpose(1, 2, 0)) for x in xs]

    def tensors():
        out = []
        for hwc in hwcs:
            t = pyneat.Tensor.from_numpy(hwc, copy=True)
            t.layout = pyneat.TensorLayout.HWC
            out.append(t)
        return out

    t0 = time.time()
    model = pyneat.Model(f"{d}/{name}.tar.gz")
    runner = model.build(tensors())
    load_s = time.time() - t0
    out = runner.run(tensors(), timeout_ms=timeout_ms)
    outs = [np.array(o.to_numpy(), dtype=np.float32, copy=True) for o in out]
    del out
    wall = []
    for _ in range(iters):
        s = time.perf_counter()
        o = runner.run(tensors(), timeout_ms=timeout_ms)
        wall.append((time.perf_counter() - s) * 1e3)
        del o
    del runner, model
    y = outs[0]
    np.save(f"{d}/{name}_board.npy", y)
    info = {"load_s": round(load_s, 2), "wall_ms_median": round(statistics.median(wall), 3) if wall else None,
            "wall_ms_min": round(min(wall), 3) if wall else None, "out_shape": list(y.shape),
            "n_outputs": len(outs)}
    with open(f"{d}/{name}_board.json", "w") as f:
        json.dump(info, f)
    print(f"{name}: {info}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--timeout-ms", type=int, default=30000)
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    rc = 0
    for n in a.names:
        try:
            run_one(a.dir, n, a.iters, a.timeout_ms)
        except Exception as e:  # noqa: BLE001
            print(f"{n}: FAILED {type(e).__name__}: {e}", flush=True)
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
