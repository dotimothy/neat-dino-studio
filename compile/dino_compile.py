#!/usr/bin/env python3
"""Quantize and compile one Grounding DINO sub-graph (or a probe) for Modalix.

Run inside the Model Compiler environment (compile/compile.sh does this):

    python3 compile/dino_compile.py --model build/onnx/dino_text_t64.onnx --precision bf16 \
        --out build/mpk/text_t64_bf16

Input names and shapes are read from the ONNX file, so the same driver serves the
four model parts and every probe.

Precisions:
  bf16   bf16 weights and activations (no calibration needed; the accurate build)
  w8a16  int8 weights, bf16 activations (needs --calib)
  int8   int8 weights and activations (needs --calib)

--calib is an .npz holding one array per ONNX input, keyed by input name, each
(N, C, H, W) or (N, H, W, C) float32.

--quant-overrides is a JSON file {node_name: "bf16" | "w8a16"} applied with
QuantizationParams.with_custom_quantization_configs, for mixed builds.

Afterwards the MPK is checked: exactly one MLA ELF and no A65 library (pyneat
refuses split MPKs). The script exits 2 otherwise.
"""
import argparse
import json
import logging
import os
import sys
import tarfile
import time

import numpy as np
import onnx

from afe.apis.defines import (
    default_quantization, quantization_scheme, bfloat16_scheme,
    RequantizationMode, CalibrationMethod, gen2_target, InputName,
)
from afe.apis.loaded_net import load_model
from afe.core.utils import convert_data_generator_to_iterable
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import ImporterParams, ModelFormat
from sima_utils.data.data_generator import DataGenerator


def log(msg=""):
    print(msg, flush=True)


def onnx_io(path):
    m = onnx.load(path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    ins = [(i.name, tuple(d.dim_value for d in i.type.tensor_type.shape.dim))
           for i in m.graph.input if i.name not in inits]
    outs = [o.name for o in m.graph.output]
    return ins, outs


def calibration_set(path, inputs, n, precision):
    data = {}
    if path and os.path.exists(path):
        z = np.load(path)
        for name, shape in inputs:
            if name not in z:
                sys.exit(f"{path}: missing array '{name}' (has {list(z.keys())})")
            a = z[name].astype(np.float32)[:n]
            if a.shape[1:] == shape[1:]:                       # NCHW -> NHWC
                a = a.transpose(0, 2, 3, 1)
            elif a.shape[1:] != (shape[2], shape[3], shape[1]):
                sys.exit(f"{path}:{name}: expected (N,{shape[1:]}) or NHWC, got {a.shape}")
            data[name] = np.ascontiguousarray(a)
        log(f"  calibration: {len(next(iter(data.values())))} samples from {path}")
    else:
        if precision != "bf16":
            sys.exit(f"--calib {path!r} not found; int8/w8a16 need real calibration data")
        rng = np.random.default_rng(0)
        for name, shape in inputs:
            data[name] = rng.random((2, shape[2], shape[3], shape[1]), dtype=np.float32)
        log("  calibration: 2 random samples (bf16 has no scales to fit)")
    return convert_data_generator_to_iterable(DataGenerator({InputName(k): v for k, v in data.items()}))


def summarize_mpk(out_dir):
    mpks = [f for f in os.listdir(out_dir) if f.endswith("_mpk.tar.gz")]
    if not mpks:
        log("  no *_mpk.tar.gz produced")
        return None
    path = os.path.join(out_dir, mpks[0])
    with tarfile.open(path) as t:
        names = t.getnames()
        manifest = next((n for n in names if n.endswith("_mpk.json")), None)
        mpk = json.load(t.extractfile(manifest)) if manifest else {}
    plugins = mpk.get("plugins", [])
    elfs = [n for n in names if n.endswith("_mla.elf")]
    a65 = [n for n in names if n.endswith("_a65.so")]
    procs = {}
    for p in plugins:
        procs[p["processor"]] = procs.get(p["processor"], 0) + 1
    log(f"  MPK: {path} ({os.path.getsize(path) / 2**20:.0f} MB)")
    log(f"  MLA ELFs: {len(elfs)} {elfs}")
    log(f"  A65 libraries: {len(a65)}")
    log(f"  plugins by processor: {procs}")
    return {"mpk": path, "mla_elfs": elfs, "a65": a65, "plugins": procs,
            "input_nodes": mpk.get("input_nodes"), "output_nodes": mpk.get("output_nodes")}


def schemes(precision):
    if precision == "bf16":
        return bfloat16_scheme(), bfloat16_scheme()
    if precision == "w8a16":
        return bfloat16_scheme(), quantization_scheme(False, True, 8)
    return quantization_scheme(True, False, 8), quantization_scheme(False, True, 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", choices=["bf16", "int8", "w8a16"], default="bf16")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--num-calib", type=int, default=64)
    ap.add_argument("--calib-method", default="min_max")
    ap.add_argument("--quant-overrides", default=None)
    ap.add_argument("--name", default=None, help="model name inside the MPK")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-compile", action="store_true", help="quantize only")
    ap.add_argument("--mla-tess", choices=["ev74", "hwc", "hwc16"], default="hwc",
                    help="where (de)tessellation runs: ev74 plugins, or on the MLA with this DRAM "
                         "layout (default; EV74 tessellation of wide token tensors takes seconds)")
    ap.add_argument("--io-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="graph input type: bf16 lets the host hand the MLA bf16 data directly")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.model))[0]
    name = args.name or f"{stem}_{args.precision}"
    out = os.path.abspath(args.out or os.path.join("build", args.precision))
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    inputs, outputs = onnx_io(args.model)
    log(f"[1/3] load {args.model}")
    log(f"  inputs {inputs}  outputs {outputs}")
    net = load_model(ImporterParams(
        format=ModelFormat.onnx, file_paths=[args.model], input_names=[n for n, _ in inputs],
        input_shapes=[s for _, s in inputs], input_types=[ScalarType.bfloat16 if args.io_dtype == "bf16" else ScalarType.float32] * len(inputs),
        layout="NCHW", output_names=outputs), target=gen2_target)

    log(f"[2/3] quantize: {args.precision}")
    act, wgt = schemes(args.precision)
    cfg = (default_quantization.with_activation_quantization(act).with_weight_quantization(wgt)
           .with_requantization_mode(RequantizationMode.sima)
           .with_calibration(CalibrationMethod.from_str(args.calib_method)))
    if args.quant_overrides:
        from afe.core.configs import QuantizationPrecision
        table = {"bf16": QuantizationPrecision.BFLOAT_16,
                 "w8a16": QuantizationPrecision.BFLOAT_16_INT8_WEIGHTS}
        with open(args.quant_overrides) as f:
            over = json.load(f)
        cfg = cfg.with_custom_quantization_configs(
            {node: {"quantization_precision": table[p]} for node, p in over.items()})
        log(f"  {len(over)} per-node precision overrides from {args.quant_overrides}")
    calib = calibration_set(args.calib, inputs, args.num_calib, args.precision)
    q = net.quantize(calibration_data=calib, quantization_config=cfg, model_name=name, log_level=logging.INFO)
    q.save(model_name=name, output_directory=out)
    log(f"  quantized model saved to {out} ({time.time() - t0:.0f} s)")

    summary = {"model": args.model, "precision": args.precision, "name": name,
               "onnx_inputs": [n for n, _ in inputs], "onnx_outputs": outputs}
    ok = True
    if not args.no_compile:
        log("[3/3] compile for Modalix")
        kw = {}
        if args.mla_tess != "ev74":
            from afe.apis.defines import TensorDRAMLayout, TensorTessellateParameters
            from sima_lmm.model.sima_analysis import get_tessellate_parameters
            lay = TensorDRAMLayout.HWC if args.mla_tess == "hwc" else TensorDRAMLayout.HWC16
            tp = TensorTessellateParameters(tile_shape=(0, 0, 0, 0), enable_mla=True, dram_layout=lay,
                                            persistent_mem_name=None, dram_shape=None)
            kw["tessellate_parameters"] = get_tessellate_parameters(q, {0: tp}, {0: tp})
            log(f"  tessellation on the MLA, DRAM layout {lay}")
        q.compile(output_path=out, batch_size=1, log_level=logging.DEBUG if os.environ.get("DINO_COMPILE_DEBUG") else logging.INFO, **kw)
        log(f"  compiled ({time.time() - t0:.0f} s total)")
        s = summarize_mpk(out) or {}
        summary.update(s)
        ok = len(s.get("mla_elfs", [])) == 1 and not s.get("a65")
        summary["single_mla_elf"] = ok
        log(f"  single MLA ELF, no A65: {'YES' if ok else 'NO'}")
    summary["seconds"] = round(time.time() - t0)
    with open(os.path.join(out, "compile_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
