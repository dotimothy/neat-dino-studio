#!/usr/bin/env python3
"""Export the Grounding DINO MLA stages to ONNX, with a real input/output pair each.

Runs the reference pipeline once on an image + prompt, records every stage's
input and output, and exports each requested stage as a single-input,
single-output ONNX graph (onnxsim-simplified). Next to <name>.onnx it writes
<name>_in.npy and <name>_ref.npy (onnxruntime output), the format the probe
tools use (compile/probes/run_probes.sh, board_check.sh).

    .venv/bin/python model/export_onnx.py --size 768 --stages backbone enc0 dec_pre0
    .venv/bin/python model/export_onnx.py --size 768 --all
Output: build/probes/s<size>_<stage>.onnx (text: build/probes/text_t64.onnx)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
import dino_mla as M  # noqa: E402
from dino import prompt as P  # noqa: E402
from dino.letterbox import letterbox  # noqa: E402

OUT = ROOT / "build" / "probes"
ALL = ["text", "backbone"] + [f"enc{k}" for k in range(7)] + ["dec_values"] + \
      [f"dec_pre{k}" for k in range(6)] + [f"dec_post{k}" for k in range(6)]

# Ops the MLA build may contain (anything else is a red flag before compiling).
ALLOWED = {"Conv", "Add", "Sub", "Mul", "Div", "Relu", "Erf", "Softmax", "Einsum", "Concat", "Split",
           "Slice", "Transpose", "Reshape", "LayerNormalization", "ReduceMean", "Pow", "Pad", "Sqrt",
           "Exp", "Reciprocal"}


PREFIX = {"base": ("s", "text_t64"), "tiny": ("t", "tiny_text_t64")}   # build names per model


def stage_name(S, st, model="base"):
    p, text = PREFIX[model]
    return text if st == "text" else f"{p}{S}_{st}"


def constants_to_initializers(model):
    g = model.graph
    keep = []
    for n in g.node:
        if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
            t = onnx.numpy_helper.to_array(n.attribute[0].t)
            g.initializer.append(onnx.numpy_helper.from_array(t, n.output[0]))
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    return model


def export(name, module, x, ref_out, simplify=True):
    import onnxruntime as ort
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.onnx"
    t0 = time.time()
    module.eval()
    with torch.no_grad():
        torch.onnx.export(module, x, str(path), input_names=["input"], output_names=["output"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    m = onnx.load(str(path))
    if simplify:
        try:
            import onnxsim
            m2, ok = onnxsim.simplify(m)
            if ok:
                m = m2
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: onnxsim failed ({str(e)[:120]}), keeping the raw export")
    m = constants_to_initializers(m)
    onnx.save(m, str(path))
    ops = sorted({n.op_type for n in m.graph.node})
    bad = sorted(set(ops) - ALLOWED)
    xi = x.numpy().astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"input": xi})[0]
    r = ref_out.numpy()
    err = float(np.abs(y - r).max() / (np.abs(r).max() + 1e-12))
    np.save(OUT / f"{name}_in.npy", xi)
    np.save(OUT / f"{name}_ref.npy", y)
    params = sum(int(np.prod(i.dims)) for i in m.graph.initializer)
    print(f"{name:16s} in {tuple(xi.shape)} out {tuple(y.shape)} nodes {len(m.graph.node):5d} "
          f"params {params / 1e6:6.1f}M ort-vs-torch {err:.1e} {time.time() - t0:5.0f}s"
          + (f"  UNEXPECTED OPS {bad}" if bad else ""))


def export_backbone(hf, S, image, only=None, fused=False, skip_existing=False, model="base"):
    """Export every MLA program of the window-layout backbone (model/backbone_plan.py,
    or the fused 9-program version in model/backbone_fused.py)."""
    import backbone_plan as BP
    from PIL import Image
    if fused:
        import backbone_fused as BF
        steps = BF.build_cpu_merge(hf, S)
    else:
        steps = BP.build(hf, S)
    px, _ = letterbox(Image.open(image), S)
    rec = {}
    orig = BP.Step

    def runner(st, x):
        y = st.module(x)
        rec[st.name] = (x.clone(), y.clone())
        return y

    # run once, recording each program's real input/output
    bufs_steps = []
    for st in steps:
        if st.op == "run":
            m = st.module
            st.module = (lambda mm, name: (lambda x: (rec.__setitem__(name, (x.clone(), mm(x).clone())) or rec[name][1])))(m, st.name)
            bufs_steps.append((st, m))
    BP.run_torch(steps, M.space_to_depth4(px), M.geometry(hf, S).N)
    for st, m in bufs_steps:
        st.module = m
        if (only and st.name not in only) or (skip_existing and (OUT / f"{stage_name(S, st.name, model)}.onnx").exists()
                                               and st.name in ("bb_stage1", "bb_stage3b", "bb_stage3c")):
            continue
        x, y = rec[st.name]
        export(stage_name(S, st.name, model), m, x, y)
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--stages", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--backbone-plan", action="store_true", help="export the window-layout backbone programs")
    ap.add_argument("--backbone-fused", action="store_true", help="export the fused 9-program backbone")
    ap.add_argument("--image", default=str(ROOT / "data/images/cats.jpg"))
    ap.add_argument("--prompt", default="a cat. a remote control. a couch. a blanket.")
    ap.add_argument("--model", default="base", choices=sorted(M.MODELS), help="grounding-dino-base or -tiny")
    a = ap.parse_args()
    stages = [s for s in (ALL if a.all else a.stages) if s != "backbone"]
    from PIL import Image
    hf = M.load_hf(a.model)
    if a.backbone_fused:
        export_backbone(hf, a.size, a.image, fused=True, model=a.model)
        if not stages:
            return
    if a.backbone_plan or a.all:
        export_backbone(hf, a.size, a.image, model=a.model)
        if not stages:
            return
    rec = {}

    def run(name, mod, x):
        y = mod(x)
        rec[name] = (mod, x.clone(), y.clone())
        return y

    pipe = M.Pipeline(hf, a.size, stages=run)
    pr = P.build(a.prompt, P.Tokenizer())
    px, _ = letterbox(Image.open(a.image), a.size)
    with torch.no_grad():
        pipe(M.space_to_depth4(px), pr)
    for st in stages:
        mod, x, y = rec[st]
        export(stage_name(a.size, st, a.model), mod, x, y)


if __name__ == "__main__":
    main()
