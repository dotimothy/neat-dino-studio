#!/usr/bin/env python3
"""Export OmDet-Turbo-Tiny as MLA programs (model/omdet_mla.py), with recorded inputs for checks and
real calibration data (build/calib/<name>.npz) for the compiler.

    .venv-sam3/bin/python model/omdet_export.py [--only bb aifi ccfm decin dec text task]

Writes build/probes/omdet_<prog>.onnx (+ _in*.npy / _ref.npy like the other exports).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import omdet_mla as O  # noqa: E402
from sam3_export import ALLOWED  # noqa: E402

OUT = ROOT / "build" / "probes"
CAL = ROOT / "build" / "calib"
SAMPLES = sorted(p for p in (ROOT / "webapp" / "web" / "samples").glob("*.jpg") if "thumb" not in p.name)
PROMPTS = ["horse . person .", "pizza . bottle . cup .", "cow .", "dog . person . boat . sunglasses .",
           "yellow bird .", "dog . red bandana . car ."]


def export(name, module, inputs: dict, calib: dict, out_names=("output",)):
    """inputs: name -> first sample; calib: name -> list of samples (all NCHW)."""
    import onnxruntime as ort
    import onnxsim
    from export_onnx import constants_to_initializers
    OUT.mkdir(parents=True, exist_ok=True)
    CAL.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.onnx"
    names = list(inputs)
    xs = tuple(inputs[n] for n in names)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(module, xs, str(path), input_names=names, output_names=list(out_names), opset_version=17,
                          dynamo=False, do_constant_folding=True)
    m = onnx.load(str(path))
    try:
        m2, ok = onnxsim.simplify(m)
        m = m2 if ok else m
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: onnxsim failed ({str(e)[:100]})")
    m = constants_to_initializers(m)
    onnx.save(m, str(path))
    feed = {n: x.numpy().astype(np.float32) for n, x in zip(names, xs)}
    ys = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)
    with torch.no_grad():
        r = module(*xs)
    r = list(r) if isinstance(r, (list, tuple)) else [r]
    err = max(float(np.abs(y - t.numpy()).max() / (np.abs(t.numpy()).max() + 1e-12)) for y, t in zip(ys, r))
    if len(names) == 1:
        np.save(OUT / f"{name}_in.npy", feed[names[0]])
    else:
        for n in names:
            np.save(OUT / f"{name}_in_{n}.npy", feed[n])
        (OUT / f"{name}_inputs.txt").write_text("\n".join(names) + "\n")
    np.save(OUT / f"{name}_ref.npy", ys[0])
    np.savez(CAL / f"{name}.npz", **{k: torch.cat(v).numpy() for k, v in calib.items()})
    ops = sorted({n.op_type for n in m.graph.node})
    bad = sorted(set(ops) - ALLOWED - {"Sigmoid", "Resize"})
    params = sum(int(np.prod(i.dims)) for i in m.graph.initializer)
    print(f"{name:16s} in {[tuple(x.shape) for x in xs]} out {[tuple(y.shape) for y in ys]} nodes {len(m.graph.node):5d} "
          f"params {params / 1e6:6.1f}M ort-vs-torch {err:.1e}" + (f"  UNEXPECTED OPS {bad}" if bad else ""), flush=True)


class Recorder:
    """Runs the reference pipeline on the sample images and keeps every program's inputs."""

    def __init__(self, P):
        self.P = P
        self.rec = {}

    def add(self, name, **kw):
        for k, v in kw.items():
            self.rec.setdefault(name, {}).setdefault(k, []).append(v.detach().clone())

    @torch.no_grad()
    def run(self):
        P = self.P
        hf = P.hf
        d = hf.decoder
        for path, prompt in zip(SAMPLES, PROMPTS):
            classes = [c.strip() for c in prompt.split(".") if c.strip()]
            rgb = np.asarray(Image.open(path).convert("RGB").resize((O.S, O.S), Image.BILINEAR), np.uint8)
            # text: the program sees TEXT_B sequences at once
            task = f"Detect {', '.join(classes)}."
            seqs = [P.text.ids(c) for c in classes] + [P.text.ids(task)]
            emb = P.text.embed(seqs[:O.TEXT_B])
            self.add("text", x=emb)
            hid = P.text_enc(emb)
            n = min(len(seqs[-1]), O.TT)
            th = torch.zeros(1, 512, 1, O.TT)
            th[0, :, 0, :n] = hid[0, :, len(seqs) - 1, :n]
            self.add("task", x=th)
            # backbone blocks: replay the plan, recording each block's input
            px = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)
            self._backbone(px)
            l0, l1, l2 = self.levels
            self.add("aifi", x=l2.reshape(1, O.D, 1, -1))
            a = P.aifi(l2.reshape(1, O.D, 1, -1)).reshape(l2.shape)
            self.add("ccfm", l0=l0, l1=l1, l2=a)
            o = P.ccfm(l0, l1, a)
            tok = [x.reshape(1, O.D, 1, -1) for x in o]
            self.add("decin", l0=tok[0], l1=tok[1], l2=tok[2])
            # decoder: run the pipeline's loop and record pre/post inputs
            self._decoder(P.decin(*tok), classes)

    def _backbone(self, px):
        bb = self.P.bb
        buf, x = None, None
        self.levels = [None] * 3
        for st in bb.steps:
            if st["op"] == "patchify":
                g = O.space_to_depth4(px).reshape(48, -1)
                tab = torch.from_numpy(st["table"])
                x = torch.where(tab[None] >= 0, g[:, tab.clamp(min=0)], torch.zeros(())).view(1, 48, -1, O.WS * O.WS)
            elif st["op"] == "gather":
                src = buf[0].reshape(buf.shape[1], -1)[st["src_off"]:st["src_off"] + st["c"]]
                tab = torch.from_numpy(st["table"]).view(-1, st["k"])
                g = torch.where(tab[None] >= 0, src[:, tab.clamp(min=0)], torch.zeros(()))
                x = g.permute(2, 0, 1).reshape(st["k"] * st["c"], -1).view(1, st["k"] * st["c"], -1, O.WS * O.WS)
            elif st["op"] == "run":
                self.add(st["name"], x=x)
                buf = bb.modules[st["name"]](x)
            elif st["op"] == "level":
                src = buf[0].reshape(buf.shape[1], -1)[st["src_off"]:st["src_off"] + st["c"]]
                h, w = st["hw"]
                self.levels[st["level"]] = src[:, torch.from_numpy(st["table"])].view(1, st["c"], h, w)

    def _decoder(self, out, classes):
        P = self.P
        d = P.hf.decoder
        cls, task, km = P.prompt(classes)
        N = sum(h * w for h, w in O.level_sizes())
        V = [O.rows(out[:, :, :, i * N:(i + 1) * N]) for i in range(6)]
        cf = O.rows(out[:, :, :, 6 * N:7 * N]).clone()
        box = O.rows(out[:, :, :, 7 * N:8 * N])[:, :4]
        cf[~P.valid] = P.cf_invalid
        sim = O.cosine_logits(cf, d.encoder_class_head(cls))
        top = torch.topk(sim.max(1).values, O.NQ).indices
        ref_logit = box[top] + P.anchor[top]
        ref_logit[~P.valid[top]] = float("inf")
        ref = ref_logit.sigmoid()
        H, TASK = S3tok(cf[top]), task
        for k in range(len(d.layers)):
            REF = torch.zeros(1, O.D, 1, O.NQ)
            REF[0, :4, 0] = ref.T
            KM = torch.zeros(1, O.D, 1, 1)
            KM[0, :O.TT, 0, 0] = km
            xin = torch.cat([H, REF, TASK, KM], 3)
            self.add(f"dec_pre{k}", x=xin)
            o = P.pre[k](xin)
            H2, T2, OFF, AW = torch.split(o, [O.NQ, O.TT, O.NQ, O.NQ], 3)
            off = O.rows(OFF)[:, :192].view(O.NQ, 8, 3, 4, 2)
            aw = torch.softmax(O.rows(AW)[:, :96].view(O.NQ, 8, 12), -1).view(O.NQ, 8, 3, 4)
            loc = ref[:, None, None, None, :2] + off / 4 * ref[:, None, None, None, 2:] * 0.5
            sampled = O.msda(V[k].view(N, 8, 32), O.level_sizes(), loc, aw)
            xp = torch.cat([H2, S3tok(sampled)], 3)
            self.add(f"dec_post{k}", x=xp)
            o = P.post[k](xp)
            H, B = torch.split(o, [O.NQ, O.NQ], 3)
            ref = torch.sigmoid(O.rows(B)[:, :4] + O.inverse_sigmoid(ref))
            TASK = T2


def S3tok(x):
    return O.S3.tok(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    want = lambda k: not a.only or k in a.only  # noqa: E731
    hf = O.load_hf()
    P = O.Pipeline(hf)
    R = Recorder(P)
    R.run()
    rec = R.rec
    print(f"recorded {len(SAMPLES)} images")
    first = lambda n: {k: v[0] for k, v in rec[n].items()}  # noqa: E731
    if want("bb"):
        for name, m in P.bb.blocks:
            export(f"omdet_{name}", m, first(name), rec[name])
    if want("aifi"):
        export("omdet_aifi", P.aifi, first("aifi"), rec["aifi"])
    if want("ccfm"):
        export("omdet_ccfm", P.ccfm, first("ccfm"), rec["ccfm"], out_names=("o0", "o1", "o2"))
    if want("decin"):
        export("omdet_decin", P.decin, first("decin"), rec["decin"])
    if want("dec"):
        for k in range(6):
            export(f"omdet_dec_pre{k}", P.pre[k], first(f"dec_pre{k}"), rec[f"dec_pre{k}"])
            export(f"omdet_dec_post{k}", P.post[k], first(f"dec_post{k}"), rec[f"dec_post{k}"])
    if want("text"):
        export("omdet_text", P.text_enc, first("text"), rec["text"])
    if want("task"):
        export("omdet_task", P.task_head, first("task"), rec["task"])


if __name__ == "__main__":
    main()
