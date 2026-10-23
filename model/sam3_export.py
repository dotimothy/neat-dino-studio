#!/usr/bin/env python3
"""Export SAM 3 (facebook/sam3) as MLA programs: one ONNX per ELF, with a real input/output set.

    .venv-sam3/bin/python model/sam3_export.py --size 672 [--only vit0 dec0 ...]

Writes build/probes/sam3_<S>_<stage>.onnx, the recorded inputs <name>_in.npy (one input) or
<name>_in_<input>.npy + <name>_inputs.txt (several), and the onnxruntime output <name>_ref.npy,
the format compile/probes/board_check.sh reads.

Programs (S = 672: 48x48 tokens, 2x2 windows of 24x24):
  vit0            patch embedding + LayerNorm + layers 0-6 (windowed)     in (1, 588, 4, 576)
  vit7            layer 7 (global)                                        in (1, 1024, 1, 2304)
  vit8 vit15 vit16 vit23 vit24 vit31                                      the same pattern
  neck4 neck2 neck1   FPN levels from the ViT output in image order       in (1, 1024, 48, 48)
  text            CLIP text encoder + projection; input = [embeddings | additive mask]
  enc             the 6 DETR encoder layers                                in x, text, tmask
  dec0..dec5      one decoder layer each                                   in h, qpos, text, tmask, enc, vis_k, rpb
  maskpre         the mask decoder's prompt cross-attention                in enc, text, tmask
  pixdec          pixel decoder + instance projection                      in e2, f2, f4
  maskdot         query . pixel for all 200 queries                        in inst, memb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import sam3_mla as S  # noqa: E402
from dino_mla import GroupNormMLA  # noqa: E402

OUT = ROOT / "build" / "probes"
ALLOWED = {"Conv", "ConvTranspose", "Add", "Sub", "Mul", "Div", "Relu", "Erf", "Softmax", "Einsum", "Concat",
           "Split", "Slice", "Transpose", "Reshape", "ReduceMean", "Pow", "Pad", "Sqrt", "Resize", "MaxPool"}


def export(name, module, inputs: dict, simplify=True):
    import onnxruntime as ort
    from export_onnx import constants_to_initializers
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.onnx"
    names = list(inputs)
    xs = tuple(inputs[n] for n in names)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(module, xs, str(path), input_names=names, output_names=["output"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    m = onnx.load(str(path))
    if simplify:
        try:
            import onnxsim
            m2, ok = onnxsim.simplify(m)
            if ok:
                m = m2
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: onnxsim failed ({str(e)[:100]})")
    m = constants_to_initializers(m)
    big = sum(int(np.prod(i.dims)) for i in m.graph.initializer) * 4 > 1.9e9
    onnx.save(m, str(path), save_as_external_data=big, location=f"{name}.data" if big else None)
    feed = {n: x.numpy().astype(np.float32) for n, x in zip(names, xs)}
    y = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    with torch.no_grad():
        r = module(*xs).numpy()
    err = float(np.abs(y - r).max() / (np.abs(r).max() + 1e-12))
    if len(names) == 1:
        np.save(OUT / f"{name}_in.npy", feed[names[0]])
    else:
        for n in names:
            np.save(OUT / f"{name}_in_{n}.npy", feed[n])
        (OUT / f"{name}_inputs.txt").write_text("\n".join(names) + "\n")
    np.save(OUT / f"{name}_ref.npy", y)
    ops = sorted({n.op_type for n in m.graph.node})
    bad = sorted(set(ops) - ALLOWED)
    params = sum(int(np.prod(i.dims)) for i in m.graph.initializer)
    print(f"{name:16s} in {[tuple(x.shape) for x in xs]} out {tuple(y.shape)} nodes {len(m.graph.node):5d} "
          f"params {params / 1e6:7.1f}M ort-vs-torch {err:.1e}" + (f"  UNEXPECTED OPS {bad}" if bad else ""), flush=True)


# ------------------------------------------------------------------ stage modules
class VitGroup(nn.Module):
    def __init__(self, layers, embed=None):
        super().__init__()
        self.embed, self.layers = embed, nn.ModuleList(layers)

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        for l in self.layers:
            x = l(x)
        return x


class NeckLevel(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        for layer in self.f.scale_layers:
            x = S.gelu(x) if isinstance(layer, nn.GELU) else layer(x)
        return self.f.proj2(self.f.proj1(x))


class TextPacked(nn.Module):
    """in (1, 1024 + T, 1, T): CLIP input embeddings on the first 1024 channels, the additive
    causal/padding mask (keys on C) on the last T."""

    def __init__(self, ts, C=1024):
        super().__init__()
        self.ts, self.C = ts, C

    def forward(self, x):
        return self.ts(x[:, :self.C], x[:, self.C:])


class Enc(nn.Module):
    def __init__(self, layers, pos):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.register_buffer("pos", pos)

    def forward(self, x, text, tmask):
        for l in self.layers:
            x = l(x, self.pos, text, tmask)
        return x


class Dec(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.l = layer

    def forward(self, h, qpos, text, tmask, enc, vis_k, rpb):
        return self.l(h, qpos, text, tmask, enc, vis_k, rpb)


class MaskPre(nn.Module):
    def __init__(self, pca, ln):
        super().__init__()
        self.pca, self.ln = pca, ln

    def forward(self, enc, text, tmask):
        return enc + self.pca(self.ln(enc), text, text, tmask)


class PixDec(nn.Module):
    """Two upsample (nearest x2) + skip + 3x3 conv + GroupNorm + ReLU stages, then the instance projection."""

    def __init__(self, md):
        super().__init__()
        pd = md.pixel_decoder
        self.convs = nn.ModuleList(pd.conv_layers[:2])
        self.norms = nn.ModuleList(GroupNormMLA(n) for n in pd.norms[:2])
        self.inst = md.instance_projection

    def forward(self, e2, f2, f4):
        p = e2
        for i, skip in enumerate((f2, f4)):
            p = F.interpolate(p, scale_factor=2.0, mode="nearest") + skip
            p = F.relu(self.norms[i](self.convs[i](p)))
        return self.inst(p)


class MaskDot(nn.Module):
    def forward(self, inst, memb):                       # inst (1,256,1,P), memb (1,256,1,Q) -> (1,Q,1,P)
        return torch.einsum("nchw,nchq->nqhw", inst, memb)


def build(hf, S_=672):
    """Run the pipeline once on a real image + prompt and export every program with its real I/O."""
    from PIL import Image
    from transformers import AutoTokenizer
    pipe = S.Pipeline(hf, S_).eval()
    G, ws = pipe.G, pipe.ws
    nW = (G // ws) ** 2
    img = Image.open(ROOT / "data/images/cats.jpg").convert("RGB").resize((S_, S_), Image.BILINEAR)
    px = torch.from_numpy((np.asarray(img, np.float32) / 255 - 0.5) / 0.5).permute(2, 0, 1)
    tokz = AutoTokenizer.from_pretrained("facebook/sam3")
    t = tokz("cat", padding="max_length", max_length=32, return_tensors="pt")
    ids, nv = t.input_ids[0], int(t.attention_mask.sum())
    progs = []                                           # (name, module, inputs dict)
    pre = f"sam3_{S_}"
    with torch.no_grad():
        # vision
        x = S.to_window_layout(S.space_to_depth(px), G, ws)
        groups = [[0, 1, 2, 3, 4, 5, 6], [7], [8, 9, 10, 11, 12, 13, 14], [15], [16, 17, 18, 19, 20, 21, 22],
                  [23], [24, 25, 26, 27, 28, 29, 30], [31]]
        for g in groups:
            glob = pipe.vit[g[0]].cos.shape[2] == 1
            mod = VitGroup([pipe.vit[i] for i in g], pipe.embed if g[0] == 0 else None).eval()
            xin = x.reshape(1, x.shape[1], 1, -1) if glob else x
            progs.append((f"{pre}_vit{g[0]}", mod, {"input": xin.clone()}))
            x = mod(xin).reshape(1, 1024, nW, ws * ws)
        sp = S.from_window_layout(x, G, ws)
        feats = []
        for name, f in zip(("neck4", "neck2", "neck1"), pipe.neck.fpn):
            mod = NeckLevel(f).eval()
            progs.append((f"{pre}_{name}", mod, {"input": sp.clone()}))
            feats.append(mod(sp))
        f4, f2, f1 = feats
        # text
        e = hf.text_encoder.text_model.embeddings
        emb = S.tok(e.token_embedding(ids) + e.position_embedding.weight[:32])
        packed = torch.cat([emb, S.text_mask(nv, 32).reshape(1, 32, 1, 32)], 1)
        mod = TextPacked(pipe.text).eval()
        progs.append((f"{pre}_text", mod, {"input": packed}))
        text = mod(packed)
        tmask = torch.where(torch.arange(32) < nv, 0.0, -1e4).view(1, 32, 1, 1)
        # encoder: the 1x level is (1, 256, G, G) row-major = the same bytes as (1, 256, 1, G*G)
        pos = S.tok(S.sine_pos_2d(G, G).reshape(256, -1).T)
        x = f1.reshape(1, 256, 1, G * G)
        mod = Enc(pipe.enc, pos).eval()
        progs.append((f"{pre}_enc", mod, {"x": x, "text": text, "tmask": tmask}))
        enc = mod(x, text, tmask)
        # decoder
        d = hf.detr_decoder
        h = torch.cat([S.tok(d.presence_token.weight), S.tok(d.query_embed.weight)], 3)
        ref = d.reference_points.weight.sigmoid()
        vis_k = enc + pos
        for i, l in enumerate(pipe.dec):
            qpos = torch.cat([torch.zeros(1, 256, 1, 1), pipe.ref_head(S.tok(S.sine_boxes(d, ref)))], 3)
            rpb = S.box_rpb(d, ref, G, G)
            mod = Dec(l).eval()
            inp = {"h": h, "qpos": qpos, "text": text, "tmask": tmask, "enc": enc, "vis_k": vis_k, "rpb": rpb}
            progs.append((f"{pre}_dec{i}", mod, inp))
            h = mod(*inp.values())
            q = pipe.out_ln(h[:, :, :, 1:])
            ref = (torch.logit(ref, eps=1e-3) + S.rows(pipe.box_head(q))).sigmoid()
        # masks
        mod = MaskPre(pipe.pca, pipe.pca_ln).eval()
        progs.append((f"{pre}_maskpre", mod, {"enc": enc, "text": text, "tmask": tmask}))
        e2 = mod(enc, text, tmask).reshape(1, 256, G, G)
        mod = PixDec(hf.mask_decoder).eval()
        progs.append((f"{pre}_pixdec", mod, {"e2": e2, "f2": f2, "f4": f4}))
        inst = mod(e2, f2, f4)
        me = q
        for j, c in enumerate(pipe.mask_embed):
            me = c(me)
            if j < len(pipe.mask_embed) - 1:
                me = F.relu(me)
        progs.append((f"{pre}_maskdot", MaskDot(), {"inst": inst.reshape(1, 256, 1, -1), "memb": me[:, :, :, :]}))
    return progs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=672)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    from transformers import Sam3Config, Sam3Model
    cfg = Sam3Config.from_pretrained("facebook/sam3")
    cfg.vision_config.backbone_config.image_size = a.size
    hf = Sam3Model.from_pretrained("facebook/sam3", config=cfg).eval()
    for name, mod, inputs in build(hf, a.size):
        if a.only and name.split("_", 2)[2] not in a.only:
            continue
        export(name, mod, inputs)


if __name__ == "__main__":
    main()
