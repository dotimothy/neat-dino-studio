"""Grounding DINO (HF `GroundingDinoForObjectDetection`) rebuilt as MLA-friendly stages.

The MLA cannot run deformable attention's sampling efficiently (GridSample is
capped at 32-px maps and ~0.3 us/sample), so the network is cut at every
deformable attention: the MLA stages produce value maps, sampling offsets and
attention logits; the CPU samples; the next stage continues. The ref-point
update between decoder layers (sigmoid/logit/sine) also runs on the CPU.

Every stage has ONE input and ONE output tensor, all (1, 256, 1, W) token
tensors (tokens along W, channels along C) packed along W, except the two
image-side ones:

  text       in (1, 768 + 64, 1, 64)   BERT embeddings | additive phrase mask
             out (1, 256, 1, 64)
  backbone   in (1, 48, S/4, S/4)      host space-to-depth of the image
             out (1, 256, 1, N)
  enc_k      in [X | S | P | TP]       X tokens, S sampled (absent for k=0),
                                       P position embedding, TP text pack
             out [X1 | V | OFF | AW | T]  (k < 6)
  enc_6      in [X | S | P | TP]
             out [V_dec0..5 | T | HEAD]   HEAD ch 0-63 class logits, 64-67 box delta
  dec_pre_k  in [H | QS0 | QS1 | TP]    H queries (900), QS sine of the ref box (512 = 2 x 256)
             out [H2 | OFF | AW]
  dec_post_k in [H2 | S | TP]
             out [H' | HEAD]              HEAD ch 0-3 delta(h); k=5: ch 4-7 delta(LN h), 8-71 logits

Text pack TP (193 columns): 0-63 text features, 64-127 text position embedding,
128-191 additive self-attention mask (key on C, query on W), 192 additive token
mask (key on C).

Everything here is plain PyTorch on (1, C, H, W) tensors using only ops the
SiMa compiler maps to the MLA: 1x1 Conv for Linear, Transpose-LN-Transpose,
two-input Einsum attention, Split/Concat head packing, Slice/Concat rolls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 256          # d_model
T_TXT = 64       # text tokens
NQ = 900         # decoder queries
TP_W = 193       # text pack width
NEG = -1e4

# --------------------------------------------------------------------------- helpers


def conv_from_linear(lin: nn.Linear, scale: float = 1.0, out_pad: int = 0) -> nn.Conv2d:
    """nn.Linear -> 1x1 Conv2d (optionally scaled, optionally zero-padded outputs)."""
    w = lin.weight.detach() * scale
    b = lin.bias.detach() * scale if lin.bias is not None else torch.zeros(w.shape[0])
    if out_pad:
        w = torch.cat([w, torch.zeros(out_pad, w.shape[1])], 0)
        b = torch.cat([b, torch.zeros(out_pad)], 0)
    c = nn.Conv2d(w.shape[1], w.shape[0], 1)
    c.weight.data.copy_(w[:, :, None, None])
    c.bias.data.copy_(b)
    return c


class LN(nn.Module):
    """LayerNorm over channels of a (1, C, H, W) tensor: Transpose -> LN(last) -> Transpose."""

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.ln = nn.LayerNorm(ln.normalized_shape, eps=ln.eps)
        self.ln.load_state_dict(ln.state_dict())

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def heads_to_h(x, hd):
    """(1, H*hd, 1, T) -> (1, hd, H, T)"""
    return torch.cat(torch.split(x, hd, dim=1), dim=2)


def h_to_heads(x):
    """(1, hd, H, T) -> (1, H*hd, 1, T)"""
    return torch.cat(torch.split(x, 1, dim=2), dim=1)


def attention(q, k, v, mask=None):
    """q (1,hd,H,Tq) k,v (1,hd,H,Tk) -> (1,hd,H,Tq); scale already folded into q.

    scores (1, Tk, H, Tq): keys on C so Softmax runs over the channel axis.
    mask broadcastable to scores.
    """
    s = torch.einsum("nchw,nchq->nqhw", q, k)
    if mask is not None:
        s = s + mask
    p = torch.softmax(s, dim=1)
    return torch.einsum("nchw,nqhc->nqhw", p, v)


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


class MHA(nn.Module):
    """GroundingDinoMultiheadAttention / BertSelfAttention in token layout."""

    def __init__(self, q: nn.Linear, k: nn.Linear, v: nn.Linear, out: nn.Linear | None, heads: int):
        super().__init__()
        self.hd = q.weight.shape[0] // heads
        self.q = conv_from_linear(q, 1.0 / math.sqrt(self.hd))
        self.k = conv_from_linear(k)
        self.v = conv_from_linear(v)
        self.out = conv_from_linear(out) if out is not None else None

    def forward(self, xq, xk, xv, mask=None):
        o = attention(heads_to_h(self.q(xq), self.hd), heads_to_h(self.k(xk), self.hd),
                      heads_to_h(self.v(xv), self.hd), mask)
        o = h_to_heads(o)
        return self.out(o) if self.out is not None else o


def wsplit(x, sizes):
    return torch.split(x, sizes, dim=3)


# --------------------------------------------------------------------------- text (BERT)

class TextStage(nn.Module):
    """BERT-base + text_projection. in (1, 832, 1, 64) -> out (1, 256, 1, 64)."""

    def __init__(self, hf):
        super().__init__()
        bert = hf.model.text_backbone
        self.emb_ln = LN(bert.embeddings.LayerNorm)
        self.layers = nn.ModuleList()
        for L in bert.encoder.layer:
            a = L.attention
            m = nn.Module()
            m.attn = MHA(a.self.query, a.self.key, a.self.value, a.output.dense, a.self.num_attention_heads)
            m.ln1 = LN(a.output.LayerNorm)
            m.fc1 = conv_from_linear(L.intermediate.dense)
            m.fc2 = conv_from_linear(L.output.dense)
            m.ln2 = LN(L.output.LayerNorm)
            self.layers.append(m)
        self.proj = conv_from_linear(hf.model.text_projection)

    def forward(self, x):
        emb, mask = x[:, :768], x[:, 768:]          # mask (1, 64 keys, 1, 64 queries)
        h = self.emb_ln(emb)
        for m in self.layers:
            h = m.ln1(h + m.attn(h, h, h, mask))
            h = m.ln2(h + m.fc2(gelu(m.fc1(h))))
        return self.proj(h)


# --------------------------------------------------------------------------- Swin backbone

def window_partition(x, ws):
    """(1, C, H, W) -> (1, C, nW, ws*ws), windows row-major, tokens row-major."""
    _, C, H, W = x.shape
    x = x.reshape(1, C, H // ws, ws, W // ws, ws).permute(0, 1, 2, 4, 3, 5)
    return x.reshape(1, C, (H // ws) * (W // ws), ws * ws)


def window_reverse(x, ws, H, W):
    """(1, C, nW, ws*ws) -> (1, C, H, W)"""
    C = x.shape[1]
    x = x.reshape(1, C, H // ws, W // ws, ws, ws).permute(0, 1, 2, 4, 3, 5)
    return x.reshape(1, C, H, W)


def roll_hw(x, s):
    """torch.roll(x, (s, s), dims=(2, 3)) with Slice + Concat (s may be negative)."""
    H, W = x.shape[2], x.shape[3]
    sh, sw = s % H, s % W
    if sh:
        x = torch.cat([x[:, :, H - sh:], x[:, :, : H - sh]], 2)
    if sw:
        x = torch.cat([x[:, :, :, W - sw:], x[:, :, :, : W - sw]], 3)
    return x


def swin_shift_mask(Hp, Wp, ws, shift):
    """Swin attention mask (nW, ws*ws keys, ws*ws queries) -> (1, keys, nW, queries) additive."""
    img = torch.zeros(1, 1, Hp, Wp)
    cnt = 0
    for hs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
        for wsl in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            img[:, :, hs, wsl] = cnt
            cnt += 1
    mw = window_partition(img, ws)[0, 0]            # (nW, ws*ws)
    m = mw[:, None, :] - mw[:, :, None]             # HF: [w, q, k] = mw[k] - mw[q]
    m = torch.where(m != 0, torch.tensor(-100.0), torch.tensor(0.0))
    return m.permute(2, 0, 1).unsqueeze(0).contiguous()   # (1, k, nW, q)


class SwinBlock(nn.Module):
    def __init__(self, layer, H, W, ws=12):
        super().__init__()
        att = layer.attention.self
        self.heads = att.num_attention_heads
        self.hd = att.attention_head_size
        self.ws = ws
        self.shift = int(layer.shift_size)
        self.H, self.W = H, W
        self.Hp, self.Wp = -(-H // ws) * ws, -(-W // ws) * ws
        self.ln1 = LN(layer.layernorm_before)
        self.ln2 = LN(layer.layernorm_after)
        self.q = conv_from_linear(att.query, 1.0 / math.sqrt(self.hd))
        self.k = conv_from_linear(att.key)
        self.v = conv_from_linear(att.value)
        self.o = conv_from_linear(layer.attention.output.dense)
        self.fc1 = conv_from_linear(layer.intermediate.dense)
        self.fc2 = conv_from_linear(layer.output.dense)
        n = ws * ws
        rpb = att.relative_position_bias_table[att.relative_position_index.view(-1)].view(n, n, -1)
        # HF bias[h, q, k] -> ours (1, k, 1, q) per head
        bias = rpb.permute(2, 1, 0).detach()        # (h, k, q)
        self.register_buffer("bias", bias[:, :, None, :].unsqueeze(1).contiguous())  # (h,1,k,1,q)
        if self.shift:
            self.register_buffer("mask", swin_shift_mask(self.Hp, self.Wp, ws, self.shift))
        else:
            self.mask = None

    def forward(self, x):
        y = self.ln1(x)
        if (self.Hp, self.Wp) != (self.H, self.W):
            y = F.pad(y, (0, self.Wp - self.W, 0, self.Hp - self.H))
        if self.shift:
            y = roll_hw(y, -self.shift)
        q, k, v = (window_partition(f(y), self.ws) for f in (self.q, self.k, self.v))
        outs = []
        for h in range(self.heads):
            sl = slice(h * self.hd, (h + 1) * self.hd)
            s = torch.einsum("nchw,nchq->nqhw", q[:, sl], k[:, sl]) + self.bias[h]
            if self.mask is not None:
                s = s + self.mask
            p = torch.softmax(s, dim=1)
            outs.append(torch.einsum("nchw,nqhc->nqhw", p, v[:, sl]))
        o = window_reverse(torch.cat(outs, 1), self.ws, self.Hp, self.Wp)
        if self.shift:
            o = roll_hw(o, self.shift)
        if (self.Hp, self.Wp) != (self.H, self.W):
            o = o[:, :, : self.H, : self.W]
        x = x + self.o(o)
        return x + self.fc2(gelu(self.fc1(self.ln2(x))))


class PatchMerge(nn.Module):
    def __init__(self, pm):
        super().__init__()
        self.ln = LN(pm.norm)
        w = pm.reduction.weight.detach()
        self.red = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
        self.red.weight.data.copy_(w[:, :, None, None])

    def forward(self, x):
        x = torch.cat([x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2], x[:, :, 0::2, 1::2], x[:, :, 1::2, 1::2]], 1)
        return self.red(self.ln(x))


class GroupNormMLA(nn.Module):
    """GroupNorm(G, C) with ReduceMean + a group-averaging 1x1 conv + Pow(-0.5)."""

    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        C, G = gn.num_channels, gn.num_groups
        avg = torch.zeros(C, C)
        g = C // G
        for i in range(G):
            avg[i * g:(i + 1) * g, i * g:(i + 1) * g] = 1.0 / g
        self.avg = nn.Conv2d(C, C, 1, bias=False)
        self.avg.weight.data.copy_(avg[:, :, None, None])
        self.eps = gn.eps
        self.register_buffer("gamma", gn.weight.detach().view(1, C, 1, 1).clone())
        self.register_buffer("beta", gn.bias.detach().view(1, C, 1, 1).clone())

    def forward(self, x):
        mu = self.avg(x.mean(dim=(2, 3), keepdim=True))
        d = x - mu
        var = self.avg((d * d).mean(dim=(2, 3), keepdim=True))
        return d * (torch.pow(var + self.eps, -0.5) * self.gamma) + self.beta


def level_shapes(S):
    return [(S // 8, S // 8), (S // 16, S // 16), (S // 32, S // 32), (-(-S // 64), -(-S // 64))]


class BackboneStage(nn.Module):
    """Swin-B + input_proj (+GN) + stride-64 level -> tokens (1, 256, 1, N).

    in: (1, 48, S/4, S/4) = space_to_depth(normalised image, 4), channel (c, py, px).
    """

    def __init__(self, hf, S):
        super().__init__()
        sw = hf.model.backbone.conv_encoder.model
        pe = sw.embeddings.patch_embeddings.projection        # Conv2d(3, 128, 4, 4)
        w = pe.weight.detach()                                # (128, 3, 4, 4)
        self.patch = nn.Conv2d(48, 128, 1)
        self.patch.weight.data.copy_(w.reshape(128, 48)[:, :, None, None])
        self.patch.bias.data.copy_(pe.bias.detach())
        self.emb_ln = LN(sw.embeddings.norm)
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        H = S // 4
        for st in sw.encoder.layers:
            self.stages.append(nn.ModuleList([SwinBlock(b, H, H) for b in st.blocks]))
            self.merges.append(PatchMerge(st.downsample) if st.downsample is not None else nn.Identity())
            H = H // 2 if st.downsample is not None else H
        self.out_ln = nn.ModuleList([LN(sw.hidden_states_norms[s]) for s in ("stage2", "stage3", "stage4")])
        ip = hf.model.input_proj_vision
        self.proj = nn.ModuleList()
        for i in range(4):
            conv, gn = ip[i][0], ip[i][1]
            c = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding)
            c.load_state_dict(conv.state_dict())
            self.proj.append(nn.Sequential(c, GroupNormMLA(gn)))

    def forward(self, x):
        h = self.emb_ln(self.patch(x))
        feats = []
        for i, (blocks, merge) in enumerate(zip(self.stages, self.merges)):
            for b in blocks:
                h = b(h)
            if i >= 1:
                feats.append(self.out_ln[i - 1](h))
            h = merge(h)
        levels = [self.proj[i](feats[i]) for i in range(3)] + [self.proj[3](feats[2])]
        return torch.cat([lv.reshape(1, D, 1, -1) for lv in levels], 3)


# --------------------------------------------------------------------------- encoder

class BiAttn(nn.Module):
    """GroundingDinoBiMultiHeadAttention (4 heads x 256) in token layout."""

    def __init__(self, a):
        super().__init__()
        self.hd = a.head_dim
        self.vq = conv_from_linear(a.vision_proj, a.scale)
        self.tk = conv_from_linear(a.text_proj)
        self.vv = conv_from_linear(a.values_vision_proj)
        self.tv = conv_from_linear(a.values_text_proj)
        self.vo = conv_from_linear(a.out_vision_proj)
        self.tout = conv_from_linear(a.out_text_proj)

    def forward(self, v, t, tmask_c):
        qv = heads_to_h(self.vq(v), self.hd)     # (1, hd, 4, N)
        kt = heads_to_h(self.tk(t), self.hd)     # (1, hd, 4, 64)
        vv = heads_to_h(self.vv(v), self.hd)
        vt = heads_to_h(self.tv(t), self.hd)
        # vision -> text: scores (1, 64, 4, N), softmax over text (C)
        s_vt = torch.einsum("nchw,nchq->nqhw", qv, kt) + tmask_c
        ov = torch.einsum("nchw,nqhc->nqhw", torch.softmax(s_vt, 1), vt)
        # text -> vision: scores (1, N, 4, 64), softmax over vision (C)
        s_tv = torch.einsum("nchw,nchq->nqhw", kt, qv)
        ot = torch.einsum("nchw,nqhc->nqhw", torch.softmax(s_tv, 1), vv)
        return self.vo(h_to_heads(ov)), self.tout(h_to_heads(ot))


class EncLayerPre(nn.Module):
    """Fusion + text enhancer + the MLA part of the deformable layer's attention."""

    def __init__(self, layer):
        super().__init__()
        f = layer.fusion_layer
        self.lnv, self.lnt = LN(f.layer_norm_vision), LN(f.layer_norm_text)
        self.bi = BiAttn(f.attn)
        # layer scale folded into the output projections (no broadcast Mul on the MLA)
        for conv, par in ((self.bi.vo, f.vision_param), (self.bi.tout, f.text_param)):
            conv.weight.data.mul_(par.detach().view(-1, 1, 1, 1))
            conv.bias.data.mul_(par.detach())
        te = layer.text_enhancer_layer
        sa = te.self_attn
        self.te_attn = MHA(sa.query, sa.key, sa.value, sa.out_proj, te.num_heads)
        self.te_ln1, self.te_ln2 = LN(te.layer_norm_before), LN(te.layer_norm_after)
        self.te_fc1, self.te_fc2 = conv_from_linear(te.fc1), conv_from_linear(te.fc2)
        da = layer.deformable_layer.self_attn
        self.value = conv_from_linear(da.value_proj)
        self.off = conv_from_linear(da.sampling_offsets)
        self.aw = conv_from_linear(da.attention_weights, out_pad=D - da.attention_weights.out_features)

    def forward(self, x, pos, t, tpos, smask, tmask_c):
        x, t = self.lnv(x), self.lnt(t)
        dv, dt = self.bi(x, t, tmask_c)
        x = x + dv
        t = t + dt
        qk = t + tpos
        t = self.te_ln1(t + self.te_attn(qk, qk, t, smask))
        t = self.te_ln2(t + self.te_fc2(F.relu(self.te_fc1(t))))
        q = x + pos
        return x, self.value(x), self.off(q), self.aw(q), t


class EncLayerPost(nn.Module):
    """Rest of the deformable layer after the CPU sampled: out_proj, residual, LN, FFN, LN."""

    def __init__(self, layer):
        super().__init__()
        d = layer.deformable_layer
        self.out = conv_from_linear(d.self_attn.output_proj)
        self.ln1, self.ln2 = LN(d.self_attn_layer_norm), LN(d.final_layer_norm)
        self.fc1, self.fc2 = conv_from_linear(d.fc1), conv_from_linear(d.fc2)

    def forward(self, x, s):
        x = self.ln1(x + self.out(s))
        return self.ln2(x + self.fc2(F.relu(self.fc1(x))))


def unpack_tp(tp):
    t, tpos, sm, km = wsplit(tp, [T_TXT, T_TXT, T_TXT, 1])
    return t, tpos, sm[:, :T_TXT], km[:, :T_TXT]     # sm (1,64,1,64)  km (1,64,1,1)


def pack_tp(t, tpos, smask, kmask):
    """Host-side helper (torch): build the (1, 256, 1, 193) text pack."""
    sm = torch.zeros(1, D, 1, T_TXT)
    sm[:, :T_TXT] = smask
    km = torch.zeros(1, D, 1, 1)
    km[:, :T_TXT] = kmask
    return torch.cat([t, tpos, sm, km], 3)


class EncStage(nn.Module):
    """k = 0: [X | P | TP] -> [X1 | V | OFF | AW | T]
       1..5: [X | S | P | TP] -> same
       6:    [X | S | P | TP] -> [V_dec0..5 | T | HEAD]"""

    def __init__(self, hf, k, N):
        super().__init__()
        self.k, self.N = k, N
        enc = hf.model.encoder.layers
        self.post = EncLayerPost(enc[k - 1]) if k > 0 else None
        if k < 6:
            self.pre = EncLayerPre(enc[k])
        else:
            m = hf.model
            self.dec_value = nn.ModuleList(conv_from_linear(L.encoder_attn.value_proj) for L in m.decoder.layers)
            self.enc_out = conv_from_linear(m.enc_output)
            self.enc_ln = LN(m.enc_output_norm)
            be = m.encoder_output_bbox_embed
            self.bbox = nn.ModuleList(conv_from_linear(l) for l in be.layers)

    def forward(self, x):
        N = self.N
        if self.k == 0:
            X, P, TP = wsplit(x, [N, N, TP_W])
        else:
            X, S, P, TP = wsplit(x, [N, N, N, TP_W])
            X = self.post(X, S)
        t, tpos, sm, km = unpack_tp(TP)
        if self.k < 6:
            X1, V, OFF, AW, T = self.pre(X, P, t, tpos, sm, km)
            return torch.cat([X1, V, OFF, AW, T], 3)
        # final: decoder value maps, class logits and box deltas of every token
        vals = [f(X) for f in self.dec_value]
        oq = self.enc_ln(self.enc_out(X))
        logits = torch.einsum("nchw,nchq->nqhw", oq, t)        # (1, 64, 1, N)
        b = oq
        for i, l in enumerate(self.bbox):
            b = l(b)
            if i < len(self.bbox) - 1:
                b = F.relu(b)
        head = torch.cat([logits, b, torch.zeros(1, D - T_TXT - 4, 1, N)], 1)
        return torch.cat(vals + [t, head], 3)


# --------------------------------------------------------------------------- decoder

class DecPre(nn.Module):
    """[H | QS0 | QS1 | TP] -> [H2 | OFF | AW]: query pos MLP, self-attn, text cross-attn, deform proj."""

    def __init__(self, hf, k):
        super().__init__()
        dec = hf.model.decoder
        rp = dec.reference_points_head
        self.rp1, self.rp2 = conv_from_linear(rp.layers[0]), conv_from_linear(rp.layers[1])
        L = dec.layers[k]
        sa, ta = L.self_attn, L.encoder_attn_text
        self.sa = MHA(sa.query, sa.key, sa.value, sa.out_proj, sa.num_attention_heads)
        self.ta = MHA(ta.query, ta.key, ta.value, ta.out_proj, ta.num_attention_heads)
        self.ln1, self.ln2 = LN(L.self_attn_layer_norm), LN(L.encoder_attn_text_layer_norm)
        da = L.encoder_attn
        self.off = conv_from_linear(da.sampling_offsets)
        self.aw = conv_from_linear(da.attention_weights, out_pad=D - da.attention_weights.out_features)

    def forward(self, x):
        H, QS0, QS1, TP = wsplit(x, [NQ, NQ, NQ, TP_W])
        t, _, _, km = unpack_tp(TP)
        qp = self.rp2(F.relu(self.rp1(torch.cat([QS0, QS1], 1))))
        q = H + qp
        H = self.ln1(H + self.sa(q, q, H))
        H = self.ln2(H + self.ta(H + qp, t, t, km))
        q = H + qp
        return torch.cat([H, self.off(q), self.aw(q)], 3)


class DecPost(nn.Module):
    """[H2 | S | TP] -> [H' | HEAD]"""

    def __init__(self, hf, k):
        super().__init__()
        self.k = k
        dec = hf.model.decoder
        L = dec.layers[k]
        self.out = conv_from_linear(L.encoder_attn.output_proj)
        self.ln1, self.ln2 = LN(L.encoder_attn_layer_norm), LN(L.final_layer_norm)
        self.fc1, self.fc2 = conv_from_linear(L.fc1), conv_from_linear(L.fc2)
        be = hf.bbox_embed[k]
        self.bbox = nn.ModuleList(conv_from_linear(l) for l in be.layers)
        if k == len(dec.layers) - 1:
            self.dec_ln = LN(dec.layer_norm)

    def mlp(self, h):
        for i, l in enumerate(self.bbox):
            h = l(h)
            if i < len(self.bbox) - 1:
                h = F.relu(h)
        return h

    def forward(self, x):
        H, S, TP = wsplit(x, [NQ, NQ, TP_W])
        H = self.ln1(H + self.out(S))
        H = self.ln2(H + self.fc2(F.relu(self.fc1(H))))
        parts = [self.mlp(H)]
        if self.k == 5:
            t = unpack_tp(TP)[0]
            hn = self.dec_ln(H)
            parts += [self.mlp(hn), torch.einsum("nchw,nchq->nqhw", hn, t)]
        used = sum(p.shape[1] for p in parts)
        head = torch.cat(parts + [torch.zeros(1, D - used, 1, NQ)], 1)
        return torch.cat([H, head], 3)


# --------------------------------------------------------------------------- host-side pieces

def sine_pos_2d(S, level_embed):
    """lvl_pos_embed_flatten for an all-ones pixel mask: (1, 256, 1, N)."""
    outs = []
    for l, (h, w) in enumerate(level_shapes(S)):
        scale = 2 * math.pi
        y = torch.arange(1, h + 1, dtype=torch.float32)[:, None].expand(h, w) / (h + 1e-6) * scale
        x = torch.arange(1, w + 1, dtype=torch.float32)[None, :].expand(h, w) / (w + 1e-6) * scale
        dim_t = torch.arange(128, dtype=torch.float32)
        dim_t = 20 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / 128)
        px, py = x[..., None] / dim_t, y[..., None] / dim_t
        px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), 3).flatten(2)
        py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), 3).flatten(2)
        pos = torch.cat((py, px), 2).permute(2, 0, 1).reshape(D, -1)       # (256, h*w)
        outs.append(pos + level_embed[l].view(D, 1))
    return torch.cat(outs, 1).view(1, D, 1, -1)


def encoder_ref_points(S):
    """(N, 2) normalised (x, y) of every token centre."""
    refs = []
    for h, w in level_shapes(S):
        ys, xs = torch.meshgrid(torch.linspace(0.5, h - 0.5, h) / h, torch.linspace(0.5, w - 0.5, w) / w,
                                indexing="ij")
        refs.append(torch.stack([xs.reshape(-1), ys.reshape(-1)], -1))
    return torch.cat(refs, 0)


def output_proposals(S):
    """(N, 4) inverse-sigmoid proposals, +inf where invalid (HF generate_encoder_output_proposals)."""
    props = []
    for l, (h, w) in enumerate(level_shapes(S)):
        gy, gx = torch.meshgrid(torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w), indexing="ij")
        grid = torch.stack([gx, gy], -1)
        grid = (grid + 0.5) / torch.tensor([w, h], dtype=torch.float32)
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** l)
        props.append(torch.cat([grid, wh], -1).view(-1, 4))
    p = torch.cat(props, 0)
    valid = ((p > 0.01) & (p < 0.99)).all(-1, keepdim=True)
    p = torch.log(p / (1 - p))
    return p.masked_fill(~valid, float("inf")), valid[:, 0]


def msda(value, shapes, loc, aw):
    """CPU-side deformable sampling reference (HF pytorch path).

    value (N, 8, 32); loc (Q, 8, L, P, 2) normalised; aw (Q, 8, L, P) softmaxed -> (Q, 256)
    """
    from transformers.models.grounding_dino.modeling_grounding_dino import MultiScaleDeformableAttention
    m = MultiScaleDeformableAttention()
    ss = torch.tensor(shapes)
    lsi = torch.cat([ss.new_zeros(1), ss.prod(1).cumsum(0)[:-1]])
    return m(value[None], ss, shapes, lsi, loc[None], aw[None], 64)[0]


def sine_box(ref):
    """get_sine_pos_embed(ref (Q,4), 128) -> (Q, 512); order (y, x, w, h) as HF (exchange_xy)."""
    from transformers.models.grounding_dino.modeling_grounding_dino import get_sine_pos_embed
    return get_sine_pos_embed(ref[None], num_pos_feats=128)[0]


@dataclass
class Geometry:
    S: int
    N: int
    shapes: list
    pos: torch.Tensor       # (1, 256, 1, N)
    ref: torch.Tensor       # (N, 2)
    proposals: torch.Tensor  # (N, 4)
    valid: torch.Tensor     # (N,)


def geometry(hf, S):
    shapes = level_shapes(S)
    N = sum(h * w for h, w in shapes)
    props, valid = output_proposals(S)
    return Geometry(S, N, shapes, sine_pos_2d(S, hf.model.level_embed.detach()), encoder_ref_points(S),
                    props, valid)


def tok_to_rows(x):
    """(1, C, 1, T) -> (T, C)"""
    return x[0, :, 0].T


def rows_to_tok(x):
    return x.T[None, :, None]


class Pipeline:
    """Runs the stage modules in torch with host-side glue: the reference for the MLA build."""

    def __init__(self, hf, S, stages=None):
        self.hf = hf
        self.g = geometry(hf, S)
        N = self.g.N
        self.text = TextStage(hf)
        self.backbone = BackboneStage(hf, S)
        self.enc = nn.ModuleList(EncStage(hf, k, N) for k in range(7))
        self.dec_pre = nn.ModuleList(DecPre(hf, k) for k in range(6))
        self.dec_post = nn.ModuleList(DecPost(hf, k) for k in range(6))
        ib = hf.model.enc_output_norm(hf.model.enc_output(torch.zeros(1, D)))   # object query of invalid tokens
        self.invalid_query = ib.detach()[0]
        self.run_stage = stages or (lambda name, mod, x: mod(x))
        self.trace = {}

    # ---- host side text embedding (BERT embeddings before LayerNorm)
    def text_input(self, prompt):
        e = self.hf.model.text_backbone.embeddings
        ids = torch.from_numpy(prompt.input_ids)
        pos = torch.from_numpy(prompt.position_ids)
        emb = e.word_embeddings(ids) + e.position_embeddings(pos) + e.token_type_embeddings(torch.zeros_like(ids))
        mask = torch.from_numpy(prompt.additive_self_mask())                # [q, k]
        return torch.cat([rows_to_tok(emb), mask.T[None, :, None, :]], 1)  # mask (1, k, 1, q)

    @torch.no_grad()
    def __call__(self, pixels48, prompt):
        from dino.prompt import sine_embed_1d
        g, N, run = self.g, self.g.N, self.run_stage
        t = run("text", self.text, self.text_input(prompt).float())
        tpos = rows_to_tok(torch.from_numpy(sine_embed_1d(prompt.position_ids)))
        smask = torch.from_numpy(prompt.additive_self_mask()).T[None, :, None, :]
        kmask = torch.from_numpy(prompt.additive_token_mask())[None, :, None, None]
        tp = pack_tp(t, tpos, smask, kmask)
        X = run("backbone", self.backbone, pixels48)
        self.trace["backbone"] = X
        x_in = torch.cat([X, g.pos, tp], 3)
        ref = g.ref                                       # (N, 2)
        for k in range(7):
            out = run(f"enc{k}", self.enc[k], x_in)
            if k == 6:
                break
            X1, V, OFF, AW, T = wsplit(out, [N, N, N, N, T_TXT])
            s = self._sample_enc(V, OFF, AW, ref)
            tp = torch.cat([T, tp[:, :, :, T_TXT:]], 3)
            x_in = torch.cat([X1, s, g.pos, tp], 3)
        vals = wsplit(out, [N] * 6 + [T_TXT, N])
        dvals, T6, head = vals[:6], vals[6], vals[7]
        tp = torch.cat([T6, tp[:, :, :, T_TXT:]], 3)
        self.trace["enc_text"] = T6
        # two-stage top-k on the host
        logits = tok_to_rows(head[:, :T_TXT])             # (N, 64)
        delta = tok_to_rows(head[:, T_TXT:T_TXT + 4])     # (N, 4)
        tm = torch.from_numpy(prompt.token_mask)
        inv_logits = (self.invalid_query @ tok_to_rows(T6).T)      # invalid tokens' class logits
        logits = torch.where(g.valid[:, None], logits, inv_logits[None, :])
        score = logits.masked_fill(~tm[None, :], float("-inf")).max(-1).values
        top = torch.topk(score, NQ).indices
        L = (delta + g.proposals)[top]                    # coordinate logits, inf where invalid
        ref4 = L.sigmoid()
        self.trace["topk"] = top
        H = rows_to_tok(self.hf.model.query_position_embeddings.weight.detach())
        for k in range(6):
            qs = rows_to_tok(sine_box(ref4))
            pre = run(f"dec_pre{k}", self.dec_pre[k], torch.cat([H, qs[:, :D], qs[:, D:], tp], 3))
            H2, OFF, AW = wsplit(pre, [NQ, NQ, NQ])
            s = self._sample_dec(dvals[k], OFF, AW, ref4)
            post = run(f"dec_post{k}", self.dec_post[k], torch.cat([H2, s, tp], 3))
            H, head = wsplit(post, [NQ, NQ])
            hr = tok_to_rows(head)
            if k < 5:
                ref_logit = torch.special.logit(ref4, eps=1e-5)
                ref4 = (hr[:, :4] + ref_logit).sigmoid()
            else:
                boxes = (hr[:, 4:8] + torch.special.logit(ref4, eps=1e-5)).sigmoid()
                logits = hr[:, 8:8 + T_TXT].masked_fill(~tm[None, :], float("-inf"))
        return logits, boxes

    def _softmax_aw(self, AW, Q):
        a = tok_to_rows(AW)[:, :128].reshape(Q, 8, 16).softmax(-1)
        return a.reshape(Q, 8, 4, 4)

    def _sample_enc(self, V, OFF, AW, ref):
        g, N = self.g, self.g.N
        off = tok_to_rows(OFF).reshape(N, 8, 4, 4, 2)
        norm = torch.tensor([[w, h] for h, w in g.shapes], dtype=torch.float32)   # (L, 2) as (W, H)
        loc = ref[:, None, None, None, :] + off / norm[None, None, :, None, :]
        s = msda(tok_to_rows(V).reshape(N, 8, 32), g.shapes, loc, self._softmax_aw(AW, N))
        return rows_to_tok(s)

    def _sample_dec(self, V, OFF, AW, ref4):
        g = self.g
        off = tok_to_rows(OFF).reshape(NQ, 8, 4, 4, 2)
        loc = ref4[:, None, None, None, :2] + off / 4 * ref4[:, None, None, None, 2:] * 0.5
        s = msda(tok_to_rows(V).reshape(g.N, 8, 32), g.shapes, loc, self._softmax_aw(AW, NQ))
        return rows_to_tok(s)


def space_to_depth4(px):
    """(3, S, S) normalised image -> (1, 48, S/4, S/4), channel order (c, py, px) as conv weight."""
    c, S, _ = px.shape
    x = px.reshape(c, S // 4, 4, S // 4, 4).transpose(0, 2, 4, 1, 3).reshape(48, S // 4, S // 4)
    return torch.from_numpy(np.ascontiguousarray(x))[None]
