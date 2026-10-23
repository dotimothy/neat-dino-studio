"""SAM 3 (facebook/sam3) rebuilt in MLA-friendly ops, starting with the Perception Encoder ViT.

Tokens use the layout of the Grounding DINO port: (1, C, H, W) with channels on C, a window
per row of H and the 24x24 tokens of a window along W (row-major). Windowed layers attend
within each row; the four global layers see the same buffer as (1, C, 1, H*W): in the MLA's
channel-last memory those are the same bytes, so the runtime just hands the next program the
other shape (no copy, no reshape on the MLA).

Rotary position embedding (axial 2-D, pairwise rotation) is folded into the weights: the
pairwise rotation is a fixed signed permutation P of each head's channels, so rot(Wq x) =
(P Wq) x is just another 1x1 conv, and q_rope = q * cos + q_rot * sin with full-shape
constant tables (no broadcast multiplies).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_from_linear(lin: nn.Linear, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None,
                     scale: float = 1.0) -> nn.Conv2d:
    w = (lin.weight if weight is None else weight).detach() * scale
    b = lin.bias if bias is None else bias
    c = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=b is not None)
    c.weight.data.copy_(w[:, :, None, None])
    if b is not None:
        c.bias.data.copy_(b.detach() * scale)
    return c


class LN(nn.Module):
    """LayerNorm over C of (1, C, H, W)."""

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.eps = ln.eps
        self.register_buffer("w", ln.weight.detach().view(1, -1, 1, 1).clone())
        self.register_buffer("b", ln.bias.detach().view(1, -1, 1, 1).clone())

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        d = x - mu
        var = (d * d).mean(1, keepdim=True)
        return d * torch.pow(var + self.eps, -0.5) * self.w + self.b


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


def pair_rotation(dim: int) -> torch.Tensor:
    """(dim, dim) matrix R with R @ x == rotate_pairwise(x): out[2i] = -x[2i+1], out[2i+1] = x[2i]."""
    r = torch.zeros(dim, dim)
    for i in range(0, dim, 2):
        r[i, i + 1] = -1.0
        r[i + 1, i] = 1.0
    return r


def rope_tables(layer, grid_hw: tuple[int, int], heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin per token for one HF Sam3ViTLayer at a token grid, as (heads*hd, tokens), row-major."""
    cfg = layer.config
    gh, gw = grid_hw
    hpos, wpos = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    pos = torch.stack([wpos.flatten(), hpos.flatten()], dim=-1).float()
    pos = pos * (cfg.window_size / gw)
    cos, sin = layer.rotary_emb(torch.zeros(1), pos)          # (1, tokens, hd)
    cos, sin = cos[0].T, sin[0].T                              # (hd, tokens)
    return cos.repeat(heads, 1), sin.repeat(heads, 1)          # same for every head


class ViTLayerW(nn.Module):
    """One Sam3ViTLayer on (1, C, nW, T) tokens. Windowed: nW windows of T = ws*ws tokens.
    Global: nW = 1 and T = all tokens (row-major over the window layout, see rope)."""

    def __init__(self, layer, n_windows: int, grid_hw: tuple[int, int], token_order=None):
        super().__init__()
        att = layer.attention
        self.heads, self.hd = att.num_attention_heads, att.head_dim
        C = att.hidden_size
        self.ln1, self.ln2 = LN(layer.layer_norm1), LN(layer.layer_norm2)
        rot = torch.block_diag(*[pair_rotation(self.hd)] * self.heads)
        s = att.scaling
        self.q = conv_from_linear(att.q_proj, scale=s)
        self.qr = conv_from_linear(att.q_proj, rot @ att.q_proj.weight, rot @ att.q_proj.bias, scale=s)
        self.k = conv_from_linear(att.k_proj)
        self.kr = conv_from_linear(att.k_proj, rot @ att.k_proj.weight, rot @ att.k_proj.bias)
        self.v = conv_from_linear(att.v_proj)
        self.o = conv_from_linear(att.o_proj)
        self.fc1 = conv_from_linear(layer.mlp.fc1)
        self.fc2 = conv_from_linear(layer.mlp.fc2)
        cos, sin = rope_tables(layer, grid_hw, self.heads)     # (C, T)
        if token_order is not None:                            # global layer on window-layout tokens
            cos, sin = cos[:, token_order], sin[:, token_order]
        T = cos.shape[1]
        self.register_buffer("cos", cos.view(1, C, 1, T).expand(1, C, n_windows, T).contiguous())
        self.register_buffer("sin", sin.view(1, C, 1, T).expand(1, C, n_windows, T).contiguous())

    def forward(self, x):
        y = self.ln1(x)
        q = self.q(y) * self.cos + self.qr(y) * self.sin
        k = self.k(y) * self.cos + self.kr(y) * self.sin
        v = self.v(y)
        outs = []
        for h in range(self.heads):
            sl = slice(h * self.hd, (h + 1) * self.hd)
            s = torch.einsum("nchw,nchq->nqhw", q[:, sl], k[:, sl])          # (1, Tk, nW, Tq)
            outs.append(torch.einsum("nchw,nqhc->nqhw", torch.softmax(s, dim=1), v[:, sl]))
        x = x + self.o(torch.cat(outs, 1))
        return x + self.fc2(gelu(self.fc1(self.ln2(x))))


def window_order(gh: int, gw: int, ws: int) -> torch.Tensor:
    """Spatial index (row-major) of each token in window layout (window-major, row-major inside)."""
    idx = torch.arange(gh * gw).view(gh // ws, ws, gw // ws, ws).permute(0, 2, 1, 3)
    return idx.reshape(-1)


# --------------------------------------------------------------------------- shared pieces
# Token layout for the 256-wide parts (DETR encoder/decoder, text features): (1, C, 1, T),
# heads moved onto H for attention; scores are (1, Tk, heads, Tq), so additive masks and the
# decoder's box bias come in that layout.

def heads_to_h(x, hd):
    return torch.cat(torch.split(x, hd, dim=1), dim=2)


def h_to_heads(x):
    return torch.cat(torch.split(x, 1, dim=2), dim=1)


def attend(q, k, v, mask=None):
    s = torch.einsum("nchw,nchq->nqhw", q, k)
    if mask is not None:
        s = s + mask
    return torch.einsum("nchw,nqhc->nqhw", torch.softmax(s, dim=1), v)


class MHA(nn.Module):
    """Sam3Attention / CLIPAttention in token layout."""

    def __init__(self, att):
        super().__init__()
        q = getattr(att, "q_proj")
        heads = getattr(att, "num_attention_heads", None) or getattr(att, "num_heads")
        self.hd = q.weight.shape[0] // heads
        self.q = conv_from_linear(att.q_proj, scale=self.hd ** -0.5)
        self.k = conv_from_linear(att.k_proj)
        self.v = conv_from_linear(att.v_proj)
        self.o = conv_from_linear(att.o_proj if hasattr(att, "o_proj") else att.out_proj)

    def forward(self, xq, xk, xv, mask=None):
        o = attend(heads_to_h(self.q(xq), self.hd), heads_to_h(self.k(xk), self.hd),
                   heads_to_h(self.v(xv), self.hd), mask)
        return self.o(h_to_heads(o))


class MLP2(nn.Module):
    """fc1 -> act -> fc2 (Sam3MLP / CLIPMLP); act 'gelu' (erf) or 'relu'."""

    def __init__(self, mlp, act):
        super().__init__()
        self.fc1, self.fc2, self.act = conv_from_linear(mlp.fc1), conv_from_linear(mlp.fc2), act

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(gelu(h) if self.act == "gelu" else F.relu(h))


class DecMLP(nn.Module):
    """Sam3DecoderMLP (2 or 3 layers, ReLU between)."""

    def __init__(self, m):
        super().__init__()
        self.l = nn.ModuleList(conv_from_linear(x) for x in (m.layer1, m.layer2, m.layer3) if x is not None)

    def forward(self, x):
        for i, c in enumerate(self.l):
            x = c(x)
            if i < len(self.l) - 1:
                x = F.relu(x)
        return x


def tok(x):
    """(N, C) or (1, N, C) -> (1, C, 1, N)."""
    x = x.reshape(-1, x.shape[-1])
    return x.T.reshape(1, x.shape[1], 1, x.shape[0])


def rows(x):
    """(1, C, 1, N) -> (N, C)."""
    return x.reshape(x.shape[1], -1).T


# --------------------------------------------------------------------------- vision
def space_to_depth(px, p=14):
    """(3, S, S) normalised image -> (1, 3*p*p, S/p, S/p), channel order (c, py, px) like Conv2d's weight."""
    c, S, _ = px.shape
    g = S // p
    return px.reshape(c, g, p, g, p).permute(0, 2, 4, 1, 3).reshape(1, c * p * p, g, g)


class ViTEmbed(nn.Module):
    """Patch embedding as a 1x1 conv on the host's space-to-depth input (window layout), the tiled
    position embedding (each 24x24 window gets the 336-px table) and the pre-layer LayerNorm."""

    def __init__(self, vit, n_windows):
        super().__init__()
        pe = vit.embeddings.patch_embeddings.projection          # Conv2d(3, 1024, 14, 14, bias=False)
        w = pe.weight.detach()
        self.patch = nn.Conv2d(w.shape[1] * w.shape[2] * w.shape[3], w.shape[0], 1, bias=False)
        self.patch.weight.data.copy_(w.reshape(w.shape[0], -1)[:, :, None, None])
        pos = vit.embeddings.position_embeddings.detach()[0]     # (576, 1024), row-major 24x24
        self.register_buffer("pos", pos.T.reshape(1, -1, 1, pos.shape[0]).expand(1, -1, n_windows, -1).contiguous())
        self.ln = LN(vit.layer_norm)

    def forward(self, x):                                         # x (1, 588, nW, 576)
        return self.ln(self.patch(x) + self.pos)


def to_window_layout(t, G, ws):
    """(1, C, G, G) -> (1, C, nW, ws*ws)."""
    C = t.shape[1]
    order = window_order(G, G, ws)
    return t.reshape(1, C, G * G)[:, :, order].reshape(1, C, (G // ws) ** 2, ws * ws)


def from_window_layout(t, G, ws):
    """(1, C, nW, ws*ws) -> (1, C, G, G)."""
    C = t.shape[1]
    order = window_order(G, G, ws)
    out = torch.empty(1, C, G * G, dtype=t.dtype)
    out[:, :, order] = t.reshape(1, C, -1)
    return out.reshape(1, C, G, G)


class Neck(nn.Module):
    """Sam3VisionNeck's 4x, 2x and 1x levels (the image path drops the 0.5x one) on (1, 1024, G, G)."""

    def __init__(self, neck):
        super().__init__()
        self.fpn = nn.ModuleList(neck.fpn_layers[:3])

    def forward(self, x):
        outs = []
        for f in self.fpn:
            h = x
            for layer in f.scale_layers:
                h = gelu(h) if isinstance(layer, nn.GELU) else layer(h)
            outs.append(f.proj2(f.proj1(h)))
        return outs


def sine_pos_2d(H, W, feats=128, temperature=10000, normalize=True, scale=2 * math.pi):
    y = torch.arange(1, H + 1, dtype=torch.float32)[:, None].expand(H, W)
    x = torch.arange(1, W + 1, dtype=torch.float32)[None, :].expand(H, W)
    if normalize:
        y = y / (y[-1:, :] + 1e-6) * scale
        x = x / (x[:, -1:] + 1e-6) * scale
    dim_t = temperature ** (2 * torch.div(torch.arange(feats, dtype=torch.float32), 2, rounding_mode="floor") / feats)
    px, py = x[..., None] / dim_t, y[..., None] / dim_t
    px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=3).flatten(2)
    py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=3).flatten(2)
    return torch.cat((py, px), dim=2).permute(2, 0, 1)[None]      # (1, 2*feats, H, W)


# --------------------------------------------------------------------------- text (CLIP)
class TextLayer(nn.Module):
    def __init__(self, layer, act):
        super().__init__()
        self.ln1, self.ln2 = LN(layer.layer_norm1), LN(layer.layer_norm2)
        self.attn = MHA(layer.self_attn)
        self.mlp = MLP2(layer.mlp, act)

    def forward(self, x, mask):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, mask)
        return x + self.mlp(self.ln2(x))


class TextStage(nn.Module):
    """CLIP text encoder (24 layers, causal) + final LayerNorm + SAM 3's text projection (1024 -> 256).
    in: embeddings (1, 1024, 1, T) and the additive mask (1, T, 1, T) (keys on C)."""

    def __init__(self, hf):
        super().__init__()
        tm = hf.text_encoder.text_model
        act = hf.config.text_config.hidden_act
        self.layers = nn.ModuleList(TextLayer(l, act) for l in tm.encoder.layers)
        self.ln = LN(tm.final_layer_norm)
        self.proj = conv_from_linear(hf.text_projection)

    def forward(self, x, mask):
        for l in self.layers:
            x = l(x, mask)
        return self.proj(self.ln(x))


def text_mask(n_valid, T=32):
    """Additive (1, Tk, 1, Tq) mask: causal and padding (CLIP), -1e4 where masked."""
    k = torch.arange(T)[:, None]
    q = torch.arange(T)[None, :]
    ok = (k <= q) & (k < n_valid)
    return torch.where(ok, 0.0, -1e4).view(1, T, 1, T)


# --------------------------------------------------------------------------- detector
class EncLayer(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = LN(l.layer_norm1), LN(l.layer_norm2), LN(l.layer_norm3)
        self.sa, self.ca = MHA(l.self_attn), MHA(l.cross_attn)
        self.mlp = MLP2(l.mlp, "relu")

    def forward(self, x, pos, text, tmask):
        y = self.ln1(x)
        x = x + self.sa(y + pos, y + pos, y)
        y = self.ln2(x)
        x = x + self.ca(y, text, text, tmask)
        return x + self.mlp(self.ln3(x))


class DecLayer(nn.Module):
    """Sam3DetrDecoderLayer (post-norm) on (1, 256, 1, 1+Q): presence token first."""

    def __init__(self, l):
        super().__init__()
        self.sa, self.ta, self.va = MHA(l.self_attn), MHA(l.text_cross_attn), MHA(l.vision_cross_attn)
        self.n1, self.n2, self.n3 = LN(l.self_attn_layer_norm), LN(l.text_cross_attn_layer_norm), LN(l.vision_cross_attn_layer_norm)
        self.mlp, self.n4 = MLP2(l.mlp, "relu"), LN(l.mlp_layer_norm)

    def forward(self, h, qpos, text, tmask, vis, vis_k, rpb):
        """qpos (1,256,1,1+Q) with zeros for the presence token; vis_k = vision + pos;
        rpb (1, HW, heads, 1+Q) additive box bias (zeros for the presence token)."""
        h = self.n1(h + self.sa(h + qpos, h + qpos, h))
        h = self.n2(h + self.ta(h + qpos, text, text, tmask))
        h = self.n3(h + self.va(h + qpos, vis_k, vis, rpb))
        return self.n4(h + self.mlp(h))


def box_rpb(dec, ref_cxcywh, H, W):
    """Box relative position bias, as HF's _get_rpb_matrix, in score layout (1, H*W, heads, 1+Q)."""
    b = torch.cat([ref_cxcywh[:, :2] - ref_cxcywh[:, 2:] / 2, ref_cxcywh[:, :2] + ref_cxcywh[:, 2:] / 2], 1)  # xyxy
    ch = torch.arange(H, dtype=torch.float32) / H
    cw = torch.arange(W, dtype=torch.float32) / W
    dy = ch.view(1, -1, 1) - b[:, None, 1::2]                    # (Q, H, 2)
    dx = cw.view(1, -1, 1) - b[:, None, 0::2]                    # (Q, W, 2)
    lg = lambda d: torch.sign(d * 8) * torch.log2((d * 8).abs() + 1.0) / math.log2(8)   # noqa: E731
    by = dec.box_rpb_embed_y(lg(dy))                              # (Q, H, heads)
    bx = dec.box_rpb_embed_x(lg(dx))                              # (Q, W, heads)
    rpb = (by[:, :, None, :] + bx[:, None, :, :]).reshape(b.shape[0], H * W, -1)   # (Q, HW, heads)
    rpb = torch.cat([torch.zeros(1, H * W, rpb.shape[2]), rpb], 0)                 # presence row
    return rpb.permute(1, 2, 0)[None]                             # (1, HW, heads, 1+Q)


def sine_boxes(dec, ref):
    return dec.position_encoding.encode_boxes(ref[None])[0]       # (Q, 512)


class Pipeline(nn.Module):
    """SAM 3 image path, MLA form, with the host steps in between (fp32 reference).

    call(px (3, S, S) normalised, input_ids (T,), n_valid) -> dict of logits (Q,), boxes (Q,4 cxcywh),
    presence (scalar logit), masks (Q, 4G, 4G)."""

    def __init__(self, hf, S=672):
        super().__init__()
        self.hf, self.S = hf, S
        vc = hf.config.vision_config.backbone_config
        self.ws, self.G = vc.window_size, S // vc.patch_size
        nW = (self.G // self.ws) ** 2
        vit = hf.vision_encoder.backbone
        self.order = window_order(self.G, self.G, self.ws)
        self.embed = ViTEmbed(vit, nW)
        self.vit = nn.ModuleList(
            ViTLayerW(l, 1, (self.G, self.G), token_order=self.order) if l.window_size == 0
            else ViTLayerW(l, nW, (self.ws, self.ws)) for l in vit.layers)
        self.neck = Neck(hf.vision_encoder.neck)
        self.text = TextStage(hf)
        self.enc = nn.ModuleList(EncLayer(l) for l in hf.detr_encoder.layers)
        self.dec = nn.ModuleList(DecLayer(l) for l in hf.detr_decoder.layers)
        d = hf.detr_decoder
        self.out_ln, self.pres_ln = LN(d.output_layer_norm), LN(d.presence_layer_norm)
        self.box_head, self.pres_head, self.ref_head = DecMLP(d.box_head), DecMLP(d.presence_head), DecMLP(d.ref_point_head)
        md = hf.mask_decoder
        self.pca, self.pca_ln = MHA(md.prompt_cross_attn), LN(md.prompt_cross_attn_norm)
        self.mask_embed = nn.ModuleList(conv_from_linear(l) for l in md.mask_embedder.layers)
        self.inst = md.instance_projection
        sc = hf.dot_product_scoring
        self.sc_mlp, self.sc_ln = DecMLP(sc.text_mlp), LN(sc.text_mlp_out_norm)
        self.sc_text, self.sc_query = conv_from_linear(sc.text_proj), conv_from_linear(sc.query_proj)
        self.sc_scale = sc.scale

    def text_features(self, input_ids, n_valid):
        e = self.hf.text_encoder.text_model.embeddings
        T = input_ids.shape[0]
        x = e.token_embedding(input_ids) + e.position_embedding.weight[:T]
        return self.text(tok(x), text_mask(n_valid, T))           # (1, 256, 1, T)

    def vision(self, px):
        G, ws = self.G, self.ws
        x = to_window_layout(space_to_depth(px), G, ws)            # host: patches in window layout
        x = self.embed(x)
        for l in self.vit:
            if l.cos.shape[2] == 1:                                # global: same bytes as (1, C, 1, G*G)
                x = l(x.reshape(1, x.shape[1], 1, -1)).reshape(x.shape)
            else:
                x = l(x)
        sp = from_window_layout(x, G, ws)                          # host gather to image order
        return self.neck(sp)                                       # [4G, 2G, G] x 256

    def forward(self, px, input_ids, n_valid):
        f4, f2, f1 = self.vision(px)
        text = self.text_features(input_ids, n_valid)              # (1, 256, 1, T)
        T = text.shape[3]
        tmask = torch.where(torch.arange(T) < n_valid, 0.0, -1e4).view(1, T, 1, 1)
        G = self.G
        pos = tok(sine_pos_2d(G, G).reshape(256, -1).T)
        x = tok(f1.reshape(256, -1).T)
        for l in self.enc:
            x = l(x, pos, text, tmask)
        enc = x
        d = self.hf.detr_decoder
        Q = d.config.num_queries
        h = torch.cat([tok(d.presence_token.weight), tok(d.query_embed.weight)], 3)
        ref = d.reference_points.weight.sigmoid()
        vis_k = enc + pos
        boxes = None
        for l in self.dec:                                         # host: sine + ref head + box bias
            qpos = self.ref_head(tok(sine_boxes(d, ref)))
            qpos = torch.cat([torch.zeros(1, 256, 1, 1), qpos], 3)
            h = l(h, qpos, text, tmask, enc, vis_k, box_rpb(d, ref, G, G))
            q = self.out_ln(h[:, :, :, 1:])
            delta = rows(self.box_head(q))
            boxes = (torch.logit(ref, eps=1e-3) + delta).sigmoid()  # inverse_sigmoid eps 1e-3 as HF
            ref = boxes
        pres = rows(self.pres_head(self.pres_ln(h[:, :, :, :1])))[0, 0].clamp(-10, 10)
        # scoring: text MLP + residual + LN, mean over valid tokens, dot with projected queries
        t = self.sc_ln(text + self.sc_mlp(text))
        pooled = t[:, :, :, :n_valid].mean(3, keepdim=True)
        logits = (rows(self.sc_query(q)) @ rows(self.sc_text(pooled)).T)[:, 0] * self.sc_scale
        logits = logits.clamp(-12, 12)
        # masks: encoder output attends to the prompt, pixel decoder, then query . pixel
        e2 = enc + self.pca(self.pca_ln(enc), text, text, tmask)
        md = self.hf.mask_decoder.pixel_decoder
        p = rows(e2).T.reshape(1, 256, G, G)
        for i, skip in enumerate((f2, f4)):
            p = F.interpolate(p, size=skip.shape[-2:], mode="nearest") + skip
            p = F.relu(md.norms[i](md.conv_layers[i](p)))
        inst = self.inst(p)                                        # (1, 256, 4G, 4G)
        me = q
        for i, c in enumerate(self.mask_embed):
            me = c(me)
            if i < len(self.mask_embed) - 1:
                me = F.relu(me)
        masks = torch.einsum("qc,chw->qhw", rows(me), inst[0])
        return {"logits": logits, "boxes": boxes, "presence": pres, "masks": masks}
