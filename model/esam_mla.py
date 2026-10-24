"""EfficientSAM (yformer/EfficientSAM, Apache-2.0; code vendored in third_party/efficient_sam) for the MLA.

Grounded EfficientSAM: Grounding DINO finds boxes from text, EfficientSAM turns each box into a mask.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))
from efficient_sam.efficient_sam import EfficientSam, PromptEncoder  # noqa: E402
from efficient_sam.efficient_sam_decoder import MaskDecoder  # noqa: E402
from efficient_sam.efficient_sam_encoder import ImageEncoderViT  # noqa: E402
from efficient_sam.two_way_transformer import TwoWayTransformer  # noqa: E402

VARIANTS = {"ti": (192, 3, "efficient_sam_vitt.pt"), "s": (384, 6, "efficient_sam_vits.pt")}
WEIGHTS = ROOT / "build" / "weights" / "efficientsam"


def load_esam(variant: str = "ti", img_size: int = 1024) -> EfficientSam:
    """EfficientSAM as build_efficient_sam makes it, but at any input size (the 14x14 pretrain position
    embedding is interpolated to the token grid, as the encoder does itself). Weights are loaded with
    weights_only=True."""
    dim, heads, ckpt = VARIANTS[variant]
    enc = ImageEncoderViT(img_size=img_size, patch_size=16, in_chans=3, patch_embed_dim=dim,
                          normalization_type="layer_norm", depth=12, num_heads=heads, mlp_ratio=4.0,
                          neck_dims=[256, 256], act_layer=nn.GELU)
    emb = enc.image_embedding_size
    sam = EfficientSam(
        image_encoder=enc,
        prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(emb, emb), input_image_size=(img_size, img_size)),
        decoder_max_num_input_points=6,
        mask_decoder=MaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048, activation=nn.GELU,
                                          normalize_before_activation=False),
            num_multimask_outputs=3, activation=nn.GELU, normalization_type="layer_norm",
            normalize_before_activation=False, iou_head_depth=2, iou_head_hidden_dim=256, upscaling_layer_dims=[64, 32]),
        pixel_mean=[0.485, 0.456, 0.406], pixel_std=[0.229, 0.224, 0.225])
    sd = torch.load(WEIGHTS / ckpt, map_location="cpu", weights_only=True)
    sam.load_state_dict(sd.get("model", sd))
    return sam.eval()


def box_masks(sam: EfficientSam, img01: torch.Tensor, boxes_xyxy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """img01 (3, H, W) in [0, 1]; boxes (Q, 4) in pixels -> best-of-3 mask logits (Q, H, W) and their IoU scores."""
    pts = boxes_xyxy.view(1, -1, 2, 2)
    lab = torch.tensor([2, 3], device=pts.device).view(1, 1, 2).expand(1, pts.shape[1], 2)
    logits, iou = sam(img01[None], pts, lab)                       # (1, Q, 3, H, W), (1, Q, 3)
    best = iou[0].argmax(1)
    q = torch.arange(pts.shape[1])
    return logits[0, q, best], iou[0, q, best]


# --------------------------------------------------------------------------- MLA form
sys.path.insert(0, str(ROOT / "model"))
import sam3_mla as S3  # noqa: E402  (conv_from_linear, LN, gelu, token layout helpers)


class BlockMLA(nn.Module):
    """One EfficientSAM encoder block (pre-norm ViT, global attention) on tokens (1, C, 1, N)."""

    def __init__(self, blk):
        super().__init__()
        a = blk.attn
        C = a.qkv.in_features
        self.heads = a.num_heads
        self.hd = C // self.heads
        w, b = a.qkv.weight.detach(), a.qkv.bias.detach()
        lin = lambda i: nn.Linear(C, C).requires_grad_(False)  # noqa: E731
        q, k, v = lin(0), lin(1), lin(2)
        for j, l in enumerate((q, k, v)):
            l.weight.data.copy_(w[j * C:(j + 1) * C])
            l.bias.data.copy_(b[j * C:(j + 1) * C])
        self.q = S3.conv_from_linear(q, scale=a.scale)
        self.k, self.v = S3.conv_from_linear(k), S3.conv_from_linear(v)
        self.o = S3.conv_from_linear(a.proj)
        self.ln1, self.ln2 = S3.LN(blk.norm1), S3.LN(blk.norm2)
        self.fc1, self.fc2 = S3.conv_from_linear(blk.mlp.fc1), S3.conv_from_linear(blk.mlp.fc2)

    def forward(self, x):
        y = self.ln1(x)
        q, k, v = self.q(y), self.k(y), self.v(y)
        outs = []
        for h in range(self.heads):
            sl = slice(h * self.hd, (h + 1) * self.hd)
            s = torch.einsum("nchw,nchq->nqhw", q[:, sl], k[:, sl])
            outs.append(torch.einsum("nchw,nqhc->nqhw", torch.softmax(s, dim=1), v[:, sl]))
        x = x + self.o(torch.cat(outs, 1))
        return x + self.fc2(S3.gelu(self.fc1(self.ln2(x))))


PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def patch_tokens(img01: torch.Tensor, p: int = 16) -> torch.Tensor:
    """(3, S, S) image in [0, 1] (already squashed to S x S) -> (1, 3*p*p, 1, (S/p)^2) patch tokens,
    channel order (c, py, px) like the patch conv's weight. The normalisation is folded into the ELF."""
    return S3.space_to_depth(img01, p).flatten(2).unsqueeze(2)


class EncoderMLA(nn.Module):
    """Patch embedding (1x1 conv on host patch tokens, ImageNet normalisation folded in) + position
    embedding + 12 global-attention blocks + the neck's 1x1 conv and LayerNorm2d.
    in (1, 768, 1, G*G) -> out (1, 256, 1, G*G); the neck's 3x3 conv runs in NeckMLA on the same bytes."""

    def __init__(self, enc):
        super().__init__()
        from efficient_sam.efficient_sam_encoder import get_abs_pos
        w, b = enc.patch_embed.proj.weight.detach(), enc.patch_embed.proj.bias.detach()   # (C, 3, p, p)
        std = torch.tensor(PIXEL_STD).view(1, 3, 1, 1)
        mean = torch.tensor(PIXEL_MEAN).view(1, 3, 1, 1)
        wn = w / std
        self.patch = nn.Conv2d(wn[0].numel(), w.shape[0], 1)
        self.patch.weight.data.copy_(wn.reshape(w.shape[0], -1)[:, :, None, None])
        self.patch.bias.data.copy_(b - (wn * mean).sum((1, 2, 3)))
        G = enc.img_size // 16
        pos = get_abs_pos(enc.pos_embed.detach(), enc.pretrain_use_cls_token, [G, G])     # (1, G, G, C)
        self.register_buffer("pos", pos.reshape(1, G * G, -1).permute(0, 2, 1).unsqueeze(2).contiguous())
        self.blocks = nn.ModuleList(BlockMLA(b_) for b_ in enc.blocks)
        self.neck_conv = enc.neck[0]
        self.neck_ln = S3.LN(enc.neck[1])

    def forward(self, x):
        x = self.patch(x) + self.pos
        for b_ in self.blocks:
            x = b_(x)
        return self.neck_ln(self.neck_conv(x))


class NeckMLA(nn.Module):
    """The neck's 3x3 conv + LayerNorm2d on (1, 256, G, G)."""

    def __init__(self, enc):
        super().__init__()
        self.conv, self.ln = enc.neck[2], S3.LN(enc.neck[3])

    def forward(self, x):
        return self.ln(self.conv(x))


TOK = 16      # decoder tokens per box: 5 output + 2 corners + 4 padding points (+ 5 masked)


def _cat_h(x, n):
    return x if n == 1 else torch.cat([x] * n, 2)


class AttnMLA(nn.Module):
    """AttentionForTwoWayAttentionBlock on (1, C, H, W) tokens (H = boxes): one einsum pair per head on
    channel slices. (Stacking the heads on H, as the SAM 3 port does, trips the quantizer's executor here.)"""

    def __init__(self, a):
        super().__init__()
        self.heads = a.num_heads
        self.hd = a.internal_dim // a.num_heads
        self.q = S3.conv_from_linear(a.q_proj, scale=a.inv_sqrt_c_per_head)
        self.k, self.v, self.o = (S3.conv_from_linear(l) for l in (a.k_proj, a.v_proj, a.out_proj))

    def forward(self, q, k, v, qp=None, kp=None, vp=None, mask=None, tile=1):
        """q/k/v: (1, C, H, W) inputs, or already projected (qp/kp/vp, possibly with H = 1 and then
        repeated `tile` times on H); mask (1, Tk, 1, 1) additive."""
        qp = self.q(q) if qp is None else qp
        kp = self.k(k) if kp is None else kp
        vp = self.v(v) if vp is None else vp
        outs = []
        for h in range(self.heads):
            sl = slice(h * self.hd, (h + 1) * self.hd)
            qh, kh, vh = qp[:, sl], kp[:, sl], vp[:, sl]
            s_ = torch.einsum("nchw,nchq->nqhw", _cat_h(qh, tile if qh.shape[2] == 1 else 1),
                              _cat_h(kh, tile if kh.shape[2] == 1 else 1))
            if mask is not None:
                s_ = s_ + mask
            outs.append(torch.einsum("nchw,nqhc->nqhw", torch.softmax(s_, dim=1),
                                     _cat_h(vh, tile if vh.shape[2] == 1 else 1)))
        return self.o(torch.cat(outs, 1))


class DecoderMLA(nn.Module):
    """EfficientSAM mask decoder (two-way transformer + upscaling + hypernetwork + IoU head) for B boxes.

    in  img (1, 256, 1, G*G)   image embedding (NeckMLA output, same bytes)
        tok (1, 256, B, 16)    per box: IoU + 4 mask tokens, the two corner prompt embeddings, the
                               4 padding-point embeddings and 5 zero tokens (host: box_tokens). The MLA
                               mis-handles a softmax over 11 channels, so the tokens are padded to 16
                               and the 5 extra keys are masked out.
    out (1, 48, B, G*G + 1)    3 multimask logits at 4G x 4G as 16 sub-pixel groups x 3 masks on C,
                               the last column holds the 3 IoU scores on channels 0..2
    """

    def __init__(self, sam, B: int):
        super().__init__()
        md, pe_enc = sam.mask_decoder, sam.prompt_encoder
        tr = md.transformer
        self.B = B
        G = pe_enc.image_embedding_size[0]
        self.G = G
        dense = pe_enc.get_dense_pe().detach()                                   # (1, 256, G, G)
        self.register_buffer("pe", dense.reshape(1, 256, 1, G * G).contiguous())
        out_tok = torch.cat([md.iou_token.weight, md.mask_tokens.weight], 0).detach()   # (5, 256)
        S_ = sam.image_encoder.img_size
        inv = torch.full((1, 4, 2), -1.0)
        inv_emb = pe_enc(inv, torch.full((1, 4), -1.0)).detach()[0]               # (4, 256) padded points
        self.register_buffer("tok_head", out_tok.clone())                       # host constants
        self.register_buffer("tok_tail", inv_emb.clone())
        self.register_buffer("kmask", torch.cat([torch.zeros(11), torch.full((TOK - 11,), -1e4)]).view(1, TOK, 1, 1))
        self.S = S_
        self.layers = nn.ModuleList()
        for l in tr.layers:
            m = nn.Module()
            m.skip_pe = l.skip_first_layer_pe
            m.self_attn = AttnMLA(l.self_attn)
            m.t2i, m.i2t = AttnMLA(l.cross_attn_token_to_image), AttnMLA(l.cross_attn_image_to_token)
            m.n1, m.n2, m.n3, m.n4 = (S3.LN(n) for n in (l.norm1, l.norm2, l.norm3, l.norm4))
            m.fc1 = S3.conv_from_linear(l.mlp.layers[0][0])
            m.fc2 = S3.conv_from_linear(l.mlp.fc)
            self.layers.append(m)
        self.final = AttnMLA(tr.final_attn_token_to_image)
        self.nf = S3.LN(tr.norm_final_attn)
        # upscaling: ConvTranspose(k=2, s=2) has no overlap, so it is a 1x1 conv to (a, b, co) channels
        ct1, gn, _ = md.final_output_upscaling_layers[0]
        ct2 = md.final_output_upscaling_layers[1][0]
        self.up1 = self._ct(ct1)
        self.gn_eps = gn.eps
        self.register_buffer("gn_w", gn.weight.detach().repeat(4).view(1, -1, 1, 1).clone())
        self.register_buffer("gn_b", gn.bias.detach().repeat(4).view(1, -1, 1, 1).clone())
        C4 = 4 * ct1.out_channels
        self.cavg = nn.Conv2d(C4, C4, 1, bias=False)
        self.cavg.weight.data.fill_(1.0 / C4)
        self.up2 = nn.ModuleList(self._ct(ct2) for _ in range(4))             # one per (a, b) group
        self.c1 = ct1.out_channels
        self.hyper = nn.ModuleList(self._mlp(md.output_hypernetworks_mlps[i]) for i in (1, 2, 3))
        self.iou = self._mlp(md.iou_prediction_head, pad_out=48, keep=(1, 2, 3))

    def _row_mean(self, x):
        """Mean over C and W of each H row (box) -> (1, C, B, 1). The MLA has no per-row reduction over
        (C, W); a one-row slice reduced over (H, W) plus a 1/C conv is the GroupNorm pattern it runs."""
        return torch.cat([self.cavg(x[:, :, b:b + 1].mean((2, 3), keepdim=True)) for b in range(x.shape[2])], 2)

    @staticmethod
    def _ct(ct):
        w = ct.weight.detach()                                                   # (cin, cout, 2, 2)
        cin, cout = w.shape[:2]
        c = nn.Conv2d(cin, 4 * cout, 1)
        c.weight.data.copy_(w.permute(2, 3, 1, 0).reshape(4 * cout, cin)[:, :, None, None])
        c.bias.data.copy_(ct.bias.detach().repeat(4))
        return c

    @staticmethod
    def _mlp(m, pad_out=None, keep=None):
        convs = [S3.conv_from_linear(s[0]) for s in m.layers]
        fw, fb = m.fc.weight.detach(), m.fc.bias.detach()
        if keep is not None:
            fw, fb = fw[list(keep)], fb[list(keep)]
        if pad_out is not None:
            fw = torch.cat([fw, fw.new_zeros(pad_out - fw.shape[0], fw.shape[1])])
            fb = torch.cat([fb, fb.new_zeros(pad_out - fb.shape[0])])
        convs.append(S3.conv_from_linear(m.fc, weight=fw, bias=fb))
        return nn.ModuleList(convs)

    @staticmethod
    def _run_mlp(convs, x):
        for c in convs[:-1]:
            x = S3.gelu(c(x))
        return convs[-1](x)

    def forward(self, img, tok):
        B = self.B
        pe = self.pe
        q, qpe = tok, tok
        keys1 = img                                                              # box-independent, H = 1
        keys = None
        for li, m in enumerate(self.layers):
            # self-attention of the tokens
            if not m.skip_pe:                                                    # this fork adds the PE to the residual too
                q = q + qpe
            q = m.n1(q + m.self_attn(q, q, q, mask=self.kmask))
            # tokens -> image
            if keys is None:                                                     # first layer: keys same for all boxes
                q = m.n2(q + m.t2i(q + qpe, None, None, kp=m.t2i.k(keys1 + pe), vp=m.t2i.v(keys1), tile=B))
            else:
                q = m.n2(q + m.t2i(q + qpe, keys + pe, keys))
            q = m.n3(q + m.fc2(S3.gelu(m.fc1(q))))
            # image -> tokens
            if keys is None:
                att = m.i2t(None, q + qpe, q, qp=m.i2t.q(keys1 + pe), mask=self.kmask, tile=B)
                keys = m.n4(_cat_h(keys1, B) + att)
            else:
                keys = m.n4(keys + m.i2t(keys + pe, q + qpe, q, mask=self.kmask))
        q = self.nf(q + self.final(q + qpe, keys + pe, keys))
        # upscaling (sub-pixel channels): up1 -> (a, b, 64); GroupNorm(1, 64) per box over C and W
        # GroupNorm(1, 64) per box = mean over (C, W) of each H row (see _row_mean)
        u = self.up1(keys)
        mu = self._row_mean(u)
        d = u - mu
        var = self._row_mean(d * d)
        u = S3.gelu(d * torch.pow(var + self.gn_eps, -0.5) * self.gn_w + self.gn_b)
        c1 = self.c1
        u2 = [S3.gelu(self.up2[g](u[:, g * c1:(g + 1) * c1])) for g in range(4)]   # each (a', b', 32)
        hyp = torch.cat([self._run_mlp(self.hyper[i], q[:, :, :, 2 + i:3 + i]) for i in range(3)], 3)  # (1,32,B,3)
        c2 = u2[0].shape[1] // 4
        masks = []
        for g in range(4):
            for s in range(4):
                masks.append(torch.einsum("nchw,nchq->nqhw", u2[g][:, s * c2:(s + 1) * c2], hyp))
        masks = torch.cat(masks, 1)                                              # (1, 48, B, G*G)
        iou = self._run_mlp(self.iou, q[:, :, :, 0:1])                           # (1, 48, B, 1)
        return torch.cat([masks, iou], 3)


@torch.no_grad()
def box_tokens(dec: "DecoderMLA", sam, boxes_xyxy: torch.Tensor, in_w: int, in_h: int) -> torch.Tensor:
    """Decoder token input: (B, 4) boxes in input-image pixels -> (1, 256, B, TOK)."""
    corners = box_prompt(sam, boxes_xyxy, in_w, in_h)                            # (1, 256, B, 2)
    B = corners.shape[2]
    head = S3.tok(dec.tok_head).expand(1, 256, B, 5)
    tail = S3.tok(dec.tok_tail).expand(1, 256, B, 4)
    pad = corners.new_zeros(1, 256, B, TOK - 11)
    return torch.cat([head, corners, tail, pad], 3).contiguous()


@torch.no_grad()
def box_prompt(sam, boxes_xyxy: torch.Tensor, in_w: int, in_h: int) -> torch.Tensor:
    """The two corner prompt embeddings of (Q, 4) boxes in input-image pixels -> (1, 256, Q, 2)."""
    S_ = sam.image_encoder.img_size
    sc = torch.tensor([S_ / in_w, S_ / in_h])
    pts = boxes_xyxy.view(1, -1, 2)[0].view(-1, 2, 2) * sc                     # (Q, 2, 2)
    lab = torch.tensor([2.0, 3.0]).expand(pts.shape[0], 2)
    emb = sam.prompt_encoder(pts, lab)                                           # (Q, 2, 256)
    return emb.permute(2, 0, 1).unsqueeze(0)


def unshuffle_masks(out: torch.Tensor, G: int) -> tuple[torch.Tensor, torch.Tensor]:
    """DecoderMLA output (1, 48, B, G*G + 1) -> masks (B, 3, 4G, 4G) and IoU (B, 3)."""
    B = out.shape[2]
    iou = out[0, :3, :, -1].T
    m = out[0, :, :, :G * G].reshape(2, 2, 2, 2, 3, B, G, G)                     # a, b, a', b', m, B, i, j
    m = m.permute(5, 4, 6, 0, 2, 7, 1, 3).reshape(B, 3, 4 * G, 4 * G)            # y = 4i + 2a + a'
    return m, iou
