<p align="center"><img src="webapp/web/assets/neat-dino.svg" width="96" alt=""></p>

<h1 align="center">Neat DINO Studio</h1>

<p align="center">Open-vocabulary object detection with <a href="https://huggingface.co/IDEA-Research/grounding-dino-base">Grounding DINO</a> on the SiMa.ai Modalix MLA.<br>
Type or say what to find — <code>person . red car . dog with a frisbee .</code> — and get boxes, from a photo or a live webcam.</p>

<p align="center"><img src="docs/studio-detect.gif" width="900" alt="Click a sample, boxes appear; lower the Box Score to see weaker matches; change the prompt and only the text-dependent part reruns"></p>

<table>
<tr>
<td width="50%"><img src="docs/studio-voice.gif" alt="Click Speak and say the request; Whisper and an LLM turn it into a prompt and the detection runs"></td>
<td width="50%"><img src="docs/studio-live.gif" alt="Live webcam mode with boxes on every frame and the frame rate in the metric strip"></td>
</tr>
<tr>
<td><b>Voice.</b> "Could you find the cats and the remote controls?" becomes <code>cat . remote control .</code>, Whisper and an LLM on the MLA.</td>
<td><b>Live.</b> The webcam with two frames in flight; time per image and FPS on top.</td>
</tr>
</table>

<sub>Recorded on a Modalix DevKit with the Fast (576 px) model. Sample photos: COCO val2017, CC BY 2.0 (see NOTICE).</sub>

**Weights:** [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base), the **Base** size of Grounding DINO: **232.3 M parameters** in total (Swin-B image backbone 86.9 M, BERT-base text encoder 108.9 M, feature enhancer, deformable decoder and heads 36.5 M), unchanged from the original and run in bf16. Apache-2.0.

## What it does

- **Any phrase, not a fixed class list.** Grounding DINO matches boxes to the words you type; each phrase gets its own color.
- **Photo or webcam.** Upload, paste or drop a photo, try one of the six samples, or open the webcam: take a full-resolution
  snapshot or go live. Changing only the prompt reruns on the same image without re-running the backbone.
- **Voice.** Click *Speak* and say what you want ("can you find the red cars and any dogs"): it stops listening when you
  stop talking, Whisper transcribes it and a small LLM turns it into `red car . dog .`, both on the MLA through LLiMa (~0.5 s).
- **Pipelined Live mode.** Two webcam frames are in flight: one runs on the MLA while the CPU samples
  the other, for ~27 % more frames per second than one at a time.
- **See where the time goes.** The strip above the image shows the time per image and in Live mode the
  frame rate, split into text, backbone, encoder and decoder (hover a part for its MLA / CPU share).
- **Thresholds on the fly.** Box and phrase score sliders filter on the page, without another run.
- **Model manager.** Download, verify, load and delete models from Hugging Face in Settings → Models. One detector runs on the MLA at a time.
  The voice models download from Settings → Voice.
- **Your camera and microphone.** Settings → Devices picks them, with a live preview and a level meter.
- **Two sizes of Base, and Tiny.** *Fast* (576 px) for the webcam, *Accurate* (768 px) for small objects, both
  Grounding DINO Base. Grounding DINO **Tiny** (576 px) is also supported: a bit quicker and a smaller bundle. All bf16.
- **SAM 3, with masks.** Meta's [SAM 3](https://ai.meta.com/research/sam3/) (Segment Anything with Concepts) runs as a third
  model family: the same prompts, voice and Live mode, and every match gets a mask as well as a box (see below).
- **OmDet-Turbo, the fast one.** [OmDet-Turbo Tiny](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf)
  (Apache-2.0) is a real-time open-vocabulary detector: ~145 ms per image on the board, half of Grounding DINO Tiny,
  and a little more accurate on our COCO subset (see below).
- **Masks for Grounding DINO too (Grounded EfficientSAM).** Turn on *Show Masks* and EfficientSAM (Ti or S) turns each
  box above the thresholds into a mask, on the MLA next to the detector (see below).

## Install (Modalix DevKit)

```bash
cd /media/nvme
curl -fsSL https://raw.githubusercontent.com/dotimothy/neat-dino-studio/main/install.sh | bash
cd neat-dino-studio && ./setup.sh          # builds the runtime, fetches the Fast model, HTTPS cert, alias
neat-dino                                  # (new shell) then open https://<board-ip>:8092/
```

`setup.sh --fetch accurate|tiny|all` fetches more models; `webapp/run.sh --daemon` runs it in the
background and `webapp/run.sh --stop` stops it (freeing the MLA). `neat-dino update` updates in place.
The page also works over plain HTTP, but browsers only allow the webcam on HTTPS; the studio
serves both on the same port with a self-signed certificate (accept the warning once).

Requirements: the NEAT SDK on the DevKit (SimaLMM runtime `sima-lmm-dev`, Python 3 with numpy,
Pillow, Flask — the `~/pyneat` venv has them), cmake and g++. About 700 MB of free MLA memory:
stop other MLA apps first.

## How it runs on the MLA

Grounding DINO Base is 232.3 M parameters: a Swin-B backbone, a BERT-base text encoder and a
6+6-layer deformable transformer. Two parts of it do not map onto the MLA: multi-scale
deformable attention samples at data-dependent points (the MLA's GridSample handles maps up to
32 px wide, at ~0.3 µs a sample), and the two-stage query selection needs a top-900 over all
image tokens. So the network is cut into MLA ELFs, and the board's Arm cores do the rest:

| On the MLA (bf16 ELFs) | On the A65 cores |
|---|---|
| Text encoder (BERT + projection), once per prompt | Tokenizing, phrase masks, text position embedding |
| Swin-B as 9 ELFs (fused stages, window shifts inside) | Swin patch merging and the return to image order (6 gathers) |
| 7 encoder ELFs (fusion, text enhancer, deformable projections) | Deformable-attention sampling (NEON, 16 threads) |
| Decoder value maps + 12 decoder ELFs | Top-900 selection, reference-box updates, sine embeddings |

`runtime/` (`dino_rt`) runs the ELFs with SimaLMM's `MLAModelWithBuffer` and reads and writes
their bf16 buffers in place, so there are no copies through the EV74 between stages. The web
app (`webapp/server.py`) talks to it as a child process; unloading a model stops it cleanly.
`dino_rt` has two *slots*, each with its own device buffers bound to the same loaded ELFs
(SimaLMM shares an ELF loaded twice), so a second frame can start while the first is still
running; the MLA goes to the older frame first, and the CPU threads are split between the slots
that are sampling. Replies come back in request order.

| | Base · Fast · 576 px | Base · Accurate · 768 px | Tiny · 576 px |
|---|---|---|---|
| Parameters | 232.3 M | 232.3 M | 172.2 M |
| Per new image (board) | 0.33 s | 0.54 s | 0.29 s |
| Prompt change, same image | 0.25 s | 0.41 s | 0.24 s |
| Live (two frames in flight), runtime | 3.9 FPS | 2.3 FPS | 4.3 FPS |
| COCO AP, board (500-image val subset) | 42.0 | 46.4 | 41.8 |
| COCO AP, FP32 PyTorch, same inputs | 41.7 | 45.8 | 41.4 |
| Bundle | 1.8 GB | 1.5 GB | 1.0 GB |

The bf16 builds on the MLA match the FP32 models. Tiny ([IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny):
Swin-T backbone, its own BERT weights) is only ~12 % faster than Base at 576: its backbone takes half the time, but most of
the time goes to the encoder's deformable sampling on the CPU, which is the same for both. Its bundles
(`dino-tiny-576-bf16` + `dino-tiny-text-t64-bf16`) download from Settings → Models or with `setup.sh --fetch tiny`.
See `report.md` for details.

## Build it yourself

Everything that produces the models is here:

```
model/dino_mla.py        Grounding DINO rebuilt in MLA-friendly ops, stage by stage (checked against HF)
model/backbone_plan.py   the window-layout Swin-B ELFs
model/export_onnx.py     ONNX per stage, with real inputs/outputs for board checks
compile/                 SiMa Model Compiler driver (bf16, MLA tessellation), probes, board checks
tools/make_bundle.py     ELFs + tables + constants -> a runtime bundle
tools/publish_hub.py     bundles -> Hugging Face (models.json)
runtime/                 dino_rt (C++, SimaLMM)
webapp/                  the studio (Flask + plain JS, no build step)
```

```bash
.venv/bin/python model/check_mirror.py --size 768           # the rebuild vs transformers, FP32
.venv/bin/python model/export_onnx.py --size 576 --all      # ONNX for every stage
compile/build_stages.sh 576                                 # compile them all (NEAT SDK container)
.venv/bin/python tools/make_bundle.py --size 576 --text     # bundles in build/bundles/
# Grounding DINO Tiny: the same steps with --model tiny (MODEL=tiny for build_stages.sh)
BOARD=sima@<board-ip> ./deploy.sh --models --build          # to a DevKit
```

## Voice (Whisper + LLiMa)

Settings → Voice → *Turn On* starts a sidecar (`voice/voice_server.py`): pyneat's GenAIServer with
Whisper Small (a16w8) and an LLM, on `127.0.0.1:9987`. Both run on the MLA next to the detector.
Click *Speak* and talk: recording stops after a short pause (0.8 s by default, adjustable in
Settings → Voice), or after 5 s if nothing was said. Holding *Speak* or the space bar records until
you let go instead.

1. The browser records WebM/Opus and posts it to `/api/voice/prompt`.
2. Whisper transcribes it (~0.3 s): "Can you find the red cars and any dogs?"
3. The LLM turns the request into phrases (~0.2 s): `red car . dog .`, also for indirect requests
   ("is there anything I could sit on" → `chair . sofa . bench . stool .`). Without an LLM,
   rules strip the request words and split on "and" and commas.
4. The prompt is filled in and the detection runs. In Live mode the next frame uses it.

The models are LLiMa folders in `/media/nvme/llima/models` (`--llima-models`), the ones NEAT
GenAI Studio downloads: `whisper-small-a16w8` and `Gemma-4-E2B-it-TextOnly-GPTQ-a16w4` (default,
best at indirect requests) or `LFM2.5-230M-a16w4` (~40 ms, literal requests only). If they are not
on the board, Settings → Voice downloads them from Hugging Face (`simaai/…`) into that folder, so
GenAI Studio can use them too; a stopped download resumes. Whisper alone is enough: without an LLM,
rules turn the request into phrases. The *Assist*
button uses the same LLM on typed text. Voice stays on across restarts until turned off.
The microphone, like the webcam, needs the page over HTTPS.

## SAM 3 (segmentation masks)

[facebook/sam3](https://huggingface.co/facebook/sam3), 840 M parameters (Perception Encoder ViT, CLIP text encoder,
DETR-style detector, mask decoder), compiled for the MLA at 672 px (2×2 windows of the ViT's 336-px window) in bf16:
21 ELFs, a 5.7 GB bundle, loaded from Settings → Models like the detectors. SAM 3 finds one concept per pass, so each
phrase of the prompt is one pass; the image features stay on the board, so extra phrases and prompt changes rerun only
the detector.

| On a Modalix DevKit, 672 px | |
|---|---|
| New image, one phrase | ~0.9 s (ViT 0.77 s) |
| Each further phrase / prompt change | ~0.1 s |
| vs PyTorch (FP32) on test images | same detections, scores within 0.004, box IoU ≥ 0.97, mask IoU ≥ 0.99 |

On the MLA: the ViT (4 groups of 7 windowed layers, 4 global layers; rotary positions folded into the weights, tokens
kept in window order), the FPN neck, the CLIP text encoder, the DETR encoder, the 6 decoder layers with their query
position, box relative-position bias and box head, the mask decoder's prompt attention and pixel decoder. On the CPU
(`runtime/src/sam3_rt.cpp`): patching, window to image order, the boxes' sine embedding and edge distances, the box
update, presence and text scoring, and query · pixel for the masks that are kept.

Build: `.venv-sam3` with transformers 5 (SAM 3's weights are gated: accept the license on Hugging Face first), then
`model/sam3_export.py --size 672`, `compile/build_programs.sh sam3_672`, `tools/make_sam3_bundle.py --size 672`, and
`tools/sam3_e2e.py` to check the board against PyTorch. The SAM 3 bundle is not on Hugging Face yet.

## OmDet-Turbo (fast open-vocabulary detection)

[omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) (Apache-2.0, 115 M parameters:
Swin-T, an RT-DETR hybrid encoder, CLIP text, a 6-layer deformable decoder) at 640 px in bf16: 29 ELFs, a 1.15 GB
bundle, loaded from Settings → Models like the other detectors. It is the open stand-in for an "efficient Grounding
DINO": IDEA's own Grounding DINO 1.5 Edge is API-only. Its encoder has no deformable attention (one plain attention
layer plus convolutions), which is where Grounding DINO spends most of its board time. Each phrase is a class; the
task prompt "Detect a, b, c." must fit 64 CLIP tokens.

| On a Modalix DevKit | OmDet-Turbo Tiny 640 | Grounding DINO Tiny 576 | Grounding DINO Base 576 |
|---|---|---|---|
| New image (runtime) | ~145 ms | ~290 ms | ~360 ms |
| Prompt change, same image | ~78 ms | | |
| COCO val2017, 500-image subset, box AP (board) | 43.5 (fp32 PyTorch 43.6) | 41.8 | 42.0 |

Board split (`runtime/src/omdet_rt.cpp`): 12 Swin-T blocks in window layout (timm rolls before padding, so the CPU's
layout tables differ from the HF Swin of Grounding DINO), the hybrid encoder's attention layer, the FPN/PAN convs
(one program with three outputs, written one level after another into the decoder's token buffer), the value
projections and encoder head, 6 + 6 decoder programs, and the CLIP text encoder (8 sequences per run) with the task
head. The CPU does the deformable sampling (3 levels), cosine class scores, the top-900 and the box updates.
Time per new image: backbone 43 ms MLA + 11 ms CPU, encoder 10, selection 8, decoder 47 MLA + 22 CPU.

Build: `model/omdet_export.py`, `compile/build_programs.sh omdet`, `tools/make_omdet_bundle.py`, and
`tools/omdet_e2e.py` / `eval/coco_eval.py --backend omdet` to check the board against PyTorch.

## Grounded EfficientSAM (masks for Grounding DINO)

[EfficientSAM](https://github.com/yformer/EfficientSAM) (Apache-2.0) in two sizes, Ti (10.2 M parameters) and S
(26.4 M), compiled at 768 px in bf16. It loads next to a Grounding DINO detector (Settings → Models, or *Show Masks*
under Thresholds) and makes masks for up to 16 boxes per image: the ones above the Box and Phrase Score thresholds.
Its image encoder starts as soon as an image arrives, while the detector runs; a prompt change reuses it.

| On a Modalix DevKit, 768 px | Ti | S |
|---|---|---|
| Image encoder (once per image) | 46 ms | 98 ms |
| Mask decoder: 1 / 4 / 8 / 16 boxes | 4 / 14 / 16 / 32 ms | the same |
| vs PyTorch (FP32), 44 Grounding DINO boxes | mask IoU mean 0.984, median 0.995 | mean 0.990, median 0.996 |
| Added per image in the studio (Base 576, 11–13 boxes) | ~120 ms | ~170 ms (estimated from the encoder) |

768 px instead of EfficientSAM's 1024: per-layer cost is about half, and the masks agree with the 1024-px ones at a
median IoU of 0.94 (Ti) / 0.96 (S); 512 px drops to 0.90 / 0.92.

Four ELFs per size (`runtime/src/esam_rt.cpp` runs them): the ViT with the patch embedding (ImageNet normalisation
folded in) and the neck's 1×1 conv; the neck's 3×3 conv on the same bytes in image order; and the mask decoder for
1 box and for 8 boxes (boxes on the H axis; the runtime times both at load and covers n boxes with the cheapest mix).
The decoder's 2×2 transposed convs are 1×1 convs to sub-pixel channels and its per-box GroupNorm one row slice per box;
the CPU does the patching, the box corners' Fourier embedding, the best-of-3 pick by predicted IoU, and a 2× bilinear
upsample (masks are 384×384 over the image, stretched like the encoder's input).

Build: `model/esam_export.py --variant ti` (weights from the EfficientSAM repo in `build/weights/efficientsam/`),
`compile/build_programs.sh esam_ti768`, `tools/make_esam_bundle.py --variant ti`, and `tools/esam_e2e.py` to check the
board against PyTorch. The code in `third_party/efficient_sam/` is upstream's (Apache-2.0).

## License

Apache-2.0. Grounding DINO © IDEA Research (Apache-2.0); BERT and Swin weights as distributed
with it. Inter and JetBrains Mono fonts under the SIL Open Font License. The voice models
(Whisper, Gemma, LFM) are not part of this repository; they are installed separately under
their own licenses. SAM 3's weights are under Meta's SAM License (https://github.com/facebookresearch/sam3/blob/main/LICENSE),
not Apache-2.0; a SAM 3 bundle carries a copy of it. EfficientSAM (code in `third_party/efficient_sam/` and weights)
© Meta / the EfficientSAM authors, Apache-2.0. OmDet-Turbo (weights) © Om AI Lab, Apache-2.0.
