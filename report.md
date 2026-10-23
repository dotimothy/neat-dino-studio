# Grounding DINO on the Modalix MLA — report

IDEA-Research/grounding-dino-base (232 M parameters, Apache-2.0), compiled for the SiMa.ai
Modalix MLA and served by Neat DINO Studio. Measured on a Modalix DevKit (eLxr, SimaLMM 0.5.0,
sima-frontend 2.1.3), 2026-10-22/23.

## Result

| | Fast · 576 px | Accurate · 768 px |
|---|---|---|
| MLA programs (bf16, 1 ELF each) | 29 + shared text encoder | 29 + shared text encoder |
| Per new image, board | **~0.33 s** | **~0.54 s** |
| COCO AP50 / APsmall, board | 65.5 / 18.4 | 69.8 / 26.3 |
| Prompt change on the same image | ~0.25 s | ~0.41 s |
| Prompt (text encoder) | 4.7 ms | 4.7 ms |
| COCO AP, 500-image val2017 subset, board | **42.0** | **46.4** |
| Same, FP32 PyTorch on GPU, same letterbox | 41.7 | 45.8 |
| Bundle on disk | 1.7 GB | 1.4 GB |

The board's bf16 results match FP32: on the test images every top-15 FP32 detection is found
again with IoU ≥ 0.97 and a score within ±0.05, and COCO AP is the same within noise.
(Grounding DINO's published COCO zero-shot numbers use 800×1333 inputs; these use a square
letterbox and the subset, so compare the board and GPU columns, not the paper.)

Time per image at 576 (ms, board): backbone 75 MLA + 7 CPU; encoder 75 MLA + 112 CPU;
decoder values 2; top-k 4; decoder 19 MLA + 28 CPU; patchify 2. At 768: backbone 112 + 12,
encoder 145 + 202, decoder 18 + 30. (Before the Swin stages were fused: backbone 90 + 25 and
140 + 45, 0.36 / 0.60 s per image.)

## Why it is cut into 48 programs

1. **Deformable attention can't run on the MLA.** The MLA's GridSample compiles only for value
   maps up to 32 px wide and 16 channels, and costs ~0.3 µs per sample (≈1 s per encoder layer).
   The sampling points spread far (coarse levels get 57–85 % of the attention and sample the
   whole map; fine levels need ±16 px for 99 %), so no fixed-window rewrite works, and a dense
   exact rewrite needs ~1 TMAC and GBs of traffic per image. The A65 cores do it instead:
   bf16 NEON gathers straight from the MLA's output buffer, 17 ms per encoder layer at 576.
2. **Top-900 query selection** needs a sort over all image tokens: done on the CPU between the
   encoder and the decoder (as SiMa's RF-DETR example does).
3. **Decoder box refinement** needs logit/sigmoid/sine of the boxes each layer: CPU, in FP32.
4. **Swin-B windows.** Window partitioning needs a 6-D reshape the MLA can't place; a 4-D
   reshape chain is placed but sends the compiler into "large tensor" mode that never finishes.
   So the tokens stay in window layout: each Swin stage (stage 3 in three groups of six blocks)
   is one program, and the move between plain and shifted windows is Slice+Concat inside it
   (free on the MLA). Patch merging and the return to image order are 6 CPU gathers (7 ms at
   576): done on the MLA with step-2 slices and a spatial reshape they compile and pass the
   compiler's check, but compute wrong values on the board.

Everything crosses between MLA and CPU in place: `dino_rt` runs bare ELFs (compiled with
MLA tessellation, plain HWC bf16 I/O) through SimaLMM's `MLAModelWithBuffer`. Through pyneat
the same hops would cost seconds (EV74 tessellation of 1×N token tensors: 5 s per call).

## Compiler issues found, and the fixes

| Symptom | Cause | Fix |
|---|---|---|
| GridSample: "Reading uninitialized L1 memory" | kernel limits (≤32 px, ≤16 ch) | CPU sampling |
| 5 s per MPK call through pyneat | EV74 tessellation of wide token tensors | `--mla-tess hwc`, run bare ELFs with SimaLMM |
| Swin attention Einsums "Cannot assign" | Einsum inputs were channel slices of reshape outputs | q/k/v convs after the partition |
| Swin compile never finishes | 4-D partition reshapes trigger large-tensor mode | window-layout blocks + CPU gathers |
| enc6 "Failed to generate code for tuple_concat" | MLP output concatenated last along W | put that part first |
| A 4-channel conv output over 7k tokens fails | small channel count | pad the conv to more outputs |
| 768 encoders "Check FAILED" | channel softmax over 12 240 channels | exp/sum over two token chunks (exact; scores in −25..15) |
| "Layer names are not unique" | two identical constants merged by onnxsim | unequal chunk sizes |

## Int8 weights (w8a16): tried, not shipped

The backbone and encoder programs were compiled with int8 weights and bf16 activations (the
decoder's small programs got no faster, so they stay bf16). At 576 on the board:

| | bf16 | w8a16 |
|---|---|---|
| Per image, one at a time | 328 ms | 313 ms (backbone 75 → 68, encoder MLA 74 → 65) |
| Live, two in flight | 3.88 FPS | 4.03 FPS |
| COCO AP (500-image subset) | **42.0** | 41.5 |
| Bundle / load time | 1.8 GB / 3 s | 2.1 GB / 6 s |

The MLA programs are compute- rather than weight-bound here (a Swin stage: 14.7 → 12.4 ms,
an encoder layer 11.7 → 10.3 ms, a decoder layer 1.25 → 1.36 ms), and the ELFs get larger,
not smaller. 5 % speed for −0.5 AP and a bigger download is not a good trade, so the hub keeps
bf16 only. `PREC=w8a16 compile/build_stages.sh` and `make_bundle.py --prec w8a16` still build it.

## Live mode: two frames in flight

One frame at a time leaves the MLA idle while the CPU samples (112 ms of the 330 ms at 576) and
the CPU idle while the MLA runs. `dino_rt` now has two slots: each has its own device buffers,
bound to the same loaded programs (SimaLMM keys programs by ELF path, so loading one twice costs
nothing). The client sends the next frame before the last reply; replies come back in order.
An arbiter gives the MLA to the older frame first, and the CPU threads are split between the
slots that are sampling at that moment.

| Board, `tools/pipe_bench.py` | One at a time | Two in flight | Per-frame latency |
|---|---|---|---|
| 576 | 3.05 FPS (328 ms) | **3.88 FPS** (258 ms) | 0.33 → 0.50 s |
| 768 | 1.84 FPS (544 ms) | **2.34 FPS** (427 ms) | 0.54 → 0.83 s |
| 768, whole web app (HTTPS, JPEG decode, post-processing) | 1.53 FPS | **2.19 FPS** | |

Results are bit-identical to one at a time (checked with two alternating images). The gain is
capped by contention, not by idle hardware: a timeline (`DINO_TRACE=file`) shows the MLA busy
71 % of the time. MLA programs run ~10 % slower while the other slot's NEON sampling runs (both
share DRAM bandwidth), sampling with 8 threads takes 24 ms vs 17 ms with 16 (the A65 has 8
cores × 2 threads), and the older frame waits ~12 ms behind long backbone programs. Back-to-back
programs are not queued in this mode, so the older frame can take the MLA between them.

## Voice prompts

Whisper Small (a16w8) and Gemma 4 E2B (a16w4) run in a sidecar process through pyneat's
GenAIServer (LLiMa), next to `dino_rt` on the MLA. On the board, the whole path
(record → transcribe → phrases) takes ~0.5 s: Whisper 0.28 s for a 3 s clip, the LLM 0.15–0.3 s.
On eight test requests Gemma 4 E2B gave the right phrases for all, including indirect ones
("is there anything I could sit on" → `chair . sofa . bench . stool .`); LFM2.5 230M answers in
~40 ms but replied "none" to that one. Loading takes ~15–20 s.

Two things matter when several MLA apps share the board:

- **One process loads at a time.** A process that connects to the MLA dispatcher while another
  is loading programs fails ("Unable to connect to the server from dispatcher", or
  `MLA_LOAD_FAILED`). The studio serialises its own loads (detector and voice); other apps can
  still collide with it.
- **Page cache sits in CMA.** After copying large model files, CmaFree dropped to ~400 MB and
  `dino_rt` could not allocate its buffers; `sync; echo 1 > /proc/sys/vm/drop_caches` freed 1.1 GB.

## Reproduce

```
model/check_mirror.py          rebuild vs transformers (FP32): identical detections at 576
model/export_onnx.py --all     every stage, with a real input/output pair
compile/build_stages.sh 576    all stages, bf16, 1 ELF each (container ghcr.io-sima-neat-sdk-latest)
tools/make_bundle.py           runtime bundle
tools/e2e_check.py             board vs FP32 on one image
eval/coco_eval.py              COCO AP (board or PyTorch)
```
