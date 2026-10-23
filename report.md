# Grounding DINO on the Modalix MLA — report

IDEA-Research/grounding-dino-base (232 M parameters, Apache-2.0), compiled for the SiMa.ai
Modalix MLA and served by Neat DINO Studio. Measured on a Modalix DevKit (eLxr, SimaLMM 0.5.0,
sima-frontend 2.1.3), 2026-10-22/23.

## Result

| | Fast · 576 px | Accurate · 768 px |
|---|---|---|
| MLA programs (bf16, 1 ELF each) | 48 (+ text encoder, shared) | 48 (+ text encoder, shared) |
| Per new image, board | **~0.36 s** | **~0.60 s** |
| Prompt change on the same image | ~0.25 s | ~0.42 s |
| Prompt (text encoder) | 4.7 ms | 4.7 ms |
| COCO AP, 500-image val2017 subset, board | **42.0** | (see below) |
| Same, FP32 PyTorch on GPU, same letterbox | 41.7 | 45.8 |
| Bundle on disk | 1.7 GB | 1.4 GB |

The board's bf16 results match FP32: on the test images every top-15 FP32 detection is found
again with IoU ≥ 0.97 and a score within ±0.05, and COCO AP is the same within noise.
(Grounding DINO's published COCO zero-shot numbers use 800×1333 inputs; these use a square
letterbox and the subset, so compare the board and GPU columns, not the paper.)

Time per image at 576 (ms, board): backbone 90 MLA + 25 CPU; encoder 75 MLA + 112 CPU;
decoder values 2; top-k 4; decoder 19 MLA + 28 CPU; patchify 2. At 768: backbone 140 + 45,
encoder 145 + 202, decoder 25 + 30.

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
   So every Swin block is its own program on tokens already in window layout, and the CPU
   moves tokens between the plain and shifted layouts (and does patch merging) — 26 gathers,
   25 ms at 576.

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

## Reproduce

```
model/check_mirror.py          rebuild vs transformers (FP32): identical detections at 576
model/export_onnx.py --all     every stage, with a real input/output pair
compile/build_stages.sh 576    all stages, bf16, 1 ELF each (container ghcr.io-sima-neat-sdk-latest)
tools/make_bundle.py           runtime bundle
tools/e2e_check.py             board vs FP32 on one image
eval/coco_eval.py              COCO AP (board or PyTorch)
```
