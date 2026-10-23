#!/usr/bin/env bash
# Compile every Grounding DINO stage of one input size to its own MLA ELF (bf16, plain HWC I/O).
#   compile/build_stages.sh 576 [stage ...]      (default: all stages + text)
#   PREC=w8a16 JOBS=6 compile/build_stages.sh 768
#   MODEL=tiny compile/build_stages.sh 576       (grounding-dino-tiny: t576_* stages + tiny_text_t64)
# Reads build/probes/<name>.onnx from model/export_onnx.py; writes build/stages/<prec>/<name>/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$ROOT"
S="$1"; shift
PREC="${PREC:-bf16}"
JOBS="${JOBS:-6}"
MODEL="${MODEL:-base}"
if [[ "$MODEL" == tiny ]]; then P=t; TEXT=tiny_text_t64; else P=s; TEXT=text_t64; fi
stages=("$@")
if [[ ${#stages[@]} -eq 0 ]]; then
  # every exported stage of this size (model/export_onnx.py --all), plus the shared text encoder
  stages=("$TEXT")
  for f in build/probes/${P}${S}_*.onnx; do
    n="$(basename "$f" .onnx)"; n="${n#${P}${S}_}"
    [[ "$n" == backbone ]] || stages+=("$n")
  done
fi
names=()
for st in "${stages[@]}"; do [[ "$st" == "$TEXT" || "$st" == text_t64 ]] && names+=("$TEXT") || names+=("${P}${S}_$st"); done
# The biggest graphs first, so the long compiles start early.
ordered=()
big() { [[ "$1" == *_bb_s4* || "$1" == *_bb_s1* || "$1" == *_bb_stage1 || "$1" == *_bbm_stage* || "$1" == "$TEXT" ]]; }
for n in "${names[@]}"; do big "$n" && ordered+=("$n"); done
for n in "${names[@]}"; do big "$n" || ordered+=("$n"); done
for n in "${ordered[@]}"; do
  out="build/stages/$PREC/$n"
  if [[ -f "$out/compile_summary.json" ]] && grep -q '"single_mla_elf": true' "$out/compile_summary.json"; then
    continue
  fi
  while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do wait -n; done
  rm -rf "$out"
  extra=()
  [[ "$PREC" != bf16 ]] && extra=(--calib "build/calib/$n.npz")
  compile/compile.sh --model "build/probes/$n.onnx" --precision "$PREC" --name "$n" --out "$out" "${extra[@]}" > /dev/null 2>&1 &
done
wait
printf '%-18s %-5s %-4s %-6s %s\n' stage elfs a65 secs plugins
for n in "${ordered[@]}"; do
  s="build/stages/$PREC/$n/compile_summary.json"
  if [[ -f "$s" ]] && grep -q mla_elfs "$s"; then
    python3 -c "import json; d=json.load(open('$s')); print(f\"{'$n':18} {len(d.get('mla_elfs',[])):<5} {len(d.get('a65',[])):<4} {d.get('seconds',0):<6} {d.get('plugins')}\")"
  else
    echo "$n: FAILED (build/stages/$PREC/$n/compile.log)"
    grep -v " INFO - " "build/stages/$PREC/$n/compile.log" | grep -m3 -iE "error|fail|Cannot" | cut -c1-200 | sed 's/^/    /'
  fi
done
