#!/usr/bin/env bash
# Compile every exported SAM 3 program of one size (model/sam3_export.py) to its own MLA ELF.
#   compile/build_sam3.sh 672 [stage ...]        JOBS=6 by default
# Reads build/probes/sam3_<S>_<stage>.onnx; writes build/stages/bf16/sam3_<S>_<stage>/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$ROOT"
S="$1"; shift
JOBS="${JOBS:-6}"
names=()
if [[ $# -gt 0 ]]; then for st in "$@"; do names+=("sam3_${S}_$st"); done
else for f in build/probes/sam3_${S}_*.onnx; do names+=("$(basename "$f" .onnx)"); done; fi
# the biggest graphs first, so the long compiles start early
ordered=()
for n in "${names[@]}"; do [[ "$n" == *_vit* || "$n" == *_text ]] && ordered+=("$n"); done
for n in "${names[@]}"; do [[ "$n" == *_vit* || "$n" == *_text ]] || ordered+=("$n"); done
for n in "${ordered[@]}"; do
  out="build/stages/bf16/$n"
  if [[ -f "$out/compile_summary.json" ]] && grep -q '"single_mla_elf": true' "$out/compile_summary.json"; then continue; fi
  while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do wait -n; done
  rm -rf "$out"
  compile/compile.sh --model "build/probes/$n.onnx" --precision bf16 --name "$n" --out "$out" > /dev/null 2>&1 &
done
wait
printf '%-22s %-5s %-4s %-6s %s\n' stage elfs a65 secs elf_mb
for n in "${ordered[@]}"; do
  s="build/stages/bf16/$n/compile_summary.json"
  if [[ -f "$s" ]] && grep -q mla_elfs "$s"; then
    mb=$(tar tzvf build/stages/bf16/$n/*_mpk.tar.gz 2>/dev/null | awk '/_mla.elf/ {printf "%.0f", $3/1048576}')
    python3 -c "import json; d=json.load(open('$s')); print(f\"{'$n':22} {len(d.get('mla_elfs',[])):<5} {len(d.get('a65',[])):<4} {d.get('seconds',0):<6} $mb\")"
  else
    echo "$n: FAILED (build/stages/bf16/$n/compile.log)"
    grep -v " INFO - " "build/stages/bf16/$n/compile.log" | grep -m3 -iE "error|fail|Cannot" | cut -c1-200 | sed 's/^/    /'
  fi
done
