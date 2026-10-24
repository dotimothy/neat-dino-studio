#!/usr/bin/env bash
# Compile every exported program with one name prefix to its own MLA ELF.
#   compile/build_programs.sh sam3_672 [stage ...]      SAM 3 (model/sam3_export.py)
#   compile/build_programs.sh esam_ti768 [stage ...]    EfficientSAM (model/esam_export.py)
# JOBS=6 by default. Reads build/probes/<prefix>_<stage>.onnx; writes build/stages/bf16/<prefix>_<stage>/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$ROOT"
P="$1"; shift
JOBS="${JOBS:-6}"
names=()
if [[ $# -gt 0 ]]; then for st in "$@"; do names+=("${P}_$st"); done
else for f in build/probes/${P}_*.onnx; do names+=("$(basename "$f" .onnx)"); done; fi
# the biggest graphs first, so the long compiles start early
ordered=()
for n in "${names[@]}"; do [[ "$n" == *_vit* || "$n" == *_text || "$n" == *_enc ]] && ordered+=("$n"); done
for n in "${names[@]}"; do [[ "$n" == *_vit* || "$n" == *_text || "$n" == *_enc ]] || ordered+=("$n"); done
for n in "${ordered[@]}"; do
  out="build/stages/bf16/$n"
  if [[ -f "$out/compile_summary.json" ]] && grep -q '"single_mla_elf": true' "$out/compile_summary.json"; then continue; fi
  while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do wait -n; done
  rm -rf "$out"
  calib=()
  [[ -f "build/calib/$n.npz" ]] && calib=(--calib "build/calib/$n.npz")     # bf16 LUT ranges need real data
  compile/compile.sh --model "build/probes/$n.onnx" --precision bf16 --name "$n" --out "$out" "${calib[@]}" > /dev/null 2>&1 &
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
