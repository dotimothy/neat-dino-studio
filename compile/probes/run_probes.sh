#!/usr/bin/env bash
# Compile probe ONNX graphs from build/probes/ in parallel (bf16) and report ELF / A65 counts.
#   compile/probes/run_probes.sh gs24 gs96 ...      (default: every build/probes/*.onnx)
#   JOBS=4 limits parallel compiles.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"
probes=("$@")
if [[ ${#probes[@]} -eq 0 ]]; then
  for f in build/probes/*.onnx; do probes+=("$(basename "$f" .onnx)"); done
fi
JOBS="${JOBS:-6}"
for p in "${probes[@]}"; do
  while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do wait -n; done
  rm -rf "build/probes/$p"
  compile/compile.sh --model "build/probes/$p.onnx" --precision bf16 --name "probe_$p" \
    --out "build/probes/$p" > /dev/null 2>&1 &
done
wait
printf '%-16s %-5s %-4s %-7s %s\n' probe elfs a65 secs plugins
for p in "${probes[@]}"; do
  s="build/probes/$p/compile_summary.json"
  if [[ -f "$s" ]] && grep -q mla_elfs "$s"; then
    python3 -c "import json; d=json.load(open('$s')); print(f\"{'$p':16} {len(d.get('mla_elfs',[])):<5} {len(d.get('a65',[])):<4} {d.get('seconds',0):<7} {d.get('plugins')}\")"
  else
    echo "$p: FAILED (build/probes/$p/compile.log)"
    grep -m4 -iE "error|Cannot assign|Traceback|Exception" "build/probes/$p/compile.log" | cut -c1-240 | sed 's/^/    /'
  fi
done
