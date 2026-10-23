#!/usr/bin/env bash
# Run compile/dino_compile.py inside the NEAT SDK container's Model Compiler venv.
#
#   compile/compile.sh --model build/onnx/dino_text_t64.onnx --precision bf16 --out build/mpk/text_t64_bf16
#   compile/compile.sh --model build/probes/gs24.onnx --precision bf16 --name probe_gs24 --out build/probes/gs24
#
# The full compiler log lands next to the output as compile.log.
#
#   CONTAINER  running NEAT SDK container (default ghcr.io-sima-neat-sdk-latest); it
#              must have the Model Compiler at /sdk-extensions/model-compiler.
#   WS_HOST    host directory the container mounts at /workspace (default
#              ~/SDK/NEAT/workspace). This repo must be inside it.
#
# XLA_FLAGS: the compiler evaluates the model with JAX/XLA on the host to write the
# MLA check file. On recent x86 CPUs XLA's LLVM vectorizer emits bf16 code it cannot
# lower ("Cannot select: v8bf16 = X86ISD::BLENDV") and every bf16 compile with a Conv
# fails with "Failed to generate mpk json file". Turning the vectorizers off for that
# host-side step avoids it; the MLA program is unaffected.
set -euo pipefail
XLA_FLAGS="${XLA_FLAGS:---xla_backend_extra_options=-vectorize-loops=false,-vectorize-slp=false}"

HERE="$(cd "$(dirname "$0")/.." && pwd -P)"
CONTAINER="${CONTAINER:-ghcr.io-sima-neat-sdk-latest}"
WS_HOST="$(cd "${WS_HOST:-$HOME/SDK/NEAT/workspace}" && pwd -P)"
REL="${HERE#"$WS_HOST"/}"
[[ "$REL" != "$HERE" ]] || { echo "$HERE is not under $WS_HOST (the container's /workspace; set WS_HOST)" >&2; exit 1; }

OUT="" PREC="bf16" prev=""
for a in "$@"; do
  [[ "$prev" == "--out" ]] && OUT="$a"
  [[ "$prev" == "--precision" ]] && PREC="$a"
  prev="$a"
done
OUT="${OUT:-build/$PREC}"
mkdir -p "$HERE/$OUT"

args=""
for a in "$@"; do args+=" $(printf '%q' "$a")"; done
set +e
docker exec -w "/workspace/$REL" -e HOME=/tmp -e "XLA_FLAGS=$XLA_FLAGS" \
  -e "SIMA_MLA_SIM_PARALLEL=${SIMA_MLA_SIM_PARALLEL:-8}" ${SIMA_MLA_COMPILE_USE_JAX:+-e SIMA_MLA_COMPILE_USE_JAX=$SIMA_MLA_COMPILE_USE_JAX} -u "$(id -u):$(id -g)" "$CONTAINER" bash -c \
  "source /sdk-extensions/model-compiler/bin/activate && python3 -u compile/dino_compile.py$args" \
  2>&1 | tee "$HERE/$OUT/compile.log" > /dev/null
rc=${PIPESTATUS[0]}
echo "compile exit code $rc  (log: $OUT/compile.log)"
exit "$rc"
