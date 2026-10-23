#!/usr/bin/env bash
# Set up Neat DINO Studio on a Modalix DevKit.
#
#   ./setup.sh                    # build the runtime, fetch the Fast model, certificate, alias, desktop icon
#   ./setup.sh --fetch accurate   # also/instead fetch: fast | accurate | all | <model name> | none
#   ./setup.sh --models-dir DIR   # keep models somewhere else (remembered in .models-dir)
#   ./setup.sh --check            # only check the board and the install, change nothing
#   ./setup.sh --no-alias --no-desktop --no-prompt
#
# Needs: the NEAT SDK's SimaLMM runtime (sima-lmm-dev, CMake package SimaLMM), cmake, a C++20
# compiler with OpenMP, and Python 3 with numpy, Pillow and Flask (the pyneat venv has them).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/webapp/models_dir.sh"

c_g=$'\e[32m' c_y=$'\e[33m' c_r=$'\e[31m' c_b=$'\e[1m' c_0=$'\e[0m'
[[ -t 1 ]] || { c_g=; c_y=; c_r=; c_b=; c_0=; }
step() { printf '\n%s▸ %s%s\n' "$c_b" "$*" "$c_0"; }
ok()   { printf '  %s✔%s %s\n' "$c_g" "$c_0" "$*"; }
warn() { printf '  %s!%s %s\n' "$c_y" "$c_0" "$*"; }
bad()  { printf '  %s✘%s %s\n' "$c_r" "$c_0" "$*"; }

fetch="" models_dir="" alias_on=1 desktop_on=1 prompt=1 check=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fetch) fetch="${2:-}"; shift 2 ;;
    --models-dir) models_dir="${2:-}"; shift 2 ;;
    --no-alias) alias_on=0; shift ;;
    --no-desktop) desktop_on=0; shift ;;
    --no-prompt) prompt=0; shift ;;
    --check) check=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -t 0 ]] || prompt=0

printf '%s' "$c_b"
cat <<'EOF'
   ✦  Neat DINO Studio — Grounding DINO on the Modalix MLA
EOF
printf '%s' "$c_0"

VENV="${PYNEAT_VENV:-$HOME/pyneat}"
PY=python3
[[ -x "$VENV/bin/python3" ]] && PY="$VENV/bin/python3"
fail=0

step "Board"
if [[ -f /usr/lib/aarch64-linux-gnu/cmake/SimaLMM/SimaLMMConfig.cmake || -n "$(ls /usr/lib/*/cmake/SimaLMM 2>/dev/null)" ]]; then
  ok "SimaLMM MLA runtime ($(dpkg-query -W -f='${Version}' sima-lmm-dev 2>/dev/null || echo installed))"
else
  bad "SimaLMM not found (install the NEAT SDK's sima-lmm-dev package)"; fail=1
fi
for t in cmake g++; do command -v $t >/dev/null && ok "$t" || { bad "$t missing"; fail=1; }; done
cma="$(awk '/CmaFree/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)"
[[ "$cma" -gt 700 ]] && ok "MLA memory free: ${cma} MB" || warn "MLA memory free: ${cma} MB (a detector needs about 700 MB; stop other MLA apps)"

step "Python"
if "$PY" -c "import numpy, PIL, flask" 2>/dev/null; then
  ok "$PY: numpy, Pillow, Flask"
else
  if ((check)); then bad "$PY lacks numpy/Pillow/Flask"; fail=1
  else
    warn "installing numpy Pillow flask into $("$PY" -c 'import sys; print(sys.prefix)')"
    "$PY" -m pip install -q numpy Pillow flask || { bad "pip install failed"; fail=1; }
  fi
fi
"$PY" -c "import tokenizers" 2>/dev/null && ok "tokenizers (fast)" || ok "tokenizer: built-in WordPiece"

step "Runtime (dino_rt)"
if ((check)); then
  [[ -x "$ROOT/runtime/build/dino_rt" ]] && ok "runtime/build/dino_rt" || { bad "runtime not built"; fail=1; }
else
  if cmake -S "$ROOT/runtime" -B "$ROOT/runtime/build" -DCMAKE_BUILD_TYPE=Release > "$ROOT/runtime/build.log" 2>&1 &&
     cmake --build "$ROOT/runtime/build" -j8 >> "$ROOT/runtime/build.log" 2>&1; then
    ok "built runtime/build/dino_rt"
  else
    bad "build failed: see runtime/build.log"; fail=1
  fi
fi

step "Models"
if [[ -n "$models_dir" ]] && ((!check)); then
  printf '%s\n' "$models_dir" > "$ROOT/.models-dir"
fi
MODELS="$(dino_models_dir "$ROOT")"
((check)) || mkdir -p "$MODELS"
ok "models directory: $MODELS"
have=$(ls -d "$MODELS"/dino-*-*/ 2>/dev/null | wc -l)
if [[ -z "$fetch" ]]; then
  if ((have > 0)); then fetch=none
  elif ((prompt)); then
    read -r -p "  Download a model now? [F]ast 576 px / [a]ccurate 768 px / [b]oth / [n]one: " r
    case "${r,,}" in a*) fetch=accurate ;; b*) fetch=all ;; n*) fetch=none ;; *) fetch=fast ;; esac
  else fetch=fast; fi
fi
if ((!check)) && [[ "$fetch" != none ]]; then
  names=()
  case "$fetch" in
    fast) names=(dino-576-bf16) ;; accurate) names=(dino-768-bf16) ;; all) names=(all) ;; *) names=("$fetch") ;;
  esac
  "$PY" "$ROOT/webapp/server.py" --models "$MODELS" --fetch "${names[@]}" || warn "download did not finish (re-run: webapp/run.sh --fetch ${fetch})"
fi
for d in "$MODELS"/dino-*/; do [[ -f "$d/manifest.json" ]] && ok "on board: $(basename "$d")"; done

step "HTTPS certificate (for the webcam)"
if ((check)); then
  [[ -f "$ROOT/webapp/certs/cert.pem" ]] && ok "webapp/certs" || warn "no certificate yet (made on first start)"
else
  bash "$ROOT/webapp/make_cert.sh" "$ROOT/webapp/certs" && ok "webapp/certs" || warn "no certificate; the webcam needs HTTPS"
fi

if ((!check)) && ((alias_on)); then
  step "Shell alias"
  line="alias neat-dino='$ROOT/webapp/run.sh'"
  if grep -qs "alias neat-dino=" "$HOME/.bashrc"; then
    sed -i "s#^alias neat-dino=.*#${line//&/\\&}#" "$HOME/.bashrc"
  else
    printf '\n%s\n' "$line" >> "$HOME/.bashrc"
  fi
  ok "neat-dino (new shells): start the studio; neat-dino --stop; neat-dino update"
fi

if ((!check)) && ((desktop_on)) && [[ -d "$HOME/Desktop" ]]; then
  step "Desktop icon"
  f="$HOME/Desktop/neat-dino-studio.desktop"
  cat > "$f" <<EOF
[Desktop Entry]
Type=Application
Name=Neat DINO Studio
Comment=Open-vocabulary detection with Grounding DINO on the MLA
Exec=bash -c '$ROOT/webapp/run.sh --daemon; xdg-open https://localhost:8092/ >/dev/null 2>&1'
Icon=$ROOT/webapp/web/assets/neat-dino-192.png
Terminal=false
Categories=Graphics;Science;
EOF
  chmod +x "$f"
  ok "$f"
fi

echo
if ((fail)); then
  bad "Setup found problems (above)."; exit 1
fi
ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
printf '%sReady.%s Start it with %sneat-dino%s (new shell) or %s/webapp/run.sh, then open https://%s:8092/\n' \
  "$c_g" "$c_0" "$c_b" "$c_0" "$ROOT" "${ip:-<board-ip>}"
