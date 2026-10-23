#!/usr/bin/env bash
# Start Neat DINO Studio on the DevKit. Extra arguments go to webapp/server.py:
#   webapp/run.sh                       # http(s)://<board>:8092/ (open https:// for the webcam)
#   webapp/run.sh --model dino-576-bf16 # load a model at start (otherwise load it in Settings -> Models)
#   webapp/run.sh --daemon              # start detached, log to webapp/server.log
#   webapp/run.sh --stop                # stop a running instance (frees the MLA cleanly)
#   webapp/run.sh --port 9000
#   webapp/run.sh --fetch fast          # download a model from Hugging Face (fast, accurate, all, or a name)
#   webapp/run.sh --list-hub            # what Hugging Face has, and what is here
#   webapp/run.sh --no-https            # plain HTTP only (the webcam then only works from localhost)
#   webapp/run.sh update                # fetch the latest source over this install, then re-run setup.sh
# Browsers allow the webcam only on HTTPS pages, so the same port also speaks HTTPS with a
# self-signed certificate (webapp/make_cert.sh); the browser warns about it once.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
# shellcheck disable=SC1091
source "$HERE/models_dir.sh"

if [[ "${1:-}" == update ]]; then
  shift
  repo="${STUDIO_REPO:-dotimothy/neat-dino-studio}" branch="${STUDIO_BRANCH:-main}"
  installer="$(mktemp)" || exit 1
  trap 'rm -f "$installer"' EXIT
  url="https://raw.githubusercontent.com/${repo}/${branch}/install.sh"
  if [[ -z "${STUDIO_ARCHIVE:-}" ]] && curl -fsSL --max-time 60 "$url" -o "$installer" 2>/dev/null \
      && grep -q STUDIO_UPDATE "$installer"; then
    echo "Using the latest install.sh (${repo}, ${branch})"
  elif [[ -f "$ROOT/install.sh" ]]; then
    cp "$ROOT/install.sh" "$installer"
  else
    echo "Could not download install.sh; get it from https://github.com/${repo}" >&2; exit 1
  fi
  STUDIO_UPDATE=1 STUDIO_SETUP=0 STUDIO_DIR="$ROOT" bash "$installer" || exit $?
  "$ROOT/setup.sh" --no-prompt "$@" || exit $?
  if pgrep -f "$HERE/[s]erver.py" > /dev/null 2>&1; then
    echo "The running studio still has the old code. Restart it: $HERE/run.sh --stop && $HERE/run.sh --daemon"
  fi
  exit 0
fi

VENV="${PYNEAT_VENV:-$HOME/pyneat}"
PY=python3
[[ -f "$VENV/bin/python3" ]] && PY="$VENV/bin/python3"
MODELS="$(dino_models_dir "$ROOT")"
PORT=8092 HTTPS=1 daemon=0
args=(--models "$MODELS")
# Model downloads don't serve anything or touch the MLA: run them straight away.
for a in "$@"; do
  if [[ "$a" == --fetch || "$a" == --list-hub ]]; then
    fetch=()
    for b in "$@"; do
      case "$b" in
        fast) fetch+=(dino-576-bf16) ;; accurate) fetch+=(dino-768-bf16) ;; tiny) fetch+=(dino-tiny-576-bf16) ;; *) fetch+=("$b") ;;
      esac
    done
    cd "$ROOT" && exec "$PY" "$HERE/server.py" "${args[@]}" "${fetch[@]}"
  fi
done
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop)
      pid="$(pgrep -f "$HERE/[s]erver.py" || true)"
      if [[ -n "$pid" ]]; then
        # shellcheck disable=SC2086
        kill -TERM $pid                 # SIGTERM: the runtime frees its MLA memory (never SIGKILL)
        for _ in $(seq 1 60); do
          alive=0; for p in $pid; do kill -0 "$p" 2>/dev/null && alive=1; done
          ((alive)) || break; sleep 0.5
        done
        ((alive)) && echo "Neat DINO Studio (pid $pid) is still shutting down" >&2 || echo "stopped Neat DINO Studio (pid $pid)"
      else echo "Neat DINO Studio is not running"; fi
      exit 0 ;;
    --daemon) daemon=1; shift ;;
    --port) [[ $# -ge 2 ]] || { echo "--port needs a value" >&2; exit 2; }
            PORT="$2"; args+=("$1" "$2"); shift 2 ;;
    --no-https) HTTPS=0; args+=("$1"); shift ;;
    *) args+=("$1"); shift ;;
  esac
done

if curl -fsS --max-time 1 "http://localhost:${PORT}/api/status" 2>/dev/null | grep -q '"dino-studio"'; then
  echo "Neat DINO Studio is already running at http://localhost:${PORT}/"; exit 0
fi
# One MLA app at a time. Bracketed first letters keep pgrep from matching itself; set
# MLA_BUSY_PATTERN to your own list. FORCE=1 starts anyway (e.g. a small LLiMa model alongside).
MLA_BUSY_PATTERN="${MLA_BUSY_PATTERN:-[n]eat-genai-studio|[l]aya/|[c]ar-scratch|[d]ofa-demo|[n]eat-change-studio/webapp|[c]yws/webapp|[l]lima run|[d]ino_rt}"
busy="$(pgrep -af "$MLA_BUSY_PATTERN" | grep -v "$HERE" || true)"
if [[ -n "$busy" && -z "${FORCE:-}" ]]; then
  echo "Another app may be using the MLA:" >&2; echo "$busy" >&2
  echo "Stop it first, or start with FORCE=1." >&2
  exit 1
fi
[[ -x "$ROOT/runtime/build/dino_rt" ]] || echo "warning: runtime/build/dino_rt is missing; run ./setup.sh" >&2
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
((HTTPS)) && { bash "$HERE/make_cert.sh" "$HERE/certs" || true; }
if ((daemon)); then
  setsid nohup "$PY" "$HERE/server.py" "${args[@]}" > "$HERE/server.log" 2>&1 < /dev/null &
  for _ in $(seq 1 30); do
    sleep 1
    if curl -fsS --max-time 1 "http://localhost:${PORT}/api/status" 2>/dev/null | grep -q '"dino-studio"'; then
      ip="$(hostname -I 2> /dev/null | awk '{print $1}' || true)"; ip="${ip:-localhost}"
      echo "Neat DINO Studio running at http://${ip}:${PORT}/  (log: $HERE/server.log)"
      ((HTTPS)) && echo "  webcam: https://${ip}:${PORT}/  (self-signed: accept the browser warning once)"
      exit 0
    fi
  done
  echo "started, but not answering yet; see $HERE/server.log" >&2; exit 0
fi
exec "$PY" "$HERE/server.py" "${args[@]}"
