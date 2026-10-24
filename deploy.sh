#!/usr/bin/env bash
# Developer deploy: copy this working tree (and locally built model bundles) to a DevKit.
#   BOARD=sima@<board-ip> ./deploy.sh [--models] [--build]
#     --models  also copy build/bundles/* into the board's models directory
#     --build   build the runtime on the board afterwards
# REMOTE (default /media/nvme/neat-dino-studio) is the install directory on the board.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BOARD="${BOARD:?set BOARD=user@board-ip}"
REMOTE="${REMOTE:-/media/nvme/neat-dino-studio}"
models=0 build=0
for a in "$@"; do
  case "$a" in --models) models=1 ;; --build) build=1 ;; *) echo "unknown option $a" >&2; exit 2 ;; esac
done
ssh "$BOARD" "mkdir -p '$REMOTE'"
rsync -a --delete \
  --exclude .git --exclude .venv --exclude .venv-sam3 --exclude .claude --exclude build --exclude data/coco --exclude 'data/images/*.jpg' \
  --exclude runtime/build --exclude webapp/certs --exclude webapp/server.log --exclude __pycache__ \
  --exclude .models-dir --exclude docs/assets \
  "$ROOT/" "$BOARD:$REMOTE/"
echo "copied the source to $BOARD:$REMOTE"
if ((models)); then
  MODELS="$(ssh "$BOARD" "cd '$REMOTE' && source webapp/models_dir.sh && dino_models_dir '$REMOTE'")"
  ssh "$BOARD" "mkdir -p '$MODELS'"
  for b in "$ROOT"/build/bundles/dino-* "$ROOT"/build/bundles/esam-*; do
    rsync -a --delete "$b" "$BOARD:$MODELS/"
    echo "copied $(basename "$b") to $MODELS"
  done
  # Page cache sits in the MLA's memory pool (CMA) after big copies: drop it before loading.
  ssh "$BOARD" "sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null || true"
fi
if ((build)); then
  ssh "$BOARD" "cd '$REMOTE' && cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build runtime/build -j8 2>&1 | tail -2"
fi
