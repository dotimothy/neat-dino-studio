#!/usr/bin/env bash
# Copy compiled probes to the DevKit, run them with pyneat, fetch outputs, compare.
#   compile/probes/board_check.sh gs24 gs96
#   BOARD=sima@192.168.91.225 REMOTE=/media/nvme/dino-dev/probes (defaults)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
BOARD="${BOARD:-sima@192.168.91.225}"
REMOTE="${REMOTE:-/media/nvme/dino-dev/probes}"
P="$ROOT/build/probes"
ssh "$BOARD" "mkdir -p $REMOTE"
files=("$ROOT/compile/probes/board_run.py")
names=()
for n in "$@"; do
  mpk=$(ls "$P/$n"/*_mpk.tar.gz 2>/dev/null | head -1 || true)
  [[ -n "$mpk" ]] || { echo "$n: no MPK in $P/$n"; continue; }
  cp -f "$mpk" "$P/$n.tar.gz"
  files+=("$P/$n.tar.gz" "$P/${n}_in.npy")
  names+=("$n")
done
[[ ${#names[@]} -gt 0 ]] || exit 1
rsync -q "${files[@]}" "$BOARD:$REMOTE/"
ssh "$BOARD" "source ~/pyneat/bin/activate && cd $REMOTE && python3 board_run.py --dir $REMOTE ${names[*]}" || true
for n in "${names[@]}"; do
  rsync -q "$BOARD:$REMOTE/${n}_board.npy" "$BOARD:$REMOTE/${n}_board.json" "$P/" 2>/dev/null || true
done
"$ROOT/.venv/bin/python" "$ROOT/compile/probes/compare.py" "${names[@]}"
