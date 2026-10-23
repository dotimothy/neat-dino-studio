#!/usr/bin/env bash
# Copy compiled probes to the DevKit, run them with pyneat, fetch outputs, compare.
#   compile/probes/board_check.sh gs24 gs96
#   BOARD=sima@<board-ip> (required), REMOTE=/media/nvme/dino-dev/probes (default)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
BOARD="${BOARD:?set BOARD=user@board-ip}"
REMOTE="${REMOTE:-/media/nvme/dino-dev/probes}"
P="$ROOT/build/probes"
ssh "$BOARD" "mkdir -p $REMOTE"
files=("$ROOT/compile/probes/board_run.py")
names=()
for n in "$@"; do
  mpk=$(ls "$P/$n"/*_mpk.tar.gz 2>/dev/null | head -1 || true)
  [[ -n "$mpk" ]] || { echo "$n: no MPK in $P/$n"; continue; }
  cp -f "$mpk" "$P/$n.tar.gz"
  files+=("$P/$n.tar.gz")
  if [[ -f "$P/${n}_inputs.txt" ]]; then
    files+=("$P/${n}_inputs.txt")
    for i in $(cat "$P/${n}_inputs.txt"); do files+=("$P/${n}_in_$i.npy"); done
  else
    files+=("$P/${n}_in.npy")
  fi
  names+=("$n")
done
[[ ${#names[@]} -gt 0 ]] || exit 1
rsync -q "${files[@]}" "$BOARD:$REMOTE/"
ssh "$BOARD" "source ~/pyneat/bin/activate && cd $REMOTE && python3 board_run.py --dir $REMOTE ${names[*]}" || true
for n in "${names[@]}"; do
  rsync -q "$BOARD:$REMOTE/${n}_board.npy" "$BOARD:$REMOTE/${n}_board.json" "$P/" 2>/dev/null || true
done
"$ROOT/.venv/bin/python" "$ROOT/compile/probes/compare.py" "${names[@]}"
