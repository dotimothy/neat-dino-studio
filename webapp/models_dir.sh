# Sourced by setup.sh and webapp/run.sh. The models directory is shared by every install and
# kept across updates (like Neat GenAI Studio's /media/nvme/llima/models). Same rule as
# default_models_dir() in webapp/server.py:
#   DINO_MODELS_DIR, else what `setup.sh --models-dir DIR` remembered in .models-dir,
#   else /media/nvme/models/dino on a DevKit, else models/ in the app directory.
dino_models_dir() {   # dino_models_dir <app dir>
  local root="$1" d=""
  if [[ -n "${DINO_MODELS_DIR:-}" ]]; then
    d="$DINO_MODELS_DIR"
  elif [[ -s "$root/.models-dir" ]]; then
    d="$(head -n 1 "$root/.models-dir")"
  fi
  if [[ -z "$d" ]]; then
    if [[ -d /media/nvme ]]; then d=/media/nvme/models/dino; else d="$root/models"; fi
  fi
  [[ "$d" == "~"* ]] && d="$HOME${d:1}"
  printf '%s\n' "$d"
}
