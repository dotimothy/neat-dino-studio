#!/usr/bin/env bash
# Install Neat DINO Studio from its own repository, on the Modalix DevKit.
#
#   cd /media/nvme
#   curl -fsSL https://raw.githubusercontent.com/dotimothy/neat-dino-studio/main/install.sh | bash
#
# This fetches the source into ./neat-dino-studio and stops there; run
# ./setup.sh in that directory next (or pass STUDIO_SETUP=1 to do it here).
#
# Environment:
#   STUDIO_DIR      Where to install. Default: ./neat-dino-studio
#                   (also the first argument: ... | bash -s -- /path/to/dir)
#   STUDIO_BRANCH   Branch or tag to install. Default: main
#   STUDIO_REPO     GitHub repository as owner/name.
#                   Default: dotimothy/neat-dino-studio
#   GITHUB_TOKEN    Token for a private repository.
#   STUDIO_ARCHIVE  A .tar.gz of the repository to install from instead of
#                   downloading (a local path or a URL), e.g. for an offline board.
#   STUDIO_SETUP=1  Run ./setup.sh once the source is in place.
#   STUDIO_UPDATE=1 Update the install already in STUDIO_DIR in place: the new
#                   source is copied over it, and models/, data/, out/,
#                   webapp/certs, webapp/uploads, webapp/cache, webapp/server.log,
#                   build/, third_party/, the virtualenvs, eval/board_* and the
#                   model, key and .env files that .gitignore lists are kept.
#                   `webapp/run.sh update` (or `neat-dino update`) does this,
#                   then re-runs setup.sh.
#
# Everything lives in main() so a download cut short cannot run half a script.
set -euo pipefail

main() {
  local repo="${STUDIO_REPO:-dotimothy/neat-dino-studio}"
  local branch="${STUDIO_BRANCH:-main}"
  local dir="${1:-${STUDIO_DIR:-neat-dino-studio}}"
  local archive="${STUDIO_ARCHIVE:-}"
  local update="${STUDIO_UPDATE:-0}"

  say()  { printf '%s\n' "$*"; }
  fail() { printf 'install: %s\n' "$*" >&2; exit 1; }
  is_install() { [[ -f "$1/setup.sh" && -f "$1/webapp/run.sh" && -f "$1/runtime/CMakeLists.txt" ]]; }

  command -v tar >/dev/null 2>&1 || fail "tar is required."

  if [[ -e "${dir}" ]]; then
    if is_install "${dir}"; then
      if [[ "${update}" != "1" ]]; then
        local abs qdir
        abs="$(cd "${dir}" && pwd)" || abs="${dir}"
        qdir="$(printf '%q' "${abs}")"
        fail "${dir} already holds an install. Update it with:
  cd ${qdir} && webapp/run.sh update      (or: neat-dino update)
or, on an install whose webapp/run.sh has no 'update' yet:
  curl -fsSL https://raw.githubusercontent.com/${repo}/${branch}/install.sh | STUDIO_UPDATE=1 STUDIO_DIR=${qdir} bash"
      fi
    elif [[ ! -d "${dir}" || -n "$(ls -A "${dir}" 2>/dev/null)" ]]; then
      fail "${dir} exists and is not empty. Choose another place with STUDIO_DIR=/path."
    else
      update=0                      # an empty directory: a fresh install
    fi
  else
    update=0
  fi

  local tmp
  tmp="$(mktemp -d)" || fail "mktemp failed."
  # shellcheck disable=SC2064  # expand now: tmp is local to main
  trap "rm -rf '${tmp}'" EXIT

  local verb="Installing"
  [[ "${update}" == "1" ]] && verb="Updating"
  if [[ -n "${archive}" && -f "${archive}" ]]; then
    say "${verb} Neat DINO Studio from ${archive}"
    cp "${archive}" "${tmp}/src.tar.gz" || fail "cannot read ${archive}"
  else
    local url="${archive}"
    local -a auth=()
    if [[ -z "${url}" ]]; then
      if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        # A private repository: only the API endpoint accepts a token.
        url="https://api.github.com/repos/${repo}/tarball/${branch}"
      else
        url="https://github.com/${repo}/archive/${branch}.tar.gz"
      fi
    fi
    [[ -n "${GITHUB_TOKEN:-}" ]] && auth=( -H "Authorization: Bearer ${GITHUB_TOKEN}" )
    say "${verb} Neat DINO Studio: fetching ${repo} (${branch})"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL ${auth[@]+"${auth[@]}"} "${url}" -o "${tmp}/src.tar.gz" \
        || fail "download failed: ${url}$([[ -z "${GITHUB_TOKEN:-}" ]] && printf ' (a private repository needs GITHUB_TOKEN)')"
    elif command -v wget >/dev/null 2>&1; then
      wget -qO "${tmp}/src.tar.gz" ${GITHUB_TOKEN:+--header="Authorization: Bearer ${GITHUB_TOKEN}"} "${url}" \
        || fail "download failed: ${url}"
    else
      fail "curl or wget is required."
    fi
  fi

  # The archive holds one top-level directory with the Studio in it. List it to
  # a file first: under pipefail, `tar | grep -q` fails when grep stops reading
  # at its first match and tar is cut off.
  tar -tzf "${tmp}/src.tar.gz" > "${tmp}/list" 2>/dev/null || fail "not a readable .tar.gz archive."
  local f
  for f in 'setup\.sh' 'webapp/run\.sh' 'runtime/CMakeLists\.txt'; do
    grep -qE "^[^/]+/${f}\$" "${tmp}/list" \
      || fail "that archive is not Neat DINO Studio (no ${f//\\/} at its top level)."
  done

  if [[ "${update}" == "1" ]]; then
    mkdir -p "${tmp}/src"
    tar -xzf "${tmp}/src.tar.gz" -C "${tmp}/src" --strip-components=1 || fail "extract failed."
    is_install "${tmp}/src" || fail "extract failed."
    say "Copying the new source over ${dir}"
    # Kept as they are: what setup, the demo, run_all.sh and you made in the install
    # (everything .gitignore lists, plus .git). Files the new source has there are
    # still copied in; nothing else in them is removed. ("dir/***" is the directory
    # and everything in it; a pattern without "/" matches at any depth.)
    local -a keep=('/models/***' '/data/***' '/out/***' '/build/***' '/third_party/***'
                   '/runtime/build/***' '/webapp/certs/***' '/webapp/uploads/***'
                   '/webapp/cache/***' /webapp/server.log '/.venv/***' 'venv/***' '/.git/***'
                   '__pycache__/***' '.pytest_cache/***' '*.py[cod]' '*.onnx' '*_mpk.tar.gz'
                   '*.ckpt*' '*.pem' '*.key' '.env' /.models-dir '.vscode/***' '.idea/***' '*.swp' '*~')
    if command -v rsync >/dev/null 2>&1; then
      local -a filters=()
      for f in "${keep[@]}"; do filters+=( --filter="P ${f}" ); done
      rsync -a --delete "${filters[@]}" "${tmp}/src/" "${dir}/" || fail "copying the new source over ${dir} failed."
    else
      # Without rsync nothing is removed: files the new source dropped stay behind.
      cp -a "${tmp}/src/." "${dir}/" || fail "copying the new source over ${dir} failed."
    fi
  else
    mkdir -p "${dir}"
    tar -xzf "${tmp}/src.tar.gz" -C "${dir}" --strip-components=1 || fail "extract failed."
  fi
  chmod +x "${dir}"/*.sh "${dir}"/webapp/*.sh "${dir}"/tools/*.py 2>/dev/null || true

  if [[ "${update}" == "1" ]]; then
    say "Updated ${dir}"
  else
    say "Installed in ${dir}"
  fi
  if [[ "${STUDIO_SETUP:-0}" == "1" ]]; then
    say "Running setup…"
    ( cd "${dir}" && ./setup.sh )
  elif [[ "${update}" != "1" ]]; then
    say ""
    say "Next:"
    say "  cd $(printf '%q' "${dir}")"
    say "  ./setup.sh        # builds the MLA runtime, fetches a model, makes the HTTPS certificate, adds the neat-dino alias"
    say "  neat-dino         # in a new shell: the studio at https://<board-ip>:8092 (or: webapp/run.sh)"
  fi
}

# "; exit": an update may replace this file while bash still reads it (the cp fallback
# writes in place), so nothing after main may be read from the new file.
main "$@"; exit $?
