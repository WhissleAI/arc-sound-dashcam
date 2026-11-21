#!/usr/bin/env bash
# Download the Arc Sound model files into ./model (or $1) and check their SHA-256 sums.
#   scripts/fetch_model.sh [DEST_DIR]
# Sources, in order:
#   1. GitHub release assets of WhissleAI/arc-sound-dashcam (tag $ARCSOUND_TAG, default v1.0.0), with the gh CLI
#      (gh auth login) or $GITHUB_TOKEN
#   2. Hugging Face $HF_REPO (default WhissleAI/arc-sound-dashcam) at $HF_REVISION (default main), with $HF_TOKEN
set -euo pipefail
GH_REPO="${GH_REPO:-WhissleAI/arc-sound-dashcam}"
TAG="${ARCSOUND_TAG:-v1.0.0}"
HF_REPO="${HF_REPO:-WhissleAI/arc-sound-dashcam}"
REV="${HF_REVISION:-main}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/model}"
FILES=(arcsound.safetensors arcsound_meta.json SHA256SUMS)
mkdir -p "$DEST"

from_github() {
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    local f
    for f in "${FILES[@]}"; do rm -f "$DEST/$f"; gh release download "$TAG" -R "$GH_REPO" -D "$DEST" -p "$f" || return 1; done
    return 0
  fi
  [ -n "${GITHUB_TOKEN:-}" ] || return 1
  local api="https://api.github.com/repos/${GH_REPO}/releases/tags/${TAG}" json id f
  json=$(curl -fsSL -H "Authorization: Bearer ${GITHUB_TOKEN}" "$api") || return 1
  for f in "${FILES[@]}"; do
    id=$(printf '%s' "$json" | python3 -c "import json,sys; print(next(a['id'] for a in json.load(sys.stdin)['assets'] if a['name']=='$f'))") || return 1
    curl -fsSL --retry 3 -H "Authorization: Bearer ${GITHUB_TOKEN}" -H "Accept: application/octet-stream" \
      -o "$DEST/$f" "https://api.github.com/repos/${GH_REPO}/releases/assets/${id}" || return 1
  done
}

from_hf() {
  [ -n "${HF_TOKEN:-}" ] || return 1
  local f
  for f in "${FILES[@]}"; do
    curl -fsSL --retry 3 -H "Authorization: Bearer ${HF_TOKEN}" -o "$DEST/$f.part" "https://huggingface.co/${HF_REPO}/resolve/${REV}/$f" \
      && mv "$DEST/$f.part" "$DEST/$f" || { rm -f "$DEST/$f.part"; return 1; }
  done
}

if from_github; then echo "fetched from GitHub release ${GH_REPO}@${TAG}"
elif from_hf; then echo "fetched from Hugging Face ${HF_REPO}@${REV}"
else
  echo "could not fetch the model: log in with 'gh auth login' (or set GITHUB_TOKEN) for the GitHub release," >&2
  echo "or set HF_TOKEN for the Hugging Face repo; both are private" >&2
  exit 1
fi
cd "$DEST"
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
echo "model files in $DEST"
