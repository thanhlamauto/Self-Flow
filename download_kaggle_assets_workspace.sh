#!/usr/bin/env bash
set -euo pipefail

# Download the ImageNet latent datasets and VAE/Inception model assets into /workspace.
# Run this on each VM/worker that needs local copies.
#
# Auth:
#   export KAGGLE_API_TOKEN='KGAT_...'
#   ./download_kaggle_assets_workspace.sh

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
DATASET_DIR="${DATASET_DIR:-${WORKSPACE_DIR}/datasets}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_DIR}/models}"

TRAIN_LATENTS_DIR="${TRAIN_LATENTS_DIR:-${DATASET_DIR}/imagenet-vae-latents-ar-v2}"
VAL_LATENTS_DIR="${VAL_LATENTS_DIR:-${DATASET_DIR}/imagenet-vae-latents-train-v3}"
VAE_DIR="${VAE_DIR:-${MODEL_DIR}/sdvae-ema-flax-default-1}"
INCEPTION_DIR="${INCEPTION_DIR:-${MODEL_DIR}/inception-v3-pytorch-default-1}"

if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ]]; then
  cat >&2 <<'EOF'
Kaggle credentials not found.

Set a token in the shell before running this script:
  export KAGGLE_API_TOKEN='KGAT_...'

Do not commit the token into the repo.
EOF
  exit 2
fi

if ! command -v kaggle >/dev/null 2>&1; then
  python3 -m pip install -U kaggle
fi

mkdir -p "${TRAIN_LATENTS_DIR}" "${VAL_LATENTS_DIR}" "${VAE_DIR}" "${INCEPTION_DIR}"

echo "[download] train latents -> ${TRAIN_LATENTS_DIR}"
kaggle datasets download \
  -d thaygiaodaysat/imagenet-vae-latents-ar-v2/1 \
  -p "${TRAIN_LATENTS_DIR}" \
  --unzip

echo "[download] val latents -> ${VAL_LATENTS_DIR}"
kaggle datasets download \
  -d thaygiaodaysat/imagenet-vae-latents-train-v3 \
  -p "${VAL_LATENTS_DIR}" \
  --unzip

download_model_version() {
  local model_ref="$1"
  local out_dir="$2"

  echo "[download] model ${model_ref} -> ${out_dir}"
  kaggle models instances versions download \
    "${model_ref}" \
    -p "${out_dir}" \
    --untar \
    -f
}

download_model_version damtrunghieu/sdvae-ema/Flax/default/1 "${VAE_DIR}"
download_model_version ctlcmleon/inception-v3/PyTorch/default/1 "${INCEPTION_DIR}"

echo "[download] done"
echo "TRAIN_LATENTS_DIR=${TRAIN_LATENTS_DIR}"
echo "VAL_LATENTS_DIR=${VAL_LATENTS_DIR}"
echo "VAE_DIR=${VAE_DIR}"
echo "INCEPTION_DIR=${INCEPTION_DIR}"
