#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CKPT_DIR_PROVIDED="${CKPT_DIR+x}"
source "${REPO_ROOT}/configs/lara_imagenet256_l.env"
if [[ -z "${CKPT_DIR_PROVIDED}" ]]; then
  CKPT_DIR="${REPO_ROOT}/checkpoints/lara_imagenet256_l"
fi
cd "${REPO_ROOT}"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    PYTHON=python
  fi
fi

SAMPLE_DIR="${SAMPLE_DIR:-${REPO_ROOT}/outputs/qualitative_samples}"
MODEL_SIZE="${MODEL_SIZE:-L}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-16}"
SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-250}"
SAMPLE_CFG_SCALE="${SAMPLE_CFG_SCALE:-4.0}"
SAMPLE_SEED="${SAMPLE_SEED:-31}"

ARGS=(
  -u sample.py
  --ckpt "${CKPT_DIR}/latest"
  --output-dir "${SAMPLE_DIR}"
  --num-fid-samples "${NUM_SAMPLES}"
  --batch-size "${SAMPLE_BATCH_SIZE}"
  --num-steps "${SAMPLE_NUM_STEPS}"
  --cfg-scale "${SAMPLE_CFG_SCALE}"
  --seed "${SAMPLE_SEED}"
  --model-size "${MODEL_SIZE}"
  --vae-model "${VAE_MODEL}"
  --use-ema
)

if [[ -n "${CLASS_IDS:-}" ]]; then
  ARGS+=(--class-ids "${CLASS_IDS}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf "%q " "${PYTHON}" "${ARGS[@]}" "$@"
  printf "\n"
  exit 0
fi

exec "${PYTHON}" "${ARGS[@]}" "$@"
