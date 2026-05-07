#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CKPT_DIR_PROVIDED="${CKPT_DIR+x}"
source "${REPO_ROOT}/configs/lara_imagenet256_l.env"
if [[ -z "${CKPT_DIR_PROVIDED}" ]]; then
  CKPT_DIR="${REPO_ROOT}/checkpoints/lara_imagenet256_xl"
fi
cd "${REPO_ROOT}"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    PYTHON=python
  fi
fi

ARGS=(
  -u train.py
  --resume
  --model-size XL
  --epochs 2000
  --vae-model "${VAE_MODEL}"
  --data-path "${TRAIN_DATA_PATH}"
  --val-data-path "${VAL_DATA_PATH}"
  --wandb-project "${WANDB_PROJECT}"
  --shortcut-predictor hybrid_depth30m
  --shortcut-predictor-hidden-size 480
  --shortcut-predictor-depth 12
  --shortcut-predictor-num-heads 8
  --shortcut-predictor-dilation-cycle 1,2,4
  --fid-steps 50000,2000000
  --ckpt-keep-steps 100000,200000,400000,800000,1000000,2000000
  --ckpt-dir "${CKPT_DIR}"
)

if [[ -n "${INCEPTION_SCORE_WEIGHTS}" ]]; then
  ARGS+=(--inception-score-weights "${INCEPTION_SCORE_WEIGHTS}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf "%q " "${PYTHON}" "${ARGS[@]}" "$@"
  printf "\n"
  exit 0
fi

exec "${PYTHON}" "${ARGS[@]}" "$@"
