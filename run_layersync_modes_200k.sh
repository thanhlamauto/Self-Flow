#!/usr/bin/env bash
set -euo pipefail

# Runs every currently supported LayerSync mode for 200k iterations each.
# steps = epochs * steps_per_epoch = 200 * 1000.

COMMON_ARGS=(
  --model-size B
  --batch-size 128
  --epochs 200
  --steps-per-epoch 1000
  --learning-rate 1e-4
  --vae-model /kaggle/input/models/damtrunghieu/sdvae-ema/flax/default/1
  --data-path /kaggle/input/datasets/thaygiaodaysat/imagenet-vae-latents-ar-v2
  --val-data-path /kaggle/input/datasets/thaygiaodaysat/imagenet-vae-latents-train-v3
  --grad-clip 1.0
  --log-freq 100
  --eval-freq 1000
  --eval-batches 4
  --sample-freq 5000
  --sample-num-steps 50
  --sample-cfg-scale 1.0
  --fid-freq 25000
  --num-fid-samples 4096
  --fid-batch-size 256
  --fid-eval-local-batch 32
  --fid-num-steps 50
  --fid-cfg-scale 1.0
  --vae-decode-batch-size 256
  --no-linear-probe
  --inception-score-weights /kaggle/input/models/ctlcmleon/inception-v3/pytorch/default/1/inception_v3_google-0cc3c7bd.pth
  --block-corr-freq 25000
  --block-corr-batches 2
  --preflight-checks
  --preflight-fid-memory-probe
  --wandb-project selfflow-jax
  --ema-decay 0.9999
  --layersync-lambda 0.2
)

MODE_NAMES=(
  "unused"
  "random-clean"
  "fixed-clean"
  "fixed-internal"
  "random-clean-residual"
  "fixed-clean-residual"
)

for MODE in 1 2 3 4 5; do
  MODE_NAME="${MODE_NAMES[$MODE]}"
  RUN_NAME="layersync-mode-${MODE}-${MODE_NAME}-200k"
  CKPT_DIR="./checkpoints/${RUN_NAME}"
  MODE_ARGS=(--layersync-mode "${MODE}")
  if [[ "${MODE}" == "1" || "${MODE}" == "4" ]]; then
    MODE_ARGS+=(--layersync-delta 6)
  else
    MODE_ARGS+=(--layersync-weak-layer 4 --layersync-strong-layer 8)
  fi

  echo "[$(date -Is)] Starting ${RUN_NAME}"
  WANDB_NAME="${RUN_NAME}" python train.py \
    "${COMMON_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    --ckpt-dir "${CKPT_DIR}"
  echo "[$(date -Is)] Finished ${RUN_NAME}"
done
