#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --model-size B
  --batch-size 128
  --epochs 400
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
)

run_variant() {
  local predictor="$1"
  local ckpt_dir="$2"

  python train.py \
    "${COMMON_ARGS[@]}" \
    --ckpt-dir "${ckpt_dir}" \
    --shortcut-predictor "${predictor}"
}

run_variant tiny ./checkpoints-depth-shortcut-ptiny
run_variant small ./checkpoints-depth-shortcut-psmall
