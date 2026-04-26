#!/usr/bin/env bash
set -euo pipefail

# Stochastic skip-in-the-loop ablations for SiT-B/2 + P-Small.
# Each run keeps direction-magnitude training enabled and adds skip-FM:
#   L += lambda_skip * L_skip-FM
#
# Recommended first run:
#   ./run_depth_shortcut_skip_in_loop_ablation.sh main
#
# Full local sweep:
#   ./run_depth_shortcut_skip_in_loop_ablation.sh all

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

DIRECTION_MAG_SKIP_ARGS=(
  --shortcut-predictor small
  --shortcut-training-mode direction-magnitude-skip
  --shortcut-lambda-dir 0.5
  --shortcut-lambda-boot 0.25
  --shortcut-lambda-mag 0.375
  --shortcut-lambda-boot-mag 0.1875
  --shortcut-mag-scale 3.0
  --shortcut-mag-abs-center 5.5
  --shortcut-mag-abs-scale 1.5
  --shortcut-mag-clip-min 3.0
  --shortcut-mag-clip-max 8.0
  --shortcut-skip-in-loop-warmup-steps 5000
  --shortcut-skip-in-loop-detach-source
)

run_ablation() {
  local name="$1"
  local skip_prob="$2"
  local lambda_skip="$3"
  local skip_gap="$4"

  python train.py \
    "${COMMON_ARGS[@]}" \
    "${DIRECTION_MAG_SKIP_ARGS[@]}" \
    --ckpt-dir "./checkpoints-depth-shortcut-small-skiploop-${name}" \
    --shortcut-skip-in-loop-prob "${skip_prob}" \
    --shortcut-lambda-skip-fm "${lambda_skip}" \
    --shortcut-skip-in-loop-gap "${skip_gap}"
}

case "${1:-main}" in
  main)
    run_ablation p010_l010_g2 0.10 0.10 2
    ;;
  all)
    run_ablation p005_l010_g2 0.05 0.10 2
    run_ablation p010_l010_g2 0.10 0.10 2
    run_ablation p020_l010_g2 0.20 0.10 2
    run_ablation p010_l005_g2 0.10 0.05 2
    run_ablation p010_l020_g2 0.10 0.20 2
    run_ablation p010_l010_g3 0.10 0.10 3
    ;;
  *)
    echo "Usage: $0 [main|all]" >&2
    exit 2
    ;;
esac
