#!/usr/bin/env bash
set -euo pipefail

# 2x2 ablation for the two schedules that may be driving the recent gain:
#   1. timestep sampling: uniform vs logit_normal
#   2. layer-pair sampling: trunc_normal vs trunc_normal_centered
#
# Usage:
#   ./run_depth_shortcut_schedule_ablation.sh control [B|L|XL]
#   ./run_depth_shortcut_schedule_ablation.sh logit_only [B|L|XL]
#   ./run_depth_shortcut_schedule_ablation.sh centered_only [B|L|XL]
#   ./run_depth_shortcut_schedule_ablation.sh logit_centered [B|L|XL]
#   ./run_depth_shortcut_schedule_ablation.sh all [B|L|XL]

usage() {
  cat <<'EOF'
Usage: ./run_depth_shortcut_schedule_ablation.sh <case> [model_size]

Cases:
  control          uniform timestep + trunc_normal pairs
  logit_only       logit_normal timestep + trunc_normal pairs
  centered_only    uniform timestep + trunc_normal_centered pairs
  logit_centered   logit_normal timestep + trunc_normal_centered pairs
  all              run all cases sequentially

Model sizes:
  B                hybrid_deep_10 baseline predictor
  L                hybrid_depth30m, depth 11, ~26M predictor
  XL               hybrid_depth30m, width 480, depth 12, heads 8, ~32M predictor
EOF
}

COMMON_ARGS=(
  --batch-size 128
  --epochs 400
  --steps-per-epoch 1000
  --learning-rate 1e-4
  --predictor-learning-rate 1e-4
  --vae-model /kaggle/input/models/damtrunghieu/sdvae-ema/flax/default/1
  --data-path /kaggle/input/datasets/thaygiaodaysat/imagenet-vae-latents-ar-v2
  --val-data-path /kaggle/input/datasets/thaygiaodaysat/imagenet-vae-latents-train-v3
  --grad-clip 1.0
  --log-freq 100
  --eval-freq 1000
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
  --wandb-project selfflow-jax
  --ema-decay 0.9999
  --shortcut-training-mode direction-magnitude
  --shortcut-lambda-dir 1
  --shortcut-lambda-boot 0.25
  --shortcut-lambda-mag 0.375
  --shortcut-lambda-boot-mag 0.1875
  --shortcut-mag-scale 3.0
  --shortcut-mag-abs-center 5.5
  --shortcut-mag-abs-scale 1.5
  --shortcut-mag-clip-min 3.0
  --shortcut-mag-clip-max 8.0
  --timestep-logit-mean 0.0
  --timestep-logit-std 1.0
  --shortcut-skip-in-loop-prob 0.0
  --shortcut-lambda-skip-fm 0.0
  --shortcut-skip-in-loop-gap-mode truncated-normal
  --shortcut-skip-in-loop-max-gap 10
  --shortcut-skip-in-loop-gap-loc 3.0
  --shortcut-skip-in-loop-gap-sigma 2.0
  --output-distill
  --output-distill-ratio 0.10
  --lambda-output-distill 0.05
  --output-distill-every 1
  --output-distill-update-mode predictor_plus_all
  --pair-center-sigma 2.0
  --direct-num-pairs 1
  --direct-joint-pairs 1
  --direct-predictor-only-pairs 0
  --shortcut-bootstrap-detach-source
  --private-loss
  --lambda-private 1.0
  --private-max-pairs 4
  --shortcut-predictor-use-timestep
  --shortcut-predictor-use-class-input
  --shortcut-predictor-class-fusion add
  --no-shortcut-predictor-normalize-input
  --private-use-residual
  --private-cosine-mode bnd
  --private-pair-mode random
  --fid-skip-timestep-mode alternate
  --weight-decay 0.1
  --shortcut-predictor-weight-decay 0.1
)

MODEL_ARGS=()
MODEL_SUFFIX=""

configure_model_args() {
  local model_size="${1:-B}"
  model_size="${model_size^^}"

  case "${model_size}" in
    B)
      MODEL_SUFFIX="b"
      MODEL_ARGS=(
        --model-size B
        --shortcut-predictor hybrid_deep_10
      )
      ;;
    L)
      MODEL_SUFFIX="l"
      MODEL_ARGS=(
        --model-size L
        --shortcut-predictor hybrid_depth30m
        --shortcut-predictor-depth 11
        --shortcut-predictor-dilation-cycle 1,2,4
      )
      ;;
    XL)
      MODEL_SUFFIX="xl"
      MODEL_ARGS=(
        --model-size XL
        --shortcut-predictor hybrid_depth30m
        --shortcut-predictor-hidden-size 480
        --shortcut-predictor-depth 12
        --shortcut-predictor-num-heads 8
        --shortcut-predictor-dilation-cycle 1,2,4
      )
      ;;
    *)
      echo "Unknown model size: ${model_size}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

run_case() {
  local case_name="$1"
  local timestep_mode
  local pair_mode
  local ckpt_suffix

  case "${case_name}" in
    control)
      timestep_mode="uniform"
      pair_mode="trunc_normal"
      ckpt_suffix="control-uniform-trunc"
      ;;
    logit_only)
      timestep_mode="logit_normal"
      pair_mode="trunc_normal"
      ckpt_suffix="logitnormal-trunc"
      ;;
    centered_only)
      timestep_mode="uniform"
      pair_mode="trunc_normal_centered"
      ckpt_suffix="uniform-centered"
      ;;
    logit_centered)
      timestep_mode="logit_normal"
      pair_mode="trunc_normal_centered"
      ckpt_suffix="logitnormal-centered"
      ;;
    *)
      echo "Unknown case: ${case_name}" >&2
      usage >&2
      exit 2
      ;;
  esac

  local run_name="depth-schedule-ab-${MODEL_SUFFIX}-${ckpt_suffix}"
  local ckpt_dir="/home/nguyenthanhlam/Self-Flow/checkpoints-${run_name}"

  echo "Running ${case_name}: model=${MODEL_SUFFIX}, timestep=${timestep_mode}, pair=${pair_mode}"
  WANDB_RUN_GROUP="depth-schedule-ablation" \
  WANDB_NAME="${run_name}" \
  python train.py \
    "${MODEL_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    --timestep-sampling-mode "${timestep_mode}" \
    --direct-pair-mode "${pair_mode}" \
    --output-distill-pair-mode "${pair_mode}" \
    --ckpt-dir "${ckpt_dir}"
}

if (($# < 1 || $# > 2)); then
  usage
  exit 2
fi

configure_model_args "${2:-B}"

if [[ "$1" == "all" ]]; then
  for case_name in control logit_only centered_only logit_centered; do
    run_case "${case_name}"
  done
else
  run_case "$1"
fi
