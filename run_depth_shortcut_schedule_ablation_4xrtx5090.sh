#!/usr/bin/env bash
set -euo pipefail

# Single-node JAX/CUDA launcher for the depth-shortcut schedule ablation on 4xRTX5090.
#
# Usage:
#   DATA_PATH=/path/to/train_ar VAL_DATA_PATH=/path/to/val_ar VAE_MODEL=/path/to/sdvae \
#     ./run_depth_shortcut_schedule_ablation_4xrtx5090.sh logit_centered [L|B|XL]
#
# Tuning knobs:
#   GPU_BATCH_SIZE=256            global train batch; 256 => local batch 64 on 4 GPUs
#   FID_EVAL_LOCAL_BATCH=4        per-GPU FID/Inception microbatch
#   VAE_DECODE_BATCH_SIZE=32      VAE decode chunk size
#   NUM_FID_SAMPLES=4096          monitoring FID sample count
#   EXPECTED_DEVICES=4            fail fast if JAX sees a different local device count
#   RESUME=1                      resume from ckpt_dir/latest

usage() {
  cat <<'EOF'
Usage: ./run_depth_shortcut_schedule_ablation_4xrtx5090.sh <case> [model_size]

Cases:
  control          uniform timestep + trunc_normal pairs
  logit_only       logit_normal timestep + trunc_normal pairs
  centered_only    uniform timestep + trunc_normal_centered pairs
  logit_centered   logit_normal timestep + trunc_normal_centered pairs
  all              run all cases sequentially

Model sizes:
  L                SiT-L + hybrid_depth30m depth 11 (default)
  B                SiT-B + hybrid_deep_10
  XL               SiT-XL + hybrid_depth30m width 480 depth 12 heads 8
EOF
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ -n "${JAX_COMPILATION_CACHE_DIR:-}" ]]; then
  mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
fi

EXPECTED_DEVICES="${EXPECTED_DEVICES:-4}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-256}"
FID_EVAL_LOCAL_BATCH="${FID_EVAL_LOCAL_BATCH:-4}"
VAE_DECODE_BATCH_SIZE="${VAE_DECODE_BATCH_SIZE:-32}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-4096}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-128}"
FID_STEPS="${FID_STEPS:-50000,20000000}"
CKPT_LATEST_FREQ="${CKPT_LATEST_FREQ:-10000}"
CKPT_ROOT="${CKPT_ROOT:-/home/nguyenthanhlam/Self-Flow/checkpoints-4xrtx5090}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-0}"

DATA_PATH="${DATA_PATH:-/home/nguyenthanhlam/kaggle_downloads/imagenet-vae-latents-ar-v2}"
VAL_DATA_PATH="${VAL_DATA_PATH:-/home/nguyenthanhlam/kaggle_downloads/imagenet-vae-latents-train-v3}"
VAE_MODEL="${VAE_MODEL:-/home/nguyenthanhlam/kaggle_downloads/models/sdvae-ema-flax-default-1}"
VAE_HF_CONFIG="${VAE_HF_CONFIG:-}"
INCEPTION_SCORE_WEIGHTS="${INCEPTION_SCORE_WEIGHTS:-/home/nguyenthanhlam/kaggle_downloads/models/inception-v3-pytorch-default-1/inception_v3_google-0cc3c7bd.pth}"

COMMON_ARGS=(
  --batch-size "${GPU_BATCH_SIZE}"
  --epochs 400
  --steps-per-epoch 1000
  --learning-rate 1e-4
  --predictor-learning-rate 1e-4
  --vae-model "${VAE_MODEL}"
  --data-path "${DATA_PATH}"
  --val-data-path "${VAL_DATA_PATH}"
  --grad-clip 1.0
  --log-freq 1000
  --eval-freq 20000
  --eval-batches 1
  --sample-freq 0
  --sample-num-steps 50
  --sample-cfg-scale 1.0
  --fid-freq 0
  --fid-steps "${FID_STEPS}"
  --num-fid-samples "${NUM_FID_SAMPLES}"
  --fid-batch-size "${FID_BATCH_SIZE}"
  --fid-eval-local-batch "${FID_EVAL_LOCAL_BATCH}"
  --fid-num-steps 50
  --fid-cfg-scale 1.0
  --vae-decode-batch-size "${VAE_DECODE_BATCH_SIZE}"
  --no-linear-probe
  --block-corr-freq 0
  --block-corr-batches 2
  --preflight-checks
  --preflight-fid-memory-probe
  --expected-device-count "${EXPECTED_DEVICES}"
  --expected-backend cuda
  --wandb-project selfflow-jax
  --official-training-mode
  --ckpt-latest-freq "${CKPT_LATEST_FREQ}"
  --ckpt-verify-step 1000
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

if [[ -n "${VAE_HF_CONFIG}" ]]; then
  COMMON_ARGS+=(--vae-hf-config "${VAE_HF_CONFIG}")
fi
if [[ -n "${INCEPTION_SCORE_WEIGHTS}" ]]; then
  COMMON_ARGS+=(--inception-score-weights "${INCEPTION_SCORE_WEIGHTS}")
fi
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "yes" ]]; then
  COMMON_ARGS+=(--resume)
fi

MODEL_ARGS=()
MODEL_SUFFIX=""

configure_model_args() {
  local model_size="${1:-L}"
  model_size="$(printf '%s' "${model_size}" | tr '[:lower:]' '[:upper:]')"

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

  local run_name="depth-schedule-ab-4x5090-${MODEL_SUFFIX}-${ckpt_suffix}-gb${GPU_BATCH_SIZE}"
  local ckpt_dir="${CKPT_ROOT}/checkpoints-${run_name}"

  echo "Running ${case_name}: model=${MODEL_SUFFIX}, timestep=${timestep_mode}, pair=${pair_mode}, batch=${GPU_BATCH_SIZE}"
  WANDB_RUN_GROUP="depth-schedule-ablation-4xrtx5090" \
  WANDB_NAME="${run_name}" \
  "${PYTHON_BIN}" train.py \
    "${MODEL_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    --timestep-sampling-mode "${timestep_mode}" \
    --direct-pair-mode "${pair_mode}" \
    --output-distill-pair-mode "${pair_mode}" \
    --ckpt-dir "${ckpt_dir}"
}

if (($# == 1)) && [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi

if (($# < 1 || $# > 2)); then
  usage
  exit 2
fi

configure_model_args "${2:-L}"

if [[ "$1" == "all" ]]; then
  for case_name in control logit_only centered_only logit_centered; do
    run_case "${case_name}"
  done
else
  run_case "$1"
fi
