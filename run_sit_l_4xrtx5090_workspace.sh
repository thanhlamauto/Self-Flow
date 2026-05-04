#!/usr/bin/env bash
set -euo pipefail

# SiT-L 4xRTX5090 workspace launcher.
# This is the L-sized equivalent of the Kaggle SiT-B command, keeping JAX/Flax
# training and applying the L-specific depth/magnitude schedule values.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_LATENTS_DIR="${TRAIN_LATENTS_DIR:-${WORKSPACE_DIR}/datasets/imagenet-vae-latents-ar-v2}"
VAL_LATENTS_DIR="${VAL_LATENTS_DIR:-${WORKSPACE_DIR}/datasets/imagenet-vae-latents-train-v3}"
VAE_MODEL="${VAE_MODEL:-${WORKSPACE_DIR}/models/sdvae-ema-flax-default-1}"
INCEPTION_SCORE_WEIGHTS="${INCEPTION_SCORE_WEIGHTS:-${WORKSPACE_DIR}/models/inception-v3-pytorch-default-1/inception_v3_google-0cc3c7bd.pth}"
CKPT_DIR="${CKPT_DIR:-${WORKSPACE_DIR}/checkpoints/depth-shortcut-L-hybrid-depth30m-outputdistill-r010-l005-classcond-centered-logitnormal-4x5090}"

GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-256}"
FID_EVAL_LOCAL_BATCH="${FID_EVAL_LOCAL_BATCH:-32}"
VAE_DECODE_BATCH_SIZE="${VAE_DECODE_BATCH_SIZE:-256}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-256}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
FID_NUM_STEPS="${FID_NUM_STEPS:-250}"
EXPECTED_DEVICES="${EXPECTED_DEVICES:-4}"

RESUME_ARGS=()
if [[ "${RESUME:-1}" == "1" || "${RESUME:-1}" == "true" || "${RESUME:-1}" == "yes" ]]; then
  RESUME_ARGS=(--resume)
fi

"${PYTHON_BIN}" train.py \
  "${RESUME_ARGS[@]}" \
  --model-size L \
  --batch-size "${GPU_BATCH_SIZE}" \
  --epochs 400 \
  --steps-per-epoch 1000 \
  --learning-rate 1e-4 \
  --predictor-learning-rate 1e-4 \
  --vae-model "${VAE_MODEL}" \
  --data-path "${TRAIN_LATENTS_DIR}" \
  --val-data-path "${VAL_LATENTS_DIR}" \
  --grad-clip 1.0 \
  --weight-decay 0.1 \
  --ema-decay 0.9999 \
  --log-freq 1000 \
  --eval-freq 20000 \
  --eval-batches 1 \
  --sample-freq 0 \
  --sample-num-steps 50 \
  --sample-cfg-scale 1.0 \
  --fid-steps 50000,100000,200000,400000 \
  --num-fid-samples "${NUM_FID_SAMPLES}" \
  --fid-batch-size "${FID_BATCH_SIZE}" \
  --fid-eval-local-batch "${FID_EVAL_LOCAL_BATCH}" \
  --fid-num-steps "${FID_NUM_STEPS}" \
  --fid-cfg-scale 1.0 \
  --vae-decode-batch-size "${VAE_DECODE_BATCH_SIZE}" \
  --no-linear-probe \
  --inception-score-weights "${INCEPTION_SCORE_WEIGHTS}" \
  --block-corr-freq 0 \
  --cfg-dropout-rate 0.1 \
  --wandb-project selfflow-jax \
  --shortcut-predictor hybrid_depth30m \
  --shortcut-predictor-depth 11 \
  --shortcut-predictor-dilation-cycle 1,2,4 \
  --shortcut-predictor-use-timestep \
  --shortcut-predictor-use-class-input \
  --shortcut-predictor-class-fusion add \
  --no-shortcut-predictor-normalize-input \
  --shortcut-predictor-weight-decay 0.1 \
  --shortcut-training-mode direction-magnitude \
  --shortcut-lambda-dir 1 \
  --shortcut-lambda-boot 0.25 \
  --shortcut-lambda-mag 0.375 \
  --shortcut-lambda-boot-mag 0.1875 \
  --shortcut-mag-scale 3.3 \
  --shortcut-mag-abs-center 5.4 \
  --shortcut-mag-abs-scale 1.0 \
  --shortcut-mag-clip-min 3.3 \
  --shortcut-mag-clip-max 7.3 \
  --shortcut-bootstrap-detach-source \
  --shortcut-skip-in-loop-prob 0.0 \
  --shortcut-lambda-skip-fm 0.0 \
  --shortcut-skip-in-loop-gap-mode truncated-normal \
  --shortcut-skip-in-loop-max-gap 20 \
  --shortcut-skip-in-loop-gap-loc 6.0 \
  --shortcut-skip-in-loop-gap-sigma 4.0 \
  --timestep-sampling-mode logit_normal \
  --timestep-logit-mean 0.0 \
  --timestep-logit-std 1.0 \
  --output-distill \
  --output-distill-ratio 0.10 \
  --lambda-output-distill 0.05 \
  --output-distill-every 1 \
  --output-distill-update-mode predictor_plus_all \
  --output-distill-pair-mode trunc_normal_centered \
  --direct-pair-mode trunc_normal_centered \
  --pair-center-sigma 4.0 \
  --direct-num-pairs 1 \
  --direct-joint-pairs 1 \
  --direct-predictor-only-pairs 0 \
  --private-loss \
  --lambda-private 1.0 \
  --private-max-pairs 4 \
  --private-use-residual \
  --private-cosine-mode bnd \
  --private-pair-mode random \
  --ckpt-keep-steps 100000,200000,400000 \
  --ckpt-verify-step 0 \
  --ckpt-latest-freq 5000 \
  --no-fid-skip-eval \
  --expected-device-count "${EXPECTED_DEVICES}" \
  --expected-backend cuda \
  --ckpt-dir "${CKPT_DIR}"
