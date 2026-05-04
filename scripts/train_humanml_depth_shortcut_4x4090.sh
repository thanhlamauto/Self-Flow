#!/usr/bin/env bash
set -euo pipefail

# 4x4090 DDP launch.
# Global batch = 4 GPUs * local batch 16 = 64, matching the reference run.

DATA_DIR="${DATA_DIR:-$PWD/dataset/HumanML3D}"
SAVE_DIR="${SAVE_DIR:-./save/humanml_mdm_depth_shortcut_hybrid_deep_12pct_outputdistill}"
NPROC="${NPROC:-4}"

torchrun --standalone --nproc_per_node="${NPROC}" -m train.train_mdm \
  --save_dir "${SAVE_DIR}" \
  --overwrite \
  --dataset humanml \
  --data_dir "${DATA_DIR}" \
  --arch trans_enc \
  --text_encoder_type clip \
  --layers 8 \
  --latent_dim 512 \
  --cond_mask_prob 0.1 \
  --mask_frames \
  --lambda_rcxyz 0.0 \
  --lambda_vel 0.0 \
  --lambda_fc 0.0 \
  --lambda_target_loc 0.0 \
  --use-depth-shortcut \
  --shortcut-predictor hybrid_deep_12pct \
  --shortcut-training-mode direction-magnitude \
  --shortcut-lambda-dir 1.0 \
  --shortcut-lambda-boot 0.25 \
  --shortcut-lambda-mag 0.375 \
  --shortcut-lambda-boot-mag 0.1875 \
  --shortcut-bootstrap-detach-source \
  --shortcut-mag-scale 2.2 \
  --shortcut-mag-abs-center 2.9 \
  --shortcut-mag-abs-scale 0.6 \
  --shortcut-mag-clip-min 0.8 \
  --shortcut-mag-clip-max 3.5 \
  --shortcut-skip-in-loop-prob 0.0 \
  --shortcut-lambda-skip-fm 0.0 \
  --shortcut-skip-in-loop-gap-mode truncated-normal \
  --shortcut-skip-in-loop-max-gap 7 \
  --shortcut-skip-in-loop-gap-loc 2.0 \
  --shortcut-skip-in-loop-gap-sigma 1.5 \
  --predictor-learning-rate 1e-4 \
  --shortcut-predictor-weight-decay 0.1 \
  --shortcut-predictor-ema-decay 0.999 \
  --no-shortcut-predictor-normalize-input \
  --shortcut-predictor-use-timestep \
  --output-distill \
  --output-distill-ratio 0.10 \
  --lambda-output-distill 0.05 \
  --output-distill-every 1 \
  --output-distill-update-mode predictor_plus_all \
  --output-distill-pair-mode trunc_normal_centered \
  --direct-pair-mode trunc_normal_centered \
  --pair-center-sigma 1.3 \
  --direct-num-pairs 1 \
  --direct-joint-pairs 1 \
  --direct-predictor-only-pairs 0 \
  --private-loss \
  --lambda-private 1.0 \
  --private-max-pairs 2 \
  --private-use-residual \
  --private-cosine-mode bnd \
  --private-pair-mode random \
  --fid-skip-timestep-mode alternate \
  --timestep-sampling-mode logit_normal \
  --timestep-logit-mean 0.0 \
  --timestep-logit-std 1.0 \
  --batch_size 16 \
  --lr 1e-4 \
  --grad-clip 1.0 \
  --weight_decay 0.1 \
  --lr_anneal_steps 0 \
  --num_steps 200000 \
  --diffusion_steps 50 \
  --noise_schedule cosine \
  --sigma_small True \
  --eval_batch_size 32 \
  --eval_split val \
  --eval_during_training \
  --eval_rep_times 3 \
  --eval_num_samples 1000 \
  --eval_steps 50k,100k,200k \
  --eval_metrics fid,precision \
  --log_interval 50 \
  --save_interval 50000 \
  --num_frames 60 \
  --resume_checkpoint "" \
  --train_platform_type WandBPlatform \
  --gen_guidance_param 2.5 \
  --avg_model_beta 0.9999 \
  --adam_beta2 0.999 \
  --seed 10 \
  --cuda True \
  --device 0
