# 4xRTX5090 JAX Runbook

This branch keeps training in JAX/Flax and adds a single-node CUDA launcher for
the depth-shortcut schedule ablation.

## Install Notes

Use a recent NVIDIA driver and a JAX CUDA wheel that supports the installed
CUDA runtime. For RTX 5090, prefer a modern CUDA 12/13 stack.

Example:

```bash
pip install -U "jax[cuda12]" flax optax grain-balsa array-record wandb
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify JAX sees all four GPUs before training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 JAX_PLATFORMS=cuda python - <<'PY'
import jax
print(jax.default_backend())
print(jax.local_device_count())
print(jax.local_devices())
PY
```

## SiT-L Schedule Ablation

The launcher defaults to SiT-L, global batch 256, four expected local devices,
preflight memory checks, sparse production logging, and smaller FID/VAE
microbatches than the TPU/Kaggle script.

```bash
DATA_PATH=/path/to/imagenet-vae-latents-ar-v2 \
VAL_DATA_PATH=/path/to/imagenet-vae-latents-train-v3 \
VAE_MODEL=/path/to/sdvae-ema-flax-default-1 \
INCEPTION_SCORE_WEIGHTS=/path/to/inception_v3_google-0cc3c7bd.pth \
./run_depth_shortcut_schedule_ablation_4xrtx5090.sh logit_centered L
```

If the preflight train step OOMs, retry with:

```bash
GPU_BATCH_SIZE=128 ./run_depth_shortcut_schedule_ablation_4xrtx5090.sh logit_centered L
```

If training passes but FID/VAE/Inception OOMs, keep the train batch and reduce:

```bash
FID_EVAL_LOCAL_BATCH=2 VAE_DECODE_BATCH_SIZE=16 \
./run_depth_shortcut_schedule_ablation_4xrtx5090.sh logit_centered L
```

Resume from the latest checkpoint bundle with:

```bash
RESUME=1 ./run_depth_shortcut_schedule_ablation_4xrtx5090.sh logit_centered L
```
