# VM Codex Guide: SiT-L on 4xRTX5090

Use this on each `/workspace` VM/worker. Keep training in JAX/Flax.

## 1. Set Kaggle Auth

Do not write the Kaggle token into files or commits. Put it only in the shell:

```bash
export KAGGLE_API_TOKEN='KGAT_...'
```

## 2. Install Runtime

```bash
cd /workspace/Self-Flow-depth-shortcut-output-distill
python3 -m pip install -U "jax[cuda12]" flax optax grain-balsa array-record wandb kaggle
python3 -m pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -r requirements.txt
```

## 3. Download Kaggle Assets To `/workspace`

Run this on both workers so each has local copies:

```bash
cd /workspace/Self-Flow-depth-shortcut-output-distill
./download_kaggle_assets_workspace.sh
```

It downloads:

- `thaygiaodaysat/imagenet-vae-latents-ar-v2` to `/workspace/datasets/imagenet-vae-latents-ar-v2`
- `thaygiaodaysat/imagenet-vae-latents-train-v3` to `/workspace/datasets/imagenet-vae-latents-train-v3`
- `damtrunghieu/sdvae-ema/Flax/default/1` to `/workspace/models/sdvae-ema-flax-default-1`
- `ctlcmleon/inception-v3/PyTorch/default/1` to `/workspace/models/inception-v3-pytorch-default-1`

## 4. Verify 4 GPUs In JAX

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 JAX_PLATFORMS=cuda python3 - <<'PY'
import jax
print(jax.default_backend())
print(jax.local_device_count())
print(jax.local_devices())
PY
```

Expected: backend `gpu`/CUDA and `4` local devices.

## 5. Run SiT-L

```bash
cd /workspace/Self-Flow-depth-shortcut-output-distill
PYTHON_BIN=python3 ./run_sit_l_4xrtx5090_workspace.sh
```

The launcher is the L-sized version of the Kaggle B command:

- `--model-size L`
- `--shortcut-predictor hybrid_depth30m`
- `--shortcut-predictor-depth 11`
- `--shortcut-predictor-dilation-cycle 1,2,4`
- `--shortcut-skip-in-loop-max-gap 20`
- `--shortcut-skip-in-loop-gap-loc 6.0`
- `--shortcut-skip-in-loop-gap-sigma 4.0`
- `--pair-center-sigma 4.0`
- `--shortcut-mag-scale 3.3`
- `--shortcut-mag-abs-center 5.4`
- `--shortcut-mag-abs-scale 1.0`
- `--shortcut-mag-clip-min 3.3`
- `--shortcut-mag-clip-max 7.3`

Default checkpoint directory:

```text
/workspace/checkpoints/depth-shortcut-L-hybrid-depth30m-outputdistill-r010-l005-classcond-centered-logitnormal-4x5090
```

## 6. Useful Overrides

Resume is enabled by default. Disable it for a fresh run:

```bash
RESUME=0 PYTHON_BIN=python3 ./run_sit_l_4xrtx5090_workspace.sh
```

If train OOMs:

```bash
GPU_BATCH_SIZE=128 PYTHON_BIN=python3 ./run_sit_l_4xrtx5090_workspace.sh
```

If only FID/VAE OOMs:

```bash
FID_EVAL_LOCAL_BATCH=8 VAE_DECODE_BATCH_SIZE=64 PYTHON_BIN=python3 ./run_sit_l_4xrtx5090_workspace.sh
```

If `/workspace` layout differs:

```bash
TRAIN_LATENTS_DIR=/custom/train \
VAL_LATENTS_DIR=/custom/val \
VAE_MODEL=/custom/sdvae \
INCEPTION_SCORE_WEIGHTS=/custom/inception_v3_google-0cc3c7bd.pth \
PYTHON_BIN=python3 ./run_sit_l_4xrtx5090_workspace.sh
```
