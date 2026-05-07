# LARA: Layer-Aware Representation Alignment for Diffusion Transformers

Anonymous review code for **LARA: Layer-Aware Representation Alignment for
Diffusion Transformers**. The repository contains the ImageNet 256x256 latent
training, evaluation, and qualitative sampling entrypoints used for the LARA
experiments.

The command-line surface is intentionally small. Paper-level settings are the
defaults in `train.py`; run scripts only pass paths, checkpoint location, model
size, predictor configuration, and fixed evaluation steps.

## Setup

Install the Python dependencies in a JAX environment with TPU or GPU support:

```bash
pip install -r requirements.txt
```

The training dataloader expects ArrayRecord files with pickled records:

- `latent`: VAE latent shaped `(4, 32, 32)`
- `label`: ImageNet class ID in `[0, 999]`

Raw images can be converted with `prepare_data_tpu.py`. Use the same VAE
variant for latent preparation, training-time FID decode, and qualitative
sampling.

## External Assets

Set external paths from the shell. Do not edit repository files with machine
specific paths.

```bash
export TRAIN_DATA_PATH=/path/to/imagenet/train_arrayrecords
export VAL_DATA_PATH=/path/to/imagenet/val_arrayrecords
export VAE_MODEL=/path/to/sd-vae-ft-ema-or-hf-id
export CKPT_DIR=/path/to/checkpoints/lara_imagenet256_l
export INCEPTION_SCORE_WEIGHTS=/path/to/inception_v3_google-0cc3c7bd.pth
```

`INCEPTION_SCORE_WEIGHTS` is optional. If it is unset, the Inception Score
worker uses its normal torchvision loading path.

## Training

Set the asset variables once, then choose the backbone scale:

```bash
export TRAIN_DATA_PATH=/path/to/imagenet/train_arrayrecords
export VAL_DATA_PATH=/path/to/imagenet/val_arrayrecords
export VAE_MODEL=/path/to/sd-vae-ft-ema-or-hf-id
export INCEPTION_SCORE_WEIGHTS=/path/to/inception_v3_google-0cc3c7bd.pth
```

Train the SiT-L/2 400K-step run:

```bash
CKPT_DIR=/path/to/checkpoints/lara_imagenet256_l \
scripts/train_lara_imagenet256_l.sh
```

Train the SiT-XL/2 2M-step run:

```bash
CKPT_DIR=/path/to/checkpoints/lara_imagenet256_xl \
scripts/train_lara_imagenet256_xl.sh
```

Train the smaller SiT-B/2 smoke or Kaggle test run:

```bash
scripts/train_lara_imagenet256_b_test.sh --no-wandb
```

These scripts resume from `CKPT_DIR/latest` when present. The L and B scripts
keep checkpoints at 100k, 200k, and 400k. The XL script keeps 100k, 200k,
400k, 800k, 1M, and 2M.

The direct `train.py` equivalents keep only non-default review arguments.
Magnitude calibration and layer-pair gap settings are scale-specific defaults
resolved from `--model-size` for B, L, and XL.

```bash
python -u train.py --resume \
  --model-size L \
  --data-path "${TRAIN_DATA_PATH}" \
  --val-data-path "${VAL_DATA_PATH}" \
  --vae-model "${VAE_MODEL}" \
  --inception-score-weights "${INCEPTION_SCORE_WEIGHTS}" \
  --shortcut-predictor hybrid_depth30m \
  --shortcut-predictor-depth 11 \
  --shortcut-predictor-dilation-cycle 1,2,4 \
  --fid-steps 50000,400000 \
  --ckpt-dir "${CKPT_DIR}"
```

```bash
python -u train.py --resume \
  --model-size XL \
  --epochs 2000 \
  --data-path "${TRAIN_DATA_PATH}" \
  --val-data-path "${VAL_DATA_PATH}" \
  --vae-model "${VAE_MODEL}" \
  --inception-score-weights "${INCEPTION_SCORE_WEIGHTS}" \
  --shortcut-predictor hybrid_depth30m \
  --shortcut-predictor-hidden-size 480 \
  --shortcut-predictor-depth 12 \
  --shortcut-predictor-num-heads 8 \
  --shortcut-predictor-dilation-cycle 1,2,4 \
  --fid-steps 50000,2000000 \
  --ckpt-keep-steps 100000,200000,400000,800000,1000000,2000000 \
  --ckpt-dir "${CKPT_DIR}"
```

Useful quick checks:

```bash
DRY_RUN=1 scripts/train_lara_imagenet256_b_test.sh

scripts/train_lara_imagenet256_b_test.sh \
  --no-wandb --mock-data --preflight-only \
  --preflight-sample-count 0 --preflight-fid-samples 0
```

## Experiments and Baselines

The LARA code lives on this branch. The controlled internal-alignment baselines
are kept as separate branches so their training code stays isolated:

- `feat/layersync-sit-paper`: LayerSync baseline.
- `feat/sit-sra-jax`: SRA baseline.

Use the same asset variables as above for all experiments. The baseline branches
do not implement `--resume` or fixed `--fid-steps`; they save the final online
and EMA checkpoints at `CKPT_DIR` and `CKPT_DIR/ema`, and use `--fid-freq` for
periodic FID.

For a branchless review ZIP, include sanitized snapshots of these two baseline
branches as sibling directories or preserve the git branches in the submitted
archive. Otherwise only the LARA branch can be run from the extracted tree.

### LARA

Current branch:

```bash
CKPT_DIR=/path/to/checkpoints/lara_imagenet256_l \
scripts/train_lara_imagenet256_l.sh
```

```bash
CKPT_DIR=/path/to/checkpoints/lara_imagenet256_xl \
scripts/train_lara_imagenet256_xl.sh
```

### LayerSync

Switch to the LayerSync branch:

```bash
git switch feat/layersync-sit-paper
```

Default SiT-L/2 controlled run:

```bash
CKPT_DIR=/path/to/checkpoints/layersync_imagenet256_l \
python -u train.py \
  --model-size L \
  --batch-size 256 \
  --epochs 400 \
  --steps-per-epoch 1000 \
  --learning-rate 1e-4 \
  --vae-model "${VAE_MODEL}" \
  --data-path "${TRAIN_DATA_PATH}" \
  --val-data-path "${VAL_DATA_PATH}" \
  --grad-clip 1.0 \
  --ema-decay 0.9999 \
  --log-freq 1000 \
  --eval-freq 20000 \
  --eval-batches 1 \
  --sample-freq 0 \
  --sample-num-steps 50 \
  --sample-cfg-scale 1.0 \
  --fid-freq 50000 \
  --num-fid-samples 50000 \
  --fid-batch-size 256 \
  --fid-eval-local-batch 32 \
  --fid-num-steps 250 \
  --fid-cfg-scale 1.0 \
  --vae-decode-batch-size 256 \
  --no-linear-probe \
  --inception-score-weights "${INCEPTION_SCORE_WEIGHTS}" \
  --block-corr-freq 0 \
  --cfg-dropout-rate 0.1 \
  --wandb-project layersync-baseline \
  --layersync-lambda 0.2 \
  --layersync-weak-layer 8 \
  --layersync-strong-layer 18 \
  --ckpt-dir "${CKPT_DIR}"
```

For SiT-XL/2, use the same command with:

```bash
--model-size XL --epochs 2000 \
--layersync-lambda 0.2 --layersync-weak-layer 8 --layersync-strong-layer 16
```

For SiT-B/2, use:

```bash
--model-size B --epochs 400 \
--layersync-lambda 0.3 --layersync-weak-layer 4 --layersync-strong-layer 7
```

### SRA

Switch to the SRA branch:

```bash
git switch feat/sit-sra-jax
```

Default SiT-L/2 controlled run:

```bash
CKPT_DIR=/path/to/checkpoints/sra_imagenet256_l \
python -u train.py \
  --model-size L \
  --batch-size 256 \
  --epochs 400 \
  --steps-per-epoch 1000 \
  --learning-rate 1e-4 \
  --vae-model "${VAE_MODEL}" \
  --data-path "${TRAIN_DATA_PATH}" \
  --val-data-path "${VAL_DATA_PATH}" \
  --grad-clip 1.0 \
  --ema-decay 0.9999 \
  --loss-type sml1 \
  --block-out-s 4 \
  --block-out-t 8 \
  --t-max 0.2 \
  --align-weight 0.2 \
  --align-decay-start-epoch 149 \
  --align-decay-denom 1000.0 \
  --align-decay-base 0.1 \
  --log-freq 1000 \
  --eval-freq 20000 \
  --eval-batches 1 \
  --sample-freq 0 \
  --sample-num-steps 50 \
  --sample-cfg-scale 1.0 \
  --fid-freq 50000 \
  --num-fid-samples 50000 \
  --fid-batch-size 256 \
  --fid-eval-local-batch 32 \
  --fid-num-steps 250 \
  --fid-cfg-scale 1.0 \
  --vae-decode-batch-size 256 \
  --no-linear-probe \
  --inception-score-weights "${INCEPTION_SCORE_WEIGHTS}" \
  --block-corr-freq 0 \
  --wandb-project sra-baseline \
  --ckpt-dir "${CKPT_DIR}"
```

For SiT-XL/2 long-run comparison, use the same command with:

```bash
--model-size XL --epochs 2000
```

## Evaluation

Evaluate `CKPT_DIR/latest` without taking optimizer steps:

```bash
scripts/eval_lara_imagenet256_l.sh --no-wandb
```

The review defaults use 50,000 generated samples, 250 denoising steps, CFG
scale 1.0, FID/sFID, Inception Score, and precision/recall. For a cheaper
sanity pass, override only the expensive counters:

```bash
scripts/eval_lara_imagenet256_l.sh \
  --no-wandb --num-fid-samples 1024 --fid-batch-size 128 --pr-max-samples 1024
```

## Qualitative Sampling

Generate PNG samples and an ADM-compatible NPZ from the EMA checkpoint:

```bash
CLASS_IDS=1,7,207,281 \
NUM_SAMPLES=64 \
SAMPLE_CFG_SCALE=4.0 \
scripts/sample_lara_qualitative.sh
```

The script uses the standard full-backbone sampler. This matches the paper's
main generation protocol; transition-interface skipping is only an appendix
diagnostic and is not enabled by the review scripts.

Outputs are written to `outputs/qualitative_samples` by default. Override
`SAMPLE_DIR`, `MODEL_SIZE`, `SAMPLE_BATCH_SIZE`, `SAMPLE_NUM_STEPS`, or
`SAMPLE_SEED` from the shell when needed.

## Paper Preset Defaults

`train.py` defaults encode the non-path LARA recipe:

- 400 epochs x 1000 steps, batch size 256, AdamW learning rate `1e-4`.
- SiT/DiT latent ImageNet setup with CFG dropout `0.1`.
- Shared layer-aware transition predictor with timestep and class conditioning.
- Raw source activations, direction plus magnitude transition loss.
- Depth-Centered Gap Sampling and magnitude calibration with scale-specific
  B/L/XL defaults resolved from `--model-size`.
- EMA transition consistency with detached source.
- Output distillation every step with ratio `0.10` and weight `0.05`.
- Residual common/private activation diversity loss with four random pairs.
- FID evaluation with 50K samples, 250 denoising steps, CFG scale `1.0`.

Pass explicit flags only for ablations or machine constraints. For example:

```bash
scripts/train_lara_imagenet256_l.sh \
  --batch-size 128 --fid-eval-local-batch 16 --vae-decode-batch-size 64
```

## Repository Layout

```text
configs/      Anonymous environment defaults for external assets
scripts/      Training, evaluation, and qualitative sampling entrypoints
src/          JAX/Flax model, LARA predictor, sampling, and metrics
train.py      Training, resume, checkpointing, preflight, and eval-only
sample.py     Standalone qualitative and NPZ sampling utility
prepare_data_tpu.py  ImageNet latent ArrayRecord preparation
```
