# Self-Transcendence Training Notes

This branch adds three training modes to `train.py`:

- `baseline`: plain SiT diffusion training
- `vae-structure-guidance`: stage 1 warm-up teacher
- `self-transcendence`: stage 2 student trained from scratch with a frozen stage-1 teacher

## Default repo-faithful knobs

- `--class-dropout-prob 0.1`
- `--guide-lambda 0.5`
- `--feature-guidance-scale 30.0`
- `--guided-layer 2` for `vae-structure-guidance`
- `--guided-layer depth//2` for `self-transcendence`
- `--guiding-layer floor(2*depth/3)`
- `--t-range 0.4 0.7`
- `--guide-stop-epochs 20` for `S/B`, `10` for `L/XL` in stage 2
- ArrayRecord records can now contain either `latent` or repo-style `moments`

## Stage 1: VAE Structure Guidance

```bash
python train.py \
  --training-mode vae-structure-guidance \
  --model-size XL \
  --epochs 40 \
  --steps-per-epoch 1000 \
  --batch-size 256 \
  --learning-rate 1e-4 \
  --class-dropout-prob 0.1 \
  --guide-lambda 0.5 \
  --guided-layer 2 \
  --t-range 0.4 0.7 \
  --data-path /path/to/train/*.ar \
  --val-data-path /path/to/val/*.ar \
  --ckpt-dir ./checkpoints/self_transcendence_stage1_xl
```

## Stage 2: Self-Transcendence

```bash
python train.py \
  --training-mode self-transcendence \
  --model-size XL \
  --epochs 80 \
  --steps-per-epoch 1000 \
  --batch-size 256 \
  --learning-rate 1e-4 \
  --class-dropout-prob 0.1 \
  --guide-lambda 0.5 \
  --feature-guidance-scale 30.0 \
  --guided-layer 6 \
  --guiding-layer 8 \
  --t-range 0.4 0.7 \
  --teacher-ckpt ./checkpoints/self_transcendence_stage1_xl \
  --teacher-use-ema \
  --data-path /path/to/train/*.ar \
  --val-data-path /path/to/val/*.ar \
  --ckpt-dir ./checkpoints/self_transcendence_stage2_xl
```

## Sampling

Use the EMA checkpoint from the selected stage:

```bash
python sample.py \
  --ckpt ./checkpoints/self_transcendence_stage2_xl/ema \
  --model-size XL \
  --class-dropout-prob 0.1 \
  --output-dir ./samples/self_transcendence_xl
```
