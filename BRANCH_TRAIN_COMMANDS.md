# Branch Train Commands

Notebook-ready training commands for the metric-enabled branches.

If you run from a normal shell instead of Kaggle/Jupyter, remove the leading `!`
from each `python` command.

## Shared Base

```bash
COMMON_BASE="\
  --model-size B \
  --batch-size 128 \
  --epochs 400 \
  --steps-per-epoch 1000 \
  --learning-rate 1e-4 \
  --vae-model /kaggle/input/models/damtrunghieu/sdvae-ema/flax/default/1 \
  --ckpt-dir ./checkpoints \
  --data-path /kaggle/input/datasets/thaygiaodaysat/imagenet-vae-latents-ar-v2 \
  --val-data-path /kaggle/input/datasets/thaygiaodaysat/imagenet-vae-latents-train-v3 \
  --grad-clip 1.0 \
  --log-freq 100 \
  --eval-freq 1000 \
  --eval-batches 4 \
  --sample-freq 5000 \
  --sample-num-steps 50 \
  --sample-cfg-scale 1.0 \
  --fid-freq 25000 \
  --num-fid-samples 4096 \
  --fid-batch-size 256 \
  --fid-eval-local-batch 32 \
  --fid-num-steps 50 \
  --fid-cfg-scale 1.0 \
  --vae-decode-batch-size 256 \
  --no-linear-probe \
  --inception-score-weights /kaggle/input/models/ctlcmleon/inception-v3/pytorch/default/1/inception_v3_google-0cc3c7bd.pth \
  --block-corr-freq 25000 \
  --block-corr-batches 2 \
  --preflight-checks \
  --preflight-fid-memory-probe"
```

## Self-Flow Family

### `feat/self-flow-i-jepa-adaln-zero-mimetic-init`

```bash
git checkout feat/self-flow-i-jepa-adaln-zero-mimetic-init
!python train.py $COMMON_BASE \
  --wandb-project selfflow-jax \
  --mask-ratio 0.25 \
  --lambda-jepa 0.5 \
  --fixed-ema-decay 0.9999 \
  --predictor-depth 4 \
  --jepa-num-targets 1 \
  --student-layer 4 \
  --teacher-layer 8
```

### `feat/self-flow-i-jepa`

```bash
git checkout feat/self-flow-i-jepa
!python train.py $COMMON_BASE \
  --wandb-project selfflow-jax \
  --mask-ratio 0.25 \
  --lambda-jepa 0.5 \
  --fixed-ema-decay 0.9999 \
  --predictor-depth 1 \
  --jepa-num-targets 1 \
  --student-layer 4 \
  --teacher-layer 8 \
  --lambda-attn-align 0.05 \
  --attn-align-warmup-steps 10000
```

### `feat/self-flow-online-target-jepa`

```bash
git checkout feat/self-flow-online-target-jepa
!python train.py $COMMON_BASE \
  --wandb-project selfflow-jax \
  --mask-ratio 0.25 \
  --lambda-jepa 0.5 \
  --predictor-depth 4 \
  --student-layer 4 \
  --teacher-layer 8
```

### `feat/self-flow-jax-original`

```bash
git checkout feat/self-flow-jax-original
!python train.py $COMMON_BASE \
  --wandb-project selfflow-jax \
  --mask-ratio 0.25 \
  --self-flow-gamma 0.5 \
  --ema-decay 0.9999 \
  --student-layer 4 \
  --teacher-layer 8
```

### `feat/self-flow-jepa`

```bash
git checkout feat/self-flow-jepa
!python train.py $COMMON_BASE \
  --wandb-project selfflow-jax \
  --mask-ratio 0.25 \
  --self-flow-gamma 0.5 \
  --ema-decay 0.9999 \
  --student-layer 4 \
  --teacher-layer 8
```

## SiT Family

### `feat/sit-vanilla-baseline`

```bash
git checkout feat/sit-vanilla-baseline
!python train.py $COMMON_BASE \
  --wandb-project sit-vanilla-jax \
  --ema-decay 0.9999
```

### `feat/layersync-sit-paper`

For `model-size B`, the branch default LayerSync pair `8:16` is invalid because
depth is only `12`. Use `4:10` explicitly.

```bash
git checkout feat/layersync-sit-paper
!python train.py $COMMON_BASE \
  --wandb-project sit-vanilla-jax \
  --ema-decay 0.9999 \
  --layersync-lambda 0.5 \
  --layersync-weak-layer 4 \
  --layersync-strong-layer 10
```

### `feat/orthogonal-vanilla-sit`

```bash
git checkout feat/orthogonal-vanilla-sit
!python train.py $COMMON_BASE \
  --wandb-project sit-vanilla-jax \
  --ema-decay 0.9999 \
  --layersync-lambda 0.5 \
  --layersync-pairs 4:10 \
  --layersync-mode stopgrad \
  --layersync-stopgrad-side strong
```

## Notes

- `--probe-layer` and `--probe-eval-batches` are omitted because all commands
  already use `--no-linear-probe`.
- `feat/self-flow-online-target-jepa` does not accept `--fixed-ema-decay` or
  `--jepa-num-targets`.
- `feat/self-flow-jax-original` and `feat/self-flow-jepa` use
  `--self-flow-gamma` and `--ema-decay`, not `--lambda-jepa` and
  `--fixed-ema-decay`.
