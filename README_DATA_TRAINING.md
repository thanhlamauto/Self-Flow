# Data and Training Notes

This branch keeps datasets, evaluator checkpoints, cached numpy files, logs, and model outputs out of Git. The code expects those files to exist locally, but they should not be committed.

## Local Data Layout

Place HumanML3D under:

```text
dataset/HumanML3D/
```

The training scripts expect the directory to include the usual HumanML3D files:

```text
dataset/HumanML3D/
  Mean.npy
  Std.npy
  new_joint_vecs/
  texts/
  train.txt
  train_val.txt
  val.txt
```

Optional local caches and evaluator assets can also live in the repo workspace:

```text
dataset/t2m_train.npy
dataset/t2m_val.npy
dataset/t2m_test.npy
kit/
save/
logs/
wandb/
```

These paths are ignored by `.gitignore` so they stay local.

## Setup Data

For a fresh 4x4090 VM with conda available, use:

```bash
bash scripts/setup_humanml3d_4090.sh
```

By default this downloads HumanML3D to `dataset/HumanML3D`. Override the path with:

```bash
DATA_DIR=/path/to/HumanML3D bash scripts/setup_humanml3d_4090.sh
```

If setting up manually, make sure `DATA_DIR` points at the final HumanML3D directory, not its parent.

## Training

Run the tuned depth-shortcut HumanML3D job with:

```bash
DATA_DIR="$PWD/dataset/HumanML3D" NPROC=4 bash scripts/train_humanml_depth_shortcut_4x4090.sh
```

Useful overrides:

```bash
SAVE_DIR=./save/my_run
RESUME_CHECKPOINT=./save/my_run/model000100000.pt
LAMBDA_PRIVATE=0.2
TORCHRUN=torchrun
WANDB_ENTITY=my_entity
WANDB_RUN_ID=my_run_id
WANDB_RESUME=allow
```

Example resume:

```bash
DATA_DIR="$PWD/dataset/HumanML3D" \
SAVE_DIR=./save/humanml_mdm_depth_shortcut_textcross_10pct_outputdistill_predictoronly \
RESUME_CHECKPOINT=./save/humanml_mdm_depth_shortcut_textcross_10pct_outputdistill_predictoronly/model000100000.pt \
NPROC=4 \
bash scripts/train_humanml_depth_shortcut_4x4090.sh
```

## Eval Then Resume

To evaluate a checkpoint on `val` and then continue training:

```bash
MODEL_PATH=./save/humanml_mdm_depth_shortcut_textcross_10pct_outputdistill_predictoronly/model000100000.pt \
DATA_DIR="$PWD/dataset/HumanML3D" \
NPROC=4 \
bash scripts/eval_val_100k_then_resume.sh
```

The script writes eval output under `SAVE_DIR` and appends training logs under `logs/`.

## Git Hygiene

Before committing, check that data stays ignored:

```bash
git status --short
git status --ignored --short
```

Commit only code, scripts, and documentation. Do not `git add dataset/`, `kit/`, `save/`, `logs/`, or `wandb/`.
