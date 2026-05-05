#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

MODEL_PATH="${MODEL_PATH:-./save/humanml_mdm_depth_shortcut_textcross_10pct_outputdistill_predictoronly/model000100000.pt}"
SAVE_DIR="${SAVE_DIR:-./save/humanml_mdm_depth_shortcut_textcross_10pct_outputdistill_predictoronly}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/dataset/HumanML3D}"
TRAIN_LOG="${TRAIN_LOG:-logs/train_humanml_depth_shortcut_4x4090_textcross_val_from100k_tmux_online.log}"
EVAL_NAME="${EVAL_NAME:-FID_test}"
EVAL_LOG="${EVAL_LOG:-${SAVE_DIR}/${EVAL_NAME}_000100000_val.log}"
NPROC="${NPROC:-4}"
export MODEL_PATH SAVE_DIR DATA_DIR TRAIN_LOG EVAL_NAME EVAL_LOG NPROC

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
unset WANDB_RUN_ID WANDB_RESUME

python - <<'PY'
import json
import os
from pathlib import Path
from types import SimpleNamespace

from diffusion import logger
from eval import eval_humanml
from train.train_platforms import WandBPlatform
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import apply_rules
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

model_path = os.environ["MODEL_PATH"]
save_dir = os.environ["SAVE_DIR"]
eval_name = os.environ["EVAL_NAME"]
eval_log = os.environ["EVAL_LOG"]
data_dir = os.environ["DATA_DIR"]

args_path = Path(model_path).with_name("args.json")
args_dict = json.loads(args_path.read_text())
args_dict.update({
    "model_path": model_path,
    "save_dir": save_dir,
    "data_dir": data_dir,
    "device": 0,
    "cuda": True,
    "seed": args_dict.get("seed", 10),
    "batch_size": 32,
    "guidance_param": args_dict.get("gen_guidance_param", 2.5),
    "eval_split": "val",
    "eval_metrics": ["fid", "precision"],
    "eval_rep_times": 3,
    "eval_num_samples": 1000,
    "autoregressive": False,
    "train_platform_type": "WandBPlatform",
    "external_mode": False,
})
args = apply_rules(SimpleNamespace(**args_dict))

fixseed(args.seed)
dist_util.setup_dist(args.device)
logger.configure()

print(f"Will save eval log to [{eval_log}]")
print("creating data loader...")
gt_loader = get_dataset_loader(
    name=args.dataset,
    batch_size=args.batch_size,
    num_frames=None,
    split=args.eval_split,
    hml_mode="gt",
)
gen_loader = get_dataset_loader(
    name=args.dataset,
    batch_size=args.batch_size,
    num_frames=None,
    split=args.eval_split,
    hml_mode="eval",
    fixed_len=args.context_len + args.pred_len,
    pred_len=args.pred_len,
    device=dist_util.dev(),
    autoregressive=args.autoregressive,
)

print("Creating model and diffusion...")
model, diffusion = create_model_and_diffusion(args, gen_loader)

print(f"Loading checkpoint from [{model_path}]...")
load_saved_model(model, model_path, use_avg=args.use_ema)
if args.guidance_param != 1:
    model = ClassifierFreeSampleModel(model)
model.to(dist_util.dev())
model.eval()

eval_motion_loaders = {
    eval_name: lambda: eval_humanml.get_mdm_loader(
        args,
        model=model,
        diffusion=diffusion,
        batch_size=args.batch_size,
        ground_truth_loader=gen_loader,
        mm_num_samples=0,
        mm_num_repeats=0,
        max_motion_length=gt_loader.dataset.opt.max_motion_length,
        num_samples_limit=args.eval_num_samples,
        scale=args.guidance_param,
    )
}

eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
eval_platform = WandBPlatform(save_dir, name=eval_name)
eval_platform.report_args(vars(args), name="Args")
eval_dict = eval_humanml.evaluation(
    eval_wrapper,
    gt_loader,
    eval_motion_loaders,
    eval_log,
    replication_times=args.eval_rep_times,
    diversity_times=300,
    mm_num_times=0,
    run_mm=False,
    eval_platform=eval_platform,
    metric_names=args.eval_metrics,
)
print(eval_dict)
eval_platform.close()
PY

set -o pipefail
DATA_DIR="${DATA_DIR}" SAVE_DIR="${SAVE_DIR}" RESUME_CHECKPOINT="${MODEL_PATH}" NPROC="${NPROC}" \
  bash scripts/train_humanml_depth_shortcut_4x4090.sh 2>&1 | tee -a "${TRAIN_LOG}"
