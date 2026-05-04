# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def main():
    args = train_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    train_platform_type = eval(args.train_platform_type) if dist_util.is_main_process() else NoPlatform
    train_platform = train_platform_type(args.save_dir)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif dist_util.is_main_process() and os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif dist_util.is_main_process() and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if dist_util.is_main_process():
        train_platform.report_args(args, name='Args')
        args_path = os.path.join(args.save_dir, 'args.json')
        with open(args_path, 'w') as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)
    dist_util.barrier()

    print("creating data loader...")

    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              device=dist_util.dev(),
                              data_dir=args.data_dir,
                              distributed=dist_util.get_world_size() > 1,)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    if dist_util.is_main_process():
        train_platform.close()

if __name__ == "__main__":
    main()
