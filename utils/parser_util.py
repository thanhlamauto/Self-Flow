from argparse import ArgumentParser
import argparse
import os
import json


def _parse_csv_ints(value):
    if value in [None, '']:
        return []
    if isinstance(value, list):
        return value
    steps = []
    for item in value.split(','):
        item = item.strip().lower()
        if not item:
            continue
        if item.endswith('k'):
            steps.append(int(float(item[:-1]) * 1000))
        else:
            steps.append(int(item))
    return steps


def _parse_csv_strings(value):
    if value in [None, '']:
        return []
    if isinstance(value, list):
        return value
    return [item.strip().lower() for item in value.split(',') if item.strip()]


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    if args.model_path != '':  # if not using external results file
        args = load_args_from_model(args, args_to_overwrite)

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    
    return apply_rules(args)

def load_args_from_model(args, args_to_overwrite):
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
    return args

def apply_rules(args):
    # For prefix completion
    if args.pred_len == 0:
        args.pred_len = args.context_len
    if hasattr(args, 'model_size') and args.model_size:
        if args.model_size.upper() == 'B':
            args.latent_dim = 512
            args.layers = 8
        else:
            raise ValueError(f'Unsupported --model-size for MDM: {args.model_size}')
    if hasattr(args, 'epochs') and args.epochs is not None and hasattr(args, 'steps_per_epoch') and args.steps_per_epoch is not None:
        args.num_steps = int(args.epochs) * int(args.steps_per_epoch)

    # For target conditioning
    if args.lambda_target_loc > 0.:
        args.multi_target_cond = True
    if hasattr(args, 'lambda_layersync'):
        if args.lambda_layersync < 0.:
            raise ValueError('--lambda_layersync must be non-negative.')
        if args.layersync_weak_layer <= 0 or args.layersync_strong_layer <= 0:
            raise ValueError('--layersync_weak_layer and --layersync_strong_layer must be positive.')
        if args.layersync_weak_layer >= args.layersync_strong_layer:
            raise ValueError('--layersync_weak_layer must be smaller than --layersync_strong_layer.')
        if hasattr(args, 'layers') and args.layersync_strong_layer > args.layers:
            raise ValueError('--layersync_strong_layer must be <= --layers.')
    if hasattr(args, 'eval_steps'):
        args.eval_steps = _parse_csv_ints(args.eval_steps)
        if any(step <= 0 for step in args.eval_steps):
            raise ValueError('--eval_steps must contain positive training steps.')
    if hasattr(args, 'eval_metrics'):
        args.eval_metrics = _parse_csv_strings(args.eval_metrics)
    if hasattr(args, 'shortcut_predictor'):
        args.shortcut_predictor = args.shortcut_predictor.replace("-", "_")
        if args.shortcut_predictor not in {"hybrid_mdm_10", "default"}:
            args.use_depth_shortcut = True
    if hasattr(args, 'shortcut_mag_scale') and args.shortcut_mag_scale <= 0:
        raise ValueError('--shortcut-mag-scale must be positive.')
    if hasattr(args, 'shortcut_mag_abs_scale') and args.shortcut_mag_abs_scale <= 0:
        raise ValueError('--shortcut-mag-abs-scale must be positive.')
    if hasattr(args, 'shortcut_mag_clip_min') and args.shortcut_mag_clip_min >= args.shortcut_mag_clip_max:
        raise ValueError('--shortcut-mag-clip-min must be smaller than --shortcut-mag-clip-max.')
    if hasattr(args, 'shortcut_lambda_skip_fm') and args.shortcut_lambda_skip_fm < 0:
        raise ValueError('--shortcut-lambda-skip-fm must be non-negative.')
    if hasattr(args, 'shortcut_skip_in_loop_prob') and not 0.0 <= args.shortcut_skip_in_loop_prob <= 1.0:
        raise ValueError('--shortcut-skip-in-loop-prob must be between 0 and 1.')
    if hasattr(args, 'shortcut_skip_in_loop_gap_mode') and args.shortcut_skip_in_loop_gap_mode == 'trunc_normal':
        args.shortcut_skip_in_loop_gap_mode = 'truncated-normal'
    if hasattr(args, 'shortcut_skip_in_loop_gap_sigma') and args.shortcut_skip_in_loop_gap_sigma <= 0:
        raise ValueError('--shortcut-skip-in-loop-gap-sigma must be positive.')
    if hasattr(args, 'timestep_logit_std') and args.timestep_logit_std <= 0:
        raise ValueError('--timestep-logit-std must be positive.')
    if hasattr(args, 'pair_center_sigma') and args.pair_center_sigma < 0:
        raise ValueError('--pair-center-sigma must be non-negative.')
    if hasattr(args, 'output_distill_ratio') and not 0.0 <= args.output_distill_ratio <= 1.0:
        raise ValueError('--output-distill-ratio must be between 0 and 1.')
    if hasattr(args, 'lambda_output_distill') and args.lambda_output_distill < 0:
        raise ValueError('--lambda-output-distill must be non-negative.')
    if hasattr(args, 'output_distill_every') and args.output_distill_every <= 0:
        raise ValueError('--output-distill-every must be positive.')
    if hasattr(args, 'direct_num_pairs') and args.direct_num_pairs != args.direct_joint_pairs + args.direct_predictor_only_pairs:
        raise ValueError('--direct-num-pairs must equal --direct-joint-pairs + --direct-predictor-only-pairs.')
    if hasattr(args, 'private_loss') and args.private_loss and args.lambda_private <= 0:
        raise ValueError('--private-loss requires --lambda-private > 0.')
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('--model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", "--batch-size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--external_mode", default=False, type=bool, help="For backward cometability, do not change or delete.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--text_encoder_type", default='clip',
                       choices=['clip', 'bert'], type=str, help="Text encoder type.")
    group.add_argument("--emb_trans_dec", action='store_true',
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--model-size", dest="model_size", default=None,
                       help="Compatibility alias for Self-Flow settings. Currently B maps to latent_dim=512,layers=8.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--mask_frames", action='store_true', help="If true, will fix Rotem's bug and mask invalid frames.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_target_loc", default=0.0, type=float, help="For HumanML only, when . L2 with target location.")
    group.add_argument("--lambda_layersync", default=0.0, type=float,
                       help="Weight for direct LayerSync cosine alignment between intermediate transformer layers.")
    group.add_argument("--layersync_weak_layer", default=3, type=int,
                       help="1-based shallow transformer layer used by LayerSync.")
    group.add_argument("--layersync_strong_layer", default=6, type=int,
                       help="1-based deeper transformer layer used by LayerSync.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--pos_embed_max_len", default=5000, type=int,
                       help="Pose embedding max length.")
    group.add_argument("--use_ema", action='store_true',
                    help="If True, will use EMA model averaging.")
    group.add_argument("--use_depth_shortcut", "--use-depth-shortcut", dest="use_depth_shortcut",
                       action='store_true', default=False,
                       help="Enable MDM depth shortcut predictor and auxiliary training losses.")
    group.add_argument("--no-use-depth-shortcut", dest="use_depth_shortcut", action='store_false',
                       help="Disable MDM depth shortcut predictor.")
    group.add_argument("--shortcut-predictor", dest="shortcut_predictor", default="hybrid_mdm_10",
                       choices=["hybrid_mdm_10", "hybrid-mdm-10", "hybrid_mdm_8", "hybrid-mdm-8", "hybrid_deep_10", "hybrid-deep-10", "hybrid_deep_12pct", "hybrid-deep-12pct", "hybrid", "default"],
                       help="Depth shortcut predictor variant. Default is a hybrid 1D MDM predictor sized around 10-12%% of MDM-B.")
    group.add_argument("--shortcut-predictor-use-timestep", dest="shortcut_predictor_use_timestep",
                       action='store_true', default=True,
                       help="Condition the shortcut predictor on the MDM timestep embedding.")
    group.add_argument("--no-shortcut-predictor-use-timestep", dest="shortcut_predictor_use_timestep",
                       action='store_false')
    group.add_argument("--shortcut-predictor-normalize-input", dest="shortcut_predictor_normalize_input",
                       action='store_true', default=False,
                       help="Feed normalized hidden directions into the shortcut predictor.")
    group.add_argument("--no-shortcut-predictor-normalize-input", dest="shortcut_predictor_normalize_input",
                       action='store_false')
    

    group.add_argument("--multi_target_cond", action='store_true', help="If true, enable multi-target conditioning (aka Sigal's model).")
    group.add_argument("--multi_encoder_type", default='single', choices=['single', 'multi', 'split'], type=str, help="Specifies the encoder type to be used for the multi joint condition.")
    group.add_argument("--target_enc_layers", default=1, type=int, help="Num target encoder layers")


    # Prefix completion model
    group.add_argument("--context_len", default=0, type=int, help="If larger than 0, will do prefix completion.")
    group.add_argument("--pred_len", default=0, type=int, help="If context_len larger than 0, will do prefix completion. If pred_len will not be specified - will use the same length as context_len")
    



def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", "--save-dir", "--ckpt-dir", dest="save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", "--learning-rate", dest="lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--grad_clip", "--grad-clip", dest="grad_clip", default=1.0, type=float,
                       help="Clip global gradient norm before optimizer step. Set <=0 to disable.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--eval_steps", default="", type=str,
                       help="Comma-separated exact training steps to run validation, e.g. 50000,100000,200000.")
    group.add_argument("--eval_metrics", default="all", type=str,
                       help=(
                           "Comma-separated validation metrics. For MDM, supported metrics are fid and precision "
                           "(HumanML/KIT precision maps to R_precision). Image metrics sfid, inception_score, "
                           "and recall are accepted but skipped because MDM has no native implementation."
                       ))
    group.add_argument("--log_interval", "--log-freq", dest="log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", "--eval-freq", dest="save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--epochs", default=None, type=int,
                       help="Compatibility alias: with --steps-per-epoch, sets --num_steps.")
    group.add_argument("--steps-per-epoch", dest="steps_per_epoch", default=None, type=int,
                       help="Compatibility alias: with --epochs, sets --num_steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", nargs='?', const="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    
    group.add_argument("--gen_during_training", action='store_true',
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=3, type=int,
                       help="Number of samples to sample while generating")
    group.add_argument("--gen_num_repetitions", default=2, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--gen_guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    
    group.add_argument("--avg_model_beta", default=0.9999, type=float, help="Average model beta (for EMA).")
    group.add_argument("--adam_beta2", default=0.999, type=float, help="Adam beta2.")
    group.add_argument("--shortcut-training-mode", default="direction-magnitude",
                       choices=["direction", "direction-magnitude"],
                       help="Depth shortcut loss preset. direction-magnitude is the default.")
    group.add_argument("--shortcut-lambda-dir", type=float, default=1.0)
    group.add_argument("--shortcut-lambda-boot", type=float, default=0.25)
    group.add_argument("--shortcut-lambda-mag", type=float, default=0.375)
    group.add_argument("--shortcut-lambda-boot-mag", type=float, default=0.1875)
    group.add_argument("--shortcut-bootstrap-detach-source", action='store_true', default=True)
    group.add_argument("--no-shortcut-bootstrap-detach-source", dest="shortcut_bootstrap_detach_source", action='store_false')
    group.add_argument("--shortcut-mag-scale", type=float, default=2.2)
    group.add_argument("--shortcut-mag-abs-center", type=float, default=2.9)
    group.add_argument("--shortcut-mag-abs-scale", type=float, default=0.6)
    group.add_argument("--shortcut-mag-clip-min", type=float, default=0.8)
    group.add_argument("--shortcut-mag-clip-max", type=float, default=3.5)
    group.add_argument("--shortcut-lambda-skip-fm", type=float, default=0.0,
                       help="Compatibility flag from the JAX shortcut run. MDM keeps skip-FM disabled.")
    group.add_argument("--shortcut-skip-in-loop-prob", type=float, default=0.0,
                       help="Compatibility flag from the JAX shortcut run. MDM keeps skip-FM disabled.")
    group.add_argument("--shortcut-skip-in-loop-gap-mode", default="truncated-normal",
                       choices=["fixed", "truncated-normal", "trunc_normal"],
                       help="Compatibility flag for disabled skip-FM scheduling.")
    group.add_argument("--shortcut-skip-in-loop-max-gap", type=int, default=10)
    group.add_argument("--shortcut-skip-in-loop-gap-loc", type=float, default=3.0)
    group.add_argument("--shortcut-skip-in-loop-gap-sigma", type=float, default=2.0)
    group.add_argument("--timestep-sampling-mode", default="uniform", choices=["uniform", "logit_normal"])
    group.add_argument("--timestep-logit-mean", type=float, default=0.0)
    group.add_argument("--timestep-logit-std", type=float, default=1.0)
    group.add_argument("--predictor-learning-rate", "--shortcut-predictor-lr", dest="shortcut_predictor_lr",
                       default=1e-4, type=float)
    group.add_argument("--shortcut-predictor-weight-decay", default=0.1, type=float)
    group.add_argument("--shortcut-predictor-ema-decay", default=0.999, type=float)
    group.add_argument("--output-distill", dest="output_distill", action='store_true', default=True)
    group.add_argument("--no-output-distill", dest="output_distill", action='store_false')
    group.add_argument("--output-distill-ratio", type=float, default=0.10)
    group.add_argument("--lambda-output-distill", type=float, default=0.05)
    group.add_argument("--output-distill-every", type=int, default=1)
    group.add_argument("--output-distill-update-mode", default="predictor_plus_all",
                       choices=["predictor_plus_all", "predictor_only"])
    group.add_argument("--output-distill-pair-mode", default="trunc_normal",
                       choices=["trunc_normal", "trunc_normal_centered", "uniform", "random"])
    group.add_argument("--direct-pair-mode", default="trunc_normal",
                       choices=["trunc_normal", "trunc_normal_centered", "uniform", "random"])
    group.add_argument("--pair-center-sigma", type=float, default=0.0)
    group.add_argument("--direct-num-pairs", type=int, default=1)
    group.add_argument("--direct-joint-pairs", type=int, default=1)
    group.add_argument("--direct-predictor-only-pairs", type=int, default=0)
    group.add_argument("--private-loss", dest="private_loss", action='store_true', default=True)
    group.add_argument("--no-private-loss", dest="private_loss", action='store_false')
    group.add_argument("--lambda-private", type=float, default=1.0)
    group.add_argument("--private-max-pairs", type=int, default=2)
    group.add_argument("--private-use-residual", dest="private_use_residual", action='store_true', default=True)
    group.add_argument("--no-private-use-residual", dest="private_use_residual", action='store_false')
    group.add_argument("--private-cosine-mode", default="bnd", choices=["bnd", "nd", "token"])
    group.add_argument("--private-pair-mode", default="random", choices=["first", "random"])
    group.add_argument("--fid-skip-timestep-mode", default="alternate", choices=["alternate", "all"],
                       help="Accepted for ImageNet shortcut command compatibility; HumanML eval ignores it.")
    group.add_argument("--shortcut-predictor-use-class-input", action='store_true', default=False,
                       help="Accepted for ImageNet shortcut command compatibility; ignored by text-conditioned MDM.")
    group.add_argument("--shortcut-predictor-class-fusion", default="add", choices=["add", "concat"],
                       help="Accepted for ImageNet shortcut command compatibility; ignored by text-conditioned MDM.")
    
    group.add_argument("--target_joint_names", default='DIMP_FINAL', type=str, help="Force single joint configuration by specifing the joints (coma separated). If None - will use the random mode for all end effectors.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=6, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")

    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")

def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--dynamic_text_path", default='', type=str,
                       help="For the autoregressive mode only! Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--target_joint_names", default='DIMP_FINAL', type=str, help="Force single joint configuration by specifing the joints (coma separated). If None - will use the random mode for all end effectors.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return apply_rules(parser.parse_args())


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)
