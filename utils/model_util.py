import torch
from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode
from data_loaders.humanml_utils import HML_EE_JOINT_NAMES

def load_model_wo_clip(model, state_dict):
    # assert (state_dict['sequence_pos_encoder.pe'][:model.sequence_pos_encoder.pe.shape[0]] == model.sequence_pos_encoder.pe).all()  # TEST
    # assert (state_dict['embed_timestep.sequence_pos_encoder.pe'][:model.embed_timestep.sequence_pos_encoder.pe.shape[0]] == model.embed_timestep.sequence_pos_encoder.pe).all()  # TEST
    del state_dict['sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    del state_dict['embed_timestep.sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([
        k.startswith('clip_model.')
        or k.startswith('depth_shortcut_predictor.')
        or 'sequence_pos_encoder' in k
        for k in missing_keys
    ])


def create_model_and_diffusion(args, data):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6
    all_goal_joint_names = []

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
        all_goal_joint_names = ['pelvis'] + HML_EE_JOINT_NAMES
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    # Compatibility with old models
    if not hasattr(args, 'pred_len'):
        args.pred_len = 0
        args.context_len = 0
    
    emb_policy = args.__dict__.get('emb_policy', 'add')
    multi_target_cond = args.__dict__.get('multi_target_cond', False)
    multi_encoder_type = args.__dict__.get('multi_encoder_type', 'multi')
    target_enc_layers = args.__dict__.get('target_enc_layers', 1)

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'text_encoder_type': args.text_encoder_type,
            'pos_embed_max_len': args.pos_embed_max_len, 'mask_frames': args.mask_frames,
            'pred_len': args.pred_len, 'context_len': args.context_len, 'emb_policy': emb_policy,
            'all_goal_joint_names': all_goal_joint_names, 'multi_target_cond': multi_target_cond, 'multi_encoder_type': multi_encoder_type, 'target_enc_layers': target_enc_layers,
            'use_depth_shortcut': getattr(args, 'use_depth_shortcut', False),
            'shortcut_predictor': getattr(args, 'shortcut_predictor', 'hybrid_mdm_10'),
            'shortcut_mag_abs_center': getattr(args, 'shortcut_mag_abs_center', 5.5),
            'shortcut_mag_abs_scale': getattr(args, 'shortcut_mag_abs_scale', 1.5),
            }



def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]
    
    if hasattr(args, 'lambda_target_loc'):
        lambda_target_loc = args.lambda_target_loc
    else:
        lambda_target_loc = 0.

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_target_loc=lambda_target_loc,
        lambda_layersync=args.lambda_layersync,
        layersync_weak_layer=args.layersync_weak_layer,
        layersync_strong_layer=args.layersync_strong_layer,
        use_depth_shortcut=getattr(args, 'use_depth_shortcut', False),
        shortcut_training_mode=getattr(args, 'shortcut_training_mode', 'direction-magnitude'),
        shortcut_lambda_dir=getattr(args, 'shortcut_lambda_dir', 1.0),
        shortcut_lambda_boot=getattr(args, 'shortcut_lambda_boot', 0.25),
        shortcut_lambda_mag=getattr(args, 'shortcut_lambda_mag', 0.375),
        shortcut_lambda_boot_mag=getattr(args, 'shortcut_lambda_boot_mag', 0.1875),
        shortcut_bootstrap_detach_source=getattr(args, 'shortcut_bootstrap_detach_source', True),
        shortcut_mag_scale=getattr(args, 'shortcut_mag_scale', 3.0),
        shortcut_mag_clip_min=getattr(args, 'shortcut_mag_clip_min', 3.0),
        shortcut_mag_clip_max=getattr(args, 'shortcut_mag_clip_max', 8.0),
        shortcut_predictor_normalize_input=getattr(args, 'shortcut_predictor_normalize_input', False),
        shortcut_predictor_use_timestep=getattr(args, 'shortcut_predictor_use_timestep', True),
        shortcut_skip_in_loop_max_gap=getattr(args, 'shortcut_skip_in_loop_max_gap', 10),
        shortcut_skip_in_loop_gap_loc=getattr(args, 'shortcut_skip_in_loop_gap_loc', 3.0),
        shortcut_skip_in_loop_gap_sigma=getattr(args, 'shortcut_skip_in_loop_gap_sigma', 2.0),
        output_distill=getattr(args, 'output_distill', True),
        output_distill_ratio=getattr(args, 'output_distill_ratio', 0.10),
        lambda_output_distill=getattr(args, 'lambda_output_distill', 0.05),
        output_distill_every=getattr(args, 'output_distill_every', 1),
        output_distill_update_mode=getattr(args, 'output_distill_update_mode', 'predictor_plus_all'),
        output_distill_pair_mode=getattr(args, 'output_distill_pair_mode', 'trunc_normal'),
        direct_pair_mode=getattr(args, 'direct_pair_mode', 'trunc_normal'),
        pair_center_sigma=getattr(args, 'pair_center_sigma', 0.0),
        direct_num_pairs=getattr(args, 'direct_num_pairs', 1),
        direct_joint_pairs=getattr(args, 'direct_joint_pairs', 1),
        direct_predictor_only_pairs=getattr(args, 'direct_predictor_only_pairs', 0),
        private_loss=getattr(args, 'private_loss', True),
        lambda_private=getattr(args, 'lambda_private', 1.0),
        private_max_pairs=getattr(args, 'private_max_pairs', 2),
        private_use_residual=getattr(args, 'private_use_residual', True),
        private_cosine_mode=getattr(args, 'private_cosine_mode', 'bnd'),
        private_pair_mode=getattr(args, 'private_pair_mode', 'random'),
    )

def load_saved_model(model, model_path, use_avg: bool=False):  # use_avg_model
    state_dict = torch.load(model_path, map_location='cpu')
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
    # if use_avg_model:
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model, loading as usual.')
    load_model_wo_clip(model, state_dict)
    return model
