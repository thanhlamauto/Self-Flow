#!/usr/bin/env python3
"""
Sample images from a trained Self-Flow diffusion model (JAX/Flax).

Usage:
    python sample.py --ckpt path/to/checkpoint --output-dir ./samples

This script generates images for FID evaluation, outputting an NPZ file
compatible with the ADM evaluation suite.
"""

import os
import math
import argparse
import functools
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import collections.abc
import matplotlib.pyplot as plt
import seaborn as sns

# Import from local src/ folder
from flax.training import checkpoints as flax_ckpt
from src.model import SelfFlowDiT
from src.sampling import denoise_loop, create_transport, FixedSampler


DIT_VARIANTS = {
    "S":  {"hidden_size": 384,  "depth": 12, "num_heads": 6},
    "B":  {"hidden_size": 768,  "depth": 12, "num_heads": 12},
    "L":  {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
}


def compute_block_cosine_matrix(block_tokens):
    """
    Compute pairwise cosine similarity matrix between blocks.

    Args:
        block_tokens: List of [B, N, D] arrays, one per block

    Returns:
        sim_mat: [L, L] array of averaged cosine similarities
    """
    # Stack to [L, B, N, D]
    H = jnp.stack(block_tokens, axis=0)
    L, B, N, D = H.shape

    # Normalize along D dimension
    H_norm = H / (jnp.linalg.norm(H, axis=-1, keepdims=True) + 1e-8)

    # Compute pairwise cosine: [L, L, B, N]
    # H_norm[:, None] has shape [L, 1, B, N, D]
    # H_norm[None, :] has shape [1, L, B, N, D]
    # After multiplication and sum: [L, L, B, N]
    sim = jnp.einsum('lbnd,mbnd->lmbn', H_norm, H_norm)

    # Average over batch and patches: [L, L]
    sim_mat = jnp.mean(sim, axis=(-2, -1))

    return sim_mat


def visualize_similarity_matrices(all_sim_mats, timesteps, output_path, model_depth):
    """
    Visualize block-wise cosine similarity matrices across timesteps.

    Args:
        all_sim_mats: [T, L, L] array of similarity matrices
        timesteps: List of timestep indices
        output_path: Path to save visualization
        model_depth: Number of blocks (L)
    """
    num_timesteps = len(timesteps)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (ax, t_idx) in enumerate(zip(axes, range(num_timesteps))):
        if t_idx >= len(all_sim_mats):
            ax.axis('off')
            continue

        sim_mat = np.array(all_sim_mats[t_idx])

        # Plot heatmap
        sns.heatmap(
            sim_mat,
            ax=ax,
            cmap='RdYlBu_r',
            vmin=0.0,
            vmax=1.0,
            square=True,
            cbar_kws={'label': 'Cosine Similarity'},
            xticklabels=range(model_depth),
            yticklabels=range(model_depth)
        )

        ax.set_title(f'Timestep {timesteps[t_idx]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Block Index', fontsize=12)
        ax.set_ylabel('Block Index', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved similarity matrix visualization to {output_path}")
    plt.close()


def visualize_spatial_heatmap(block_tokens, block_a, block_b, grid_size, output_path):
    """
    Visualize spatial heatmap of cosine similarity for a pair of blocks.

    Args:
        block_tokens: List of [B, N, D] arrays
        block_a, block_b: Indices of blocks to compare
        grid_size: Spatial grid size (H_patch = W_patch)
        output_path: Path to save visualization
    """
    # Get tokens for the two blocks
    h_a = block_tokens[block_a]  # [B, N, D]
    h_b = block_tokens[block_b]  # [B, N, D]

    # Normalize
    h_a_norm = h_a / (jnp.linalg.norm(h_a, axis=-1, keepdims=True) + 1e-8)
    h_b_norm = h_b / (jnp.linalg.norm(h_b, axis=-1, keepdims=True) + 1e-8)

    # Compute cosine similarity per patch: [B, N]
    patch_sim = jnp.sum(h_a_norm * h_b_norm, axis=-1)

    # Average over batch: [N]
    patch_sim_avg = jnp.mean(patch_sim, axis=0)

    # Reshape to spatial grid: [H_patch, W_patch]
    patch_map = patch_sim_avg.reshape(grid_size, grid_size)

    # Plot
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        np.array(patch_map),
        cmap='RdYlBu_r',
        vmin=0.0,
        vmax=1.0,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(f'Spatial Similarity: Block {block_a} vs Block {block_b}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Patch X', fontsize=12)
    plt.ylabel('Patch Y', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved spatial heatmap to {output_path}")
    plt.close()


def _model_config_for_size(model_size):
    """Return the full model-init config dict for a DiT variant name (S/B/L/XL)."""
    variant = DIT_VARIANTS[model_size.upper()]
    return dict(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=variant["hidden_size"],
        depth=variant["depth"],
        num_heads=variant["num_heads"],
        mlp_ratio=4.0,
        num_classes=1001,
        learn_sigma=True,
        compatibility_mode=True,
    )


def create_npz_from_samples(samples, output_path):
    """Save samples to NPZ file for ADM evaluation."""
    samples = np.stack(samples, axis=0)
    np.savez(output_path, arr_0=samples)
    print(f"Saved {len(samples)} samples to {output_path}")


def load_vae(vae_model="stabilityai/sd-vae-ft-mse", dtype=jnp.bfloat16):
    """Load the SD-VAE for decoding latents to images."""
    from diffusers.models import FlaxAutoencoderKL

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        vae_model,
        from_pt=True,
        dtype=dtype,
    )
    scale_factor = 0.18215
    shift_factor = 0.0
    return vae, vae_params, scale_factor, shift_factor


def load_model(ckpt_path=None, model_size="XL"):
    """Load the DiT backbone from a flax.training.checkpoints checkpoint.

    This SiT baseline expects flat parameter trees (both online and EMA).
    For convenience, we also tolerate a few older shapes when loading:
      - Nested {"backbone": ...} checkpoints: extract "backbone".
      - Flat checkpoints that include "feature_head": drop that key.
    """
    config = _model_config_for_size(model_size)
    model = SelfFlowDiT(**config, per_token=False)

    # Initialize parameters with random key
    key = jax.random.PRNGKey(0)
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    dummy_x = jnp.ones((1, n_patches, patch_dim))
    dummy_t = jnp.ones((1,))
    dummy_vec = jnp.ones((1,), dtype=jnp.int32)

    variables = model.init(key, dummy_x, timesteps=dummy_t, vector=dummy_vec, deterministic=True)
    params = variables["params"]

    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")

        raw = flax_ckpt.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
        if raw is not None:
            if isinstance(raw, collections.abc.Mapping) and "backbone" in raw:
                raw = dict(raw["backbone"])
            elif isinstance(raw, collections.abc.Mapping) and "feature_head" in raw:
                raw = {k: v for k, v in raw.items() if k != "feature_head"}
            params = raw
    
    return model, params


def sample_with_block_analysis(
    model,
    params,
    vae,
    vae_params,
    scale_factor,
    shift_factor,
    rng,
    class_labels,
    batch_size,
    num_steps,
    cfg_scale,
    guidance_low,
    guidance_high,
    analyze_timesteps=[1, 4, 8, 32, 64, 127],
):
    """
    Sample images while collecting block-wise cosine similarity at specified timesteps.

    Returns:
        images: Generated images [B, H, W, C]
        all_sim_mats: List of similarity matrices at analyze_timesteps
        collected_timesteps: Actual timesteps where analysis was performed
    """
    latent_channels = 4
    latent_size = 32
    patch_size = 2

    # Generate noise
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(
        noise_rng,
        (batch_size, latent_channels, latent_size, latent_size),
        dtype=jnp.bfloat16
    )

    # Patchify
    x = rearrange(
        noise,
        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        p1=patch_size, p2=patch_size
    )
    token_h = latent_size // patch_size
    token_w = latent_size // patch_size

    # CFG setup
    use_cfg = cfg_scale > 1.0
    if use_cfg:
        x = jnp.concatenate([x, x], axis=0)
        null_labels = jnp.full_like(class_labels, 1000)
        class_labels_cfg = jnp.concatenate([null_labels, class_labels], axis=0)
    else:
        class_labels_cfg = class_labels

    # Setup transport and sampler
    from src.sampling import Config
    args = Config()
    args.num_steps = num_steps
    transport = create_transport(
        args.transport.path_type,
        args.transport.prediction,
        args.transport.loss_weight,
        args.transport.train_eps,
        args.transport.sample_eps,
    )

    sampler = FixedSampler(transport)
    t0, t1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        diffusion_form=args.sde.diffusion_form,
        sde=True,
        eval=True,
        reverse=False,
        last_step_size=args.sde.last_step_size,
    )

    # Create timestep schedule
    timesteps = jnp.linspace(t0, t1, num_steps)
    dt = timesteps[1] - timesteps[0]

    # Prepare to collect similarity matrices
    all_sim_mats = []
    collected_timesteps = []

    # Manual sampling loop to collect block tokens
    current_x = x
    rng, sample_rng = jax.random.split(rng)

    print(f"\nSampling with block analysis at timesteps: {analyze_timesteps}")

    for step_idx in tqdm(range(num_steps - 1), desc="Sampling"):
        t = timesteps[step_idx]
        t_batch = jnp.ones(current_x.shape[0]) * t

        # Check if we should analyze this timestep
        should_analyze = step_idx in analyze_timesteps

        # Forward pass
        if should_analyze:
            # Get block tokens for analysis
            pred, block_tokens = model.apply(
                {"params": params},
                current_x,
                timesteps=t_batch,
                vector=class_labels_cfg,
                deterministic=True,
                return_block_tokens=True
            )

            # Extract conditional branch for CFG
            if use_cfg:
                # Assuming conditional is the second half
                block_tokens_cond = [bt[batch_size:] for bt in block_tokens]
            else:
                block_tokens_cond = block_tokens

            # Compute similarity matrix
            sim_mat = compute_block_cosine_matrix(block_tokens_cond)
            all_sim_mats.append(sim_mat)
            collected_timesteps.append(step_idx)
        else:
            # Normal forward without block tokens
            pred = model.apply(
                {"params": params},
                current_x,
                timesteps=t_batch,
                vector=class_labels_cfg,
                deterministic=True
            )

        # Apply CFG if needed
        if use_cfg:
            pred_u, pred_c = jnp.split(pred, 2, axis=0)
            pred = pred_u + cfg_scale * (pred_c - pred_u)
            # For drift calculation, we only use CFG-combined prediction
            pred_for_update = jnp.concatenate([pred, pred], axis=0)
        else:
            pred_for_update = pred

        # Reverse the prediction (matching the wrapper in denoise_loop)
        pred_for_update = -pred_for_update

        # Euler-Maruyama step
        rng, step_rng = jax.random.split(rng)
        w_cur = jax.random.normal(step_rng, current_x.shape)
        dw = w_cur * jnp.sqrt(dt)

        # Compute drift and diffusion
        drift = pred_for_update
        diffusion_val = transport.path_sampler.compute_diffusion(
            current_x, t_batch, form=args.sde.diffusion_form, norm=args.sde.diffusion_norm
        )

        mean_x = current_x + drift * dt
        current_x = mean_x + jnp.sqrt(2 * diffusion_val) * dw

    # Last step
    t_last = timesteps[-1]
    t_last_batch = jnp.ones(current_x.shape[0]) * t_last
    pred_last = model.apply(
        {"params": params},
        current_x,
        timesteps=t_last_batch,
        vector=class_labels_cfg,
        deterministic=True
    )

    if use_cfg:
        pred_u, pred_c = jnp.split(pred_last, 2, axis=0)
        pred_last = pred_u + cfg_scale * (pred_c - pred_u)

    pred_last = -pred_last
    current_x = current_x + pred_last * args.sde.last_step_size

    # Extract conditional samples if CFG
    if use_cfg:
        samples = current_x[batch_size:]
    else:
        samples = current_x

    # Unpatchify
    samples = rearrange(
        samples,
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h=token_h, w=token_w,
        p1=patch_size, p2=patch_size, c=latent_channels
    )

    # Decode latents to images
    latents = samples / scale_factor + shift_factor
    latents = jnp.transpose(latents, (0, 2, 3, 1))  # (B, H, W, C)

    images = vae.apply({"params": vae_params}, latents, method=vae.decode).sample
    images = jnp.transpose(images, (0, 2, 3, 1))
    images = (images + 1.0) / 2.0
    images = jnp.clip(images, 0.0, 1.0)
    images = (images * 255.0).astype(jnp.uint8)

    return images, all_sim_mats, collected_timesteps


def build_sample_step(model, vae, scale_factor, shift_factor):
    """Build JIT-compiled sampling function."""
    
    @functools.partial(jax.jit, static_argnames=("batch_size", "num_steps"))
    def sample_batch_jit(
        params,
        vae_params,
        rng,
        class_labels,
        batch_size,
        num_steps,
        cfg_scale,
        guidance_low,
        guidance_high,
    ):
        latent_channels = 4
        latent_size = 32
        patch_size = 2
        
        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(
            noise_rng, 
            (batch_size, latent_channels, latent_size, latent_size),
            dtype=jnp.bfloat16
        )
        
        # Patchify matching the training dataloader: each token in (p1 p2 c) order.
        x = rearrange(
            noise,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_size, p2=patch_size
        )
        token_h = latent_size // patch_size
        token_w = latent_size // patch_size
        
        # Enable CFG
        use_cfg = cfg_scale > 1.0
        if use_cfg:
            x = jnp.concatenate([x, x], axis=0)
            null_labels = jnp.full_like(class_labels, 1000)
            class_labels = jnp.concatenate([null_labels, class_labels], axis=0)
            
        def model_fn(z_x, t):
            # z_x has dynamic shape inside compilation, but tracing fixes it
            return model.apply(
                {"params": params},
                z_x,
                timesteps=t,
                vector=class_labels,
                deterministic=True
            )
            
        rng, denoise_rng = jax.random.split(rng)
        samples = denoise_loop(
            model_fn=model_fn,
            x=x,
            rng=denoise_rng,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            mode="SDE",
            reverse=False,
        )
        
        if use_cfg:
            samples = samples[batch_size:]
        
        # Unpatchify: inverse of (p1 p2 c) train patchify → NCHW latent.
        samples = rearrange(
            samples,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=token_h, w=token_w,
            p1=patch_size, p2=patch_size, c=latent_channels
        )
        
        # Decode latents using VAE
        latents = samples / scale_factor + shift_factor
        latents = jnp.transpose(latents, (0, 2, 3, 1))  # (B, H, W, C)
        
        images = vae.apply({"params": vae_params}, latents, method=vae.decode).sample
        images = jnp.transpose(images, (0, 2, 3, 1))  # Diffusers Flax VAE decode returns NCHW
        images = (images + 1.0) / 2.0
        images = jnp.clip(images, 0.0, 1.0)
        
        images = (images * 255.0).astype(jnp.uint8)
        return images
        
    return sample_batch_jit


def main():
    parser = argparse.ArgumentParser(description="Sample images from vanilla SiT model (JAX)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./samples", help="Output directory")
    parser.add_argument("--num-fid-samples", type=int, default=50000, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-steps", type=int, default=250, help="Number of diffusion steps")
    parser.add_argument("--mode", type=str, default="SDE", choices=["SDE"], help="Sampling mode")
    parser.add_argument("--seed", type=int, default=31, help="Random seed")
    parser.add_argument("--save-images", action="store_true", default=True, help="Save individual PNG images")
    parser.add_argument("--no-save-images", action="store_false", dest="save_images")
    parser.add_argument("--model-size", type=str, default="XL", choices=["S", "B", "L", "XL"], help="DiT backbone size: S, B, L, XL")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse",
                        choices=["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"],
                        help="HuggingFace VAE model ID")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale (1.0 = no guidance)")
    parser.add_argument("--guidance-low", type=float, default=0.0, help="Lower guidance bound")
    parser.add_argument("--guidance-high", type=float, default=0.7, help="Upper guidance bound")
    parser.add_argument("--analyze-blocks", action="store_true", help="Analyze block-wise cosine similarity")
    parser.add_argument("--analyze-timesteps", type=str, default="1,4,8,32,64,127",
                        help="Comma-separated timesteps to analyze (default: 1,4,8,32,64,127)")
    args = parser.parse_args()
    
    print(f"Generating {args.num_fid_samples} samples")
    print(f"Mode: {args.mode}, Steps: {args.num_steps}, CFG: {args.cfg_scale}")
    
    rng = jax.random.PRNGKey(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (output_dir / "images").mkdir(exist_ok=True)
        
    model, params = load_model(args.ckpt, model_size=args.model_size)
    vae, vae_params, scale_factor, shift_factor = load_vae(vae_model=args.vae_model)

    # Parse analyze timesteps
    analyze_timesteps = [int(t) for t in args.analyze_timesteps.split(',')]

    # Get model depth
    config = _model_config_for_size(args.model_size)
    model_depth = config["depth"]
    grid_size = config["input_size"] // config["patch_size"]

    if args.analyze_blocks:
        print(f"\n{'='*60}")
        print(f"BLOCK ANALYSIS MODE ENABLED")
        print(f"Model depth: {model_depth} blocks")
        print(f"Analyzing timesteps: {analyze_timesteps}")
        print(f"{'='*60}\n")

    sample_step_fn = build_sample_step(model, vae, scale_factor, shift_factor)

    total_samples = args.num_fid_samples
    num_batches = math.ceil(total_samples / args.batch_size)

    all_samples = []
    all_sim_mats_collection = []  # Collect similarity matrices from all batches
    
    for batch_idx in tqdm(range(num_batches), desc="Sampling"):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total_samples)
        needed = batch_end - batch_start

        rng, class_rng, step_rng = jax.random.split(rng, 3)
        # Keep JIT shapes static: always run with the full batch size, then slice.
        class_labels = jax.random.randint(class_rng, (args.batch_size,), 0, 1000)

        if args.analyze_blocks and batch_idx == 0:
            # Run with block analysis for first batch only
            print(f"\nRunning batch {batch_idx} with block analysis...")
            images, batch_sim_mats, collected_timesteps = sample_with_block_analysis(
                model=model,
                params=params,
                vae=vae,
                vae_params=vae_params,
                scale_factor=scale_factor,
                shift_factor=shift_factor,
                rng=step_rng,
                class_labels=class_labels,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                guidance_low=args.guidance_low,
                guidance_high=args.guidance_high,
                analyze_timesteps=analyze_timesteps,
            )
            all_sim_mats_collection.extend(batch_sim_mats)
            print(f"Collected {len(batch_sim_mats)} similarity matrices at timesteps: {collected_timesteps}")
        else:
            # Normal sampling
            images = sample_step_fn(
                params=params,
                vae_params=vae_params,
                rng=step_rng,
                class_labels=class_labels,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                guidance_low=args.guidance_low,
                guidance_high=args.guidance_high,
            )

        # JAX arrays to NumPy
        images_np = np.asarray(images)[:needed]
        all_samples.append(images_np)

        if args.save_images:
            for i, img in enumerate(images_np):
                global_idx = batch_start + i
                Image.fromarray(img).save(output_dir / "images" / f"{global_idx:06d}.png")
                
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = all_samples[:args.num_fid_samples]
    
    npz_path = output_dir / f"samples_{len(all_samples)}.npz"
    create_npz_from_samples(list(all_samples), npz_path)

    print(f"Done! NPZ saved at: {npz_path}")

    # Visualize block analysis if enabled
    if args.analyze_blocks and len(all_sim_mats_collection) > 0:
        print(f"\n{'='*60}")
        print(f"VISUALIZING BLOCK ANALYSIS RESULTS")
        print(f"{'='*60}\n")

        # Save similarity matrices as numpy array
        sim_mats_array = np.stack([np.array(m) for m in all_sim_mats_collection], axis=0)
        sim_mats_path = output_dir / "block_similarity_matrices.npz"
        np.savez(sim_mats_path, similarity_matrices=sim_mats_array, timesteps=np.array(analyze_timesteps[:len(all_sim_mats_collection)]))
        print(f"Saved similarity matrices to: {sim_mats_path}")

        # Visualize similarity matrices
        viz_path = output_dir / "block_similarity_heatmaps.png"
        visualize_similarity_matrices(
            all_sim_mats_collection,
            analyze_timesteps[:len(all_sim_mats_collection)],
            viz_path,
            model_depth
        )

        # Visualize spatial heatmaps for select block pairs
        # We'll use the last analyzed timestep for spatial visualization
        if len(all_sim_mats_collection) > 0:
            print("\nGenerating spatial heatmaps for selected block pairs...")

            # Get block tokens at the last analyzed timestep
            # We need to run one more forward pass to get block tokens
            print("Running forward pass to get block tokens for spatial analysis...")
            rng, spatial_rng = jax.random.split(rng)
            test_labels = jnp.array([281])  # Cat class for example

            # Generate initial noise
            noise = jax.random.normal(
                spatial_rng,
                (1, 4, 32, 32),
                dtype=jnp.bfloat16
            )
            x_test = rearrange(noise, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=2, p2=2)

            # Forward pass
            t_test = jnp.array([0.5])
            _, block_tokens_test = model.apply(
                {"params": params},
                x_test,
                timesteps=t_test,
                vector=test_labels,
                deterministic=True,
                return_block_tokens=True
            )

            # Visualize shallow vs deep block pairs
            pairs = [
                (0, 1),  # Adjacent shallow blocks
                (0, model_depth-1),  # Shallowest vs deepest
                (model_depth//2-1, model_depth//2),  # Middle adjacent blocks
            ]

            for block_a, block_b in pairs:
                spatial_path = output_dir / f"spatial_similarity_block{block_a}_vs_block{block_b}.png"
                visualize_spatial_heatmap(
                    block_tokens_test,
                    block_a,
                    block_b,
                    grid_size,
                    spatial_path
                )

        print(f"\n{'='*60}")
        print(f"BLOCK ANALYSIS COMPLETE")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
