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
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import orbax.checkpoint as ocp

# Import from local src/ folder
from src.model import SelfFlowPerTokenDiT
from src.sampling import denoise_loop


def create_npz_from_samples(samples, output_path):
    """Save samples to NPZ file for ADM evaluation."""
    samples = np.stack(samples, axis=0)
    np.savez(output_path, arr_0=samples)
    print(f"Saved {len(samples)} samples to {output_path}")


def load_vae(dtype=jnp.bfloat16):
    """Load the SD-VAE for decoding latents to images."""
    from diffusers.models import FlaxAutoencoderKL

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-ema",
        from_pt=True,
        dtype=dtype,
    )
    scale_factor = 0.18215
    shift_factor = 0.0
    return vae, vae_params, scale_factor, shift_factor


def load_model(ckpt_path=None):
    """Load the Self-Flow model from checkpoint."""
    # Create model with DiT-XL/2 settings
    config = dict(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1001,
        learn_sigma=True,
        compatibility_mode=True,
    )
    model = SelfFlowPerTokenDiT(**config)
    
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
        # Assuming Flax native checkpoints via Orbax:
        options = ocp.CheckpointManagerOptions()
        checkpoint_manager = ocp.CheckpointManager(ckpt_path, options=options)
        
        # Orbax usually requires the target to know the shape
        restored = checkpoint_manager.restore(checkpoint_manager.latest_step(), args=ocp.args.StandardRestore(params))
        if restored is not None:
             params = restored
    
    return model, params


def build_sample_step(model, vae, scale_factor, shift_factor):
    """Build JIT-compiled sampling function."""
    
    @jax.jit
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
            mode="SDE"
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
        images = (images + 1.0) / 2.0
        images = jnp.clip(images, 0.0, 1.0)
        
        images = (images * 255.0).astype(jnp.uint8)
        return images
        
    return sample_batch_jit


def main():
    parser = argparse.ArgumentParser(description="Sample images from Self-Flow model (JAX)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./samples", help="Output directory")
    parser.add_argument("--num-fid-samples", type=int, default=50000, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-steps", type=int, default=250, help="Number of diffusion steps")
    parser.add_argument("--mode", type=str, default="SDE", choices=["SDE"], help="Sampling mode")
    parser.add_argument("--seed", type=int, default=31, help="Random seed")
    parser.add_argument("--save-images", action="store_true", default=True, help="Save individual PNG images")
    parser.add_argument("--no-save-images", action="store_false", dest="save_images")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale (1.0 = no guidance)")
    parser.add_argument("--guidance-low", type=float, default=0.0, help="Lower guidance bound")
    parser.add_argument("--guidance-high", type=float, default=0.7, help="Upper guidance bound")
    args = parser.parse_args()
    
    print(f"Generating {args.num_fid_samples} samples")
    print(f"Mode: {args.mode}, Steps: {args.num_steps}, CFG: {args.cfg_scale}")
    
    rng = jax.random.PRNGKey(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (output_dir / "images").mkdir(exist_ok=True)
        
    model, params = load_model(args.ckpt)
    vae, vae_params, scale_factor, shift_factor = load_vae()
    
    sample_step_fn = build_sample_step(model, vae, scale_factor, shift_factor)
    
    total_samples = args.num_fid_samples
    num_batches = math.ceil(total_samples / args.batch_size)
    
    all_samples = []
    
    for batch_idx in tqdm(range(num_batches), desc="Sampling"):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total_samples)
        current_batch_size = batch_end - batch_start
        
        rng, class_rng, step_rng = jax.random.split(rng, 3)
        class_labels = jax.random.randint(class_rng, (current_batch_size,), 0, 1000)
        
        images = sample_step_fn(
            params=params,
            vae_params=vae_params,
            rng=step_rng,
            class_labels=class_labels,
            batch_size=current_batch_size,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
        )
        
        # JAX arrays to NumPy
        images_np = np.asarray(images)
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

if __name__ == "__main__":
    main()
