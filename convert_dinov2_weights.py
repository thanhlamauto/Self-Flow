#!/usr/bin/env python3
"""
Convert DINOv2 ViT-B/14 weights from PyTorch to Flax format.

Run once on a GPU/CPU machine:
    python convert_dinov2_weights.py --output dinov2_vitb14_flax.pkl

Then upload the .pkl file to Kaggle as a dataset for TPU training.
"""

import argparse
import pickle

import numpy as np
import torch


def convert_dinov2_weights(pt_state_dict):
    """Convert PyTorch DINOv2 state_dict to flat Flax param dict.

    Flax key convention uses '/' separators matching the module names
    in src/dinov2_flax.py.

    Conversions:
        - Dense/Linear weight: (out, in) → kernel (in, out)  [transpose]
        - Conv2d weight: (out, in, h, w) → kernel (h, w, in, out)  [permute]
        - LayerNorm weight → scale, bias → bias  [rename only]
        - LayerScale gamma → direct copy
        - cls_token, pos_embed → direct copy
    """
    flax_params = {}

    for pt_key, tensor in pt_state_dict.items():
        arr = tensor.detach().cpu().numpy()

        # --- Global tokens / embeddings ---
        if pt_key == "cls_token":
            # Shape: (1, 1, 768)
            flax_params["cls_token"] = arr
            continue

        if pt_key == "pos_embed":
            # Shape: (1, 257, 768)
            flax_params["pos_embed"] = arr
            continue

        # Skip mask_token (not used in inference)
        if pt_key == "mask_token":
            continue

        # --- Patch embedding (Conv2d) ---
        if pt_key == "patch_embed.proj.weight":
            # PyTorch Conv2d: (out, in, h, w) → Flax Conv: (h, w, in, out)
            flax_params["patch_embed/Conv_0/kernel"] = arr.transpose(2, 3, 1, 0)
            continue
        if pt_key == "patch_embed.proj.bias":
            flax_params["patch_embed/Conv_0/bias"] = arr
            continue

        # --- Transformer blocks ---
        # PyTorch: blocks.{i}.{submodule}
        if pt_key.startswith("blocks."):
            parts = pt_key.split(".")
            block_idx = parts[1]
            flax_prefix = f"blocks_{block_idx}"
            rest = ".".join(parts[2:])

            # LayerScale
            if rest == "ls1.gamma":
                flax_params[f"{flax_prefix}/ls1_gamma"] = arr
                continue
            if rest == "ls2.gamma":
                flax_params[f"{flax_prefix}/ls2_gamma"] = arr
                continue

            # LayerNorm 1 (pre-attention)
            if rest == "norm1.weight":
                flax_params[f"{flax_prefix}/norm1/scale"] = arr
                continue
            if rest == "norm1.bias":
                flax_params[f"{flax_prefix}/norm1/bias"] = arr
                continue

            # LayerNorm 2 (pre-MLP)
            if rest == "norm2.weight":
                flax_params[f"{flax_prefix}/norm2/scale"] = arr
                continue
            if rest == "norm2.bias":
                flax_params[f"{flax_prefix}/norm2/bias"] = arr
                continue

            # Attention QKV (combined Dense)
            if rest == "attn.qkv.weight":
                # (out=dim*3, in=dim) → kernel (in, out)
                flax_params[f"{flax_prefix}/attn/qkv/kernel"] = arr.T
                continue
            if rest == "attn.qkv.bias":
                flax_params[f"{flax_prefix}/attn/qkv/bias"] = arr
                continue

            # Attention output projection
            if rest == "attn.proj.weight":
                flax_params[f"{flax_prefix}/attn/proj/kernel"] = arr.T
                continue
            if rest == "attn.proj.bias":
                flax_params[f"{flax_prefix}/attn/proj/bias"] = arr
                continue

            # MLP fc1
            if rest == "mlp.fc1.weight":
                flax_params[f"{flax_prefix}/mlp/fc1/kernel"] = arr.T
                continue
            if rest == "mlp.fc1.bias":
                flax_params[f"{flax_prefix}/mlp/fc1/bias"] = arr
                continue

            # MLP fc2
            if rest == "mlp.fc2.weight":
                flax_params[f"{flax_prefix}/mlp/fc2/kernel"] = arr.T
                continue
            if rest == "mlp.fc2.bias":
                flax_params[f"{flax_prefix}/mlp/fc2/bias"] = arr
                continue

            print(f"  [SKIP] Unhandled block key: {pt_key}")
            continue

        # --- Final LayerNorm ---
        if pt_key == "norm.weight":
            flax_params["norm/scale"] = arr
            continue
        if pt_key == "norm.bias":
            flax_params["norm/bias"] = arr
            continue

        # Skip head (DINOv2 classification head — not used)
        if pt_key.startswith("head."):
            continue

        print(f"  [SKIP] Unhandled key: {pt_key}")

    return flax_params


def main():
    parser = argparse.ArgumentParser(description="Convert DINOv2 ViT-B/14 PyTorch weights to Flax pickle.")
    parser.add_argument("--output", type=str, default="dinov2_vitb14_flax.pkl",
                        help="Output pickle file path")
    parser.add_argument("--model", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vitb14", "dinov2_vits14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="DINOv2 model variant to download")
    args = parser.parse_args()

    print(f"Downloading {args.model} from torch.hub...")
    pt_model = torch.hub.load("facebookresearch/dinov2", args.model)
    pt_model.eval()
    pt_state = pt_model.state_dict()

    print(f"Converting {len(pt_state)} PyTorch parameters to Flax format...")
    flax_params = convert_dinov2_weights(pt_state)

    print(f"Converted {len(flax_params)} Flax parameters.")

    # Validation: print shape summary
    total_params = 0
    for key in sorted(flax_params.keys()):
        shape = flax_params[key].shape
        total_params += np.prod(shape)
        print(f"  {key}: {shape}")

    print(f"\nTotal parameters: {total_params:,} ({total_params * 4 / 1e6:.1f} MB float32)")

    with open(args.output, "wb") as f:
        pickle.dump(flax_params, f)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
