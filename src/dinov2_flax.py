"""
DINOv2 Vision Transformer (ViT-B/14) in Flax.

Minimal implementation matching the PyTorch DINOv2 architecture for weight loading.
Used as a frozen feature extractor for REPA alignment during SiT training.
"""

import pickle
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


class PatchEmbed(nn.Module):
    """Convert image patches to embeddings via Conv2D."""
    embed_dim: int = 768
    patch_size: int = 14

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C) NHWC
        x = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True,
        )(x)
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)  # (B, num_patches, embed_dim)


class Attention(nn.Module):
    """Multi-head self-attention with combined QKV projection (matches PyTorch DINOv2)."""
    dim: int = 768
    num_heads: int = 12

    @nn.compact
    def __call__(self, x):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5

        # Combined QKV projection — matches PyTorch: self.qkv = nn.Linear(dim, dim*3)
        qkv = nn.Dense(self.dim * 3, use_bias=True, name="qkv")(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = jax.nn.softmax(attn, axis=-1)

        x = attn @ v  # (B, heads, N, head_dim)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(B, N, C)

        # Output projection — matches PyTorch: self.proj = nn.Linear(dim, dim)
        x = nn.Dense(self.dim, use_bias=True, name="proj")(x)
        return x


class MLP(nn.Module):
    """Feed-forward network: Dense → GELU → Dense."""
    dim: int = 768
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x):
        hidden_dim = int(self.dim * self.mlp_ratio)
        x = nn.Dense(hidden_dim, use_bias=True, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, use_bias=True, name="fc2")(x)
        return x


class Block(nn.Module):
    """Transformer block with LayerScale (DINOv2 architecture)."""
    dim: int = 768
    num_heads: int = 12

    @nn.compact
    def __call__(self, x):
        # Pre-norm attention + LayerScale
        y = nn.LayerNorm(name="norm1")(x)
        y = Attention(dim=self.dim, num_heads=self.num_heads, name="attn")(y)
        ls1 = self.param("ls1_gamma", nn.initializers.ones, (self.dim,))
        x = x + ls1 * y

        # Pre-norm MLP + LayerScale
        y = nn.LayerNorm(name="norm2")(x)
        y = MLP(dim=self.dim, name="mlp")(y)
        ls2 = self.param("ls2_gamma", nn.initializers.ones, (self.dim,))
        x = x + ls2 * y

        return x


class DINOv2ViT(nn.Module):
    """DINOv2 Vision Transformer feature extractor.

    Default config is ViT-B/14:
        embed_dim=768, depth=12, num_heads=12, patch_size=14
        Input: (B, 224, 224, 3) NHWC
        Output: (B, 256, 768) patch tokens (CLS excluded)
    """
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    patch_size: int = 14
    img_size: int = 224

    def setup(self):
        num_patches = (self.img_size // self.patch_size) ** 2  # 256 for 224/14
        self.cls_token = self.param(
            "cls_token", nn.initializers.zeros, (1, 1, self.embed_dim)
        )
        self.pos_embed = self.param(
            "pos_embed", nn.initializers.zeros, (1, num_patches + 1, self.embed_dim)
        )

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (B, 224, 224, 3) NHWC normalized images
        Returns:
            (B, 256, 768) patch token features (CLS token excluded)
        """
        B = x.shape[0]

        # Patch embedding
        x = PatchEmbed(embed_dim=self.embed_dim, patch_size=self.patch_size, name="patch_embed")(x)

        # Prepend CLS token
        cls_tokens = jnp.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (B, 257, 768)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer blocks
        for i in range(self.depth):
            x = Block(dim=self.embed_dim, num_heads=self.num_heads, name=f"blocks_{i}")(x)

        # Final layer norm
        x = nn.LayerNorm(name="norm")(x)

        # Return patch tokens only (exclude CLS)
        return x[:, 1:, :]  # (B, 256, 768)


def _unflatten_params(flat_dict):
    """Convert flat param dict with '/' separators to nested dict for Flax."""
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split("/")
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = jnp.array(value)
    return nested


def _interpolate_pos_embed(pos_embed, target_num_patches):
    """Interpolate pos_embed from source grid to target grid size.

    DINOv2 ViT-B/14 default is img_size=518 → 37x37=1369 patches,
    but we use img_size=224 → 16x16=256 patches.

    Args:
        pos_embed: (1, 1+src_patches, dim) — CLS + patch positional embeddings
        target_num_patches: number of target patches (e.g. 256 for 224/14)
    Returns:
        (1, 1+target_num_patches, dim)
    """
    cls_pos = pos_embed[:, :1, :]  # (1, 1, dim)
    patch_pos = pos_embed[:, 1:, :]  # (1, src_patches, dim)

    src_patches = patch_pos.shape[1]
    if src_patches == target_num_patches:
        return pos_embed

    src_grid = int(src_patches ** 0.5)
    tgt_grid = int(target_num_patches ** 0.5)
    dim = patch_pos.shape[-1]

    # (1, src_grid, src_grid, dim) → resize → (1, tgt_grid, tgt_grid, dim)
    patch_pos = patch_pos.reshape(1, src_grid, src_grid, dim)
    patch_pos = jax.image.resize(
        patch_pos, (1, tgt_grid, tgt_grid, dim), method="bilinear"
    )
    patch_pos = patch_pos.reshape(1, target_num_patches, dim)

    return jnp.concatenate([cls_pos, patch_pos], axis=1)


def load_dinov2_params(pkl_path, img_size=224, patch_size=14):
    """Load converted DINOv2 Flax params from pickle file.

    Automatically interpolates pos_embed if the source resolution
    differs from the target (e.g. 518→224).

    Args:
        pkl_path: path to dinov2_vitb14_flax.pkl (output of convert_dinov2_weights.py)
        img_size: target image size (default 224)
        patch_size: patch size (default 14)

    Returns:
        Nested Flax param dict ready for model.apply({"params": params}, ...)
    """
    with open(pkl_path, "rb") as f:
        flat_params = pickle.load(f)

    # Interpolate pos_embed to target resolution
    target_num_patches = (img_size // patch_size) ** 2  # 256 for 224/14
    if "pos_embed" in flat_params:
        pos = jnp.array(flat_params["pos_embed"])
        if pos.shape[1] != target_num_patches + 1:
            pos = _interpolate_pos_embed(pos, target_num_patches)
            flat_params["pos_embed"] = pos

    return _unflatten_params(flat_params)
