"""DINOv2 Vision Transformer (ViT-B/14) in Flax."""

from __future__ import annotations

import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp


class PatchEmbed(nn.Module):
    """Convert image patches to embeddings via Conv2D."""

    embed_dim: int = 768
    patch_size: int = 14

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True,
        )(x)
        batch, height, width, channels = x.shape
        return x.reshape(batch, height * width, channels)


class Attention(nn.Module):
    """Multi-head self-attention with combined QKV projection."""

    dim: int = 768
    num_heads: int = 12

    @nn.compact
    def __call__(self, x):
        batch, num_tokens, channels = x.shape
        head_dim = self.dim // self.num_heads
        scale = head_dim**-0.5

        qkv = nn.Dense(self.dim * 3, use_bias=True, name="qkv")(x)
        qkv = qkv.reshape(batch, num_tokens, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = jax.nn.softmax(attn, axis=-1)

        x = attn @ v
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(batch, num_tokens, channels)
        return nn.Dense(self.dim, use_bias=True, name="proj")(x)


class MLP(nn.Module):
    """Feed-forward network used in DINOv2 blocks."""

    dim: int = 768
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x):
        hidden_dim = int(self.dim * self.mlp_ratio)
        x = nn.Dense(hidden_dim, use_bias=True, name="fc1")(x)
        x = nn.gelu(x)
        return nn.Dense(self.dim, use_bias=True, name="fc2")(x)


class Block(nn.Module):
    """Transformer block with LayerScale."""

    dim: int = 768
    num_heads: int = 12

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(name="norm1")(x)
        y = Attention(dim=self.dim, num_heads=self.num_heads, name="attn")(y)
        ls1 = self.param("ls1_gamma", nn.initializers.ones, (self.dim,))
        x = x + ls1 * y

        y = nn.LayerNorm(name="norm2")(x)
        y = MLP(dim=self.dim, name="mlp")(y)
        ls2 = self.param("ls2_gamma", nn.initializers.ones, (self.dim,))
        x = x + ls2 * y
        return x


class DINOv2ViT(nn.Module):
    """DINOv2 ViT-B/14 feature extractor returning patch tokens."""

    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    patch_size: int = 14
    img_size: int = 224

    def setup(self):
        num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.zeros,
            (1, 1, self.embed_dim),
        )
        self.pos_embed = self.param(
            "pos_embed",
            nn.initializers.zeros,
            (1, num_patches + 1, self.embed_dim),
        )

    @nn.compact
    def __call__(self, x):
        batch = x.shape[0]
        x = PatchEmbed(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            name="patch_embed",
        )(x)
        cls_tokens = jnp.broadcast_to(self.cls_token, (batch, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + self.pos_embed

        for idx in range(self.depth):
            x = Block(dim=self.embed_dim, num_heads=self.num_heads, name=f"blocks_{idx}")(x)

        x = nn.LayerNorm(name="norm")(x)
        return x[:, 1:, :]


def _unflatten_params(flat_dict):
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split("/")
        current = nested
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = jnp.array(value)
    return nested


def _interpolate_pos_embed(pos_embed, target_num_patches):
    cls_pos = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]

    source_num_patches = patch_pos.shape[1]
    if source_num_patches == target_num_patches:
        return pos_embed

    source_grid = int(source_num_patches**0.5)
    target_grid = int(target_num_patches**0.5)
    dim = patch_pos.shape[-1]

    patch_pos = patch_pos.reshape(1, source_grid, source_grid, dim)
    patch_pos = jax.image.resize(
        patch_pos,
        (1, target_grid, target_grid, dim),
        method="bilinear",
    )
    patch_pos = patch_pos.reshape(1, target_num_patches, dim)
    return jnp.concatenate([cls_pos, patch_pos], axis=1)


def load_dinov2_params(pkl_path, img_size=224, patch_size=14):
    """Load converted DINOv2 Flax params from a pickle file."""

    with open(pkl_path, "rb") as handle:
        flat_params = pickle.load(handle)

    target_num_patches = (img_size // patch_size) ** 2
    if "pos_embed" in flat_params:
        pos_embed = jnp.array(flat_params["pos_embed"])
        if pos_embed.shape[1] != target_num_patches + 1:
            flat_params["pos_embed"] = _interpolate_pos_embed(pos_embed, target_num_patches)

    return _unflatten_params(flat_params)
