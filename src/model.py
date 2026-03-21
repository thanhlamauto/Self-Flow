"""
Self-Flow Model (Flax version).

This module contains the SelfFlowPerTokenDiT model, a Diffusion Transformer
with per-token timestep conditioning for Self-Flow training, implemented in Flax.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = jnp.einsum("m,d->md", pos, omega)
    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


class PatchedPatchEmbed(nn.Module):
    """Simplified Sequence to Patch Embedding using Linear layer."""
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(self.embed_dim, use_bias=self.bias, name="proj")(x)


def modulate(x, shift, scale):
    """Standard modulation with unsqueeze for (N, D) conditioning."""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def modulate_per_token(x, shift, scale):
    """Per-token modulation for (N, T, D) conditioning."""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    hidden_size: int
    frequency_embedding_size: int = 256

    def timestep_embedding(self, t, dim, max_period=10000.0):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    @nn.compact
    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        x = nn.Dense(self.hidden_size)(t_freq)
        x = nn.swish(x)
        x = nn.Dense(self.hidden_size)(x)
        return x


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""
    num_classes: int
    hidden_size: int
    dropout_prob: float

    @nn.compact
    def __call__(self, labels, deterministic: bool = True, force_drop_ids=None):
        use_cfg_embedding = self.dropout_prob > 0
        embedding_table = nn.Embed(
            num_embeddings=self.num_classes + use_cfg_embedding, 
            features=self.hidden_size
        )

        use_dropout = self.dropout_prob > 0
        if (not deterministic and use_dropout) or (force_drop_ids is not None):
            if force_drop_ids is None:
                rng = self.make_rng('dropout')
                drop_ids = jax.random.uniform(rng, labels.shape) < self.dropout_prob
            else:
                drop_ids = force_drop_ids == 1
            labels = jnp.where(drop_ids, self.num_classes, labels)

        return embedding_table(labels)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    per_token: bool = False

    @nn.compact
    def __call__(self, x, c):
        norm1 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        norm2 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        
        if self.per_token:
            batch_size, seq_len, hidden_dim = c.shape
            c_flat = c.reshape(-1, hidden_dim)
            modulation_flat = nn.Sequential([
                nn.swish,
                nn.Dense(6 * self.hidden_size)
            ])(c_flat)
            modulation = modulation_flat.reshape(batch_size, seq_len, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)
            
            x_norm = modulate_per_token(norm1(x), shift_msa, scale_msa)
            # Self Attention
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, qkv_features=self.hidden_size, out_features=self.hidden_size
            )(x_norm, x_norm)
            x = x + gate_msa * attn
            
            x_norm2 = modulate_per_token(norm2(x), shift_mlp, scale_mlp)
            mlp_fn = nn.Sequential([
                nn.Dense(mlp_hidden_dim),
                lambda z: nn.gelu(z, approximate=True),
                nn.Dense(self.hidden_size)
            ])
            x = x + gate_mlp * mlp_fn(x_norm2)
        else:
            modulation = nn.Sequential([
                nn.swish,
                nn.Dense(6 * self.hidden_size)
            ])(c)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=1)
            
            x_norm = modulate(norm1(x), shift_msa, scale_msa)
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, qkv_features=self.hidden_size, out_features=self.hidden_size
            )(x_norm, x_norm)
            x = x + gate_msa[:, None, :] * attn
            
            x_norm2 = modulate(norm2(x), shift_mlp, scale_mlp)
            mlp_fn = nn.Sequential([
                nn.Dense(mlp_hidden_dim),
                lambda z: nn.gelu(z, approximate=True),
                nn.Dense(self.hidden_size)
            ])
            x = x + gate_mlp[:, None, :] * mlp_fn(x_norm2)
            
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""
    hidden_size: int
    patch_size: int
    out_channels: int
    per_token: bool = False

    @nn.compact
    def __call__(self, x, c):
        norm_final = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        linear = nn.Dense(self.patch_size * self.patch_size * self.out_channels)
        
        if self.per_token:
            batch_size, seq_len, hidden_dim = c.shape
            c_flat = c.reshape(-1, hidden_dim)
            modulation_flat = nn.Sequential([
                nn.swish,
                nn.Dense(2 * self.hidden_size)
            ])(c_flat)
            modulation = modulation_flat.reshape(batch_size, seq_len, -1)
            shift, scale = jnp.split(modulation, 2, axis=-1)
            
            x = modulate_per_token(norm_final(x), shift, scale)
            x = linear(x)
        else:
            modulation = nn.Sequential([
                nn.swish,
                nn.Dense(2 * self.hidden_size)
            ])(c)
            shift, scale = jnp.split(modulation, 2, axis=1)
            
            x = modulate(norm_final(x), shift, scale)
            x = linear(x)
            
        return x


class SimpleHead(nn.Module):
    """Simple projection head for self-distillation."""
    in_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.in_dim + self.out_dim)(x)
        x = nn.swish(x)
        x = nn.Dense(self.out_dim)(x)
        return x


def _build_repeat_segment_routes(segment_len: int, max_repeats: int) -> jax.Array:
    """Enumerate every legal fixed-length route with exactly one repeated block."""
    if segment_len < 2:
        raise ValueError(f"repeat segment must contain at least 2 blocks, got {segment_len}")
    if max_repeats < 2:
        raise ValueError(f"max_repeats must be >= 2, got {max_repeats}")

    routes = []
    base_route = tuple(range(segment_len))
    for total_repeats in range(2, max_repeats + 1):
        for repeat_offset in range(segment_len - total_repeats + 1):
            route = (
                base_route[:repeat_offset]
                + (base_route[repeat_offset],) * total_repeats
                + base_route[repeat_offset + total_repeats:]
            )
            routes.append(route)

    if not routes:
        raise ValueError(
            f"no legal repeat routes for segment_len={segment_len} max_repeats={max_repeats}"
        )
    return jnp.asarray(routes, dtype=jnp.int32)


class SelfFlowDiT(nn.Module):
    """Base Self-Flow DiT model."""
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    learn_sigma: bool = False
    compatibility_mode: bool = False
    per_token: bool = False
    repeat_block_route: bool = False
    repeat_block_start: int = 4
    repeat_block_end: int = 9
    repeat_block_prob: float = 0.5
    repeat_block_max_repeats: int = 3

    def setup(self):
        self.out_channels_val = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        self.pos_embed_val = pos_embed[None, ...] # (1, num_patches, hidden_size)

        self.patch_embed = PatchedPatchEmbed(
            img_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.hidden_size,
            name="patch_embed",
        )
        self.t_embedder = TimestepEmbedder(hidden_size=self.hidden_size, name="t_embedder")
        self.y_embedder = LabelEmbedder(
            num_classes=self.num_classes,
            hidden_size=self.hidden_size,
            dropout_prob=0.0,
            name="y_embedder",
        )
        self.blocks = tuple(
            DiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                per_token=self.per_token,
                name=f"block_{i}",
            )
            for i in range(self.depth)
        )
        self.feature_head = SimpleHead(
            in_dim=self.hidden_size,
            out_dim=self.hidden_size,
            name="feature_head",
        )
        self.final_layer_mod = FinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels_val,
            per_token=self.per_token,
            name="final_layer_mod",
        )

        if self.repeat_block_route:
            if not (1 <= int(self.repeat_block_start) <= int(self.repeat_block_end) <= self.depth):
                raise ValueError(
                    f"repeat block window must satisfy 1 <= start <= end <= depth ({self.depth}), "
                    f"got {self.repeat_block_start}:{self.repeat_block_end}"
                )
            if not (0.0 <= float(self.repeat_block_prob) <= 1.0):
                raise ValueError(f"repeat_block_prob must be in [0, 1], got {self.repeat_block_prob}")
            self.repeat_window_start_idx = int(self.repeat_block_start) - 1
            self.repeat_window_end_idx = int(self.repeat_block_end)
            self.repeat_window_len = self.repeat_window_end_idx - self.repeat_window_start_idx
            if self.repeat_window_len < 2:
                raise ValueError(
                    f"repeat block window must contain at least 2 blocks, got {self.repeat_block_start}:{self.repeat_block_end}"
                )
            if self.repeat_block_max_repeats < 2:
                raise ValueError(
                    f"repeat_block_max_repeats must be >= 2, got {self.repeat_block_max_repeats}"
                )
            if self.repeat_block_max_repeats > self.repeat_window_len:
                raise ValueError(
                    f"repeat_block_max_repeats ({self.repeat_block_max_repeats}) exceeds repeat window length "
                    f"({self.repeat_window_len})"
                )
            self.repeat_segment_base = jnp.arange(self.repeat_window_len, dtype=jnp.int32)
            self.repeat_segment_bank = _build_repeat_segment_routes(
                segment_len=self.repeat_window_len,
                max_repeats=int(self.repeat_block_max_repeats),
            )
        else:
            self.repeat_window_start_idx = 0
            self.repeat_window_end_idx = 0
            self.repeat_window_len = 0
            self.repeat_segment_base = jnp.zeros((0,), dtype=jnp.int32)
            self.repeat_segment_bank = jnp.zeros((0, 0), dtype=jnp.int32)

    def _resolve_repeat_segment_route(
        self,
        repeat_route_index: Optional[jax.Array],
        deterministic: bool,
    ) -> jax.Array:
        if (
            deterministic
            or not self.repeat_block_route
            or self.repeat_window_len == 0
        ):
            return self.repeat_segment_base

        if repeat_route_index is None:
            return self.repeat_segment_base

        repeat_route_index = jnp.asarray(repeat_route_index, dtype=jnp.int32)
        repeat_route_index = jnp.clip(repeat_route_index, 0, self.repeat_segment_bank.shape[0])
        use_repeat = repeat_route_index > 0
        repeat_idx = jnp.maximum(repeat_route_index - 1, 0)
        repeat_route = self.repeat_segment_bank[repeat_idx]
        return jnp.where(use_repeat, repeat_route, self.repeat_segment_base)

    def _capture_block_outputs(
        self,
        hidden: jax.Array,
        layer_idx: int,
        return_features: bool,
        return_raw_features: bool,
        return_raw_feature_layers,
        multi_raw_features,
        block_summaries,
        zs,
    ):
        if block_summaries is not None:
            block_summaries.append(jnp.mean(hidden, axis=1))
        if layer_idx == return_features:
            zs = self.feature_head(hidden)
        elif layer_idx == return_raw_features:
            zs = hidden
        if multi_raw_features is not None:
            for feature_idx, target_layer in enumerate(return_raw_feature_layers):
                if layer_idx == target_layer:
                    multi_raw_features[feature_idx] = hidden
        return block_summaries, zs, multi_raw_features

    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        vector: jax.Array,
        x_ids: Optional[jax.Array] = None,
        return_features: bool = False,
        return_raw_features: bool = False,
        return_raw_feature_layers: Optional[tuple[int, ...]] = None,
        return_block_summaries: bool = False,
        repeat_route_index: Optional[jax.Array] = None,
        deterministic: bool = True,
    ):
        """Forward pass with compatibility mode handling."""
        assert not (return_raw_features and return_features)
        assert not (return_features and return_raw_feature_layers)
        assert not (return_raw_features and return_raw_feature_layers)
        # return_block_summaries can be combined with either mode; callers must
        # handle the expanded return tuple shape.

        if return_raw_feature_layers is not None:
            return_raw_feature_layers = tuple(int(layer) for layer in return_raw_feature_layers)
            if len(set(return_raw_feature_layers)) != len(return_raw_feature_layers):
                raise ValueError(
                    f"return_raw_feature_layers must be unique, got {return_raw_feature_layers}"
                )
            for layer in return_raw_feature_layers:
                if not (1 <= layer <= self.depth):
                    raise ValueError(
                        f"return_raw_feature_layers must be in [1, {self.depth}], got {return_raw_feature_layers}"
                    )

        # PyTorch implementation explicitly negates timesteps
        timesteps = 1.0 - timesteps

        # Patch Embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed_val

        if self.per_token:
            batch_size, seq_len, _ = x.shape
            if timesteps.ndim == 1:
                t_emb = self.t_embedder(timesteps)
                t_emb = jnp.tile(t_emb[:, None, :], (1, seq_len, 1))
            elif timesteps.ndim == 2:
                t_flat = timesteps.reshape(-1)
                t_emb_flat = self.t_embedder(t_flat)
                t_emb = t_emb_flat.reshape(batch_size, seq_len, -1)
            else:
                raise ValueError(f"Unsupported per-token timestep rank: {timesteps.ndim}")
            
            y_emb = self.y_embedder(vector, deterministic=deterministic)
            y_emb = jnp.tile(y_emb[:, None, :], (1, seq_len, 1))
        else:
            t_emb = self.t_embedder(timesteps)
            y_emb = self.y_embedder(vector, deterministic=deterministic)

        c = t_emb + y_emb

        zs = None
        multi_raw_features = (
            [None] * len(return_raw_feature_layers)
            if return_raw_feature_layers is not None
            else None
        )
        block_summaries = [] if return_block_summaries else None
        layer_idx = 0

        repeat_start = self.repeat_window_start_idx
        repeat_end = self.repeat_window_end_idx

        for block in self.blocks[:repeat_start]:
            x = block(x, c)
            layer_idx += 1
            block_summaries, zs, multi_raw_features = self._capture_block_outputs(
                x,
                layer_idx,
                return_features,
                return_raw_features,
                return_raw_feature_layers,
                multi_raw_features,
                block_summaries,
                zs,
            )

        if self.repeat_block_route and self.repeat_window_len > 0:
            if self.is_mutable_collection("params"):
                for block in self.blocks[repeat_start:repeat_end]:
                    _ = block(x, c)
            segment_route = self._resolve_repeat_segment_route(
                repeat_route_index=repeat_route_index,
                deterministic=deterministic,
            )
            segment_branches = tuple(
                (
                    lambda mdl, hidden, relative_idx=relative_idx:
                    mdl.blocks[repeat_start + relative_idx](hidden, c)
                )
                for relative_idx in range(self.repeat_window_len)
            )
            for route_idx in range(self.repeat_window_len):
                x = nn.switch(segment_route[route_idx], segment_branches, self, x)
                layer_idx += 1
                block_summaries, zs, multi_raw_features = self._capture_block_outputs(
                    x,
                    layer_idx,
                    return_features,
                    return_raw_features,
                    return_raw_feature_layers,
                    multi_raw_features,
                    block_summaries,
                    zs,
                )
        else:
            for block in self.blocks[repeat_start:repeat_end]:
                x = block(x, c)
                layer_idx += 1
                block_summaries, zs, multi_raw_features = self._capture_block_outputs(
                    x,
                    layer_idx,
                    return_features,
                    return_raw_features,
                    return_raw_feature_layers,
                    multi_raw_features,
                    block_summaries,
                    zs,
                )

        for block in self.blocks[repeat_end:]:
            x = block(x, c)
            layer_idx += 1
            block_summaries, zs, multi_raw_features = self._capture_block_outputs(
                x,
                layer_idx,
                return_features,
                return_raw_features,
                return_raw_feature_layers,
                multi_raw_features,
                block_summaries,
                zs,
            )

        x = self.final_layer_mod(x, c)

        x = self._shufflechannel(x)
        
        # PyTorch implementation negates the final prediction
        x = -x

        if return_block_summaries:
            block_summaries = jnp.stack(block_summaries, axis=0)  # (depth, B, D)

        if multi_raw_features is not None:
            multi_raw_features = tuple(multi_raw_features)
            if any(feature is None for feature in multi_raw_features):
                raise RuntimeError(
                    f"Failed to capture all requested raw feature layers: {return_raw_feature_layers}"
                )
            if return_block_summaries:
                return x, multi_raw_features, block_summaries
            return x, multi_raw_features

        if return_features or return_raw_features:
            if return_block_summaries:
                return x, zs, block_summaries
            return x, zs
        if return_block_summaries:
            return x, block_summaries
        return x

    def _shufflechannel(self, x):
        """Reorder channels/patches to match expected output format."""
        p = self.patch_size
        x = rearrange(x, "b l (c p q) -> b l (c p q)", p=p, q=p, c=self.out_channels_val) # equivalent to rearranging in torch
        # wait, the PyTorch implementation says:
        # x = rearrange(x, "b l (p q c) -> b l (c p q)", p=p, q=p, c=self.out_channels)
        x = rearrange(x, "b l (p q c) -> b l (c p q)", p=p, q=p, c=self.out_channels_val)
        if self.learn_sigma:
            x, _ = jnp.split(x, 2, axis=2)
        return x


class SelfFlowPerTokenDiT(SelfFlowDiT):
    """
    Self-Flow DiT with per-token timestep conditioning.
    Main model used for Self-Flow inference on ImageNet.
    """
    per_token: bool = True


# Thin alias for clarity in the vanilla SiT baseline.
# Use as: SiTDiT(..., per_token=False)
SiTDiT = SelfFlowDiT
