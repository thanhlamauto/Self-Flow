"""
Self-Flow Model (Flax version).

This module contains the SelfFlowPerTokenDiT model, a Diffusion Transformer
with per-token timestep conditioning for Self-Flow training, implemented in Flax.
"""

import math
from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
from src.depth_shortcut import (
    DepthShortcutPredictor,
    log_token_magnitudes,
    l2_normalize_tokens,
    predictor_config_from_name,
    restore_l2_ema_magnitude,
)

XAVIER_UNIFORM = nn.initializers.xavier_uniform()
ZERO_INIT = nn.initializers.zeros
NORMAL_002 = nn.initializers.normal(stddev=0.02)


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
        return nn.Dense(
            self.embed_dim,
            use_bias=self.bias,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="proj",
        )(x)


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
        x = nn.Dense(
            self.hidden_size,
            kernel_init=NORMAL_002,
            bias_init=ZERO_INIT,
        )(t_freq)
        x = nn.swish(x)
        x = nn.Dense(
            self.hidden_size,
            kernel_init=NORMAL_002,
            bias_init=ZERO_INIT,
        )(x)
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
            features=self.hidden_size,
            embedding_init=NORMAL_002,
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
            modulation_flat = nn.Dense(
                6 * self.hidden_size,
                kernel_init=ZERO_INIT,
                bias_init=ZERO_INIT,
            )(nn.swish(c_flat))
            modulation = modulation_flat.reshape(batch_size, seq_len, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)
            
            x_norm = modulate_per_token(norm1(x), shift_msa, scale_msa)
            # Self Attention
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_size,
                out_features=self.hidden_size,
                kernel_init=XAVIER_UNIFORM,
                out_kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                out_bias_init=ZERO_INIT,
            )(x_norm, x_norm)
            x = x + gate_msa * attn
            
            x_norm2 = modulate_per_token(norm2(x), shift_mlp, scale_mlp)
            mlp_fn = nn.Sequential([
                nn.Dense(
                    mlp_hidden_dim,
                    kernel_init=XAVIER_UNIFORM,
                    bias_init=ZERO_INIT,
                ),
                lambda z: nn.gelu(z, approximate=True),
                nn.Dense(
                    self.hidden_size,
                    kernel_init=XAVIER_UNIFORM,
                    bias_init=ZERO_INIT,
                ),
            ])
            x = x + gate_mlp * mlp_fn(x_norm2)
        else:
            modulation = nn.Dense(
                6 * self.hidden_size,
                kernel_init=ZERO_INIT,
                bias_init=ZERO_INIT,
            )(nn.swish(c))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=1)
            
            x_norm = modulate(norm1(x), shift_msa, scale_msa)
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_size,
                out_features=self.hidden_size,
                kernel_init=XAVIER_UNIFORM,
                out_kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                out_bias_init=ZERO_INIT,
            )(x_norm, x_norm)
            x = x + gate_msa[:, None, :] * attn
            
            x_norm2 = modulate(norm2(x), shift_mlp, scale_mlp)
            mlp_fn = nn.Sequential([
                nn.Dense(
                    mlp_hidden_dim,
                    kernel_init=XAVIER_UNIFORM,
                    bias_init=ZERO_INIT,
                ),
                lambda z: nn.gelu(z, approximate=True),
                nn.Dense(
                    self.hidden_size,
                    kernel_init=XAVIER_UNIFORM,
                    bias_init=ZERO_INIT,
                ),
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
        linear = nn.Dense(
            self.patch_size * self.patch_size * self.out_channels,
            kernel_init=ZERO_INIT,
            bias_init=ZERO_INIT,
        )
        
        if self.per_token:
            batch_size, seq_len, hidden_dim = c.shape
            c_flat = c.reshape(-1, hidden_dim)
            modulation_flat = nn.Dense(
                2 * self.hidden_size,
                kernel_init=ZERO_INIT,
                bias_init=ZERO_INIT,
            )(nn.swish(c_flat))
            modulation = modulation_flat.reshape(batch_size, seq_len, -1)
            shift, scale = jnp.split(modulation, 2, axis=-1)
            
            x = modulate_per_token(norm_final(x), shift, scale)
            x = linear(x)
        else:
            modulation = nn.Dense(
                2 * self.hidden_size,
                kernel_init=ZERO_INIT,
                bias_init=ZERO_INIT,
            )(nn.swish(c))
            shift, scale = jnp.split(modulation, 2, axis=1)
            
            x = modulate(norm_final(x), shift, scale)
            x = linear(x)
            
        return x


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
    class_dropout_prob: float = 0.1

    def setup(self):
        self.out_channels_val = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        self.pos_embed_val = pos_embed[None, ...] # (1, num_patches, hidden_size)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        vector: jax.Array,
        x_ids: Optional[jax.Array] = None,
        return_features: bool = False,
        return_raw_features: bool | int | Sequence[int] = False,
        return_block_summaries: bool = False,
        return_hidden_states: bool = False,
        depth_shortcut_predictor_params: Optional[dict] = None,
        depth_shortcut_l2_ema: Optional[jax.Array] = None,
        depth_shortcut_variant: str = "tiny",
        depth_shortcut_skip_every_other: bool = False,
        depth_shortcut_predict_magnitude: bool = False,
        depth_shortcut_mag_scale: float = 3.0,
        depth_shortcut_mag_abs_center: float = 5.5,
        depth_shortcut_mag_abs_scale: float = 1.5,
        depth_shortcut_mag_clip_min: float = 3.0,
        depth_shortcut_mag_clip_max: float = 8.0,
        depth_shortcut_timesteps: int = 50,
        deterministic: bool = True,
    ):
        """Forward pass with compatibility mode handling."""
        assert not (return_raw_features and return_features)
        # return_block_summaries can be combined with either mode; callers must
        # handle the expanded return tuple shape.

        raw_layers = ()
        raw_single = False
        raw_positions = None
        if isinstance(return_raw_features, Sequence) and not isinstance(return_raw_features, (str, bytes)):
            raw_layers = tuple(int(layer) for layer in return_raw_features)
            raw_single = len(raw_layers) == 1
        elif isinstance(return_raw_features, int) and not isinstance(return_raw_features, bool):
            raw_layers = (int(return_raw_features),)
            raw_single = True

        if raw_layers:
            if any(layer <= 0 for layer in raw_layers):
                raise ValueError(f"return_raw_features layers must be >= 1, got {raw_layers}")
            if any(layer > self.depth for layer in raw_layers):
                raise ValueError(
                    f"return_raw_features layers must be <= model depth ({self.depth}), got {raw_layers}"
                )
            if not raw_single:
                raw_positions = {}
                for idx, layer in enumerate(raw_layers):
                    raw_positions.setdefault(layer, []).append(idx)

        original_timesteps = timesteps
        # PyTorch implementation explicitly negates timesteps
        timesteps = 1.0 - timesteps

        # Patch Embedding
        x = PatchedPatchEmbed(
            img_size=self.input_size, 
            patch_size=self.patch_size, 
            in_channels=self.in_channels, 
            embed_dim=self.hidden_size
        )(x)
        x = x + self.pos_embed_val

        t_embedder = TimestepEmbedder(hidden_size=self.hidden_size)
        y_embedder = LabelEmbedder(
            num_classes=self.num_classes,
            hidden_size=self.hidden_size,
            dropout_prob=self.class_dropout_prob,
        )

        if self.per_token:
            batch_size, seq_len, _ = x.shape
            if timesteps.ndim == 1:
                t_emb = t_embedder(timesteps)
                t_emb = jnp.tile(t_emb[:, None, :], (1, seq_len, 1))
            elif timesteps.ndim == 2:
                t_flat = timesteps.reshape(-1)
                t_emb_flat = t_embedder(t_flat)
                t_emb = t_emb_flat.reshape(batch_size, seq_len, -1)
            else:
                raise ValueError(f"Unsupported per-token timestep rank: {timesteps.ndim}")
            
            y_emb = y_embedder(vector, deterministic=deterministic)
            y_emb = jnp.tile(y_emb[:, None, :], (1, seq_len, 1))
        else:
            t_emb = t_embedder(timesteps)
            y_emb = y_embedder(vector, deterministic=deterministic)

        c = t_emb + y_emb

        shortcut_enabled = (
            depth_shortcut_skip_every_other
            and depth_shortcut_predictor_params is not None
            and (depth_shortcut_predict_magnitude or depth_shortcut_l2_ema is not None)
        )
        shortcut_sources = tuple(range(1, self.depth, 2))
        shortcut_skip_until = 0
        shortcut_q = None
        shortcut_predictor = None
        if shortcut_enabled:
            predictor_cfg = predictor_config_from_name(depth_shortcut_variant, self.hidden_size)
            shortcut_predictor = DepthShortcutPredictor(
                hidden_size=self.hidden_size,
                depth=self.depth,
                num_tokens=self.num_patches,
                mag_abs_center=depth_shortcut_mag_abs_center,
                mag_abs_scale=depth_shortcut_mag_abs_scale,
                **predictor_cfg,
            )
            # Training discretizes tau as q / (T - 1). Sampling uses continuous tau,
            # so round back to the nearest discrete index for EMA magnitude bins.
            shortcut_q = jnp.rint(original_timesteps * max(int(depth_shortcut_timesteps) - 1, 1)).astype(jnp.int32)
            shortcut_q = jnp.clip(shortcut_q, 0, int(depth_shortcut_timesteps) - 1)

        zs = None
        raw_zs = [None] * len(raw_layers) if raw_layers and not raw_single else None
        block_summaries = [] if return_block_summaries else None
        hidden_states = [x] if return_hidden_states else None
        for i in range(self.depth):
            layer_idx = i + 1
            if shortcut_enabled and layer_idx <= shortcut_skip_until:
                continue

            x = DiTBlock(
                hidden_size=self.hidden_size, 
                num_heads=self.num_heads, 
                mlp_ratio=self.mlp_ratio,
                per_token=self.per_token
            )(x, c)

            if shortcut_enabled and layer_idx in shortcut_sources and (layer_idx + 2) <= self.depth:
                target_layer = layer_idx + 2
                u_source_short = l2_normalize_tokens(x)
                m_source_short = log_token_magnitudes(x) if depth_shortcut_predict_magnitude else None
                pred_short = shortcut_predictor.apply(
                    {"params": depth_shortcut_predictor_params},
                    u_source_short,
                    jnp.asarray(layer_idx, dtype=jnp.int32),
                    jnp.asarray(target_layer, dtype=jnp.int32),
                    t_emb,
                    m_source_short,
                )
                if depth_shortcut_predict_magnitude:
                    y_short, delta_m_short = pred_short
                else:
                    y_short = pred_short
                u_short = l2_normalize_tokens(y_short)
                if depth_shortcut_predict_magnitude:
                    m_target_short = jnp.clip(
                        m_source_short + float(depth_shortcut_mag_scale) * delta_m_short,
                        float(depth_shortcut_mag_clip_min),
                        float(depth_shortcut_mag_clip_max),
                    )
                    x = jnp.exp(m_target_short) * u_short
                else:
                    x = restore_l2_ema_magnitude(
                        u_short,
                        depth_shortcut_l2_ema,
                        jnp.asarray(target_layer, dtype=jnp.int32),
                        shortcut_q,
                    )
                shortcut_skip_until = target_layer

            if return_hidden_states:
                hidden_states.append(x)

            if return_block_summaries:
                # Token-pooled summary per block: (B, D)
                block_summaries.append(jnp.mean(x, axis=1))
            
            if (i + 1) == return_features:
                zs = x
            if raw_layers and ((i + 1) in raw_positions if raw_positions is not None else (i + 1) in raw_layers):
                if raw_single:
                    zs = x
                else:
                    for idx in raw_positions[i + 1]:
                        raw_zs[idx] = x

        x = FinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels_val,
            per_token=self.per_token
        )(x, c)

        x = self._shufflechannel(x)
        
        # PyTorch implementation negates the final prediction
        x = -x

        if return_block_summaries:
            block_summaries = jnp.stack(block_summaries, axis=0)  # (depth, B, D)

        if return_hidden_states:
            hidden_states = tuple(hidden_states)
            if return_block_summaries:
                return x, hidden_states, t_emb, block_summaries
            return x, hidden_states, t_emb
        if return_features:
            if return_block_summaries:
                return x, zs, block_summaries
            return x, zs
        if raw_layers:
            raw_out = zs if raw_single else tuple(raw_zs)
            if return_block_summaries:
                return x, raw_out, block_summaries
            return x, raw_out
        if return_raw_features:
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
