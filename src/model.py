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
            modulation_flat = nn.Sequential([
                nn.swish,
                nn.Dense(
                    6 * self.hidden_size,
                    kernel_init=ZERO_INIT,
                    bias_init=ZERO_INIT,
                ),
            ])(c_flat)
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
            modulation = nn.Sequential([
                nn.swish,
                nn.Dense(
                    6 * self.hidden_size,
                    kernel_init=ZERO_INIT,
                    bias_init=ZERO_INIT,
                ),
            ])(c)
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
            modulation_flat = nn.Sequential([
                nn.swish,
                nn.Dense(
                    2 * self.hidden_size,
                    kernel_init=ZERO_INIT,
                    bias_init=ZERO_INIT,
                ),
            ])(c_flat)
            modulation = modulation_flat.reshape(batch_size, seq_len, -1)
            shift, scale = jnp.split(modulation, 2, axis=-1)
            
            x = modulate_per_token(norm_final(x), shift, scale)
            x = linear(x)
        else:
            modulation = nn.Sequential([
                nn.swish,
                nn.Dense(
                    2 * self.hidden_size,
                    kernel_init=ZERO_INIT,
                    bias_init=ZERO_INIT,
                ),
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
        x = nn.Dense(
            self.in_dim + self.out_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
        )(x)
        x = nn.swish(x)
        x = nn.Dense(
            self.out_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
        )(x)
        return x


class CommonSpatialCNNProjector(nn.Module):
    """Lightweight CNN projector for A_common before the spatial auxiliary loss."""

    grid_size: int
    width: int = 256
    depth: int = 2
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        if self.width <= 0:
            raise ValueError(f"CommonSpatialCNNProjector width must be > 0, got {self.width}")
        if self.depth <= 0:
            raise ValueError(f"CommonSpatialCNNProjector depth must be > 0, got {self.depth}")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(
                f"CommonSpatialCNNProjector kernel_size must be a positive odd integer, got {self.kernel_size}"
            )

        b, num_tokens, channels = x.shape
        expected_tokens = self.grid_size * self.grid_size
        if num_tokens != expected_tokens:
            raise ValueError(
                f"CommonSpatialCNNProjector expected {expected_tokens} tokens for a "
                f"{self.grid_size}x{self.grid_size} grid, got {num_tokens}"
            )

        x = x.reshape(b, self.grid_size, self.grid_size, channels)
        x = nn.Conv(
            features=self.width,
            kernel_size=(1, 1),
            padding="SAME",
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="input_proj",
        )(x)
        x = nn.silu(x)

        for block_idx in range(self.depth):
            residual = x
            h = nn.Conv(
                features=self.width,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding="SAME",
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                name=f"block_{block_idx}_conv1",
            )(x)
            h = nn.silu(h)
            h = nn.Conv(
                features=self.width,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding="SAME",
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                name=f"block_{block_idx}_conv2",
            )(h)
            x = nn.silu(h + residual)

        return x.reshape(b, num_tokens, self.width)


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
    common_spatial_projector: str = "identity"
    common_spatial_projector_width: int = 256
    common_spatial_projector_depth: int = 2
    common_spatial_projector_kernel_size: int = 3

    def setup(self):
        self.out_channels_val = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        self.pos_embed_val = pos_embed[None, ...] # (1, num_patches, hidden_size)
        self.feature_head = SimpleHead(in_dim=self.hidden_size, out_dim=self.hidden_size)
        if self.common_spatial_projector not in ("identity", "cnn"):
            raise ValueError(
                "common_spatial_projector must be 'identity' or 'cnn', "
                f"got {self.common_spatial_projector!r}"
            )
        if self.common_spatial_projector == "cnn":
            self.common_spatial_projector_head = CommonSpatialCNNProjector(
                grid_size=self.grid_size,
                width=self.common_spatial_projector_width,
                depth=self.common_spatial_projector_depth,
                kernel_size=self.common_spatial_projector_kernel_size,
                name="common_spatial_projector",
            )
        else:
            self.common_spatial_projector_head = None

    def project_common_spatial(self, common_tokens: jax.Array) -> jax.Array:
        """Optionally project A_common through a locality-preserving CNN head."""
        if self.common_spatial_projector_head is None:
            return common_tokens
        return self.common_spatial_projector_head(common_tokens)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        vector: jax.Array,
        x_ids: Optional[jax.Array] = None,
        return_features: bool = False,
        return_raw_features: bool = False,
        return_block_summaries: bool = False,
        return_activations: bool = False,
        deterministic: bool = True,
    ):
        """Forward pass with compatibility mode handling."""
        assert not (return_raw_features and return_features)
        # return_block_summaries can be combined with either mode; callers must
        # handle the expanded return tuple shape.

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

        if self.common_spatial_projector_head is not None and self.is_mutable_collection("params"):
            # Initialize projector params even though the head is only used by the aux-loss path.
            _ = self.project_common_spatial(jnp.zeros_like(x))

        t_embedder = TimestepEmbedder(hidden_size=self.hidden_size)
        y_embedder = LabelEmbedder(num_classes=self.num_classes, hidden_size=self.hidden_size, dropout_prob=0.0)

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

        zs = None
        block_summaries = [] if return_block_summaries else None
        activations = [] if return_activations else None
        for i in range(self.depth):
            x = DiTBlock(
                hidden_size=self.hidden_size, 
                num_heads=self.num_heads, 
                mlp_ratio=self.mlp_ratio,
                per_token=self.per_token
            )(x, c)

            if return_block_summaries:
                # Token-pooled summary per block: (B, D)
                block_summaries.append(jnp.mean(x, axis=1))

            if return_activations:
                # Full post-block hidden state: (B, N, D)
                activations.append(x)
            
            if (i + 1) == return_features:
                zs = self.feature_head(x)
            elif (i + 1) == return_raw_features:
                zs = x

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
        if return_activations:
            activations = jnp.stack(activations, axis=0)  # (depth, B, N, D)

        outputs = [x]
        if return_features or return_raw_features:
            outputs.append(zs)
        if return_block_summaries:
            outputs.append(block_summaries)
        if return_activations:
            outputs.append(activations)
        if len(outputs) > 1:
            return tuple(outputs)
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
