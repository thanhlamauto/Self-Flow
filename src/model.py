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
from flax.core import FrozenDict, freeze, unfreeze
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


def _mimetic_target_matrix(rng, width: int, alpha: float, beta: float, diag_sign: float, dtype):
    z = jax.random.normal(rng, (width, width), dtype=jnp.float32) / jnp.sqrt(float(width))
    eye = jnp.eye(width, dtype=jnp.float32)
    target = alpha * z + diag_sign * beta * eye
    return target.astype(dtype)


def _truncated_svd_factors(target, rank: int):
    """Return A [d, rank], B [rank, d] with A @ B approximating target."""
    u, s, vh = jnp.linalg.svd(target, full_matrices=False)
    s_root = jnp.sqrt(jnp.clip(s[:rank], a_min=0.0))
    left = u[:, :rank] * s_root[None, :]
    right = s_root[:, None] * vh[:rank, :]
    return left, right


def apply_mimetic_attention_init(
    params,
    *,
    rng,
    start_layer: int = 1,
    end_layer: Optional[int] = None,
    qk_alpha: float = 0.7,
    qk_beta: float = 0.7,
    vo_alpha: float = 0.4,
    vo_beta: float = 0.4,
):
    """Apply mimetic self-attention init to a contiguous range of backbone blocks.

    The paper assumes single-head full-rank value/projection matrices. In this
    multi-head Flax model, each head uses rank-`head_dim` factors, so we apply
    the same SVD construction per head and keep the top-`head_dim` components.
    """
    block_names = sorted(
        (name for name in params.keys() if name.startswith("DiTBlock_")),
        key=lambda name: int(name.split("_")[1]),
    )
    if not block_names:
        return params

    total_layers = len(block_names)
    end_layer = total_layers if end_layer is None else end_layer
    if not (1 <= start_layer <= total_layers):
        raise ValueError(f"mimetic start_layer={start_layer} out of range [1, {total_layers}]")
    if not (1 <= end_layer <= total_layers):
        raise ValueError(f"mimetic end_layer={end_layer} out of range [1, {total_layers}]")
    if start_layer > end_layer:
        raise ValueError(
            f"mimetic start_layer ({start_layer}) must be <= end_layer ({end_layer})"
        )

    mutable = unfreeze(params)
    layer_rngs = jax.random.split(rng, end_layer - start_layer + 1)

    for layer_offset, layer_rng in enumerate(layer_rngs, start=start_layer - 1):
        block_name = block_names[layer_offset]
        attn = mutable[block_name]["MultiHeadDotProductAttention_0"]

        query_kernel = attn["query"]["kernel"]
        value_kernel = attn["value"]["kernel"]
        width, num_heads, head_dim = query_kernel.shape
        dtype = query_kernel.dtype

        head_rngs = jax.random.split(layer_rng, num_heads * 2)
        query_heads = []
        key_heads = []
        value_heads = []
        out_heads = []

        for head in range(num_heads):
            qk_target = _mimetic_target_matrix(
                head_rngs[2 * head], width, qk_alpha, qk_beta, +1.0, dtype
            )
            q_factor, k_factor_t = _truncated_svd_factors(qk_target, head_dim)

            vo_target = _mimetic_target_matrix(
                head_rngs[2 * head + 1], width, vo_alpha, vo_beta, -1.0, dtype
            )
            v_factor, out_factor = _truncated_svd_factors(vo_target, head_dim)

            query_heads.append(q_factor.astype(dtype))
            key_heads.append(k_factor_t.T.astype(dtype))
            value_heads.append(v_factor.astype(dtype))
            out_heads.append(out_factor.astype(dtype))

        attn["query"]["kernel"] = jnp.stack(query_heads, axis=1)
        attn["key"]["kernel"] = jnp.stack(key_heads, axis=1)
        attn["value"]["kernel"] = jnp.stack(value_heads, axis=1)
        attn["out"]["kernel"] = jnp.stack(out_heads, axis=0)

        attn["query"]["bias"] = jnp.zeros_like(attn["query"]["bias"])
        attn["key"]["bias"] = jnp.zeros_like(attn["key"]["bias"])
        attn["value"]["bias"] = jnp.zeros_like(attn["value"]["bias"])
        attn["out"]["bias"] = jnp.zeros_like(attn["out"]["bias"])

    if isinstance(params, FrozenDict):
        return freeze(mutable)
    return mutable


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
                nn.Dense(
                    6 * self.hidden_size,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                )
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
                nn.Dense(
                    6 * self.hidden_size,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                )
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
        linear = nn.Dense(
            self.patch_size * self.patch_size * self.out_channels,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )

        if self.per_token:
            batch_size, seq_len, hidden_dim = c.shape
            c_flat = c.reshape(-1, hidden_dim)
            modulation_flat = nn.Sequential([
                nn.swish,
                nn.Dense(
                    2 * self.hidden_size,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                )
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
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                )
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

    def setup(self):
        self.out_channels_val = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        self.pos_embed_val = pos_embed[None, ...] # (1, num_patches, hidden_size)
        self.feature_head = SimpleHead(in_dim=self.hidden_size, out_dim=self.hidden_size)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        vector: jax.Array,
        x_ids: Optional[jax.Array] = None,
        return_features: bool = False,
        return_raw_features: bool = False,
        deterministic: bool = True,
    ):
        """Forward pass with compatibility mode handling."""
        assert not (return_raw_features and return_features)

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
        for i in range(self.depth):
            x = DiTBlock(
                hidden_size=self.hidden_size, 
                num_heads=self.num_heads, 
                mlp_ratio=self.mlp_ratio,
                per_token=self.per_token
            )(x, c)
            
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

        if return_features or return_raw_features:
            return x, zs
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
