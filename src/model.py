"""
Self-Flow Model (Flax version).

This module contains the SelfFlowPerTokenDiT model, a Diffusion Transformer
with per-token timestep conditioning for Self-Flow training, implemented in Flax.
"""

import math
from typing import Any, Mapping, Optional

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
    def __call__(
        self,
        x,
        c,
        *,
        decompose_mlp: bool = False,
        shared_ffn_w1: Optional[jax.Array] = None,
        shared_ffn_w2: Optional[jax.Array] = None,
        capture_mlp_input: bool = False,
    ):
        norm1 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        norm2 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)

        if decompose_mlp and (shared_ffn_w1 is None or shared_ffn_w2 is None):
            raise ValueError("Decomposed FFN requires shared_ffn_w1 and shared_ffn_w2.")

        def apply_mlp(mlp_input):
            captured = mlp_input if capture_mlp_input else None
            if not decompose_mlp:
                mlp_fn = nn.Sequential([
                    nn.Dense(mlp_hidden_dim),
                    lambda z: nn.gelu(z, approximate=True),
                    nn.Dense(self.hidden_size),
                ])
                return mlp_fn(mlp_input), captured

            residual_w1 = self.param(
                "dw1",
                nn.initializers.lecun_normal(),
                (self.hidden_size, mlp_hidden_dim),
            )
            bias1 = self.param(
                "b1",
                nn.initializers.zeros,
                (mlp_hidden_dim,),
            )
            residual_w2 = self.param(
                "dw2",
                nn.initializers.lecun_normal(),
                (mlp_hidden_dim, self.hidden_size),
            )
            bias2 = self.param(
                "b2",
                nn.initializers.zeros,
                (self.hidden_size,),
            )

            shared_w1 = shared_ffn_w1.astype(mlp_input.dtype)
            shared_w2 = shared_ffn_w2.astype(mlp_input.dtype)
            residual_w1 = residual_w1.astype(mlp_input.dtype)
            residual_w2 = residual_w2.astype(mlp_input.dtype)

            u_common = jnp.einsum("...d,dm->...m", mlp_input, shared_w1)
            u_res = jnp.einsum("...d,dm->...m", mlp_input, residual_w1)
            u = u_common + u_res + bias1.astype(mlp_input.dtype)
            activated = nn.gelu(u, approximate=True)

            o_common = jnp.einsum("...m,md->...d", activated, shared_w2)
            o_res = jnp.einsum("...m,md->...d", activated, residual_w2)
            return o_common + o_res + bias2.astype(mlp_input.dtype), captured
        
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
            mlp_out, captured = apply_mlp(x_norm2)
            x = x + gate_mlp * mlp_out
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
            mlp_out, captured = apply_mlp(x_norm2)
            x = x + gate_mlp[:, None, :] * mlp_out
            
        return x, captured


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


class SharedLowRankMatrix(nn.Module):
    """Shared low-rank matrix A @ B used across selected FFN layers."""
    in_features: int
    rank: int
    out_features: int
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self) -> jax.Array:
        a = self.param(
            "A",
            nn.initializers.normal(stddev=self.init_scale),
            (self.in_features, self.rank),
        )
        b = self.param(
            "B",
            nn.initializers.normal(stddev=self.init_scale),
            (self.rank, self.out_features),
        )
        return jnp.matmul(a, b)


def model_init_kwargs_from_config(
    config: Mapping[str, Any],
    *,
    per_token: bool = False,
) -> dict[str, Any]:
    """Build model kwargs from the shared training config."""
    return dict(
        input_size=config["input_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        num_classes=config["num_classes"],
        learn_sigma=config["learn_sigma"],
        compatibility_mode=config["compatibility_mode"],
        per_token=per_token,
        selected_ffn_layers=tuple(config.get("selected_ffn_layers", ())),
        rank_r1=config.get("rank_r1", 0),
        rank_r2=config.get("rank_r2", 0),
        gram_dim=config.get("gram_dim", 0),
    )


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
    selected_ffn_layers: tuple[int, ...] = ()
    rank_r1: int = 0
    rank_r2: int = 0
    gram_dim: int = 0

    def setup(self):
        self.out_channels_val = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = self.patch_size * self.patch_size * self.in_channels
        self.mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.selected_ffn_layers_val = tuple(int(layer) for layer in self.selected_ffn_layers)
        self.selected_ffn_layer_set = frozenset(self.selected_ffn_layers_val)
        self.ffn_decomposition_enabled = bool(
            self.selected_ffn_layers_val and self.rank_r1 > 0 and self.rank_r2 > 0
        )
        
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        self.pos_embed_val = pos_embed[None, ...] # (1, num_patches, hidden_size)
        self.feature_head = SimpleHead(in_dim=self.hidden_size, out_dim=self.hidden_size)
        if self.ffn_decomposition_enabled:
            self.shared_ffn_w1 = SharedLowRankMatrix(
                in_features=self.hidden_size,
                rank=self.rank_r1,
                out_features=self.mlp_hidden_dim,
                name="shared_ffn_w1",
            )
            self.shared_ffn_w2 = SharedLowRankMatrix(
                in_features=self.mlp_hidden_dim,
                rank=self.rank_r2,
                out_features=self.hidden_size,
                name="shared_ffn_w2",
            )
        else:
            self.shared_ffn_w1 = None
            self.shared_ffn_w2 = None

        if self.gram_dim > 0:
            self.gram_common_head = nn.Dense(self.gram_dim, name="gram_common_head")
            self.gram_target_head = nn.Dense(self.gram_dim, name="gram_target_head")
        else:
            self.gram_common_head = None
            self.gram_target_head = None

    def _ensure_gram_heads_initialized(self, dtype: jnp.dtype) -> None:
        if self.gram_dim <= 0:
            return
        dummy_common = jnp.zeros((1, 1, self.mlp_hidden_dim), dtype=dtype)
        dummy_target = jnp.zeros((1, 1, self.patch_dim), dtype=dtype)
        _ = self.gram_common_head(dummy_common)
        _ = self.gram_target_head(dummy_target)

    def compute_common_ffn1(self, hidden: jax.Array) -> jax.Array:
        if not self.ffn_decomposition_enabled:
            raise ValueError("FFN decomposition is not enabled for this model.")
        shared_w1 = self.shared_ffn_w1().astype(hidden.dtype)
        return jnp.einsum("...d,dm->...m", hidden, shared_w1)

    def project_gram_common_from_hidden(self, hidden: jax.Array) -> jax.Array:
        if self.gram_dim <= 0:
            raise ValueError("gram_dim must be > 0 to use the Gram common projector.")
        return self.gram_common_head(self.compute_common_ffn1(hidden))

    def project_gram_target(self, z0_tokens: jax.Array) -> jax.Array:
        if self.gram_dim <= 0:
            raise ValueError("gram_dim must be > 0 to use the Gram target projector.")
        return self.gram_target_head(z0_tokens)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        vector: jax.Array,
        x_ids: Optional[jax.Array] = None,
        return_features: bool = False,
        return_raw_features: bool = False,
        return_ffn_inputs: bool = False,
        return_block_summaries: bool = False,
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
        self._ensure_gram_heads_initialized(x.dtype)

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
        shared_ffn_w1 = self.shared_ffn_w1() if self.ffn_decomposition_enabled else None
        shared_ffn_w2 = self.shared_ffn_w2() if self.ffn_decomposition_enabled else None
        ffn_input_positions = None
        ffn_inputs = None
        if return_ffn_inputs:
            ffn_input_positions = {
                layer_idx: pos for pos, layer_idx in enumerate(self.selected_ffn_layers_val)
            }
            ffn_inputs = [None] * len(self.selected_ffn_layers_val)
        block_summaries = [] if return_block_summaries else None
        for i in range(self.depth):
            layer_idx = i + 1
            capture_mlp_input = return_ffn_inputs and layer_idx in ffn_input_positions
            x, captured_ffn_input = DiTBlock(
                hidden_size=self.hidden_size, 
                num_heads=self.num_heads, 
                mlp_ratio=self.mlp_ratio,
                per_token=self.per_token
            )(
                x,
                c,
                decompose_mlp=self.ffn_decomposition_enabled and layer_idx in self.selected_ffn_layer_set,
                shared_ffn_w1=shared_ffn_w1,
                shared_ffn_w2=shared_ffn_w2,
                capture_mlp_input=capture_mlp_input,
            )

            if return_block_summaries:
                # Token-pooled summary per block: (B, D)
                block_summaries.append(jnp.mean(x, axis=1))
            
            if capture_mlp_input:
                ffn_inputs[ffn_input_positions[layer_idx]] = captured_ffn_input

            if layer_idx == return_features:
                zs = self.feature_head(x)
            elif layer_idx == return_raw_features:
                zs = x

        hidden_shape = x.shape
        if return_ffn_inputs:
            if ffn_inputs:
                ffn_inputs = jnp.stack(ffn_inputs, axis=0)
            else:
                ffn_inputs = jnp.zeros((0,) + hidden_shape, dtype=x.dtype)

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

        outputs = [x]
        if return_features or return_raw_features:
            outputs.append(zs)
        if return_ffn_inputs:
            outputs.append(ffn_inputs)
        if return_block_summaries:
            outputs.append(block_summaries)
        if len(outputs) == 1:
            return x
        return tuple(outputs)

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
