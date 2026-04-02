from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange


def token_haar_detail_scores(clean_tokens: jax.Array, *, in_channels: int = 4) -> jax.Array:
    """Return normalized per-token Haar detail scores in [0, 1].

    clean_tokens is expected in the repo's patchified latent layout:
      [B, N, patch_size * patch_size * in_channels]
    with token order (p1, p2, c). The current SiT baseline uses patch_size=2.
    """
    patch_area_channels = clean_tokens.shape[-1]
    if patch_area_channels % in_channels != 0:
        raise ValueError(
            f"Token dim {patch_area_channels} is not divisible by in_channels={in_channels}"
        )

    patch_area = patch_area_channels // in_channels
    patch_size = int(round(math.sqrt(patch_area)))
    if patch_size * patch_size != patch_area or patch_size != 2:
        raise ValueError(
            "token_haar_detail_scores expects patch_size=2 tokens "
            f"(got token dim {patch_area_channels}, inferred patch_size={patch_size})"
        )

    patches = rearrange(
        clean_tokens,
        "b n (p1 p2 c) -> b n p1 p2 c",
        p1=patch_size,
        p2=patch_size,
        c=in_channels,
    )

    a = patches[:, :, 0, 0, :]
    b = patches[:, :, 0, 1, :]
    c = patches[:, :, 1, 0, :]
    d = patches[:, :, 1, 1, :]

    # One-level orthonormal Haar coefficients inside each token patch.
    lh = 0.5 * (a - b + c - d)
    hl = 0.5 * (a + b - c - d)
    hh = 0.5 * (a - b - c + d)
    energy = jnp.mean(lh * lh + hl * hl + hh * hh, axis=-1)

    e_min = jnp.min(energy, axis=-1, keepdims=True)
    e_max = jnp.max(energy, axis=-1, keepdims=True)
    denom = jnp.maximum(e_max - e_min, 1e-6)
    return (energy - e_min) / denom


def alpha_from_layer_ids(
    layer_ids: jax.Array,
    *,
    depth: int,
    alpha_max: float,
) -> jax.Array:
    layer_ids = jnp.asarray(layer_ids, dtype=jnp.float32)
    if depth <= 1:
        return jnp.zeros_like(layer_ids)
    s = (layer_ids - 1.0) / float(depth - 1)
    return jnp.asarray(alpha_max, dtype=jnp.float32) * 0.5 * (1.0 - jnp.cos(jnp.pi * s))


def build_layer_token_weights(
    detail_scores: jax.Array,
    layer_ids: jax.Array,
    *,
    depth: int,
    alpha_max: float,
    eps: float = 1e-6,
) -> jax.Array:
    """Build mean-normalized token weights for each selected layer."""
    alpha = alpha_from_layer_ids(layer_ids, depth=depth, alpha_max=alpha_max)
    weights = 1.0 + alpha[:, None, None] * detail_scores[None, :, :]
    mean = jnp.mean(weights, axis=-1, keepdims=True)
    return jax.lax.stop_gradient(weights / (mean + eps))


def masked_cosine_loss(
    projected_tokens: jax.Array,
    teacher_tokens: jax.Array,
    layer_weights: jax.Array,
    *,
    eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    """Weighted cosine loss averaged over selected layers."""
    proj = projected_tokens / (jnp.linalg.norm(projected_tokens, axis=-1, keepdims=True) + eps)
    teacher = teacher_tokens / (jnp.linalg.norm(teacher_tokens, axis=-1, keepdims=True) + eps)
    cosine = jnp.sum(proj * teacher[None, :, :, :], axis=-1)
    token_loss = 1.0 - cosine
    weighted = layer_weights * token_loss
    per_layer = jnp.sum(weighted, axis=(1, 2)) / (jnp.sum(layer_weights, axis=(1, 2)) + eps)
    return jnp.mean(per_layer), per_layer


class LayerTimeProjector(nn.Module):
    hidden_dim: int
    out_dim: int
    num_layers: int
    cond_dim: int | None = None

    @nn.compact
    def __call__(
        self,
        hidden_stack: jax.Array,
        time_embed: jax.Array,
        layer_ids: jax.Array,
    ) -> jax.Array:
        """Project hidden states into teacher-token space.

        hidden_stack: [K, B, N, D]
        time_embed:   [B, D]
        layer_ids:    [K] (1-based transformer block indices)
        """
        cond_dim = int(self.cond_dim or self.hidden_dim)
        layer_embed = nn.Embed(
            num_embeddings=self.num_layers,
            features=self.hidden_dim,
            name="layer_embed",
        )(jnp.asarray(layer_ids, dtype=jnp.int32) - 1)

        layer_embed = jnp.broadcast_to(
            layer_embed[:, None, :],
            (hidden_stack.shape[0], hidden_stack.shape[1], self.hidden_dim),
        )
        time_embed = jnp.broadcast_to(
            time_embed[None, :, :],
            (hidden_stack.shape[0], hidden_stack.shape[1], self.hidden_dim),
        )

        cond = jnp.concatenate([layer_embed, time_embed], axis=-1)
        cond = nn.Dense(cond_dim, name="cond_proj")(cond)
        cond = nn.silu(cond)

        gamma = nn.Dense(self.hidden_dim, name="gamma")(cond)
        beta = nn.Dense(self.hidden_dim, name="beta")(cond)

        h = nn.LayerNorm(epsilon=1e-6, name="norm")(hidden_stack)
        h = h * (1.0 + gamma[:, :, None, :]) + beta[:, :, None, :]
        h = nn.Dense(self.hidden_dim, name="fc1")(h)
        h = nn.silu(h)
        return nn.Dense(self.out_dim, name="fc2")(h)
