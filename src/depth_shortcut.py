"""Depth Shortcut Predictor modules and helpers for DiT hidden states."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


ZERO_INIT = nn.initializers.zeros
XAVIER_UNIFORM = nn.initializers.xavier_uniform()
NORMAL_002 = nn.initializers.normal(stddev=0.02)


def l2_normalize_tokens(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Normalize each token vector to unit L2 direction."""
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


class AdaLNConvNeXtBlock(nn.Module):
    """Small ConvNeXt-style block conditioned by AdaLN shift/scale."""

    width: int
    cond_dim: int
    expansion: int = 2

    @nn.compact
    def __call__(self, h: jax.Array, c: jax.Array) -> jax.Array:
        gamma_beta = nn.Dense(
            2 * self.width,
            kernel_init=ZERO_INIT,
            bias_init=ZERO_INIT,
            name="adaln_mod",
        )(nn.gelu(c, approximate=True))
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)

        x = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name="ln")(h)
        x = x * (1.0 + gamma[:, None, None, :]) + beta[:, None, None, :]
        x = nn.Conv(
            features=self.width,
            kernel_size=(3, 3),
            padding="SAME",
            feature_group_count=self.width,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="dwconv",
        )(x)
        x = nn.Dense(
            self.width * self.expansion,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="pw1",
        )(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(
            self.width,
            kernel_init=ZERO_INIT,
            bias_init=ZERO_INIT,
            name="pw2",
        )(x)
        return h + x


class DepthShortcutPredictor(nn.Module):
    """Predicts target hidden-state directions from source directions."""

    hidden_size: int = 768
    depth: int = 12
    num_tokens: int = 256
    width: int = 256
    num_blocks: int = 2
    expansion: int = 2
    layer_cond_dim: int = 16
    cond_hidden_dim: int = 512
    cond_dim: int | None = None
    gamma_out_init: float = 0.05

    @property
    def grid_size(self) -> int:
        return int(self.num_tokens ** 0.5)

    @nn.compact
    def __call__(
        self,
        u_source: jax.Array,
        source_layer: jax.Array,
        target_layer: jax.Array,
        timestep_embed: jax.Array,
    ) -> jax.Array:
        batch_size = u_source.shape[0]
        cond_dim = int(self.cond_dim or self.width)
        source_layer = jnp.asarray(source_layer, dtype=jnp.int32)
        target_layer = jnp.asarray(target_layer, dtype=jnp.int32)
        delta = target_layer - source_layer

        state_embed = nn.Dense(
            self.layer_cond_dim,
            use_bias=False,
            kernel_init=NORMAL_002,
            name="state_layer_proj",
        )
        delta_embed = nn.Dense(
            self.layer_cond_dim,
            use_bias=False,
            kernel_init=NORMAL_002,
            name="delta_layer_proj",
        )

        h_a = state_embed(jax.nn.one_hot(source_layer, self.depth + 1))
        h_b = state_embed(jax.nn.one_hot(target_layer, self.depth + 1))
        h_delta = delta_embed(jax.nn.one_hot(delta - 1, self.depth))
        h_depth = jnp.concatenate([h_a, h_b, h_b - h_a, h_delta], axis=-1)
        h_depth = jnp.broadcast_to(h_depth[None, :], (batch_size, h_depth.shape[-1]))

        t_cond = jax.lax.stop_gradient(timestep_embed)
        c_in = jnp.concatenate([h_depth, t_cond], axis=-1)
        c = nn.Dense(
            self.cond_hidden_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="cond_mlp_0",
        )(c_in)
        c = nn.gelu(c, approximate=True)
        c = nn.Dense(
            cond_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="cond_mlp_1",
        )(c)

        h = nn.Dense(
            self.width,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="in_proj",
        )(u_source)
        h = h.reshape(batch_size, self.grid_size, self.grid_size, self.width)
        for idx in range(self.num_blocks):
            h = AdaLNConvNeXtBlock(
                width=self.width,
                cond_dim=cond_dim,
                expansion=self.expansion,
                name=f"blocks_{idx}",
            )(h, c)
        h = h.reshape(batch_size, self.num_tokens, self.width)
        delta_y = nn.Dense(
            self.hidden_size,
            kernel_init=ZERO_INIT,
            bias_init=ZERO_INIT,
            name="out_proj",
        )(h)
        gamma_out = self.param(
            "gamma_out",
            nn.initializers.constant(self.gamma_out_init),
            (),
        )
        return u_source + gamma_out * delta_y


def predictor_config_from_name(name: str, hidden_size: int) -> dict:
    """Return predictor size settings for tiny/small/base."""
    name = str(name).lower()
    if name in {"tiny", "p-tiny", "ptiny"}:
        return {"width": 256, "num_blocks": 2, "expansion": 2}
    if name in {"small", "p-small", "psmall"}:
        return {"width": 384, "num_blocks": 4, "expansion": 2}
    if name in {"base", "p-base", "pbase"}:
        return {"width": int(hidden_size), "num_blocks": 4, "expansion": 2}
    if name in {"large", "p-large", "plarge"}:
        return {"width": int(hidden_size), "num_blocks": 6, "expansion": 2}
    raise ValueError(f"Unknown shortcut predictor variant: {name!r}")


def restore_l2_ema_magnitude(
    predicted_direction: jax.Array,
    l2_ema: jax.Array,
    target_layer: jax.Array,
    timestep_indices: jax.Array,
) -> jax.Array:
    """Restore token magnitudes for inference ablations from 5-bin L2 EMA."""
    bins = jnp.clip(timestep_indices // 10, 0, l2_ema.shape[1] - 1)
    radii = l2_ema[target_layer, bins]  # [B, P] for batched q, [P] for scalar q.
    if radii.ndim == 1:
        radii = radii[None, :]
    return predicted_direction * radii[:, :, None]
