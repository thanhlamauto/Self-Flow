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
    x_f32 = x.astype(jnp.float32)
    return x_f32 / (jnp.linalg.norm(x_f32, axis=-1, keepdims=True) + eps)


def log_token_magnitudes(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Tokenwise log L2 magnitudes."""
    x_f32 = x.astype(jnp.float32)
    return jnp.log(jnp.linalg.norm(x_f32, axis=-1, keepdims=True) + eps)


def magnitude_input_features(
    m_source: jax.Array,
    abs_center: float = 5.5,
    abs_scale: float = 1.5,
    eps: float = 1e-6,
) -> jax.Array:
    """Build absolute and within-image contrast log-magnitude channels."""
    m_source = m_source.astype(jnp.float32)
    m_abs = (m_source - float(abs_center)) / float(abs_scale)
    token_mean = jnp.mean(m_source, axis=1, keepdims=True)
    token_std = jnp.std(m_source, axis=1, keepdims=True)
    m_spatial = (m_source - token_mean) / (token_std + eps)
    return jnp.concatenate([m_abs, m_spatial], axis=-1)


class AdaLNConvNeXtBlock(nn.Module):
    """Small ConvNeXt-style block conditioned by AdaLN shift/scale."""

    width: int
    cond_dim: int
    expansion: int = 2
    dilation: int = 1

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
            kernel_dilation=(self.dilation, self.dilation),
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


class AttentionHybridShortcutBlock(nn.Module):
    """ConvNeXt-style block with a lightweight token-attention bottleneck."""

    width: int
    cond_dim: int
    expansion: int = 1
    attn_dim: int = 128
    num_heads: int = 4
    dilation: int = 1
    attn_gamma_init: float = 0.05

    @nn.compact
    def __call__(self, h: jax.Array, c: jax.Array) -> jax.Array:
        if self.attn_dim % self.num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")

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
            kernel_dilation=(self.dilation, self.dilation),
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="dwconv",
        )(x)

        batch_size, height, grid_width, channels = x.shape
        seq_len = height * grid_width
        tokens = x.reshape(batch_size, seq_len, channels)
        head_dim = self.attn_dim // self.num_heads
        q = nn.Dense(
            self.attn_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="attn_q",
        )(tokens)
        k = nn.Dense(
            self.attn_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="attn_k",
        )(tokens)
        v = nn.Dense(
            self.attn_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="attn_v",
        )(tokens)
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        attn_logits = jnp.einsum("bhnd,bhmd->bhnm", q, k) * (head_dim ** -0.5)
        attn = nn.softmax(attn_logits, axis=-1)
        attn_out = jnp.einsum("bhnm,bhmd->bhnd", attn, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.attn_dim)
        attn_out = nn.Dense(
            self.width,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="attn_out",
        )(attn_out)
        attn_out = attn_out.reshape(batch_size, height, grid_width, self.width)
        gamma_attn = self.param(
            "gamma_attn",
            nn.initializers.constant(self.attn_gamma_init),
            (),
        )
        x = x + gamma_attn * attn_out

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


class MagnitudeHead(nn.Module):
    """Predicts log-magnitude residuals conditioned like the shared backbone."""

    width: int
    cond_dim: int
    mag_abs_center: float = 5.5
    mag_abs_scale: float = 1.5

    @nn.compact
    def __call__(self, h: jax.Array, m_source: jax.Array, c: jax.Array) -> jax.Array:
        batch_size, height, width, channels = h.shape
        m_features = magnitude_input_features(
            m_source,
            abs_center=self.mag_abs_center,
            abs_scale=self.mag_abs_scale,
        )
        m_grid = m_features.reshape(batch_size, height, width, 2)
        x = jnp.concatenate([h, m_grid], axis=-1)
        mag_channels = channels + 2

        gamma_beta = nn.Dense(
            2 * mag_channels,
            kernel_init=ZERO_INIT,
            bias_init=ZERO_INIT,
            name="adaln_mod",
        )(nn.gelu(c, approximate=True))
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)

        x = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name="ln")(x)
        x = x * (1.0 + gamma[:, None, None, :]) + beta[:, None, None, :]
        x = nn.Conv(
            features=mag_channels,
            kernel_size=(3, 3),
            padding="SAME",
            feature_group_count=mag_channels,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="dwconv",
        )(x)
        x = nn.Dense(
            max(channels // 4, 1),
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            name="pw1",
        )(x)
        x = nn.gelu(x, approximate=True)
        raw_delta_m = nn.Dense(
            1,
            kernel_init=ZERO_INIT,
            bias_init=ZERO_INIT,
            name="pw2",
        )(x)
        return jnp.tanh(raw_delta_m).reshape(batch_size, height * width, 1)


class DepthShortcutPredictor(nn.Module):
    """Predicts target hidden-state directions from source directions."""

    hidden_size: int = 768
    depth: int = 12
    num_tokens: int = 256
    width: int = 256
    num_blocks: int = 2
    expansion: int = 2
    arch: str = "convnext"
    dilation_schedule: tuple[int, ...] | None = None
    attn_dim: int | None = None
    num_heads: int | None = None
    attn_gamma_init: float = 0.05
    layer_cond_dim: int = 16
    cond_hidden_dim: int = 512
    cond_dim: int | None = None
    gamma_out_init: float = 0.05
    mag_abs_center: float = 5.5
    mag_abs_scale: float = 1.5

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
        m_source: jax.Array | None = None,
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
        dilation_schedule = self.dilation_schedule or tuple([1] * self.num_blocks)
        if len(dilation_schedule) != self.num_blocks:
            raise ValueError("dilation_schedule length must match num_blocks")
        for idx in range(self.num_blocks):
            dilation = int(dilation_schedule[idx])
            if self.arch in {"convnext", "dilated_convnext"}:
                h = AdaLNConvNeXtBlock(
                    width=self.width,
                    cond_dim=cond_dim,
                    expansion=self.expansion,
                    dilation=dilation,
                    name=f"blocks_{idx}",
                )(h, c)
            elif self.arch == "attn_hybrid":
                if self.attn_dim is None or self.num_heads is None:
                    raise ValueError("attn_hybrid requires attn_dim and num_heads")
                h = AttentionHybridShortcutBlock(
                    width=self.width,
                    cond_dim=cond_dim,
                    expansion=self.expansion,
                    attn_dim=int(self.attn_dim),
                    num_heads=int(self.num_heads),
                    dilation=dilation,
                    attn_gamma_init=self.attn_gamma_init,
                    name=f"blocks_{idx}",
                )(h, c)
            else:
                raise ValueError(f"Unknown shortcut predictor arch: {self.arch!r}")
        h_grid = h
        h = h_grid.reshape(batch_size, self.num_tokens, self.width)
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
        y = u_source + gamma_out * delta_y
        if m_source is None:
            return y
        delta_m = MagnitudeHead(
            width=self.width,
            cond_dim=cond_dim,
            mag_abs_center=self.mag_abs_center,
            mag_abs_scale=self.mag_abs_scale,
            name="mag_head",
        )(h_grid, m_source, c)
        return y, delta_m


PREDICTOR_VARIANTS = {
    "convnext_tiny": {
        "arch": "convnext",
        "width": 256,
        "num_blocks": 2,
        "expansion": 2,
        "dilation_schedule": (1, 1),
        "attn_dim": None,
        "num_heads": None,
    },
    "convnext_small": {
        "arch": "convnext",
        "width": 384,
        "num_blocks": 4,
        "expansion": 2,
        "dilation_schedule": (1, 1, 1, 1),
        "attn_dim": None,
        "num_heads": None,
    },
    "convnext_base": {
        "arch": "convnext",
        "width": None,
        "num_blocks": 4,
        "expansion": 2,
        "dilation_schedule": (1, 1, 1, 1),
        "attn_dim": None,
        "num_heads": None,
    },
    "convnext_large": {
        "arch": "convnext",
        "width": None,
        "num_blocks": 6,
        "expansion": 2,
        "dilation_schedule": (1, 1, 1, 1, 1, 1),
        "attn_dim": None,
        "num_heads": None,
    },
    "dilated_tiny": {
        "arch": "dilated_convnext",
        "width": 256,
        "num_blocks": 2,
        "expansion": 2,
        "dilation_schedule": (1, 2),
        "attn_dim": None,
        "num_heads": None,
    },
    "dilated_small": {
        "arch": "dilated_convnext",
        "width": 384,
        "num_blocks": 4,
        "expansion": 2,
        "dilation_schedule": (1, 2, 3, 1),
        "attn_dim": None,
        "num_heads": None,
    },
    "dilated_base": {
        "arch": "dilated_convnext",
        "width": None,
        "num_blocks": 4,
        "expansion": 2,
        "dilation_schedule": (1, 2, 3, 1),
        "attn_dim": None,
        "num_heads": None,
    },
    "dilated_large": {
        "arch": "dilated_convnext",
        "width": None,
        "num_blocks": 6,
        "expansion": 2,
        "dilation_schedule": (1, 2, 3, 1, 2, 3),
        "attn_dim": None,
        "num_heads": None,
    },
    "attn_hybrid_tiny": {
        "arch": "attn_hybrid",
        "width": 256,
        "num_blocks": 2,
        "expansion": 1,
        "dilation_schedule": (1, 1),
        "attn_dim": 128,
        "num_heads": 4,
    },
    "attn_hybrid_small": {
        "arch": "attn_hybrid",
        "width": 384,
        "num_blocks": 4,
        "expansion": 1,
        "dilation_schedule": (1, 1, 1, 1),
        "attn_dim": 192,
        "num_heads": 6,
    },
    "attn_hybrid_base": {
        "arch": "attn_hybrid",
        "width": None,
        "num_blocks": 4,
        "expansion": 1,
        "dilation_schedule": (1, 1, 1, 1),
        "attn_dim": None,
        "num_heads": 8,
    },
    "attn_hybrid_large": {
        "arch": "attn_hybrid",
        "width": None,
        "num_blocks": 6,
        "expansion": 1,
        "dilation_schedule": (1, 1, 1, 1, 1, 1),
        "attn_dim": None,
        "num_heads": 8,
    },
}


PREDICTOR_VARIANT_ALIASES = {
    "tiny": "convnext_tiny",
    "p_tiny": "convnext_tiny",
    "ptiny": "convnext_tiny",
    "small": "convnext_small",
    "p_small": "convnext_small",
    "psmall": "convnext_small",
    "base": "convnext_base",
    "p_base": "convnext_base",
    "pbase": "convnext_base",
    "large": "convnext_large",
    "p_large": "convnext_large",
    "plarge": "convnext_large",
}


def predictor_variant_names() -> tuple[str, ...]:
    """Return accepted CLI names for shortcut predictor variants."""
    names = set(PREDICTOR_VARIANTS) | set(PREDICTOR_VARIANT_ALIASES)
    names |= {name.replace("_", "-") for name in names}
    return tuple(sorted(names))


def canonical_predictor_variant_name(name: str) -> str:
    """Normalize legacy and architecture-specific predictor variant names."""
    normalized = str(name).lower().replace("-", "_")
    normalized = PREDICTOR_VARIANT_ALIASES.get(normalized, normalized)
    if normalized not in PREDICTOR_VARIANTS:
        raise ValueError(f"Unknown shortcut predictor variant: {name!r}")
    return normalized


def predictor_size_bucket(name: str) -> str:
    """Return tiny/small/base/large for a predictor variant."""
    canonical = canonical_predictor_variant_name(name)
    return canonical.rsplit("_", 1)[-1]


def predictor_config_from_name(name: str, hidden_size: int) -> dict:
    """Return predictor architecture and size settings."""
    canonical = canonical_predictor_variant_name(name)
    cfg = dict(PREDICTOR_VARIANTS[canonical])
    if cfg["width"] is None:
        cfg["width"] = int(hidden_size)
    if cfg["attn_dim"] is None and cfg["arch"] == "attn_hybrid":
        cfg["attn_dim"] = max(int(cfg["width"]) // 2, 1)
    return cfg


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
