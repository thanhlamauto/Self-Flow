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


def modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class ShortcutDiTBlock(nn.Module):
    """DiT-style adaLN-Zero block for shortcut predictor tokens."""

    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        norm1 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name="norm1")
        norm2 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name="norm2")
        modulation = nn.Dense(
            6 * self.hidden_size,
            kernel_init=ZERO_INIT,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="adaLN_modulation",
        )(nn.swish(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            modulation,
            6,
            axis=1,
        )

        x_norm = modulate(norm1(x), shift_msa, scale_msa)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            kernel_init=XAVIER_UNIFORM,
            out_kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            out_bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="attn",
        )(x_norm, x_norm)
        x = x + gate_msa[:, None, :] * attn

        x_norm = modulate(norm2(x), shift_mlp, scale_mlp)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        x_mlp = nn.Dense(
            mlp_hidden_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="mlp_fc1",
        )(x_norm)
        x_mlp = nn.gelu(x_mlp, approximate=True)
        x_mlp = nn.Dense(
            self.hidden_size,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="mlp_fc2",
        )(x_mlp)
        return x + gate_mlp[:, None, :] * x_mlp


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
            dtype=jnp.bfloat16,
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
            dtype=jnp.bfloat16,
            name="dwconv",
        )(x)
        x = nn.Dense(
            self.width * self.expansion,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="pw1",
        )(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(
            self.width,
            kernel_init=ZERO_INIT,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
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
            dtype=jnp.bfloat16,
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
            dtype=jnp.bfloat16,
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
            dtype=jnp.bfloat16,
            name="attn_q",
        )(tokens)
        k = nn.Dense(
            self.attn_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="attn_k",
        )(tokens)
        v = nn.Dense(
            self.attn_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
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
            dtype=jnp.bfloat16,
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
            dtype=jnp.bfloat16,
            name="pw1",
        )(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(
            self.width,
            kernel_init=ZERO_INIT,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="pw2",
        )(x)
        return h + x


class DeepDilatedShortcutBlock(nn.Module):
    """Deep conditional residual block with dilated depthwise conv, optional attention, and MLP."""

    width: int
    grid_size: int
    mlp_ratio: float = 4.0
    dilation: int = 1
    use_attention: bool = False
    num_heads: int = 6
    adaln_zero: bool = True

    @nn.compact
    def __call__(self, h: jax.Array, c: jax.Array) -> jax.Array:
        mod_init = ZERO_INIT if self.adaln_zero else XAVIER_UNIFORM

        def modulation(name: str):
            shift, scale, gate = jnp.split(
                nn.Dense(
                    3 * self.width,
                    kernel_init=mod_init,
                    bias_init=ZERO_INIT,
                    dtype=jnp.bfloat16,
                    name=name,
                )(nn.silu(c)),
                3,
                axis=-1,
            )
            return shift, scale, gate

        shift_conv, scale_conv, gate_conv = modulation("conv_adaln")
        x = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name="conv_ln")(h)
        x = modulate(x, shift_conv, scale_conv)
        batch_size, num_tokens, channels = x.shape
        if num_tokens != self.grid_size * self.grid_size:
            raise ValueError("num_tokens must equal grid_size * grid_size for deep shortcut predictor")
        x_grid = x.reshape(batch_size, self.grid_size, self.grid_size, channels)
        x_grid = nn.Conv(
            features=self.width,
            kernel_size=(3, 3),
            padding="SAME",
            feature_group_count=self.width,
            kernel_dilation=(self.dilation, self.dilation),
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="dwconv",
        )(x_grid)
        x_grid = nn.Dense(
            self.width,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="pwconv",
        )(x_grid)
        x = x_grid.reshape(batch_size, num_tokens, self.width)
        h = h + gate_conv[:, None, :] * x

        if self.use_attention:
            if self.width % self.num_heads != 0:
                raise ValueError("width must be divisible by num_heads")
            shift_attn, scale_attn, gate_attn = modulation("attn_adaln")
            x = modulate(
                nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name="attn_ln")(h),
                shift_attn,
                scale_attn,
            )
            x_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.width,
                out_features=self.width,
                kernel_init=XAVIER_UNIFORM,
                out_kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                out_bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name="attn",
            )(x, x)
            h = h + gate_attn[:, None, :] * x_attn

        shift_mlp, scale_mlp, gate_mlp = modulation("mlp_adaln")
        x = modulate(
            nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name="mlp_ln")(h),
            shift_mlp,
            scale_mlp,
        )
        x = nn.Dense(
            int(self.width * self.mlp_ratio),
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="mlp_fc1",
        )(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(
            self.width,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="mlp_fc2",
        )(x)
        return h + gate_mlp[:, None, :] * x


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
            dtype=jnp.bfloat16,
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
            dtype=jnp.bfloat16,
            name="dwconv",
        )(x)
        x = nn.Dense(
            max(channels // 4, 1),
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="pw1",
        )(x)
        x = nn.gelu(x, approximate=True)
        raw_delta_m = nn.Dense(
            1,
            kernel_init=ZERO_INIT,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
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
    mlp_ratio: float = 4.0
    attn_gamma_init: float = 0.05
    layer_cond_dim: int = 16
    cond_hidden_dim: int = 512
    cond_dim: int | None = None
    grid_size_override: int | None = None
    residual_output: bool = True
    attention_every: int = 4
    adaln_zero: bool = True
    gamma_out_init: float = 0.05
    mag_abs_center: float = 5.5
    mag_abs_scale: float = 1.5
    dt_use_h0_target: bool = False
    dt_h0_fusion: str = "concat"
    dt_cond_mode: str = "concat_mlp"
    dt_max_time_bins: int = 4096
    dt_use_layer_cond: bool = True

    @property
    def grid_size(self) -> int:
        return int(self.grid_size_override or int(self.num_tokens ** 0.5))

    @nn.compact
    def __call__(
        self,
        u_source: jax.Array,
        source_layer: jax.Array,
        target_layer: jax.Array,
        timestep_embed: jax.Array,
        m_source: jax.Array | None = None,
        detach_timestep_embed: bool = True,
        use_timestep_embed: bool = True,
        h0_tgt: jax.Array | None = None,
        timestep_tgt_embed: jax.Array | None = None,
        t_src_idx: jax.Array | None = None,
        t_tgt_idx: jax.Array | None = None,
        delta_t_idx: jax.Array | None = None,
    ) -> jax.Array:
        batch_size = u_source.shape[0]
        cond_dim = int(self.cond_dim or self.width)
        if self.dt_h0_fusion not in {"add", "concat"}:
            raise ValueError(f"Unknown h0 fusion mode: {self.dt_h0_fusion!r}")
        if self.dt_cond_mode not in {"sum", "concat_mlp", "factorized"}:
            raise ValueError(f"Unknown depth-time conditioning mode: {self.dt_cond_mode!r}")
        source_layer = jnp.asarray(source_layer, dtype=jnp.int32)
        target_layer = jnp.asarray(target_layer, dtype=jnp.int32)
        delta = target_layer - source_layer

        detach_timestep = jnp.asarray(detach_timestep_embed, dtype=jnp.bool_)
        if use_timestep_embed:
            t_cond = jax.lax.cond(
                detach_timestep,
                jax.lax.stop_gradient,
                lambda x: x,
                timestep_embed,
            )
        else:
            t_cond = jnp.zeros_like(timestep_embed)
        if timestep_tgt_embed is None:
            t_tgt_cond = t_cond
        elif use_timestep_embed:
            t_tgt_cond = jax.lax.cond(
                detach_timestep,
                jax.lax.stop_gradient,
                lambda x: x,
                timestep_tgt_embed,
            )
        else:
            t_tgt_cond = jnp.zeros_like(timestep_embed)
        t_delta_cond = t_cond - t_tgt_cond

        def time_idx_embed(name: str, idx: jax.Array | None):
            if idx is None:
                return jnp.zeros((batch_size, cond_dim), dtype=t_cond.dtype)
            idx = jnp.asarray(idx, dtype=jnp.int32)
            idx = jnp.clip(idx, 0, int(self.dt_max_time_bins) - 1)
            emb = nn.Embed(
                num_embeddings=int(self.dt_max_time_bins),
                features=cond_dim,
                embedding_init=NORMAL_002,
                dtype=jnp.bfloat16,
                name=name,
            )(idx)
            if emb.ndim == 1:
                emb = jnp.broadcast_to(emb[None, :], (batch_size, cond_dim))
            return emb

        def cond_mlp(name: str, x: jax.Array, out_dim: int) -> jax.Array:
            h_cond = nn.Dense(
                out_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name=f"{name}_0",
            )(x)
            h_cond = nn.gelu(h_cond, approximate=True)
            return nn.Dense(
                out_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name=f"{name}_1",
            )(h_cond)

        if self.arch in {"deep_dilated_mlp", "hybrid_deep"}:
            t_proj = nn.Dense(
                cond_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name="cond_t_proj",
            )(t_cond)
            layer_embed = nn.Embed(
                num_embeddings=self.depth + 1,
                features=cond_dim,
                embedding_init=NORMAL_002,
                dtype=jnp.bfloat16,
                name="cond_layer_embed",
            )
            delta_embed = nn.Embed(
                num_embeddings=self.depth,
                features=cond_dim,
                embedding_init=NORMAL_002,
                dtype=jnp.bfloat16,
                name="cond_delta_embed",
            )
            if self.dt_use_layer_cond:
                e_a = layer_embed(source_layer)
                e_b = layer_embed(target_layer)
                e_delta_layer = delta_embed(jnp.clip(delta - 1, 0, self.depth - 1))
                if e_a.ndim == 1:
                    e_a = jnp.broadcast_to(e_a[None, :], (batch_size, cond_dim))
                    e_b = jnp.broadcast_to(e_b[None, :], (batch_size, cond_dim))
                    e_delta_layer = jnp.broadcast_to(e_delta_layer[None, :], (batch_size, cond_dim))
            else:
                e_a = jnp.zeros((batch_size, cond_dim), dtype=t_cond.dtype)
                e_b = jnp.zeros_like(e_a)
                e_delta_layer = jnp.zeros_like(e_a)
            e_t = t_proj + time_idx_embed("cond_t_src_idx_embed", t_src_idx)
            e_s = nn.Dense(
                cond_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name="cond_t_tgt_proj",
            )(t_tgt_cond) + time_idx_embed("cond_t_tgt_idx_embed", t_tgt_idx)
            e_delta_t = nn.Dense(
                cond_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name="cond_t_delta_proj",
            )(t_delta_cond) + time_idx_embed("cond_delta_t_idx_embed", delta_t_idx)
            if self.dt_cond_mode == "sum":
                c = e_a + e_b + e_delta_layer + e_t + e_s + e_delta_t
            elif self.dt_cond_mode == "concat_mlp":
                c = cond_mlp(
                    "cond_concat_mlp",
                    jnp.concatenate([e_a, e_b, e_delta_layer, e_t, e_s, e_delta_t], axis=-1),
                    cond_dim,
                )
            else:
                e_layer = cond_mlp(
                    "cond_layer_mlp",
                    jnp.concatenate([e_a, e_b, e_delta_layer], axis=-1),
                    cond_dim,
                )
                e_time = cond_mlp(
                    "cond_time_mlp",
                    jnp.concatenate([e_t, e_s, e_delta_t], axis=-1),
                    cond_dim,
                )
                c = (
                    cond_mlp(
                        "cond_factorized_fuse_mlp",
                        jnp.concatenate([e_layer, e_time, e_layer * e_time], axis=-1),
                        cond_dim,
                    )
                )
            c = nn.Dense(
                cond_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name="cond_out",
            )(nn.gelu(c, approximate=True))
        else:
            state_embed = nn.Dense(
                self.layer_cond_dim,
                use_bias=False,
                kernel_init=NORMAL_002,
                dtype=jnp.bfloat16,
                name="state_layer_proj",
            )
            delta_embed = nn.Dense(
                self.layer_cond_dim,
                use_bias=False,
                kernel_init=NORMAL_002,
                dtype=jnp.bfloat16,
                name="delta_layer_proj",
            )

            if self.dt_use_layer_cond:
                h_a_raw = state_embed(jax.nn.one_hot(source_layer, self.depth + 1))
                h_b_raw = state_embed(jax.nn.one_hot(target_layer, self.depth + 1))
                h_delta_raw = delta_embed(jax.nn.one_hot(delta - 1, self.depth))
                e_a = nn.Dense(
                    self.cond_hidden_dim,
                    kernel_init=XAVIER_UNIFORM,
                    bias_init=ZERO_INIT,
                    dtype=jnp.bfloat16,
                    name="cond_layer_src_proj",
                )(jnp.broadcast_to(h_a_raw[None, :], (batch_size, h_a_raw.shape[-1])))
                e_b = nn.Dense(
                    self.cond_hidden_dim,
                    kernel_init=XAVIER_UNIFORM,
                    bias_init=ZERO_INIT,
                    dtype=jnp.bfloat16,
                    name="cond_layer_tgt_proj",
                )(jnp.broadcast_to(h_b_raw[None, :], (batch_size, h_b_raw.shape[-1])))
                e_delta_layer = nn.Dense(
                    self.cond_hidden_dim,
                    kernel_init=XAVIER_UNIFORM,
                    bias_init=ZERO_INIT,
                    dtype=jnp.bfloat16,
                    name="cond_layer_delta_proj",
                )(jnp.broadcast_to(h_delta_raw[None, :], (batch_size, h_delta_raw.shape[-1])))
            else:
                e_a = jnp.zeros((batch_size, self.cond_hidden_dim), dtype=t_cond.dtype)
                e_b = jnp.zeros_like(e_a)
                e_delta_layer = jnp.zeros_like(e_a)
            e_t = nn.Dense(
                self.cond_hidden_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name="cond_time_src_proj",
            )(t_cond)
            e_s = nn.Dense(
                self.cond_hidden_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name="cond_time_tgt_proj",
            )(t_tgt_cond)
            e_delta_t = nn.Dense(
                self.cond_hidden_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name="cond_time_delta_proj",
            )(t_delta_cond)
            if self.dt_cond_mode == "sum":
                c_in = e_a + e_b + e_delta_layer + e_t + e_s + e_delta_t
            elif self.dt_cond_mode == "concat_mlp":
                c_in = jnp.concatenate([e_a, e_b, e_delta_layer, e_t, e_s, e_delta_t], axis=-1)
            else:
                e_layer = cond_mlp(
                    "cond_layer_mlp",
                    jnp.concatenate([e_a, e_b, e_delta_layer], axis=-1),
                    self.cond_hidden_dim,
                )
                e_time = cond_mlp(
                    "cond_time_mlp",
                    jnp.concatenate([e_t, e_s, e_delta_t], axis=-1),
                    self.cond_hidden_dim,
                )
                c_in = jnp.concatenate([e_layer, e_time, e_layer * e_time], axis=-1)
            c = nn.Dense(
                self.cond_hidden_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name="cond_mlp_0",
            )(c_in)
            c = nn.gelu(c, approximate=True)
            c = nn.Dense(
                cond_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name="cond_mlp_1",
            )(c)

        predictor_input = u_source
        if self.dt_use_h0_target and h0_tgt is not None:
            h0_cond = nn.Dense(
                self.hidden_size,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
                dtype=jnp.bfloat16,
                name="h0_tgt_proj",
            )(h0_tgt.astype(jnp.bfloat16))
            if self.dt_h0_fusion == "add":
                predictor_input = predictor_input + h0_cond
            else:
                predictor_input = nn.Dense(
                    self.hidden_size,
                    kernel_init=XAVIER_UNIFORM,
                    bias_init=ZERO_INIT,
                    dtype=jnp.bfloat16,
                    name="h0_concat_proj",
                )(jnp.concatenate([predictor_input, h0_cond], axis=-1))

        h = nn.Dense(
            self.width,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="in_proj",
        )(predictor_input)
        if self.arch == "dit2":
            if self.num_heads is None:
                raise ValueError("dit2 predictor requires num_heads")
            for idx in range(self.num_blocks):
                h = ShortcutDiTBlock(
                    hidden_size=self.width,
                    num_heads=int(self.num_heads),
                    mlp_ratio=float(self.mlp_ratio),
                    name=f"blocks_{idx}",
                )(h, c)
            h_grid = h.reshape(batch_size, self.grid_size, self.grid_size, self.width)
        else:
            h = h.reshape(batch_size, self.grid_size, self.grid_size, self.width)
            dilation_schedule = self.dilation_schedule or tuple([1] * self.num_blocks)
            if len(dilation_schedule) != self.num_blocks:
                raise ValueError("dilation_schedule length must match num_blocks")
            for idx in range(self.num_blocks):
                dilation = int(dilation_schedule[idx])
                if self.arch in {"deep_dilated_mlp", "hybrid_deep"}:
                    h = h.reshape(batch_size, self.num_tokens, self.width)
                    use_attention = (
                        self.arch == "hybrid_deep"
                        and self.attention_every > 0
                        and ((idx + 1) % self.attention_every == 0)
                    )
                    h = DeepDilatedShortcutBlock(
                        width=self.width,
                        grid_size=self.grid_size,
                        mlp_ratio=float(self.mlp_ratio),
                        dilation=dilation,
                        use_attention=use_attention,
                        num_heads=int(self.num_heads or 1),
                        adaln_zero=bool(self.adaln_zero),
                        name=f"blocks_{idx}",
                    )(h, c)
                    h = h.reshape(batch_size, self.grid_size, self.grid_size, self.width)
                elif self.arch in {"convnext", "dilated_convnext"}:
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
        if self.arch in {"deep_dilated_mlp", "hybrid_deep"}:
            h = nn.LayerNorm(epsilon=1e-6, name="final_ln")(h)
        delta_y = nn.Dense(
            self.hidden_size,
            kernel_init=ZERO_INIT,
            bias_init=ZERO_INIT,
            dtype=jnp.bfloat16,
            name="out_proj",
        )(h)
        gamma_out = self.param(
            "gamma_out",
            nn.initializers.constant(self.gamma_out_init),
            (),
        )
        y = u_source + gamma_out * delta_y if self.residual_output else delta_y
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
    "dit2_tiny": {
        "arch": "dit2",
        "width": 256,
        "num_blocks": 2,
        "expansion": 2,
        "dilation_schedule": None,
        "attn_dim": None,
        "num_heads": 4,
        "mlp_ratio": 4.0,
    },
    "dit2_small": {
        "arch": "dit2",
        "width": 384,
        "num_blocks": 2,
        "expansion": 2,
        "dilation_schedule": None,
        "attn_dim": None,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "dit2_base": {
        "arch": "dit2",
        "width": 384,
        "num_blocks": 4,
        "expansion": 2,
        "dilation_schedule": None,
        "attn_dim": None,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "dit2_wide_base": {
        "arch": "dit2",
        "width": 512,
        "num_blocks": 3,
        "expansion": 2,
        "dilation_schedule": None,
        "attn_dim": None,
        "num_heads": 8,
        "mlp_ratio": 4.0,
    },
    "dit2_base_v2": {
        "arch": "dit2",
        "width": 320,
        "num_blocks": 6,
        "expansion": 2,
        "dilation_schedule": None,
        "attn_dim": None,
        "num_heads": 5,
        "mlp_ratio": 4.0,
    },
    "dit2_deep_256": {
        "arch": "dit2",
        "width": 256,
        "num_blocks": 8,
        "expansion": 2,
        "dilation_schedule": None,
        "attn_dim": None,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "residual_output": True,
    },
    "dit2_large": {
        "arch": "dit2",
        "width": 384,
        "num_blocks": 6,
        "expansion": 2,
        "dilation_schedule": None,
        "attn_dim": None,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
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
    "deep_dilated_mlp": {
        "arch": "deep_dilated_mlp",
        "width": 384,
        "num_blocks": 12,
        "expansion": 2,
        "dilation_schedule": (1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4),
        "attn_dim": None,
        "num_heads": None,
        "mlp_ratio": 4.0,
        "cond_dim": 32,
        "grid_size_override": 16,
        "residual_output": True,
        "attention_every": 0,
        "adaln_zero": True,
    },
    "hybrid_deep": {
        "arch": "hybrid_deep",
        "width": 384,
        "num_blocks": 12,
        "expansion": 2,
        "dilation_schedule": (1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4),
        "attn_dim": None,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "cond_dim": 32,
        "grid_size_override": 16,
        "residual_output": True,
        "attention_every": 4,
        "adaln_zero": True,
    },
    "hybrid_deep_10": {
        "arch": "hybrid_deep",
        "width": 384,
        "num_blocks": 10,
        "expansion": 2,
        "dilation_schedule": (1, 2, 4, 1, 2, 4, 1, 2, 4, 1),
        "attn_dim": None,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "cond_dim": 32,
        "grid_size_override": 16,
        "residual_output": True,
        "attention_every": 4,
        "adaln_zero": True,
    },
    "hybrid_deep_30m": {
        "arch": "hybrid_deep",
        "width": 448,
        "num_blocks": 12,
        "expansion": 2,
        "dilation_schedule": (1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4),
        "attn_dim": None,
        "num_heads": 7,
        "mlp_ratio": 4.0,
        "cond_dim": 64,
        "grid_size_override": 16,
        "residual_output": True,
        "attention_every": 4,
        "adaln_zero": True,
    },
}


PREDICTOR_VARIANT_ALIASES = {
    "tiny": "convnext_tiny",
    "p_tiny": "convnext_tiny",
    "ptiny": "convnext_tiny",
    "dit_tiny": "dit2_tiny",
    "dit_t": "dit2_tiny",
    "small": "convnext_small",
    "p_small": "convnext_small",
    "psmall": "convnext_small",
    "dit_small": "dit2_small",
    "dit_s": "dit2_small",
    "base": "convnext_base",
    "p_base": "convnext_base",
    "pbase": "convnext_base",
    "dit_base": "dit2_base",
    "dit_b": "dit2_base",
    "dit_wide_base": "dit2_wide_base",
    "dit_wb": "dit2_wide_base",
    "dit_base_v2": "dit2_base_v2",
    "dit_b_v2": "dit2_base_v2",
    "dit_deep_256": "dit2_deep_256",
    "dit_d256": "dit2_deep_256",
    "large": "convnext_large",
    "p_large": "convnext_large",
    "plarge": "convnext_large",
    "dit_large": "dit2_large",
    "dit_l": "dit2_large",
    "hybrid_depth10": "hybrid_deep_10",
    "hybrid_depth_10": "hybrid_deep_10",
    "hybrid_depth30m": "hybrid_deep_30m",
    "hybrid_depth_30m": "hybrid_deep_30m",
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
    if canonical in {"deep_dilated_mlp", "hybrid_deep", "hybrid_deep_10", "hybrid_deep_30m"}:
        return "large"
    if canonical in {"dit2_base_v2", "dit2_deep_256"}:
        return "base"
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


def apply_predictor_config_overrides(
    cfg: dict,
    *,
    arch: str = "existing",
    hidden_size: int | None = None,
    depth: int | None = None,
    mlp_ratio: float | None = None,
    dilation_cycle: tuple[int, ...] | None = None,
    grid_size: int | None = None,
    residual_output: bool | None = None,
    attention_every: int | None = None,
    num_heads: int | None = None,
    adaln_zero: bool | None = None,
) -> dict:
    """Apply optional CLI overrides to a predictor config."""
    cfg = dict(cfg)
    if arch != "existing":
        if arch not in PREDICTOR_VARIANTS:
            raise ValueError(f"Unknown shortcut predictor arch override: {arch!r}")
        cfg.update(dict(PREDICTOR_VARIANTS[arch]))
    if hidden_size is not None:
        cfg["width"] = int(hidden_size)
    if depth is not None:
        cfg["num_blocks"] = int(depth)
    if mlp_ratio is not None:
        cfg["mlp_ratio"] = float(mlp_ratio)
    if dilation_cycle:
        blocks = int(cfg.get("num_blocks", len(dilation_cycle)))
        cfg["dilation_schedule"] = tuple(int(dilation_cycle[idx % len(dilation_cycle)]) for idx in range(blocks))
    if grid_size is not None:
        cfg["grid_size_override"] = int(grid_size)
    if residual_output is not None:
        cfg["residual_output"] = bool(residual_output)
    if attention_every is not None:
        cfg["attention_every"] = int(attention_every)
    if num_heads is not None:
        cfg["num_heads"] = int(num_heads)
    if adaln_zero is not None:
        cfg["adaln_zero"] = bool(adaln_zero)
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
