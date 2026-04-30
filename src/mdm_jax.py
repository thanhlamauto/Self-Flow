"""JAX/Flax implementation of the core Motion Diffusion Model.

This ports the MDM transformer path used by ``motion-diffusion-model`` without
pulling in PyTorch, CLIP, BERT, or SMPL. Text conditioning is supported through
precomputed text embeddings; action conditioning is supported through an
embedding table.
"""

from __future__ import annotations

import math
from typing import Optional

import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state


Array = jax.Array


def sinusoidal_embedding(timesteps: Array, dim: int, max_period: float = 10000.0) -> Array:
    """Create sinusoidal timestep embeddings for integer or float timesteps."""
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = timesteps.astype(jnp.float32)[:, None] * freqs[None]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
    return emb


class MlpBlock(nn.Module):
    hidden_dim: int
    out_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: Array, *, deterministic: bool) -> Array:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(self.out_dim)(x)
        x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        return x


class TransformerEncoderBlock(nn.Module):
    latent_dim: int
    num_heads: int
    ff_size: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: Array, attention_mask: Optional[Array], *, deterministic: bool) -> Array:
        h = nn.LayerNorm(epsilon=1e-5)(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.latent_dim,
            out_features=self.latent_dim,
            dropout_rate=self.dropout,
        )(h, h, mask=attention_mask, deterministic=deterministic)
        x = x + nn.Dropout(self.dropout)(h, deterministic=deterministic)
        h = nn.LayerNorm(epsilon=1e-5)(x)
        x = x + MlpBlock(self.ff_size, self.latent_dim, self.dropout)(h, deterministic=deterministic)
        return x


class MDMJax(nn.Module):
    """Motion Diffusion Model in Flax.

    Input and output shape follows the PyTorch MDM convention:
    ``[batch, njoints, nfeats, nframes]``.
    """

    njoints: int
    nfeats: int
    num_actions: int = 1
    latent_dim: int = 256
    ff_size: int = 1024
    num_layers: int = 8
    num_heads: int = 4
    dropout: float = 0.1
    cond_mode: str = "no_cond"
    cond_mask_prob: float = 0.0
    text_embed_dim: int = 512
    max_frames: int = 196
    diffusion_steps: int = 1000

    @nn.compact
    def __call__(
        self,
        x: Array,
        timesteps: Array,
        *,
        mask: Optional[Array] = None,
        action: Optional[Array] = None,
        text_embed: Optional[Array] = None,
        force_uncond: bool = False,
        deterministic: bool = True,
    ) -> Array:
        batch, njoints, nfeats, nframes = x.shape
        if njoints != self.njoints or nfeats != self.nfeats:
            raise ValueError(
                f"Expected motion shape [B,{self.njoints},{self.nfeats},T], got {x.shape}"
            )
        if nframes > self.max_frames:
            raise ValueError(f"nframes={nframes} exceeds max_frames={self.max_frames}")

        h = jnp.transpose(x, (0, 3, 1, 2)).reshape(batch, nframes, njoints * nfeats)
        h = nn.Dense(self.latent_dim, name="pose_embedding")(h)

        time_emb = sinusoidal_embedding(timesteps, self.latent_dim)
        time_emb = nn.Dense(self.latent_dim, name="time_embed_0")(time_emb)
        time_emb = nn.silu(time_emb)
        cond = nn.Dense(self.latent_dim, name="time_embed_1")(time_emb)

        if "text" in self.cond_mode:
            if text_embed is None:
                raise ValueError("cond_mode includes text but text_embed was not provided")
            text_cond = nn.Dense(self.latent_dim, name="text_embedding")(text_embed)
            cond = cond + self._mask_cond(text_cond, force_uncond, deterministic)

        if "action" in self.cond_mode:
            if action is None:
                raise ValueError("cond_mode includes action but action was not provided")
            action_ids = action.reshape((batch,)).astype(jnp.int32)
            action_cond = nn.Embed(
                num_embeddings=self.num_actions,
                features=self.latent_dim,
                name="action_embedding",
            )(action_ids)
            cond = cond + self._mask_cond(action_cond, force_uncond, deterministic)

        pos = self.param(
            "positional_encoding",
            lambda *_: sinusoidal_embedding(jnp.arange(self.max_frames + 1), self.latent_dim),
            (self.max_frames + 1, self.latent_dim),
        )
        cond_token = cond[:, None, :]
        h = jnp.concatenate([cond_token, h], axis=1)
        h = h + pos[None, : nframes + 1, :]
        h = nn.Dropout(self.dropout)(h, deterministic=deterministic)

        attention_mask = None
        if mask is not None:
            frame_mask = _normalize_frame_mask(mask, batch, nframes)
            token_mask = jnp.concatenate([jnp.ones((batch, 1), dtype=bool), frame_mask], axis=1)
            attention_mask = nn.make_attention_mask(token_mask, token_mask)

        for _ in range(self.num_layers):
            h = TransformerEncoderBlock(
                latent_dim=self.latent_dim,
                num_heads=self.num_heads,
                ff_size=self.ff_size,
                dropout=self.dropout,
            )(h, attention_mask, deterministic=deterministic)

        h = h[:, 1:, :]
        h = nn.LayerNorm(epsilon=1e-5)(h)
        h = nn.Dense(njoints * nfeats, name="pose_final")(h)
        h = h.reshape(batch, nframes, njoints, nfeats)
        return jnp.transpose(h, (0, 2, 3, 1))

    def _mask_cond(self, cond: Array, force_uncond: bool, deterministic: bool) -> Array:
        if force_uncond:
            return jnp.zeros_like(cond)
        if deterministic or self.cond_mask_prob <= 0:
            return cond
        keep = jax.random.bernoulli(
            self.make_rng("dropout"),
            p=1.0 - self.cond_mask_prob,
            shape=(cond.shape[0], 1),
        )
        return cond * keep.astype(cond.dtype)


def _normalize_frame_mask(mask: Array, batch: int, nframes: int) -> Array:
    mask = jnp.asarray(mask)
    if mask.ndim == 4:
        mask = mask.reshape(batch, -1)[:, :nframes]
    elif mask.ndim == 2:
        mask = mask[:, :nframes]
    else:
        raise ValueError(f"Expected mask rank 2 or 4, got shape {mask.shape}")
    return mask.astype(bool)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> np.ndarray:
    steps = int(num_steps)
    t = np.linspace(0, steps, steps + 1, dtype=np.float64) / steps
    alphas_cumprod = np.cos((t + s) / (1.0 + s) * np.pi / 2.0) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0, 0.999)


@struct.dataclass
class GaussianDiffusionJax:
    betas: Array
    sqrt_alphas_cumprod: Array
    sqrt_one_minus_alphas_cumprod: Array

    @classmethod
    def create(cls, num_steps: int = 1000, schedule: str = "cosine") -> "GaussianDiffusionJax":
        if schedule == "cosine":
            betas_np = cosine_beta_schedule(num_steps)
        elif schedule == "linear":
            scale = 1000.0 / float(num_steps)
            betas_np = np.linspace(scale * 0.0001, scale * 0.02, num_steps, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported beta schedule {schedule!r}")
        alphas = 1.0 - betas_np
        alphas_cumprod = np.cumprod(alphas, axis=0)
        return cls(
            betas=jnp.asarray(betas_np, dtype=jnp.float32),
            sqrt_alphas_cumprod=jnp.asarray(np.sqrt(alphas_cumprod), dtype=jnp.float32),
            sqrt_one_minus_alphas_cumprod=jnp.asarray(
                np.sqrt(1.0 - alphas_cumprod), dtype=jnp.float32
            ),
        )

    @property
    def num_steps(self) -> int:
        return int(self.betas.shape[0])

    def q_sample(self, x_start: Array, timesteps: Array, noise: Array) -> Array:
        shape = (timesteps.shape[0],) + (1,) * (x_start.ndim - 1)
        a = self.sqrt_alphas_cumprod[timesteps].reshape(shape)
        b = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(shape)
        return a * x_start + b * noise


class MDMTrainState(train_state.TrainState):
    diffusion: GaussianDiffusionJax


def create_mdm_train_state(
    rng: Array,
    model: MDMJax,
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> MDMTrainState:
    dummy_x = jnp.zeros((1, model.njoints, model.nfeats, model.max_frames), dtype=jnp.float32)
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    variables = model.init({"params": rng, "dropout": rng}, dummy_x, dummy_t, deterministic=False)
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    return MDMTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        diffusion=GaussianDiffusionJax.create(model.diffusion_steps),
    )


def mdm_loss_and_metrics(
    apply_fn,
    params,
    diffusion: GaussianDiffusionJax,
    batch: dict[str, Array],
    rng: Array,
    *,
    deterministic: bool = False,
) -> tuple[Array, dict[str, Array]]:
    motion = batch["motion"]
    local_batch = motion.shape[0]
    rng, t_rng, noise_rng, drop_rng = jax.random.split(rng, 4)
    timesteps = jax.random.randint(t_rng, (local_batch,), 0, diffusion.num_steps)
    noise = jax.random.normal(noise_rng, motion.shape, dtype=motion.dtype)
    x_t = diffusion.q_sample(motion, timesteps, noise)

    pred = apply_fn(
        {"params": params},
        x_t,
        timesteps,
        mask=batch.get("mask"),
        action=batch.get("action"),
        text_embed=batch.get("text_embed"),
        deterministic=deterministic,
        rngs={"dropout": drop_rng},
    )
    loss_mask = batch.get("mask")
    if loss_mask is not None:
        loss_mask = _normalize_frame_mask(loss_mask, local_batch, motion.shape[-1])
        loss_mask = loss_mask[:, None, None, :].astype(motion.dtype)
        denom = jnp.maximum(jnp.sum(loss_mask) * motion.shape[1] * motion.shape[2], 1.0)
        mse = jnp.sum(((pred - noise) ** 2) * loss_mask) / denom
    else:
        mse = jnp.mean((pred - noise) ** 2)
    metrics = {
        "loss": mse,
        "pred_abs_mean": jnp.mean(jnp.abs(pred)),
        "noise_abs_mean": jnp.mean(jnp.abs(noise)),
    }
    return mse, metrics


def train_step(state: MDMTrainState, batch: dict[str, Array], rng: Array):
    def loss_fn(params):
        return mdm_loss_and_metrics(state.apply_fn, params, state.diffusion, batch, rng)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {f"train/{k}": v for k, v in metrics.items()}
    metrics["train/loss"] = loss
    return state, metrics
