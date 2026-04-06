"""Activation-only regularizer helpers for DiT training."""

from typing import Sequence
import jax
import jax.numpy as jnp


def build_layer_pairs(selected_layers: Sequence[int], pair_stride: int) -> tuple[tuple[int, int], ...]:
    """Create adjacent pairs while stepping through the layer list by pair_stride."""
    if pair_stride <= 0:
        raise ValueError("pair_stride must be greater than 0")

    layers = tuple(int(layer) for layer in selected_layers)
    return tuple(
        (layers[idx], layers[idx + 1])
        for idx in range(0, len(layers) - 1, pair_stride)
    )


def _safe_l2_normalize(x: jax.Array, eps: float) -> jax.Array:
    x = x.astype(jnp.float32)
    eps_arr = jnp.asarray(eps, dtype=x.dtype)
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps_arr)


def safe_cosine_similarity(x: jax.Array, y: jax.Array, eps: float) -> jax.Array:
    """Cosine similarity with explicit norm clamping for stability."""
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    eps_arr = jnp.asarray(eps, dtype=x.dtype)
    numerator = jnp.sum(x * y, axis=-1)
    x_norm = jnp.maximum(jnp.linalg.norm(x, axis=-1), eps_arr)
    y_norm = jnp.maximum(jnp.linalg.norm(y, axis=-1), eps_arr)
    sim = numerator / (x_norm * y_norm)
    return jnp.clip(sim, -1.0, 1.0)


def normalize_hidden(x: jax.Array, eps: float) -> jax.Array:
    """Layer-normalize then L2-normalize hidden states over the feature axis."""
    x = x.astype(jnp.float32)
    eps_arr = jnp.asarray(eps, dtype=x.dtype)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    variance = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    layer_normed = centered * jax.lax.rsqrt(variance + eps_arr)
    return _safe_l2_normalize(layer_normed, eps)


def decompose_pair(
    y1: jax.Array,
    y2: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Project each layer onto the counterpart direction, then take the residual."""
    yhat1 = normalize_hidden(y1, eps)
    yhat2 = normalize_hidden(y2, eps)
    a1 = jnp.sum(yhat1 * yhat2, axis=-1, keepdims=True) * yhat2
    a2 = jnp.sum(yhat2 * yhat1, axis=-1, keepdims=True) * yhat1
    b1 = yhat1 - a1
    b2 = yhat2 - a2
    return a1, a2, b1, b2


def compute_pair_losses(y1: jax.Array, y2: jax.Array, eps: float) -> dict[str, jax.Array]:
    """Compute shared, private, and cross-separation losses for one hidden-state pair."""
    a1, a2, b1, b2 = decompose_pair(y1, y2, eps)

    shared_cos = safe_cosine_similarity(a1, a2, eps)
    private_cos = safe_cosine_similarity(b1, b2, eps)
    # Cross-layer separation avoids the trivial zero loss that arises when
    # comparing a layer's shared component against its own residual.
    sep1_cos = safe_cosine_similarity(a1, b2, eps)
    sep2_cos = safe_cosine_similarity(a2, b1, eps)

    return {
        "loss_shared": jnp.mean(1.0 - shared_cos),
        "loss_private": jnp.mean(jnp.square(private_cos)),
        "loss_sep": 0.5 * (
            jnp.mean(jnp.square(sep1_cos)) +
            jnp.mean(jnp.square(sep2_cos))
        ),
    }


def zero_regularizer_metrics(dtype=jnp.float32) -> dict[str, jax.Array]:
    """Return zero-valued logging scalars for the disabled path."""
    zero = jnp.array(0.0, dtype=dtype)
    return {
        "loss_reg": zero,
        "loss_shared": zero,
        "loss_private": zero,
        "loss_sep": zero,
        "num_pairs_used": zero,
    }


def regularizer_is_active(
    global_step: jax.Array,
    *,
    enabled: bool,
    start_step: int,
    end_step: int | None,
    apply_every: int,
    num_pairs: int,
) -> jax.Array:
    """Return whether the activation regularizer should run on this step."""
    if not enabled or num_pairs <= 0:
        return jnp.asarray(False)

    step = jnp.asarray(global_step)
    active = step >= jnp.asarray(start_step, dtype=step.dtype)
    if end_step is not None:
        active = jnp.logical_and(active, step <= jnp.asarray(end_step, dtype=step.dtype))
    if apply_every > 1:
        delta = step - jnp.asarray(start_step, dtype=step.dtype)
        active = jnp.logical_and(active, jnp.equal(jnp.mod(delta, apply_every), 0))
    return active


def compute_regularizer_metrics(
    hidden_states: jax.Array | None,
    *,
    selected_layers: Sequence[int],
    layer_pairs: Sequence[tuple[int, int]],
    lambda_shared: float,
    lambda_private: float,
    lambda_sep: float,
    eps: float,
) -> dict[str, jax.Array]:
    """Aggregate pairwise activation losses across the configured layer pairs."""
    metrics = zero_regularizer_metrics()
    if hidden_states is None or not layer_pairs:
        return metrics

    layer_to_index = {
        int(layer): idx for idx, layer in enumerate(tuple(int(layer) for layer in selected_layers))
    }
    pair_losses = [
        compute_pair_losses(
            hidden_states[layer_to_index[layer_l]],
            hidden_states[layer_to_index[layer_k]],
            eps,
        )
        for layer_l, layer_k in layer_pairs
    ]

    shared = jnp.mean(jnp.stack([loss["loss_shared"] for loss in pair_losses], axis=0))
    private = jnp.mean(jnp.stack([loss["loss_private"] for loss in pair_losses], axis=0))
    sep = jnp.mean(jnp.stack([loss["loss_sep"] for loss in pair_losses], axis=0))
    loss_reg = (
        jnp.asarray(lambda_shared, dtype=jnp.float32) * shared +
        jnp.asarray(lambda_private, dtype=jnp.float32) * private +
        jnp.asarray(lambda_sep, dtype=jnp.float32) * sep
    )
    return {
        "loss_reg": loss_reg,
        "loss_shared": shared,
        "loss_private": private,
        "loss_sep": sep,
        "num_pairs_used": jnp.asarray(float(len(pair_losses)), dtype=jnp.float32),
    }
