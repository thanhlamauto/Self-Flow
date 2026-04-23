"""Common/private activation decomposition losses."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _layer_logit_normal_weights(
    num_layers: int,
    center_layer: float | jax.Array,
    logit_sigma: float | jax.Array,
    dtype: jnp.dtype,
) -> jax.Array:
    """Nonnegative layer weights from a Gaussian over logit-depth."""
    indices = jnp.arange(num_layers, dtype=dtype)
    denom_layers = jnp.maximum(num_layers, 1)
    u = (indices + jnp.asarray(0.5, dtype=dtype)) / denom_layers
    eps = jnp.asarray(1e-6, dtype=dtype)
    u = jnp.clip(u, eps, jnp.asarray(1.0, dtype=dtype) - eps)
    z = jnp.log(u / (jnp.asarray(1.0, dtype=dtype) - u))

    center = jnp.asarray(center_layer, dtype=dtype)
    center = jnp.clip(
        center,
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(max(num_layers - 1, 0), dtype=dtype),
    )
    u_center = (center + jnp.asarray(0.5, dtype=dtype)) / denom_layers
    u_center = jnp.clip(u_center, eps, jnp.asarray(1.0, dtype=dtype) - eps)
    mu = jnp.log(u_center / (jnp.asarray(1.0, dtype=dtype) - u_center))

    sigma = jnp.maximum(jnp.asarray(logit_sigma, dtype=dtype), eps)
    log_weights = -0.5 * jnp.square((z - mu) / sigma)
    weights = jnp.exp(log_weights - jnp.max(log_weights))
    return weights / jnp.maximum(jnp.sum(weights), eps)


def collect_activations(activations: Any) -> jax.Array:
    """Normalize layer activations into a stacked [L, B, N, D] tensor."""
    if hasattr(activations, "ndim"):
        if activations.ndim != 4:
            raise ValueError(f"Expected rank-4 activations, got shape {activations.shape}")
        return activations
    if isinstance(activations, (list, tuple)):
        if not activations:
            raise ValueError("Expected at least one activation tensor.")
        stacked = jnp.stack(activations, axis=0)
        if stacked.ndim != 4:
            raise ValueError(f"Expected stacked rank-4 activations, got shape {stacked.shape}")
        return stacked
    raise TypeError(f"Unsupported activation container type: {type(activations)!r}")


def _normalize_channels(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def compute_common_private(
    activations: Any,
    *,
    agg: str = "mean",
    logit_normal_center_layer: float = 0.0,
    logit_normal_sigma: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Compute A_common and private residuals B_i = A_i - stop_gradient(A_common)."""
    activations = _normalize_channels(collect_activations(activations))
    num_layers = activations.shape[0]
    if agg == "mean":
        common = jnp.mean(activations, axis=0)
    elif agg == "logit_normal":
        weights = _layer_logit_normal_weights(
            num_layers,
            logit_normal_center_layer,
            logit_normal_sigma,
            activations.dtype,
        )
        common = jnp.tensordot(weights, activations, axes=(0, 0))
    else:
        raise ValueError(f"Unknown common aggregation {agg!r}; expected 'mean' or 'logit_normal'.")
    private = activations - jax.lax.stop_gradient(common)[None, ...]
    return common, private


def _pairwise_cosines(private: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Return cosine similarities for all layer pairs i < j."""
    num_layers = private.shape[0]
    if num_layers < 2:
        return jnp.array(0.0, dtype=private.dtype)

    flattened = private.reshape(num_layers, -1)
    normalized = flattened / jnp.maximum(
        jnp.linalg.norm(flattened, axis=-1, keepdims=True),
        eps,
    )
    cosine_matrix = normalized @ normalized.T
    upper_indices = jnp.triu_indices(num_layers, k=1)
    return cosine_matrix[upper_indices]


def _mean_pairwise_cosine_squared(
    private: jax.Array,
    *,
    rng: jax.Array | None = None,
    max_pairs: int = 0,
) -> tuple[jax.Array, jax.Array]:
    pairwise_cosines = _pairwise_cosines(private)
    if pairwise_cosines.ndim == 0:
        return pairwise_cosines, pairwise_cosines

    selected_cosines = pairwise_cosines
    if max_pairs and max_pairs > 0 and pairwise_cosines.shape[0] > max_pairs:
        if rng is None:
            raise ValueError("An RNG key is required when sampling private-layer pairs.")
        indices = jax.random.permutation(rng, pairwise_cosines.shape[0])[:max_pairs]
        selected_cosines = pairwise_cosines[indices]

    return jnp.mean(jnp.square(selected_cosines)), jnp.mean(pairwise_cosines)


def compute_private_activation_loss(
    activations: Any,
    *,
    rng: jax.Array | None = None,
    max_pairs: int = 0,
    common_agg: str = "mean",
    common_logit_normal_center_layer: float = 0.0,
    common_logit_normal_sigma: float = 1.0,
) -> dict[str, jax.Array]:
    """Compute the private diversity loss and diagnostics from block activations."""
    common, private = compute_common_private(
        activations,
        agg=common_agg,
        logit_normal_center_layer=common_logit_normal_center_layer,
        logit_normal_sigma=common_logit_normal_sigma,
    )
    loss_private, avg_pairwise_cosine = _mean_pairwise_cosine_squared(
        private,
        rng=rng,
        max_pairs=max_pairs,
    )
    common_norm = jnp.mean(jnp.linalg.norm(common.reshape(common.shape[0], -1), axis=-1))
    private_norms = jnp.linalg.norm(private.reshape(private.shape[0], private.shape[1], -1), axis=-1)
    return {
        "loss_private": loss_private,
        "norm_common": common_norm,
        "avg_private_norm": jnp.mean(private_norms),
        "avg_pairwise_private_cosine": avg_pairwise_cosine,
    }
