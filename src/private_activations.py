"""Helpers for direct private-loss regularization on DiT layer outputs."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def collect_activations(activations: Any) -> jax.Array:
    """Normalize activations into a stacked ``[L, B, N, D]`` tensor."""
    if hasattr(activations, "ndim"):
        if activations.ndim != 4:
            raise ValueError(f"Expected activations with rank 4, got shape {activations.shape}")
        return activations
    if isinstance(activations, (list, tuple)):
        if not activations:
            raise ValueError("Expected at least one activation tensor.")
        stacked = jnp.stack(activations, axis=0)
        if stacked.ndim != 4:
            raise ValueError(f"Expected stacked activations with rank 4, got shape {stacked.shape}")
        return stacked
    raise TypeError(f"Unsupported activation container type: {type(activations)!r}")


def _normalize_channels(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def _pairwise_cosines(
    activations: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Return cosine similarities for all layer pairs ``i < j``."""
    num_layers = activations.shape[0]
    if num_layers < 2:
        return jnp.array(0.0, dtype=activations.dtype)

    normalized = _normalize_channels(activations, eps=eps)
    flattened = normalized.reshape(num_layers, -1)
    layer_norms = jnp.linalg.norm(flattened, axis=-1, keepdims=True)
    flattened = flattened / jnp.maximum(layer_norms, eps)
    cosine_matrix = flattened @ flattened.T
    upper_indices = jnp.triu_indices(num_layers, k=1)
    return cosine_matrix[upper_indices]


def _mean_pairwise_cosine_squared(
    activations: jax.Array,
    eps: float = 1e-8,
    rng: jax.Array | None = None,
    max_pairs: int = 0,
) -> jax.Array:
    """Average squared cosine similarity over all or sampled layer pairs."""
    pairwise_cosines = _pairwise_cosines(activations, eps=eps)
    if pairwise_cosines.ndim == 0:
        return pairwise_cosines

    if max_pairs and max_pairs > 0 and pairwise_cosines.shape[0] > max_pairs:
        if rng is None:
            raise ValueError("An RNG key is required when sampling private-layer pairs.")
        indices = jax.random.permutation(rng, pairwise_cosines.shape[0])[:max_pairs]
        pairwise_cosines = pairwise_cosines[indices]

    return jnp.mean(jnp.square(pairwise_cosines))


def compute_direct_private_loss_metrics(
    activations: Any,
    *,
    private_pair_rng: jax.Array | None = None,
    private_max_pairs: int = 0,
) -> dict[str, jax.Array]:
    """Compute private diversity metrics directly on post-block hidden states."""
    activations = collect_activations(activations)

    private_loss = _mean_pairwise_cosine_squared(
        activations,
        rng=private_pair_rng,
        max_pairs=private_max_pairs,
    )

    activation_norms = jnp.linalg.norm(
        activations.reshape(activations.shape[0], activations.shape[1], -1),
        axis=-1,
    )
    avg_activation_norm = jnp.mean(activation_norms)

    pairwise_cosines = _pairwise_cosines(activations)
    if pairwise_cosines.ndim == 0:
        avg_pairwise_cosine = pairwise_cosines
    else:
        avg_pairwise_cosine = jnp.mean(pairwise_cosines)

    return {
        "loss_private": private_loss,
        "avg_activation_norm": avg_activation_norm,
        "avg_pairwise_activation_cosine": avg_pairwise_cosine,
    }
