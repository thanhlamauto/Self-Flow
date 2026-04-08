"""Helper utilities for common/private activation decomposition in DiT."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def collect_activations(activations: Any) -> jax.Array:
    """Normalize activations into a stacked `[L, B, N, D]` tensor."""
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


def compute_common_private(activations: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute differentiable common activation and private residuals."""
    activations = collect_activations(activations)
    common = jnp.mean(activations, axis=0)
    common_anchor = jax.lax.stop_gradient(common)
    private = activations - common_anchor[None, ...]
    return common, common_anchor, private


def gram_matrix(x: jax.Array) -> jax.Array:
    """Compute batched Gram matrices from `[B, N, D]` to `[B, N, N]`."""
    if x.ndim != 3:
        raise ValueError(f"Expected input with rank 3, got shape {x.shape}")
    return jnp.einsum("bnd,bmd->bnm", x, x)


def _mean_pairwise_cosine_squared(private: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Average squared cosine similarity over layer pairs."""
    num_layers = private.shape[0]
    if num_layers < 2:
        return jnp.array(0.0, dtype=private.dtype)

    flattened = private.reshape(num_layers, -1)
    norms = jnp.linalg.norm(flattened, axis=-1, keepdims=True)
    normalized = flattened / jnp.maximum(norms, eps)
    cosine_matrix = normalized @ normalized.T
    upper_indices = jnp.triu_indices(num_layers, k=1)
    pairwise_cosines = cosine_matrix[upper_indices]
    return jnp.mean(jnp.square(pairwise_cosines))


def compute_aux_losses(
    activations: Any,
    spatial_target: jax.Array,
) -> dict[str, jax.Array]:
    """Compute auxiliary losses and logging metrics for activation decomposition."""
    activations = collect_activations(activations)
    if spatial_target.ndim != 3:
        raise ValueError(f"Expected spatial target with rank 3, got shape {spatial_target.shape}")

    common, common_anchor, private = compute_common_private(activations)
    common_loss = jnp.mean(jnp.square(activations - common_anchor[None, ...]))

    gram_common = gram_matrix(common)
    gram_target = gram_matrix(spatial_target)
    spatial_loss = jnp.mean(jnp.square(gram_common - gram_target))

    private_loss = _mean_pairwise_cosine_squared(private)

    common_norm = jnp.mean(jnp.linalg.norm(common.reshape(common.shape[0], -1), axis=-1))
    private_norms = jnp.linalg.norm(private.reshape(private.shape[0], private.shape[1], -1), axis=-1)
    avg_private_norm = jnp.mean(private_norms)

    num_layers = private.shape[0]
    flattened_private = private.reshape(num_layers, -1)
    norms = jnp.linalg.norm(flattened_private, axis=-1, keepdims=True)
    normalized_private = flattened_private / jnp.maximum(norms, 1e-8)
    cosine_matrix = normalized_private @ normalized_private.T
    upper_indices = jnp.triu_indices(num_layers, k=1)
    if num_layers < 2:
        avg_pairwise_cosine = jnp.array(0.0, dtype=private.dtype)
    else:
        avg_pairwise_cosine = jnp.mean(cosine_matrix[upper_indices])

    return {
        "common_activation": common,
        "private_activations": private,
        "loss_common": common_loss,
        "loss_spatial": spatial_loss,
        "loss_private": private_loss,
        "norm_common": common_norm,
        "avg_private_norm": avg_private_norm,
        "avg_pairwise_private_cosine": avg_pairwise_cosine,
    }
