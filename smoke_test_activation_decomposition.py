#!/usr/bin/env python3
"""Smoke tests for shared-subspace activation decomposition helpers."""

import numpy as np
import jax
import jax.numpy as jnp

from src.activation_decomposition import project_onto_basis, shared_subspace_basis


def _primal_svd_basis(mean_activations: jax.Array, rank: int) -> tuple[jax.Array, int]:
    effective_rank = min(
        max(int(rank), 1),
        mean_activations.shape[-2],
        mean_activations.shape[-1],
    )
    _, _, vh = jnp.linalg.svd(mean_activations.astype(jnp.float32), full_matrices=False)
    basis = jnp.swapaxes(vh[:, :effective_rank, :], 1, 2).astype(mean_activations.dtype)
    return basis, effective_rank


def _projector_from_basis(basis: jax.Array) -> jax.Array:
    basis_gram = jnp.einsum("bdk,bdj->bkj", basis, basis).astype(jnp.float32)
    basis_gram_pinv = jnp.linalg.pinv(basis_gram, hermitian=True).astype(basis.dtype)
    return jnp.einsum("bdk,bkj,bej->bde", basis, basis_gram_pinv, basis)


def test_dual_gram_projector_matches_primal_svd():
    mean_rng, act_rng = jax.random.split(jax.random.PRNGKey(0))
    mean_activations = jax.random.normal(mean_rng, (3, 5, 8), dtype=jnp.float32)
    activations = jax.random.normal(act_rng, (4, 3, 5, 8), dtype=jnp.float32)

    dual_basis, effective_rank = shared_subspace_basis(mean_activations, rank=3)
    primal_basis, primal_rank = _primal_svd_basis(mean_activations, rank=3)

    assert effective_rank == primal_rank

    dual_projector = np.asarray(jax.device_get(_projector_from_basis(dual_basis)))
    primal_projector = np.asarray(jax.device_get(_projector_from_basis(primal_basis)))
    assert np.allclose(dual_projector, primal_projector, atol=2e-4, rtol=2e-4), (
        dual_projector,
        primal_projector,
    )

    dual_projection = np.asarray(jax.device_get(project_onto_basis(activations, dual_basis)))
    primal_projection = np.asarray(jax.device_get(project_onto_basis(activations, primal_basis)))
    assert np.allclose(dual_projection, primal_projection, atol=2e-4, rtol=2e-4), (
        dual_projection,
        primal_projection,
    )


if __name__ == "__main__":
    test_dual_gram_projector_matches_primal_svd()
    print("OK")
