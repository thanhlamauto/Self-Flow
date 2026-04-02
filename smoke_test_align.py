#!/usr/bin/env python3
"""
Lightweight smoke tests for wavelet-weighted alignment helpers.

These tests are CPU-friendly but still require JAX to be installed.
"""

import numpy as np
import jax.numpy as jnp

from src.align import (
    alpha_from_layer_ids,
    build_layer_token_weights,
    masked_cosine_loss,
    token_haar_detail_scores,
)


def test_token_haar_detail_scores_high_vs_flat():
    flat = np.zeros((1, 1, 16), dtype=np.float32)
    checker_patch = np.array(
        [
            [
                [[1.0] * 4, [-1.0] * 4],
                [[-1.0] * 4, [1.0] * 4],
            ]
        ],
        dtype=np.float32,
    )
    checker = checker_patch.reshape(1, 1, 16)
    tokens = jnp.asarray(np.concatenate([flat, checker], axis=1))
    scores = np.asarray(token_haar_detail_scores(tokens))
    assert scores.shape == (1, 2)
    assert np.isclose(scores[0, 0], 0.0)
    assert np.isclose(scores[0, 1], 1.0)


def test_layer_weights_mean_one():
    detail = jnp.asarray([[0.0, 0.5, 1.0], [0.2, 0.2, 0.2]], dtype=jnp.float32)
    layer_ids = jnp.asarray([1, 6, 12], dtype=jnp.int32)
    weights = np.asarray(
        build_layer_token_weights(detail, layer_ids, depth=12, alpha_max=0.6)
    )
    means = np.mean(weights, axis=-1)
    assert np.allclose(means, 1.0, atol=1e-5), means


def test_masked_cosine_loss_averages_layers():
    teacher = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float32)
    proj = jnp.asarray(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[0.0, 1.0], [1.0, 0.0]]],
        ],
        dtype=jnp.float32,
    )
    weights = jnp.ones((2, 1, 2), dtype=jnp.float32)
    loss, per_layer = masked_cosine_loss(proj, teacher, weights)
    per_layer = np.asarray(per_layer)
    assert np.allclose(per_layer, np.array([0.0, 1.0], dtype=np.float32))
    assert np.isclose(float(loss), 0.5)


def test_alpha_schedule_increases():
    alpha = np.asarray(alpha_from_layer_ids(jnp.asarray([1, 6, 12]), depth=12, alpha_max=0.6))
    assert alpha[0] <= alpha[1] <= alpha[2]
    assert np.isclose(alpha[-1], 0.6, atol=1e-6)


if __name__ == "__main__":
    test_token_haar_detail_scores_high_vs_flat()
    test_layer_weights_mean_one()
    test_masked_cosine_loss_averages_layers()
    test_alpha_schedule_increases()
    print("OK")
