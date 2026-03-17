#!/usr/bin/env python3
"""
Lightweight smoke tests for metric helpers (CPU-friendly).

These do NOT require a TPU. They aim to catch logic regressions:
  - masked pad samples do not bias Gaussian sums
  - shared extractor trims padded tail samples
  - PR kNN chunking runs end-to-end
  - reservoir sampling is deterministic and not prefix-only
  - Pearson correlation helper is symmetric with unit diagonal
  - eval chunk RNG derivation changes across chunks
"""

import numpy as np
import jax
import jax.numpy as jnp

from src.metrics import (
    ReservoirSampler,
    extract_inception_features_host_images,
    gaussian_batch_sums_pmap,
    gaussian_sums_add,
    finalize_gaussian_sums,
    init_gaussian_sums,
    make_eval_chunk_rngs,
    pearson_corrcoef_rows,
    precision_recall_knn,
)


def test_gaussian_masking_count():
    devices = jax.device_count()
    local_b = 4
    dim = 8
    feats = jnp.ones((devices, local_b, dim), dtype=jnp.float32)
    # Mask out last 2 samples on each device
    valid = jnp.array([1, 1, 0, 0], dtype=jnp.bool_)
    valid_mask = jnp.broadcast_to(valid, (devices, local_b))
    cnt, s, sxx = gaussian_batch_sums_pmap(feats, valid_mask)
    cnt0 = float(jax.device_get(cnt[0]))
    assert cnt0 == devices * 2.0, (devices, cnt0)

    acc = init_gaussian_sums(dim)
    acc = gaussian_sums_add(acc, cnt[0], s[0], sxx[0])
    mu, cov, n = finalize_gaussian_sums(acc)
    assert n == int(devices * 2), (n, devices)
    assert np.allclose(mu, 1.0), mu


def test_precision_recall_runs():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(256, 32)).astype(np.float32)
    fake = rng.normal(size=(256, 32)).astype(np.float32)
    p, r = precision_recall_knn(real, fake, k=3, chunk=64)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0


def test_shared_extractor_trims_pad():
    devices = jax.device_count()
    local_b = 3
    n = devices * local_b + 1
    imgs = np.zeros((n, 4, 4, 3), dtype=np.float32)
    for i in range(n):
        imgs[i, ...] = i / 10.0

    def fake_inception(inp):
        return inp[:, :, :1, :1, :1]

    acts = extract_inception_features_host_images(
        imgs,
        fake_inception,
        num_devices=devices,
        local_batch=local_b,
        mode="pooled",
    )
    expected = np.array([i / 10.0 * 2.0 - 1.0 for i in range(n)], dtype=np.float32)
    assert acts.shape == (n, 1), acts.shape
    assert np.allclose(acts[:, 0], expected), (acts[:, 0], expected)


def test_reservoir_sampler_deterministic():
    data = np.arange(400, dtype=np.float32).reshape(100, 4)
    s1 = ReservoirSampler(10, seed=123)
    s2 = ReservoirSampler(10, seed=123)
    for start in range(0, len(data), 7):
        batch = data[start:start + 7]
        s1.add(batch)
        s2.add(batch)
    out1 = s1.get()
    out2 = s2.get()
    assert np.array_equal(out1, out2)
    assert out1.shape == (10, 4)
    assert not np.array_equal(out1, data[:10]), out1


def test_pearson_corr_rows():
    x = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    corr = pearson_corrcoef_rows(x)
    assert np.allclose(corr, corr.T), corr
    assert np.allclose(np.diag(corr), 1.0), corr
    assert corr[0, 1] > 0.999, corr
    assert corr[0, 2] < -0.999, corr


def test_eval_chunk_rngs_are_unique():
    base = jax.random.split(jax.random.PRNGKey(0), jax.device_count())
    c0, s0 = make_eval_chunk_rngs(base, 0)
    c1, s1 = make_eval_chunk_rngs(base, 1)
    assert not np.array_equal(np.asarray(jax.device_get(c0)), np.asarray(jax.device_get(c1)))
    assert not np.array_equal(np.asarray(jax.device_get(s0)), np.asarray(jax.device_get(s1)))


if __name__ == "__main__":
    test_gaussian_masking_count()
    test_precision_recall_runs()
    test_shared_extractor_trims_pad()
    test_reservoir_sampler_deterministic()
    test_pearson_corr_rows()
    test_eval_chunk_rngs_are_unique()
    print("OK")
