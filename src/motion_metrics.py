"""Feature metrics for motion distributions."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


Array = jax.Array


def masked_motion_mean(motion: Array, mask: Optional[Array] = None) -> Array:
    """Simple deterministic motion embedding by masked temporal averaging."""
    x = jnp.asarray(motion, dtype=jnp.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected motion [B,J,F,T], got shape {x.shape}")
    x = jnp.transpose(x, (0, 3, 1, 2)).reshape(x.shape[0], x.shape[-1], -1)
    if mask is None:
        return jnp.mean(x, axis=1)
    m = jnp.asarray(mask)
    if m.ndim == 4:
        m = m.reshape(x.shape[0], -1)[:, : x.shape[1]]
    elif m.ndim != 2:
        raise ValueError(f"Expected mask rank 2 or 4, got shape {m.shape}")
    m = m.astype(x.dtype)
    denom = jnp.maximum(jnp.sum(m, axis=1, keepdims=True), 1.0)
    return jnp.sum(x * m[:, :, None], axis=1) / denom


def activation_statistics(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    acts = np.asarray(activations, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"Expected activations [N,D], got shape {acts.shape}")
    if acts.shape[0] < 2:
        raise ValueError("At least two activations are required for covariance")
    return np.mean(acts, axis=0), np.cov(acts, rowvar=False)


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Stable Frechet distance between two Gaussian feature distributions."""
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))
    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean shapes differ: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariance shapes differ: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0], dtype=np.float64) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0.0, atol=1e-3):
            raise ValueError(f"Frechet distance has imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def motion_fid_from_activations(real_activations: np.ndarray, fake_activations: np.ndarray) -> float:
    real_stats = activation_statistics(real_activations)
    fake_stats = activation_statistics(fake_activations)
    return frechet_distance(real_stats[0], real_stats[1], fake_stats[0], fake_stats[1])


def extract_motion_activations(
    loader,
    embed_fn: Callable[[Array, Optional[Array]], Array] = masked_motion_mean,
) -> np.ndarray:
    chunks = []
    for batch in loader:
        feats = embed_fn(batch["motion"], batch.get("mask"))
        chunks.append(np.asarray(jax.device_get(feats), dtype=np.float32))
    if not chunks:
        raise ValueError("No batches were produced by the loader")
    return np.concatenate(chunks, axis=0)
