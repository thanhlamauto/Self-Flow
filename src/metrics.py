from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import functools
import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass(frozen=True)
class GaussianSums:
    """Streaming Gaussian sufficient statistics.

    Stores:
      - count: scalar float32
      - sum:   [D] float32
      - sum_xxt: [D, D] float32  (sum of outer-products)
    """

    count: Array
    sum: Array
    sum_xxt: Array


class ReservoirSampler:
    """Deterministic reservoir sampler for host-side monitoring subsets."""

    def __init__(self, max_items: int, *, seed: int = 0):
        self.max_items = int(max_items)
        self.num_seen = 0
        self._rng = np.random.default_rng(seed)
        self._items: Optional[np.ndarray] = None

    def add(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float32)
        if self.max_items <= 0 or batch.size == 0:
            return
        if batch.ndim != 2:
            raise ValueError(f"ReservoirSampler expects 2D batch, got shape {batch.shape}")
        if self._items is None:
            self._items = np.empty((self.max_items, batch.shape[1]), dtype=batch.dtype)
        for row in batch:
            if self.num_seen < self.max_items:
                self._items[self.num_seen] = row
            else:
                idx = int(self._rng.integers(0, self.num_seen + 1))
                if idx < self.max_items:
                    self._items[idx] = row
            self.num_seen += 1

    def get(self) -> Optional[np.ndarray]:
        if self._items is None:
            return None
        kept = min(self.num_seen, self.max_items)
        return self._items[:kept].copy()


def init_gaussian_sums(dim: int, *, dtype=jnp.float32) -> GaussianSums:
    return GaussianSums(
        count=jnp.array(0.0, dtype=dtype),
        sum=jnp.zeros((dim,), dtype=dtype),
        sum_xxt=jnp.zeros((dim, dim), dtype=dtype),
    )


def _masked_flatten_feats(feats: Array, valid_mask: Array) -> Array:
    """Apply mask and flatten leading dimensions to (N, D)."""
    valid_f = valid_mask.astype(feats.dtype)
    x = feats * valid_f[..., None]
    return x.reshape((-1, x.shape[-1]))


@functools.partial(jax.pmap, axis_name="batch")
def gaussian_batch_sums_pmap(feats: Array, valid_mask: Array) -> Tuple[Array, Array, Array]:
    """Compute global (across devices) sums for a batch of features.

    Args:
      feats: (local_batch, D) per device
      valid_mask: (local_batch,) per device boolean (False for pad)

    Returns (replicated on each device):
      count: scalar float32
      sum:   [D] float32
      sum_xxt: [D, D] float32
    """
    x = _masked_flatten_feats(feats, valid_mask)
    count = jnp.sum(valid_mask.astype(jnp.float32))
    s = jnp.sum(x, axis=0)
    sxx = jnp.einsum("nd,ne->de", x, x)
    count = jax.lax.psum(count, axis_name="batch")
    s = jax.lax.psum(s, axis_name="batch")
    sxx = jax.lax.psum(sxx, axis_name="batch")
    return count, s, sxx


@functools.partial(jax.pmap, axis_name="batch")
def gaussian_spatial_batch_sums_pmap(
    spatial_feats: Array,
    valid_mask: Array,
) -> Tuple[Array, Array, Array]:
    """Compute global sums for spatial features.

    Args:
      spatial_feats: (local_batch, H, W, D) per device
      valid_mask: (local_batch,) per device boolean

    Treats each spatial location as an independent sample, i.e. reshapes to
    (local_batch*H*W, D) and repeats valid_mask across H*W.
    """
    lb, h, w, d = spatial_feats.shape
    x = spatial_feats.reshape((lb * h * w, d))
    valid = jnp.repeat(valid_mask, h * w)
    valid_f = valid.astype(x.dtype)
    x = x * valid_f[:, None]
    count = jnp.sum(valid.astype(jnp.float32))
    s = jnp.sum(x, axis=0)
    sxx = jnp.einsum("nd,ne->de", x, x)
    count = jax.lax.psum(count, axis_name="batch")
    s = jax.lax.psum(s, axis_name="batch")
    sxx = jax.lax.psum(sxx, axis_name="batch")
    return count, s, sxx


def gaussian_sums_add(acc: GaussianSums, batch_count: Array, batch_sum: Array, batch_sum_xxt: Array) -> GaussianSums:
    return GaussianSums(
        count=acc.count + batch_count,
        sum=acc.sum + batch_sum,
        sum_xxt=acc.sum_xxt + batch_sum_xxt,
    )


def finalize_gaussian_sums(acc: GaussianSums) -> Tuple[np.ndarray, np.ndarray, int]:
    """Finalize to (mu, cov, count_int) on host."""
    count = float(jax.device_get(acc.count))
    if count <= 0:
        raise ValueError("Cannot finalize Gaussian stats with count=0")
    s = np.asarray(jax.device_get(acc.sum), dtype=np.float64)
    sxx = np.asarray(jax.device_get(acc.sum_xxt), dtype=np.float64)
    mu = s / count
    cov = sxx / count - np.outer(mu, mu)
    cov = (cov + cov.T) / 2.0  # symmetrize
    return mu.astype(np.float64), cov.astype(np.float64), int(round(count))


def inception_preprocess_batched(imgs: Array) -> Array:
    """Resize+normalize images for JAX Inception FID network.

    Args:
      imgs: (devices, local_batch, H, W, 3) float32 in [0,1]
    Returns:
      (devices, local_batch, 299, 299, 3) float32 in [-1,1]
    """
    target_shape = (imgs.shape[0], imgs.shape[1], 299, 299, imgs.shape[-1])
    imgs_299 = jax.image.resize(imgs.astype(jnp.float32), target_shape, method="bilinear")
    return imgs_299 * 2.0 - 1.0


def global_valid_mask(num_devices: int, local_batch: int, valid_global: int) -> Array:
    """Boolean mask shaped (devices, local_batch) for a padded global batch."""
    total = int(num_devices) * int(local_batch)
    valid = int(valid_global)
    if valid < 0 or valid > total:
        raise ValueError(f"valid_global must be in [0, {total}], got {valid}")
    return (jnp.arange(total) < jnp.int32(valid)).reshape(int(num_devices), int(local_batch))


def make_valid_mask(local_batch: int, valid_local: int) -> Array:
    """(local_batch,) bool mask for a per-device batch with pad tail."""
    return (jnp.arange(local_batch) < jnp.int32(valid_local))


def trim_sharded_batch_to_host(arr: Array, valid_global: int) -> np.ndarray:
    """Flatten leading (devices, local_batch) dims and trim padded tail on host."""
    host = np.asarray(jax.device_get(arr))
    flat = host.reshape((host.shape[0] * host.shape[1],) + host.shape[2:])
    return flat[: int(valid_global)]


def apply_inception_to_decoded_sharded(
    imgs_sharded: Array,
    inception_fn,
    *,
    mode: str = "pooled",
    valid_global: Optional[int] = None,
) -> Tuple[Array, Optional[Array], Array]:
    """Run preprocess + Inception on a decoded sharded image batch.

    Args:
      imgs_sharded: (devices, local_batch, H, W, 3) float32 in [0,1]
      inception_fn: callable from get_inception_network()
      mode: "pooled" or "pooled+spatial"
      valid_global: number of valid samples before padding; defaults to full batch

    Returns:
      pooled: (devices, local_batch, 1, 1, D)
      spatial: (devices, local_batch, Hs, Ws, D) or None
      valid_mask: (devices, local_batch) bool
    """
    num_devices, local_batch = imgs_sharded.shape[:2]
    if valid_global is None:
        valid_global = num_devices * local_batch
    inp = inception_preprocess_batched(imgs_sharded)
    out = inception_fn(inp)
    if mode == "pooled":
        pooled = out
        spatial = None
    elif mode == "pooled+spatial":
        pooled, spatial = out
    else:
        raise ValueError(f"Unsupported mode {mode!r}")
    valid_mask = global_valid_mask(num_devices, local_batch, int(valid_global))
    return pooled, spatial, valid_mask


def extract_inception_features_host_images(
    imgs_nhwc: np.ndarray,
    inception_fn,
    *,
    num_devices: int,
    local_batch: int,
    mode: str = "pooled",
) -> Tuple[np.ndarray, Optional[np.ndarray]] | np.ndarray:
    """Pad/shard host images, run Inception, and trim padded outputs."""
    imgs = np.asarray(imgs_nhwc, dtype=np.float32)
    if imgs.ndim != 4:
        raise ValueError(f"Expected NHWC images, got shape {imgs.shape}")
    if imgs.shape[0] == 0:
        empty = np.empty((0, 2048), dtype=np.float32)
        return empty if mode == "pooled" else (empty, np.empty((0, 0, 0, 2048), dtype=np.float32))

    global_batch = int(num_devices) * int(local_batch)
    pooled_all = []
    spatial_all = [] if mode == "pooled+spatial" else None
    for start in range(0, imgs.shape[0], global_batch):
        chunk = imgs[start:start + global_batch]
        valid = int(chunk.shape[0])
        if valid < global_batch:
            pad_shape = (global_batch - valid,) + imgs.shape[1:]
            chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=np.float32)], axis=0)
        imgs_sharded = jnp.array(
            chunk.reshape(int(num_devices), int(local_batch), *chunk.shape[1:]),
            dtype=jnp.float32,
        )
        pooled, spatial, _ = apply_inception_to_decoded_sharded(
            imgs_sharded,
            inception_fn,
            mode=mode,
            valid_global=valid,
        )
        pooled_all.append(trim_sharded_batch_to_host(pooled, valid).reshape(valid, -1))
        if mode == "pooled+spatial":
            spatial_all.append(trim_sharded_batch_to_host(spatial, valid))

    pooled_host = np.concatenate(pooled_all, axis=0)
    if mode == "pooled":
        return pooled_host
    return pooled_host, np.concatenate(spatial_all, axis=0)


def make_eval_chunk_rngs(eval_rng: Array, chunk_idx: int) -> Tuple[Array, Array]:
    """Deterministically derive distinct class/sample RNGs for each eval chunk."""
    idx = int(chunk_idx)
    fold_in_batched = jax.vmap(lambda key, data: jax.random.fold_in(key, data), in_axes=(0, None))
    class_rng = fold_in_batched(eval_rng, jnp.uint32(idx * 2))
    sample_rng = fold_in_batched(eval_rng, jnp.uint32(idx * 2 + 1))
    return class_rng, sample_rng


def pearson_corrcoef_rows(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Pearson correlation matrix between rows of x."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {arr.shape}")
    centered = arr - np.mean(arr, axis=1, keepdims=True)
    denom = np.linalg.norm(centered, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    corr = (centered @ centered.T) / (denom @ denom.T)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float64)


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


def knn_radii(x: np.ndarray, k: int, *, chunk: int = 512) -> np.ndarray:
    """Return per-point radius to k-th nearest neighbor within x (excluding self)."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    radii2 = np.empty((n,), dtype=np.float32)
    k_eff = int(k)
    if k_eff <= 0:
        raise ValueError("k must be > 0")
    for i in range(0, n, chunk):
        xi = x[i:i + chunk]
        m = xi.shape[0]
        best = np.full((m, k_eff + 1), np.inf, dtype=np.float32)
        for j in range(0, n, chunk):
            xj = x[j:j + chunk]
            d2 = _pairwise_sq_dists(xi, xj)
            if i == j:
                # exclude self matches
                diag = np.arange(m)
                d2[diag, diag] = np.inf
            # take k+1 smallest in this block
            blk = np.partition(d2, kth=min(k_eff, d2.shape[1] - 1), axis=1)[:, : k_eff + 1]
            merged = np.concatenate([best, blk], axis=1)
            best = np.partition(merged, kth=k_eff, axis=1)[:, : k_eff + 1]
        radii2[i:i + m] = best[:, k_eff]
    return np.sqrt(np.maximum(radii2, 0.0))


def _nearest_with_index(a: np.ndarray, b: np.ndarray, *, chunk: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """For each row in a, find nearest neighbor in b. Returns (min_dist, argmin)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = a.shape[0]
    min_d2 = np.full((n,), np.inf, dtype=np.float32)
    argmin = np.full((n,), -1, dtype=np.int32)
    for i in range(0, n, chunk):
        ai = a[i:i + chunk]
        d2_min = np.full((ai.shape[0],), np.inf, dtype=np.float32)
        idx_min = np.full((ai.shape[0],), -1, dtype=np.int32)
        for j in range(0, b.shape[0], chunk):
            bj = b[j:j + chunk]
            d2 = _pairwise_sq_dists(ai, bj)
            idx = np.argmin(d2, axis=1).astype(np.int32)
            val = d2[np.arange(d2.shape[0]), idx]
            better = val < d2_min
            d2_min[better] = val[better]
            idx_min[better] = idx[better] + j
        min_d2[i:i + ai.shape[0]] = d2_min
        argmin[i:i + ai.shape[0]] = idx_min
    return np.sqrt(np.maximum(min_d2, 0.0)), argmin


def precision_recall_knn(
    real: np.ndarray,
    fake: np.ndarray,
    *,
    k: int = 3,
    chunk: int = 512,
) -> Tuple[float, float]:
    """kNN-manifold precision/recall on feature distributions."""
    real = np.asarray(real, dtype=np.float32)
    fake = np.asarray(fake, dtype=np.float32)
    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("real/fake must be 2D arrays")
    if real.shape[0] < (k + 1) or fake.shape[0] < (k + 1):
        raise ValueError("Not enough samples for kNN radii (need >= k+2)")

    r_real = knn_radii(real, k, chunk=chunk)
    r_fake = knn_radii(fake, k, chunk=chunk)

    d_fr, idx_fr = _nearest_with_index(fake, real, chunk=chunk)
    d_rf, idx_rf = _nearest_with_index(real, fake, chunk=chunk)

    precision = float(np.mean(d_fr <= r_real[idx_fr]))
    recall = float(np.mean(d_rf <= r_fake[idx_rf]))
    return precision, recall
