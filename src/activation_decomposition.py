"""Helper utilities for common/private activation decomposition in DiT."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

DEFAULT_SPATIAL_WINDOW_SIZE = 3
DEFAULT_SPATIAL_WINDOW_STRIDE = 1
DEFAULT_MAX_TIMESTEP_BLUR_SIGMA = 3.0


def _layer_logit_normal_weights(
    num_layers: int,
    center_layer: float | jax.Array,
    logit_sigma: float | jax.Array,
    dtype: jnp.dtype,
    weight_kind: str = "rbf",
) -> jax.Array:
    """Depth weights for ``A_common = sum_i w_i A_i`` with ``sum_i w_i = 1``.

    For each layer index ``i`` we use a depth coordinate ``u_i = (i + 0.5) / L`` in ``(0,1)`` and
    ``z_i = logit(u_i)``. The **center** is ``u_c = (k + 0.5) / L`` for ``center_layer`` ``k``
    (clipped to ``[0, L-1]``), with ``mu = logit(u_c)``. So the profile is **centered at layer k**
    on this depth axis (not “``w_k`` is the center” as a random variable — ``w`` is fixed each
    step, not sampled i.i.d. from a distribution).

    **weight_kind**

    - ``"rbf"`` (default): ``w_i ∝ exp(-0.5 * ((z_i - mu) / sigma)^2)``, then normalized. Same
      convex-combination scaling as the mean (``sum w_i = 1``); ``sigma → ∞`` → uniform → mean.
    - ``"pdf"``: ``w_i`` proportional to the **logit-normal** density at ``u_i``,
      i.e. ``∝ phi((z_i - mu) / sigma) / (u_i (1 - u_i))`` with ``phi`` the standard normal PDF,
      then normalized. This is the strict continuous definition evaluated on the discrete ``u_i``
      lattice; boundary layers get extra mass from the Jacobian (often less stable in training).
    """
    L = num_layers
    i = jnp.arange(L, dtype=dtype)
    u = (i + 0.5) / jnp.maximum(L, 1)
    eps = jnp.asarray(1e-6, dtype=dtype)
    u = jnp.clip(u, eps, jnp.asarray(1.0, dtype=dtype) - eps)
    z = jnp.log(u / (jnp.asarray(1.0, dtype=dtype) - u))

    k = jnp.asarray(center_layer, dtype=dtype)
    k = jnp.clip(k, jnp.asarray(0.0, dtype=dtype), jnp.maximum(L - 1, 0).astype(dtype))
    u_c = (k + 0.5) / jnp.maximum(L, 1)
    u_c = jnp.clip(u_c, eps, jnp.asarray(1.0, dtype=dtype) - eps)
    mu = jnp.log(u_c / (jnp.asarray(1.0, dtype=dtype) - u_c))

    s = jnp.maximum(jnp.asarray(logit_sigma, dtype=dtype), eps)
    if weight_kind == "rbf":
        log_w = -0.5 * jnp.square((z - mu) / s)
    elif weight_kind == "pdf":
        inv_sqrt_2pi = jnp.asarray(0.3989422804014327, dtype=dtype)
        log_phi = -0.5 * jnp.square((z - mu) / s) - jnp.log(s) + jnp.log(inv_sqrt_2pi)
        log_w = log_phi - jnp.log(u) - jnp.log(jnp.asarray(1.0, dtype=dtype) - u)
    else:
        raise ValueError(f"Unknown logit-normal weight_kind {weight_kind!r}; expected 'rbf' or 'pdf'.")
    w = jnp.exp(log_w - jnp.max(log_w))
    return w / jnp.maximum(jnp.sum(w), eps)


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


def compute_common_private(
    activations: Any,
    *,
    agg: str = "mean",
    logit_normal_center_layer: float = 0.0,
    logit_normal_sigma: float = 1.0,
    logit_normal_weight_kind: str = "rbf",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute differentiable common activation and private residuals.

    ``agg="mean"`` uses ``A_common = (1/L) sum_i tilde{A}_i`` (default), with ``tilde{A}_i`` the
    channel-normalized per-layer activations. ``agg="logit_normal"`` uses
    ``A_common = sum_i w_i tilde{A}_i`` with ``w_i >= 0`` and ``sum_i w_i = 1`` — same **convex
    combination volume** as the mean (no extra ``/L`` or global scale beyond ``w``). Weights are
    built from depth ``u_i = (i+0.5)/L`` centered at layer ``logit_normal_center_layer``; see
    ``_layer_logit_normal_weights`` for ``rbf`` vs strict ``pdf`` logit-normal.
    """
    activations = collect_activations(activations)
    activations = _normalize_channels(activations)
    num_layers = activations.shape[0]
    if agg == "mean":
        common = jnp.mean(activations, axis=0)
    elif agg == "logit_normal":
        w = _layer_logit_normal_weights(
            num_layers,
            logit_normal_center_layer,
            logit_normal_sigma,
            activations.dtype,
            weight_kind=logit_normal_weight_kind,
        )
        common = jnp.tensordot(w, activations, axes=(0, 0))
    else:
        raise ValueError(f"Unknown common aggregation {agg!r}; expected 'mean' or 'logit_normal'.")
    common_anchor = jax.lax.stop_gradient(common)
    private = activations - common_anchor[None, ...]
    return common, common_anchor, private


def gram_matrix(x: jax.Array) -> jax.Array:
    """Compute batched Gram matrices from `[B, N, D]` to `[B, N, N]`."""
    if x.ndim != 3:
        raise ValueError(f"Expected input with rank 3, got shape {x.shape}")
    return jnp.einsum("bnd,bmd->bnm", x, x)


def tokens_to_grid(x: jax.Array) -> jax.Array:
    """Reshape `[B, N, C]` tokens into a square spatial grid `[B, H, W, C]`."""
    if x.ndim != 3:
        raise ValueError(f"Expected tokens with rank 3, got shape {x.shape}")
    batch, num_tokens, channels = x.shape
    grid_size = math.isqrt(num_tokens)
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Expected a square token grid, got N={num_tokens}")
    return x.reshape(batch, grid_size, grid_size, channels)


def _normalize_channels(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def _gaussian_kernel_1d(
    sigmas: jax.Array,
    max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
    truncate: float = 3.0,
    eps: float = 1e-6,
) -> jax.Array:
    """Build normalized per-example 1D Gaussian kernels `[B, K]`."""
    if sigmas.ndim != 1:
        raise ValueError(f"Expected sigma vector with rank 1, got shape {sigmas.shape}")
    radius = max(1, int(math.ceil(truncate * max_sigma)))
    offsets = jnp.arange(-radius, radius + 1, dtype=sigmas.dtype)
    safe_sigmas = jnp.maximum(sigmas[:, None], eps)
    kernels = jnp.exp(-0.5 * jnp.square(offsets[None, :] / safe_sigmas))
    delta_kernel = (offsets == 0).astype(sigmas.dtype)[None, :]
    kernels = jnp.where(sigmas[:, None] <= eps, delta_kernel, kernels)
    return kernels / jnp.sum(kernels, axis=-1, keepdims=True)


def _apply_separable_gaussian_blur(
    grid: jax.Array,
    sigmas: jax.Array,
    max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
) -> jax.Array:
    """Apply per-example separable Gaussian blur to `[B, H, W, C]` grids."""
    if grid.ndim != 4:
        raise ValueError(f"Expected spatial grid with rank 4, got shape {grid.shape}")
    if sigmas.ndim != 1:
        raise ValueError(f"Expected sigma vector with rank 1, got shape {sigmas.shape}")
    if grid.shape[0] != sigmas.shape[0]:
        raise ValueError(
            "Grid batch size and sigma batch size must match, "
            f"got {grid.shape[0]} vs {sigmas.shape[0]}"
        )

    kernels = _gaussian_kernel_1d(sigmas.astype(grid.dtype), max_sigma=max_sigma)
    radius = (kernels.shape[-1] - 1) // 2
    height, width = grid.shape[1:3]

    padded_h = jnp.pad(grid, ((0, 0), (radius, radius), (0, 0), (0, 0)), mode="reflect")
    stacked_h = jnp.stack(
        [padded_h[:, idx:idx + height, :, :] for idx in range(kernels.shape[-1])],
        axis=1,
    )
    blurred_h = jnp.sum(stacked_h * kernels[:, :, None, None, None], axis=1)

    padded_w = jnp.pad(blurred_h, ((0, 0), (0, 0), (radius, radius), (0, 0)), mode="reflect")
    stacked_w = jnp.stack(
        [padded_w[:, :, idx:idx + width, :] for idx in range(kernels.shape[-1])],
        axis=1,
    )
    return jnp.sum(stacked_w * kernels[:, :, None, None, None], axis=1)


def extract_sliding_windows(
    grid: jax.Array,
    window_size: int,
    stride: int = DEFAULT_SPATIAL_WINDOW_STRIDE,
) -> jax.Array:
    """Extract all valid sliding windows as `[B, num_windows, window_area, C]`."""
    if grid.ndim != 4:
        raise ValueError(f"Expected a spatial grid with rank 4, got shape {grid.shape}")
    if window_size <= 0:
        raise ValueError(f"Window size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"Stride must be positive, got {stride}")

    batch, height, width, channels = grid.shape
    if window_size > height or window_size > width:
        raise ValueError(
            f"Window size {window_size} exceeds grid size {(height, width)}"
        )

    out_h = (height - window_size) // stride + 1
    out_w = (width - window_size) // stride + 1
    window_tokens = []
    for dy in range(window_size):
        for dx in range(window_size):
            window_tokens.append(
                grid[
                    :,
                    dy:dy + out_h * stride:stride,
                    dx:dx + out_w * stride:stride,
                    :,
                ]
            )

    stacked = jnp.stack(window_tokens, axis=3)  # [B, out_h, out_w, window_area, C]
    return stacked.reshape(batch, out_h * out_w, window_size * window_size, channels)


def window_gram_matrix(
    window_tokens: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Compute normalized local Gram matrices for `[B, W, M, C]` window tokens."""
    if window_tokens.ndim != 4:
        raise ValueError(
            f"Expected window tokens with rank 4, got shape {window_tokens.shape}"
        )
    normalized = _normalize_channels(window_tokens, eps=eps)
    return jnp.einsum("bwmc,bwnc->bwmn", normalized, normalized)


def local_window_gram_loss(
    feature_tokens: jax.Array,
    target_tokens: jax.Array,
    window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    stride: int = DEFAULT_SPATIAL_WINDOW_STRIDE,
    eps: float = 1e-8,
    target_blur_sigmas: jax.Array | None = None,
    target_blur_max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compare local Gram structure between feature and target sliding windows."""
    feature_grid = tokens_to_grid(feature_tokens)
    target_grid = tokens_to_grid(target_tokens)
    if feature_grid.shape[:3] != target_grid.shape[:3]:
        raise ValueError(
            "Feature and target grids must match in batch/height/width, "
            f"got {feature_grid.shape[:3]} vs {target_grid.shape[:3]}"
        )
    if target_blur_sigmas is not None:
        target_grid = _apply_separable_gaussian_blur(
            target_grid,
            target_blur_sigmas,
            max_sigma=target_blur_max_sigma,
        )

    feature_windows = extract_sliding_windows(feature_grid, window_size, stride=stride)
    target_windows = extract_sliding_windows(target_grid, window_size, stride=stride)
    feature_grams = window_gram_matrix(feature_windows, eps=eps)
    target_grams = window_gram_matrix(target_windows, eps=eps)

    window_losses = jnp.mean(jnp.abs(feature_grams - target_grams), axis=(-2, -1))
    spatial_loss = jnp.mean(window_losses)
    blur_sigma_mean = (
        jnp.mean(target_blur_sigmas.astype(feature_tokens.dtype))
        if target_blur_sigmas is not None
        else jnp.array(0.0, dtype=feature_tokens.dtype)
    )
    spatial_metrics = {
        "spatial_num_windows": jnp.asarray(feature_windows.shape[1], dtype=feature_tokens.dtype),
        "spatial_window_area": jnp.asarray(feature_windows.shape[2], dtype=feature_tokens.dtype),
        "spatial_blur_sigma_mean": blur_sigma_mean,
    }
    return spatial_loss, spatial_metrics


def _pairwise_cosines(private: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Return cosine similarities for all layer pairs `i < j`."""
    num_layers = private.shape[0]
    if num_layers < 2:
        return jnp.array(0.0, dtype=private.dtype)

    flattened = private.reshape(num_layers, -1)
    norms = jnp.linalg.norm(flattened, axis=-1, keepdims=True)
    normalized = flattened / jnp.maximum(norms, eps)
    cosine_matrix = normalized @ normalized.T
    upper_indices = jnp.triu_indices(num_layers, k=1)
    return cosine_matrix[upper_indices]


def _mean_pairwise_cosine_squared(
    private: jax.Array,
    eps: float = 1e-8,
    rng: jax.Array | None = None,
    max_pairs: int = 0,
) -> jax.Array:
    """Average squared cosine similarity over all or sampled layer pairs."""
    pairwise_cosines = _pairwise_cosines(private, eps=eps)
    if pairwise_cosines.ndim == 0:
        return pairwise_cosines

    if max_pairs and max_pairs > 0 and pairwise_cosines.shape[0] > max_pairs:
        if rng is None:
            raise ValueError("An RNG key is required when sampling private-layer pairs.")
        indices = jax.random.permutation(rng, pairwise_cosines.shape[0])[:max_pairs]
        pairwise_cosines = pairwise_cosines[indices]

    return jnp.mean(jnp.square(pairwise_cosines))


def compute_aux_losses(
    activations: Any,
    spatial_target: jax.Array | None = None,
    timesteps: jax.Array | None = None,
    private_pair_rng: jax.Array | None = None,
    private_max_pairs: int = 0,
    spatial_window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    spatial_window_stride: int = DEFAULT_SPATIAL_WINDOW_STRIDE,
    spatial_blur_by_timestep: bool = False,
    spatial_blur_max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
    common_agg: str = "mean",
    common_logit_normal_center_layer: float = 0.0,
    common_logit_normal_sigma: float = 1.0,
    common_logit_normal_weight_kind: str = "rbf",
) -> dict[str, jax.Array]:
    """Compute A_common/private metrics for activation decomposition.

    Spatial-Gram arguments are retained for compatibility with existing call sites, but the
    returned auxiliary bundle now only contains the common activation plus private-loss metrics.
    """
    activations = collect_activations(activations)

    common, _, private = compute_common_private(
        activations,
        agg=common_agg,
        logit_normal_center_layer=common_logit_normal_center_layer,
        logit_normal_sigma=common_logit_normal_sigma,
        logit_normal_weight_kind=common_logit_normal_weight_kind,
    )

    private_loss = _mean_pairwise_cosine_squared(
        private,
        rng=private_pair_rng,
        max_pairs=private_max_pairs,
    )

    common_norm = jnp.mean(jnp.linalg.norm(common.reshape(common.shape[0], -1), axis=-1))
    private_norms = jnp.linalg.norm(private.reshape(private.shape[0], private.shape[1], -1), axis=-1)
    avg_private_norm = jnp.mean(private_norms)

    pairwise_cosines = _pairwise_cosines(private)
    if pairwise_cosines.ndim == 0:
        avg_pairwise_cosine = pairwise_cosines
    else:
        avg_pairwise_cosine = jnp.mean(pairwise_cosines)

    return {
        "common_activation": common,
        "private_activations": private,
        "loss_private": private_loss,
        "norm_common": common_norm,
        "avg_private_norm": avg_private_norm,
        "avg_pairwise_private_cosine": avg_pairwise_cosine,
    }
