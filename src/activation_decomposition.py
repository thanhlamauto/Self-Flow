"""Helper utilities for common/private activation decomposition in DiT."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

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
) -> jax.Array:
    """Nonnegative weights over layer index, peak near ``center_layer`` (logit-normal in depth).

    For layer ``i`` we map ``u_i = (i + 0.5) / L`` in ``(0, 1)`` and take the logit-normal
    density with underlying Gaussian ``N(mu, sigma^2)`` on ``logit(u)``, with ``mu`` chosen
    from the same mapping of ``center_layer``. Weights are normalized to sum to ``1`` so
    ``sum_i w_i A_i`` is a convex combination. Larger ``sigma`` yields a flatter profile
    (less mass concentrated near the center layer).
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
    inv_sqrt_2pi = jnp.asarray(0.3989422804014327, dtype=dtype)  # 1/sqrt(2*pi)
    log_phi = -0.5 * jnp.square((z - mu) / s) - jnp.log(s) + jnp.log(inv_sqrt_2pi)
    log_w = log_phi - jnp.log(u) - jnp.log(jnp.asarray(1.0, dtype=dtype) - u)
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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute differentiable common activation and private residuals.

    ``agg="mean"`` uses ``A_common = mean_i A_i`` (default). ``agg="logit_normal"`` uses
    ``A_common = sum_i w_i A_i`` with ``w`` from a logit-normal depth profile centered at
    ``logit_normal_center_layer`` (0-based layer index; fractional values interpolate the
    center in ``(0,1)`` via ``(k+0.5)/L``). ``logit_normal_sigma`` is the Gaussian standard
    deviation on the logit scale (larger => flatter weights).
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


def _per_token_cosine_two_layers(z_a: jax.Array, z_b: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Cosine similarity per ``(batch, token)`` between two `[B, N, D]` tensors."""
    na = jnp.linalg.norm(z_a, axis=-1, keepdims=True)
    nb = jnp.linalg.norm(z_b, axis=-1, keepdims=True)
    a = z_a / jnp.maximum(na, eps)
    b = z_b / jnp.maximum(nb, eps)
    return jnp.sum(a * b, axis=-1)


def _selected_pair_token_cosines(
    private: jax.Array,
    pair_indices: Tuple[Tuple[int, int], ...],
    eps: float = 1e-8,
) -> jax.Array:
    """Stack per-token cosines for fixed (early, late) layer pairs -> `[P, B, N]`."""
    stacks = [
        _per_token_cosine_two_layers(private[i], private[j], eps=eps) for i, j in pair_indices
    ]
    return jnp.stack(stacks, axis=0)


def _layer_pair_per_token_cosines(private: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Per-patch cosines between layer pairs. private `[L, B, N, D]` -> `[P, B, N]` for `i<j`."""
    num_layers = private.shape[0]
    if num_layers < 2:
        raise ValueError("_layer_pair_per_token_cosines requires at least two layers.")

    norms = jnp.linalg.norm(private, axis=-1, keepdims=True)
    p = private / jnp.maximum(norms, eps)
    cos_lm = jnp.einsum("lbnd,mbnd->lmbn", p, p)
    li, mi = jnp.triu_indices(num_layers, k=1)
    return cos_lm[li, mi]


def _private_loss_per_token_mi_proxy(
    private: jax.Array,
    eps: float = 1e-8,
    rng: jax.Array | None = None,
    max_pairs: int = 0,
    private_layer_pairs: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> tuple[jax.Array, jax.Array]:
    """Per-token MI proxy: L2-normalize along feature dim ``D``, cosine dot per ``(batch, token)``.

    Matches the usual block-wise formulation: for each pair of layers ``(i, j)``, compute
    ``hat(f)_i[b,n,:]^T hat(f)_j[b,n,:]`` (cosine similarity), then average over all pairs,
    batch, and tokens — i.e. mean cosine, not ``|cos|`` or ``cos^2``.

    If ``private_layer_pairs`` is set (0-based indices into the layer stack), only those pairs
    are used. Otherwise all ``i<j`` pairs, optionally subsampled when ``max_pairs > 0``.

    Returns ``(loss_private, avg_pairwise_cos)``; the metric is the mean cosine over all chosen
    pairs before subsampling; ``loss_private`` uses the subsampled subset when applicable.
    """
    num_layers = private.shape[0]
    if num_layers < 2:
        z = jnp.array(0.0, dtype=private.dtype)
        return z, z

    if private_layer_pairs:
        pair_cos = _selected_pair_token_cosines(private, private_layer_pairs, eps=eps)
    else:
        pair_cos = _layer_pair_per_token_cosines(private, eps=eps)
    avg_pairwise_cos = jnp.mean(pair_cos)

    if max_pairs and max_pairs > 0 and pair_cos.shape[0] > max_pairs:
        if rng is None:
            raise ValueError("An RNG key is required when sampling private-layer pairs.")
        indices = jax.random.permutation(rng, pair_cos.shape[0])[:max_pairs]
        pair_cos = pair_cos[indices]

    loss_private = jnp.mean(pair_cos)
    return loss_private, avg_pairwise_cos


def compute_aux_losses(
    activations: Any,
    spatial_target: jax.Array,
    timesteps: jax.Array | None = None,
    private_pair_rng: jax.Array | None = None,
    private_max_pairs: int = 0,
    private_layer_pairs: Optional[Tuple[Tuple[int, int], ...]] = None,
    spatial_window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    spatial_window_stride: int = DEFAULT_SPATIAL_WINDOW_STRIDE,
    spatial_blur_by_timestep: bool = False,
    spatial_blur_max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
    learnable_common_tensor: bool = False,
    common_activation: jax.Array | None = None,
    common_agg: str = "mean",
    common_logit_normal_center_layer: float = 0.0,
    common_logit_normal_sigma: float = 1.0,
) -> dict[str, jax.Array]:
    """Compute auxiliary losses and logging metrics for activation decomposition."""
    activations = collect_activations(activations)
    if spatial_target.ndim != 3:
        raise ValueError(f"Expected spatial target with rank 3, got shape {spatial_target.shape}")
    if spatial_blur_by_timestep:
        if timesteps is None:
            raise ValueError("Timesteps are required when timestep-dependent spatial blur is enabled.")
        if timesteps.ndim != 1:
            raise ValueError(f"Expected timesteps with rank 1, got shape {timesteps.shape}")
        if timesteps.shape[0] != spatial_target.shape[0]:
            raise ValueError(
                "Timesteps batch size and spatial target batch size must match, "
                f"got {timesteps.shape[0]} vs {spatial_target.shape[0]}"
            )
        blur_sigmas = spatial_blur_max_sigma * jnp.clip(1.0 - timesteps, 0.0, 1.0)
    else:
        blur_sigmas = None

    if learnable_common_tensor:
        if common_activation is None:
            raise ValueError("common_activation is required when learnable_common_tensor is True.")
        batch, num_tokens, _ = spatial_target.shape
        if common_activation.shape[0] != num_tokens:
            raise ValueError(
                "common_activation num_patches must match spatial_target, "
                f"got N={common_activation.shape[0]} vs {num_tokens}"
            )
        if common_activation.shape[-1] != activations.shape[-1]:
            raise ValueError(
                "common_activation hidden dim must match layer activations, "
                f"got D={common_activation.shape[-1]} vs {activations.shape[-1]}"
            )
        # spatial_target is latent patches (B, N, patch_dim); common lives in DiT hidden space (B, N, D).
        # Same layout as the mean-activations branch — Gram windows only need matching B,N grid.
        common = jnp.broadcast_to(
            common_activation[None, :, :],
            (batch, num_tokens, common_activation.shape[-1]),
        ).astype(spatial_target.dtype)
        private = _normalize_channels(activations)
    else:
        common, common_anchor, private = compute_common_private(
            activations,
            agg=common_agg,
            logit_normal_center_layer=common_logit_normal_center_layer,
            logit_normal_sigma=common_logit_normal_sigma,
        )

    spatial_loss, spatial_metrics = local_window_gram_loss(
        common,
        spatial_target,
        window_size=spatial_window_size,
        stride=spatial_window_stride,
        target_blur_sigmas=blur_sigmas,
        target_blur_max_sigma=spatial_blur_max_sigma,
    )

    private_loss, avg_pairwise_cosine = _private_loss_per_token_mi_proxy(
        private,
        rng=private_pair_rng,
        max_pairs=private_max_pairs,
        private_layer_pairs=private_layer_pairs,
    )

    common_norm = jnp.mean(jnp.linalg.norm(common.reshape(common.shape[0], -1), axis=-1))
    private_norms = jnp.linalg.norm(private.reshape(private.shape[0], private.shape[1], -1), axis=-1)
    avg_private_norm = jnp.mean(private_norms)

    return {
        "common_activation": common,
        "private_activations": private,
        "loss_spatial": spatial_loss,
        "loss_private": private_loss,
        "spatial_metrics": spatial_metrics,
        "norm_common": common_norm,
        "avg_private_norm": avg_private_norm,
        "avg_pairwise_private_cosine": avg_pairwise_cosine,
    }
