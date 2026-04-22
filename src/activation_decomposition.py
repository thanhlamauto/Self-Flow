"""Helper utilities for layer-delta activation regularization in DiT."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

DEFAULT_SPATIAL_WINDOW_SIZE = 3
DEFAULT_SPATIAL_WINDOW_STRIDE = 1
DEFAULT_SPATIAL_ALIGN_MAX_LAYERS = 2
DEFAULT_SPATIAL_LOSS_TYPE = "local_gram"
DEFAULT_SPATIAL_TARGET_SVD_RANK = 0
DEFAULT_PRIVATE_MAX_LAYERS = 6
DEFAULT_PRIVATE_MAX_PAIRS = 0
DEFAULT_MAX_TIMESTEP_BLUR_SIGMA = 3.0
DEFAULT_TIMESTEP_BLUR_SCHEDULE = "linear"
DEFAULT_TIMESTEP_BLUR_EXP_RATE = 5.0


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


def compute_private_deltas(
    activations: Any,
    eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    """Return normalized layer activations and consecutive layer deltas.

    Layer deltas follow the user's requested definition:
    `private_i = activation_i - activation_{i-1}`.
    """
    activations = collect_activations(activations)
    if activations.shape[0] < 2:
        raise ValueError(
            "Need at least two activation tensors to form consecutive layer deltas, "
            f"got {activations.shape[0]}"
        )
    normalized_activations = _normalize_channels(activations, eps=eps)
    private = normalized_activations[1:] - jax.lax.stop_gradient(normalized_activations[:-1])
    return normalized_activations, private


def _timestep_dependent_blur_sigmas(
    timesteps: jax.Array,
    max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
    schedule: str = DEFAULT_TIMESTEP_BLUR_SCHEDULE,
    exp_rate: float = DEFAULT_TIMESTEP_BLUR_EXP_RATE,
) -> jax.Array:
    """Map `tau` in `[0, 1]` to blur sigma with selectable decay curvature."""
    tau = jnp.clip(timesteps, 0.0, 1.0)
    if schedule == "linear":
        blur_scale = 1.0 - tau
    elif schedule == "exp-concave":
        if exp_rate <= 0:
            raise ValueError("Exponential blur rate must be > 0 for exp-concave schedule.")
        rate = jnp.asarray(exp_rate, dtype=tau.dtype)
        denom = jnp.maximum(jnp.expm1(rate), jnp.asarray(1e-6, dtype=tau.dtype))
        blur_scale = 1.0 - (jnp.expm1(rate * tau) / denom)
    elif schedule == "exp-convex":
        if exp_rate <= 0:
            raise ValueError("Exponential blur rate must be > 0 for exp-convex schedule.")
        rate = jnp.asarray(exp_rate, dtype=tau.dtype)
        denom = jnp.maximum(jnp.expm1(rate), jnp.asarray(1e-6, dtype=tau.dtype))
        blur_scale = jnp.expm1(rate * (1.0 - tau)) / denom
    else:
        raise ValueError(
            f"Unknown timestep blur schedule {schedule!r}; "
            "expected 'linear', 'exp-concave', or 'exp-convex'."
        )
    return jnp.asarray(max_sigma, dtype=tau.dtype) * jnp.clip(blur_scale, 0.0, 1.0)


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
        raise ValueError(f"Window size {window_size} exceeds grid size {(height, width)}")

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

    stacked = jnp.stack(window_tokens, axis=3)
    return stacked.reshape(batch, out_h * out_w, window_size * window_size, channels)


def window_gram_matrix(
    window_tokens: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Compute normalized local Gram matrices for `[B, W, M, C]` window tokens."""
    if window_tokens.ndim != 4:
        raise ValueError(f"Expected window tokens with rank 4, got shape {window_tokens.shape}")
    normalized = _normalize_channels(window_tokens, eps=eps)
    return jnp.einsum("bwmc,bwnc->bwmn", normalized, normalized)


def _apply_truncated_svd_to_tokens(
    tokens: jax.Array,
    rank: int = DEFAULT_SPATIAL_TARGET_SVD_RANK,
) -> tuple[jax.Array, jax.Array]:
    """Apply batched truncated SVD to `[B, N, D]` tokens and return effective rank."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with rank 3, got shape {tokens.shape}")

    max_rank = min(tokens.shape[1], tokens.shape[2])
    if rank <= 0:
        return tokens, jnp.array(0.0, dtype=tokens.dtype)

    effective_rank = min(rank, max_rank)
    if effective_rank >= max_rank:
        return tokens, jnp.asarray(effective_rank, dtype=tokens.dtype)

    u, singular_values, vh = jnp.linalg.svd(tokens, full_matrices=False)
    u = u[:, :, :effective_rank]
    singular_values = singular_values[:, :effective_rank]
    vh = vh[:, :effective_rank, :]
    truncated_tokens = jnp.einsum("bnr,br,brd->bnd", u, singular_values, vh)
    return truncated_tokens, jnp.asarray(effective_rank, dtype=tokens.dtype)


def _prepare_spatial_target_tokens(
    target_tokens: jax.Array,
    *,
    target_blur_sigmas: jax.Array | None = None,
    target_blur_max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
    target_svd_rank: int = DEFAULT_SPATIAL_TARGET_SVD_RANK,
) -> tuple[jax.Array, jax.Array]:
    """Apply optional blur then truncated SVD to the spatial target tokens."""
    if target_tokens.ndim != 3:
        raise ValueError(f"Expected target tokens with rank 3, got shape {target_tokens.shape}")

    if target_blur_sigmas is not None:
        target_grid = tokens_to_grid(target_tokens)
        target_grid = _apply_separable_gaussian_blur(
            target_grid,
            target_blur_sigmas,
            max_sigma=target_blur_max_sigma,
        )
        target_tokens = target_grid.reshape(target_tokens.shape)

    return _apply_truncated_svd_to_tokens(target_tokens, rank=target_svd_rank)


def local_window_gram_loss_from_grids(
    feature_grid: jax.Array,
    target_grid: jax.Array,
    window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    stride: int = DEFAULT_SPATIAL_WINDOW_STRIDE,
    eps: float = 1e-8,
    sample_mask: jax.Array | None = None,
    target_blur_sigmas: jax.Array | None = None,
    target_blur_max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
    target_svd_rank: int = DEFAULT_SPATIAL_TARGET_SVD_RANK,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compare local Gram structure between two `[B, H, W, D]` grids."""
    if feature_grid.ndim != 4:
        raise ValueError(f"Expected feature grid with rank 4, got shape {feature_grid.shape}")
    if target_grid.ndim != 4:
        raise ValueError(f"Expected target grid with rank 4, got shape {target_grid.shape}")
    if feature_grid.shape[:3] != target_grid.shape[:3]:
        raise ValueError(
            "Feature and target grids must match in batch/height/width, "
            f"got {feature_grid.shape[:3]} vs {target_grid.shape[:3]}"
        )

    if target_blur_sigmas is not None or target_svd_rank > 0:
        batch, height, width, channels = target_grid.shape
        target_tokens = target_grid.reshape(batch, height * width, channels)
        target_tokens, effective_rank = _prepare_spatial_target_tokens(
            target_tokens,
            target_blur_sigmas=target_blur_sigmas,
            target_blur_max_sigma=target_blur_max_sigma,
            target_svd_rank=target_svd_rank,
        )
        target_grid = target_tokens.reshape(target_grid.shape)
    else:
        effective_rank = jnp.array(0.0, dtype=target_grid.dtype)

    feature_windows = extract_sliding_windows(feature_grid, window_size, stride=stride)
    target_windows = extract_sliding_windows(target_grid, window_size, stride=stride)
    feature_grams = window_gram_matrix(feature_windows, eps=eps)
    target_grams = window_gram_matrix(target_windows, eps=eps)

    window_losses = jnp.mean(jnp.abs(feature_grams - target_grams), axis=(-2, -1))
    example_losses = jnp.mean(window_losses, axis=-1)
    if sample_mask is not None:
        if sample_mask.ndim != 1:
            raise ValueError(f"Expected sample mask with rank 1, got shape {sample_mask.shape}")
        if sample_mask.shape[0] != feature_grid.shape[0]:
            raise ValueError(
                "Sample mask batch size and feature batch size must match, "
                f"got {sample_mask.shape[0]} vs {feature_grid.shape[0]}"
            )
        mask = sample_mask.astype(feature_grid.dtype)
        active_count = jnp.sum(mask)
        spatial_loss = jnp.where(
            active_count > 0,
            jnp.sum(example_losses * mask) / active_count,
            jnp.array(0.0, dtype=feature_grid.dtype),
        )
        active_fraction = active_count / jnp.maximum(
            jnp.asarray(feature_grid.shape[0], dtype=feature_grid.dtype),
            jnp.array(1.0, dtype=feature_grid.dtype),
        )
        blur_sigma_mean = (
            jnp.where(
                active_count > 0,
                jnp.sum(target_blur_sigmas.astype(feature_grid.dtype) * mask) / active_count,
                jnp.array(0.0, dtype=feature_grid.dtype),
            )
            if target_blur_sigmas is not None
            else jnp.array(0.0, dtype=feature_grid.dtype)
        )
    else:
        spatial_loss = jnp.mean(example_losses)
        active_fraction = jnp.array(1.0, dtype=feature_grid.dtype)
        blur_sigma_mean = (
            jnp.mean(target_blur_sigmas.astype(feature_grid.dtype))
            if target_blur_sigmas is not None
            else jnp.array(0.0, dtype=feature_grid.dtype)
        )
    spatial_metrics = {
        "spatial_num_windows": jnp.asarray(feature_windows.shape[1], dtype=feature_grid.dtype),
        "spatial_window_area": jnp.asarray(feature_windows.shape[2], dtype=feature_grid.dtype),
        "spatial_active_fraction": active_fraction,
        "spatial_blur_sigma_mean": blur_sigma_mean,
        "spatial_target_svd_rank": effective_rank,
    }
    return spatial_loss, spatial_metrics


def local_window_gram_loss(
    feature_tokens: jax.Array,
    target_tokens: jax.Array,
    window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    stride: int = DEFAULT_SPATIAL_WINDOW_STRIDE,
    eps: float = 1e-8,
    sample_mask: jax.Array | None = None,
    target_blur_sigmas: jax.Array | None = None,
    target_blur_max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
    target_svd_rank: int = DEFAULT_SPATIAL_TARGET_SVD_RANK,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Convert `[B, N, D]` tokens to `[B, H, W, D]` before the local Gram loss."""
    feature_grid = tokens_to_grid(feature_tokens)
    target_grid = tokens_to_grid(target_tokens)
    return local_window_gram_loss_from_grids(
        feature_grid,
        target_grid,
        window_size=window_size,
        stride=stride,
        eps=eps,
        sample_mask=sample_mask,
        target_blur_sigmas=target_blur_sigmas,
        target_blur_max_sigma=target_blur_max_sigma,
        target_svd_rank=target_svd_rank,
    )


def global_linear_cka_loss(
    feature_tokens: jax.Array,
    target_tokens: jax.Array,
    eps: float = 1e-8,
    sample_mask: jax.Array | None = None,
    target_blur_sigmas: jax.Array | None = None,
    target_blur_max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
    target_svd_rank: int = DEFAULT_SPATIAL_TARGET_SVD_RANK,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute global token-space linear CKA on `[B, N, D]` tokens.

    This path intentionally avoids token-wise L2 normalization before CKA. We only
    center across tokens, then compute the standard linear CKA in token space
    using `N x N` Gram matrices because `N < D` in the current DiT setups.
    """
    if feature_tokens.ndim != 3:
        raise ValueError(f"Expected feature tokens with rank 3, got shape {feature_tokens.shape}")
    if target_tokens.ndim != 3:
        raise ValueError(f"Expected target tokens with rank 3, got shape {target_tokens.shape}")
    if feature_tokens.shape != target_tokens.shape:
        raise ValueError(
            "Feature and target tokens must match in batch/token/channel dims, "
            f"got {feature_tokens.shape} vs {target_tokens.shape}"
        )

    target_tokens, effective_rank = _prepare_spatial_target_tokens(
        target_tokens,
        target_blur_sigmas=target_blur_sigmas,
        target_blur_max_sigma=target_blur_max_sigma,
        target_svd_rank=target_svd_rank,
    )

    feature_centered = feature_tokens - jnp.mean(feature_tokens, axis=1, keepdims=True)
    target_centered = target_tokens - jnp.mean(target_tokens, axis=1, keepdims=True)

    cross_gram = jnp.einsum("bnd,bmd->bnm", feature_centered, target_centered)
    feature_gram = jnp.einsum("bnd,bmd->bnm", feature_centered, feature_centered)
    target_gram = jnp.einsum("bnd,bmd->bnm", target_centered, target_centered)

    numerator = jnp.sum(jnp.square(cross_gram), axis=(-2, -1))
    denom_feature = jnp.sqrt(jnp.sum(jnp.square(feature_gram), axis=(-2, -1)) + eps)
    denom_target = jnp.sqrt(jnp.sum(jnp.square(target_gram), axis=(-2, -1)) + eps)
    cka = numerator / jnp.maximum(denom_feature * denom_target, eps)
    cka = jnp.clip(cka, 0.0, 1.0)
    example_losses = 1.0 - cka

    if sample_mask is not None:
        if sample_mask.ndim != 1:
            raise ValueError(f"Expected sample mask with rank 1, got shape {sample_mask.shape}")
        if sample_mask.shape[0] != feature_tokens.shape[0]:
            raise ValueError(
                "Sample mask batch size and feature batch size must match, "
                f"got {sample_mask.shape[0]} vs {feature_tokens.shape[0]}"
            )
        mask = sample_mask.astype(feature_tokens.dtype)
        active_count = jnp.sum(mask)
        spatial_loss = jnp.where(
            active_count > 0,
            jnp.sum(example_losses * mask) / active_count,
            jnp.array(0.0, dtype=feature_tokens.dtype),
        )
        active_fraction = active_count / jnp.maximum(
            jnp.asarray(feature_tokens.shape[0], dtype=feature_tokens.dtype),
            jnp.array(1.0, dtype=feature_tokens.dtype),
        )
        blur_sigma_mean = (
            jnp.where(
                active_count > 0,
                jnp.sum(target_blur_sigmas.astype(feature_tokens.dtype) * mask) / active_count,
                jnp.array(0.0, dtype=feature_tokens.dtype),
            )
            if target_blur_sigmas is not None
            else jnp.array(0.0, dtype=feature_tokens.dtype)
        )
    else:
        spatial_loss = jnp.mean(example_losses)
        active_fraction = jnp.array(1.0, dtype=feature_tokens.dtype)
        blur_sigma_mean = (
            jnp.mean(target_blur_sigmas.astype(feature_tokens.dtype))
            if target_blur_sigmas is not None
            else jnp.array(0.0, dtype=feature_tokens.dtype)
        )

    spatial_metrics = {
        "spatial_num_windows": jnp.array(1.0, dtype=feature_tokens.dtype),
        "spatial_window_area": jnp.asarray(feature_tokens.shape[1], dtype=feature_tokens.dtype),
        "spatial_active_fraction": active_fraction,
        "spatial_blur_sigma_mean": blur_sigma_mean,
        "spatial_target_svd_rank": effective_rank,
    }
    return spatial_loss, spatial_metrics


def _pairwise_cosine_matrix(private: jax.Array, eps: float = 1e-8) -> jax.Array:
    num_layers = private.shape[0]
    if num_layers == 0:
        return jnp.zeros((0, 0), dtype=private.dtype)
    flattened = private.reshape(num_layers, -1)
    norms = jnp.linalg.norm(flattened, axis=-1, keepdims=True)
    normalized = flattened / jnp.maximum(norms, eps)
    return normalized @ normalized.T


def _pairwise_cosines(private: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Return cosine similarities for all layer pairs `i < j`."""
    num_layers = private.shape[0]
    if num_layers < 2:
        return jnp.array(0.0, dtype=private.dtype)
    cosine_matrix = _pairwise_cosine_matrix(private, eps=eps)
    upper_indices = jnp.triu_indices(num_layers, k=1)
    return cosine_matrix[upper_indices]


def _sample_layer_mask(
    num_layers: int,
    max_layers: int,
    dtype: jnp.dtype,
    rng: jax.Array | None = None,
) -> jax.Array:
    if num_layers == 0:
        return jnp.zeros((0,), dtype=dtype)
    if max_layers and max_layers > 0 and num_layers > max_layers:
        if rng is None:
            raise ValueError("An RNG key is required when sampling layer indices.")
        indices = jax.random.permutation(rng, num_layers)[:max_layers]
        return jnp.zeros((num_layers,), dtype=dtype).at[indices].set(1.0)
    return jnp.ones((num_layers,), dtype=dtype)


def _sample_private_layer_mask(
    num_layers: int,
    max_layers: int,
    dtype: jnp.dtype,
    rng: jax.Array | None = None,
) -> jax.Array:
    """Sample private layers while always keeping layer 1 (index 0 in private deltas)."""
    if num_layers == 0:
        return jnp.zeros((0,), dtype=dtype)

    if max_layers <= 0 or max_layers >= num_layers:
        return jnp.ones((num_layers,), dtype=dtype)

    sampled_layers = max(1, min(max_layers, num_layers))
    selection_mask = jnp.zeros((num_layers,), dtype=dtype).at[0].set(1.0)
    if sampled_layers == 1:
        return selection_mask

    if rng is None:
        raise ValueError("An RNG key is required when sampling private layers.")

    remaining_indices = jnp.arange(1, num_layers, dtype=jnp.int32)
    extra_indices = jax.random.permutation(rng, remaining_indices)[:sampled_layers - 1]
    return selection_mask.at[extra_indices].set(1.0)


def _masked_mean(values: jax.Array, mask: jax.Array, dtype: jnp.dtype) -> jax.Array:
    if values.shape[0] == 0:
        return jnp.array(0.0, dtype=dtype)
    count = jnp.sum(mask)
    return jnp.where(
        count > 0,
        jnp.sum(values * mask) / count,
        jnp.array(0.0, dtype=dtype),
    )


def _mean_masked_pairwise_cosine_squared(
    private: jax.Array,
    selection_mask: jax.Array,
    max_pairs: int = DEFAULT_PRIVATE_MAX_PAIRS,
    eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    """Average squared cosine similarity over ordered sampled pairs.

    Pair order follows the requested deterministic traversal over sampled layers:
    `(1, l2), (1, l3), ..., (l2, l3), ...`, truncated once `max_pairs` is reached.
    """
    num_layers = private.shape[0]
    if num_layers < 2:
        zero = jnp.array(0.0, dtype=private.dtype)
        return zero, zero

    cosine_matrix = _pairwise_cosine_matrix(private, eps=eps)
    pair_i = []
    pair_j = []
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            pair_i.append(i)
            pair_j.append(j)

    if not pair_i:
        zero = jnp.array(0.0, dtype=private.dtype)
        return zero, zero

    pair_i = jnp.asarray(pair_i, dtype=jnp.int32)
    pair_j = jnp.asarray(pair_j, dtype=jnp.int32)
    ordered_pair_mask = selection_mask[pair_i] * selection_mask[pair_j]
    if max_pairs > 0:
        pair_rank = jnp.cumsum(ordered_pair_mask.astype(jnp.int32))
        ordered_pair_mask = ordered_pair_mask * (pair_rank <= max_pairs).astype(private.dtype)

    pair_values = jnp.square(cosine_matrix[pair_i, pair_j])
    pair_count = jnp.sum(ordered_pair_mask)
    loss = jnp.where(
        pair_count > 0,
        jnp.sum(pair_values * ordered_pair_mask) / pair_count,
        jnp.array(0.0, dtype=private.dtype),
    )
    return loss, pair_count


def _resolve_spatial_window(
    layer_index: int,
    *,
    model_size: str | None,
    default_window_size: int,
    default_stride: int,
) -> tuple[int, int]:
    if default_window_size > 0 and default_stride > 0:
        return default_window_size, default_stride

    if model_size is not None and model_size.upper() == "B":
        if 1 <= layer_index <= 4:
            return 2, 2
        if 5 <= layer_index <= 10:
            return 8, 8
        if layer_index == 11:
            return 4, 4

    return DEFAULT_SPATIAL_WINDOW_SIZE, DEFAULT_SPATIAL_WINDOW_STRIDE


def _mean_sampled_spatial_alignment_loss(
    activations: jax.Array,
    *,
    spatial_align_rng: jax.Array | None,
    spatial_align_max_layers: int,
    spatial_loss_type: str,
    spatial_window_size: int,
    spatial_window_stride: int,
    spatial_sample_mask: jax.Array | None,
    blur_sigmas: jax.Array | None,
    spatial_blur_max_sigma: float,
    spatial_target_svd_rank: int,
    model_size: str | None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    dtype = activations.dtype
    candidate_layers = activations[1:-1]
    target_layer = jax.lax.stop_gradient(activations[-1])
    num_candidates = candidate_layers.shape[0]

    if num_candidates == 0:
        zero = jnp.array(0.0, dtype=dtype)
        return zero, {
            "spatial_num_windows": zero,
            "spatial_window_area": zero,
            "spatial_active_fraction": zero,
            "spatial_blur_sigma_mean": zero,
            "spatial_target_svd_rank": zero,
            "spatial_selected_layers": zero,
        }

    selection_mask = _sample_layer_mask(
        num_candidates,
        spatial_align_max_layers,
        dtype=dtype,
        rng=spatial_align_rng,
    )

    losses = []
    num_windows = []
    window_areas = []
    active_fractions = []
    blur_sigma_means = []
    target_svd_ranks = []
    for layer_offset in range(num_candidates):
        layer_index = layer_offset + 1
        if spatial_loss_type == "local_gram":
            window_size, stride = _resolve_spatial_window(
                layer_index,
                model_size=model_size,
                default_window_size=spatial_window_size,
                default_stride=spatial_window_stride,
            )
            layer_loss, layer_metrics = local_window_gram_loss(
                candidate_layers[layer_offset],
                target_layer,
                window_size=window_size,
                stride=stride,
                sample_mask=spatial_sample_mask,
                target_blur_sigmas=blur_sigmas,
                target_blur_max_sigma=spatial_blur_max_sigma,
                target_svd_rank=spatial_target_svd_rank,
            )
        elif spatial_loss_type == "global_linear_cka":
            layer_loss, layer_metrics = global_linear_cka_loss(
                candidate_layers[layer_offset],
                target_layer,
                sample_mask=spatial_sample_mask,
                target_blur_sigmas=blur_sigmas,
                target_blur_max_sigma=spatial_blur_max_sigma,
                target_svd_rank=spatial_target_svd_rank,
            )
        else:
            raise ValueError(
                f"Unknown spatial loss type {spatial_loss_type!r}; "
                "expected 'local_gram' or 'global_linear_cka'."
            )
        losses.append(layer_loss)
        num_windows.append(layer_metrics["spatial_num_windows"])
        window_areas.append(layer_metrics["spatial_window_area"])
        active_fractions.append(layer_metrics["spatial_active_fraction"])
        blur_sigma_means.append(layer_metrics["spatial_blur_sigma_mean"])
        target_svd_ranks.append(layer_metrics["spatial_target_svd_rank"])

    losses = jnp.stack(losses)
    num_windows = jnp.stack(num_windows)
    window_areas = jnp.stack(window_areas)
    active_fractions = jnp.stack(active_fractions)
    blur_sigma_means = jnp.stack(blur_sigma_means)
    target_svd_ranks = jnp.stack(target_svd_ranks)
    selected_layers = jnp.sum(selection_mask)

    spatial_loss = jnp.where(
        selected_layers > 0,
        jnp.sum(losses * selection_mask) / selected_layers,
        jnp.array(0.0, dtype=dtype),
    )
    spatial_metrics = {
        "spatial_num_windows": _masked_mean(num_windows, selection_mask, dtype),
        "spatial_window_area": _masked_mean(window_areas, selection_mask, dtype),
        "spatial_active_fraction": _masked_mean(active_fractions, selection_mask, dtype),
        "spatial_blur_sigma_mean": _masked_mean(blur_sigma_means, selection_mask, dtype),
        "spatial_target_svd_rank": _masked_mean(target_svd_ranks, selection_mask, dtype),
        "spatial_selected_layers": selected_layers,
    }
    return spatial_loss, spatial_metrics


def compute_aux_losses(
    activations: Any,
    timesteps: jax.Array | None = None,
    private_layer_rng: jax.Array | None = None,
    private_max_layers: int = DEFAULT_PRIVATE_MAX_LAYERS,
    private_max_pairs: int = DEFAULT_PRIVATE_MAX_PAIRS,
    compute_private_loss: bool = True,
    spatial_align_rng: jax.Array | None = None,
    spatial_align_max_layers: int = DEFAULT_SPATIAL_ALIGN_MAX_LAYERS,
    compute_spatial_loss: bool = True,
    spatial_window_size: int = 0,
    spatial_window_stride: int = 0,
    spatial_timestep_range: tuple[float, float] | None = None,
    spatial_blur_by_timestep: bool = False,
    spatial_blur_max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
    spatial_blur_schedule: str = DEFAULT_TIMESTEP_BLUR_SCHEDULE,
    spatial_blur_exp_rate: float = DEFAULT_TIMESTEP_BLUR_EXP_RATE,
    spatial_loss_type: str = DEFAULT_SPATIAL_LOSS_TYPE,
    spatial_target_svd_rank: int = DEFAULT_SPATIAL_TARGET_SVD_RANK,
    model_size: str | None = None,
) -> dict[str, jax.Array]:
    """Compute auxiliary losses for consecutive-layer deltas and spatial alignment."""
    activations = collect_activations(activations)
    normalized_activations, private = compute_private_deltas(activations)
    if spatial_loss_type == "local_gram":
        spatial_activations = normalized_activations
    elif spatial_loss_type == "global_linear_cka":
        spatial_activations = activations
    else:
        raise ValueError(
            f"Unknown spatial loss type {spatial_loss_type!r}; "
            "expected 'local_gram' or 'global_linear_cka'."
        )

    needs_timesteps = compute_spatial_loss and (
        spatial_blur_by_timestep or spatial_timestep_range is not None
    )
    if needs_timesteps:
        if timesteps is None:
            raise ValueError(
                "Timesteps are required when timestep-dependent spatial blur or timestep gating is enabled."
            )
        if timesteps.ndim != 1:
            raise ValueError(f"Expected timesteps with rank 1, got shape {timesteps.shape}")
        if timesteps.shape[0] != spatial_activations.shape[1]:
            raise ValueError(
                "Timesteps batch size and activation batch size must match, "
                f"got {timesteps.shape[0]} vs {spatial_activations.shape[1]}"
            )

    if compute_spatial_loss and spatial_blur_by_timestep:
        blur_sigmas = _timestep_dependent_blur_sigmas(
            timesteps,
            max_sigma=spatial_blur_max_sigma,
            schedule=spatial_blur_schedule,
            exp_rate=spatial_blur_exp_rate,
        )
    else:
        blur_sigmas = None

    if compute_spatial_loss and spatial_timestep_range is not None:
        spatial_tau_min, spatial_tau_max = spatial_timestep_range
        spatial_tau_min = jnp.asarray(spatial_tau_min, dtype=timesteps.dtype)
        spatial_tau_max = jnp.asarray(spatial_tau_max, dtype=timesteps.dtype)
        spatial_sample_mask = jnp.logical_and(timesteps >= spatial_tau_min, timesteps <= spatial_tau_max)
    else:
        spatial_sample_mask = None

    if compute_spatial_loss:
        spatial_loss, spatial_metrics = _mean_sampled_spatial_alignment_loss(
            spatial_activations,
            spatial_align_rng=spatial_align_rng,
            spatial_align_max_layers=spatial_align_max_layers,
            spatial_loss_type=spatial_loss_type,
            spatial_window_size=spatial_window_size,
            spatial_window_stride=spatial_window_stride,
            spatial_sample_mask=spatial_sample_mask,
            blur_sigmas=blur_sigmas,
            spatial_blur_max_sigma=spatial_blur_max_sigma,
            spatial_target_svd_rank=spatial_target_svd_rank,
            model_size=model_size,
        )
    else:
        zero = jnp.array(0.0, dtype=normalized_activations.dtype)
        spatial_loss = zero
        spatial_metrics = {
            "spatial_num_windows": zero,
            "spatial_window_area": zero,
            "spatial_active_fraction": zero,
            "spatial_blur_sigma_mean": zero,
            "spatial_target_svd_rank": zero,
            "spatial_selected_layers": zero,
        }

    if compute_private_loss:
        private_selection_mask = _sample_private_layer_mask(
            private.shape[0],
            private_max_layers,
            dtype=private.dtype,
            rng=private_layer_rng,
        )
        private_loss, private_selected_pairs = _mean_masked_pairwise_cosine_squared(
            private,
            private_selection_mask,
            max_pairs=private_max_pairs,
        )
    else:
        private_selection_mask = jnp.zeros((private.shape[0],), dtype=private.dtype)
        private_loss = jnp.array(0.0, dtype=private.dtype)
        private_selected_pairs = jnp.array(0.0, dtype=private.dtype)

    target_activation = jax.lax.stop_gradient(spatial_activations[-1])
    target_norm = jnp.mean(jnp.linalg.norm(target_activation.reshape(target_activation.shape[0], -1), axis=-1))
    private_norms = jnp.linalg.norm(private.reshape(private.shape[0], private.shape[1], -1), axis=-1)
    avg_private_norm = jnp.mean(private_norms)

    pairwise_cosines = _pairwise_cosines(private)
    if pairwise_cosines.ndim == 0:
        avg_pairwise_cosine = pairwise_cosines
    else:
        avg_pairwise_cosine = jnp.mean(pairwise_cosines)

    return {
        "target_activation": target_activation,
        "private_activations": private,
        "loss_spatial": spatial_loss,
        "loss_private": private_loss,
        "spatial_metrics": spatial_metrics,
        "target_norm": target_norm,
        "avg_private_norm": avg_private_norm,
        "avg_pairwise_private_cosine": avg_pairwise_cosine,
        "private_selected_layers": jnp.sum(private_selection_mask),
        "private_selected_pairs": private_selected_pairs,
    }
