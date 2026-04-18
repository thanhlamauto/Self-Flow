"""Shared-subspace activation decomposition losses for DiT flow matching."""

from __future__ import annotations

import math
from typing import Any, Callable

import jax
import jax.image
import jax.numpy as jnp

DEFAULT_SPATIAL_WINDOW_SIZE = 3
DEFAULT_SPATIAL_WINDOW_STRIDE = 1
DEFAULT_ACTIVATION_WINDOW_SIZE = 4
DEFAULT_SHARED_SUBSPACE_RANK = 6
DEFAULT_PRIVATE_MAX_PAIRS = DEFAULT_ACTIVATION_WINDOW_SIZE
DEFAULT_COARSE_TARGET_SIZE = 4
DEFAULT_PATCH_SIZE = 2
DEFAULT_LATENT_CHANNELS = 4


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


def token_layer_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """LayerNorm over the hidden dimension without learned affine parameters."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps)


def sample_activation_window(
    activations: jax.Array,
    *,
    rng: jax.Array | None,
    window_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Sample ``m`` consecutive layers from ``[L, B, N, D]`` activations."""
    total_layers = activations.shape[0]
    if total_layers <= 0:
        raise ValueError("Expected at least one activation layer.")

    selected_size = min(max(int(window_size), 1), total_layers)
    max_start = total_layers - selected_size
    if max_start > 0:
        if rng is None:
            raise ValueError("An RNG key is required when sampling a layer window.")
        start = jax.random.randint(rng, shape=(), minval=0, maxval=max_start + 1)
    else:
        start = jnp.array(0, dtype=jnp.int32)
    window = jax.lax.dynamic_slice_in_dim(activations, start, selected_size, axis=0)
    return window, start


def sample_activation_subset(
    activations: jax.Array,
    *,
    rng: jax.Array | None,
    sample_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Sample ``k`` distinct layers from ``[L, B, N, D]`` activations."""
    total_layers = activations.shape[0]
    if total_layers <= 0:
        raise ValueError("Expected at least one activation layer.")

    selected_size = min(max(int(sample_size), 1), total_layers)
    if selected_size < total_layers:
        if rng is None:
            raise ValueError("An RNG key is required when sampling random layers.")
        layer_indices = jnp.sort(jax.random.permutation(rng, total_layers)[:selected_size])
    else:
        layer_indices = jnp.arange(total_layers, dtype=jnp.int32)
    subset = jnp.take(activations, layer_indices, axis=0)
    return subset, layer_indices


def sample_layer_pairs(
    num_layers: int,
    *,
    rng: jax.Array | None,
    max_pairs: int,
) -> tuple[jax.Array, int]:
    """Sample distinct layer pairs ``(i, j)`` with ``i < j``."""
    if num_layers < 2:
        return jnp.zeros((0, 2), dtype=jnp.int32), 0

    upper_i, upper_j = jnp.triu_indices(num_layers, k=1)
    num_available = int(upper_i.shape[0])
    selected_count = num_available if max_pairs <= 0 else min(max(int(max_pairs), 1), num_available)
    if selected_count < num_available:
        if rng is None:
            raise ValueError("An RNG key is required when sampling random layer pairs.")
        selected_indices = jnp.sort(jax.random.permutation(rng, num_available)[:selected_count])
    else:
        selected_indices = jnp.arange(num_available, dtype=jnp.int32)
    return jnp.stack([upper_i[selected_indices], upper_j[selected_indices]], axis=-1), selected_count


def extract_sliding_windows(
    grid: jax.Array,
    window_size: int,
    stride: int = DEFAULT_SPATIAL_WINDOW_STRIDE,
) -> jax.Array:
    """Extract all valid sliding windows as ``[B, num_windows, window_area, C]``."""
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


def tokens_to_grid(tokens: jax.Array) -> jax.Array:
    """Reshape ``[B, N, C]`` tokens into a square NHWC grid."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor with rank 3, got shape {tokens.shape}")
    batch, num_tokens, channels = tokens.shape
    token_grid = math.isqrt(num_tokens)
    if token_grid * token_grid != num_tokens:
        raise ValueError(f"Expected a square token grid, got N={num_tokens}")
    return tokens.reshape(batch, token_grid, token_grid, channels)


def patchified_latents_to_grid(
    latents: jax.Array,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    latent_channels: int = DEFAULT_LATENT_CHANNELS,
) -> jax.Array:
    """Convert patchified latent tokens ``[B, N, p*p*C]`` into an NHWC latent grid."""
    if latents.ndim != 3:
        raise ValueError(f"Expected patchified latents with rank 3, got shape {latents.shape}")
    batch, num_tokens, patch_dim = latents.shape
    token_grid = math.isqrt(num_tokens)
    if token_grid * token_grid != num_tokens:
        raise ValueError(f"Expected a square token grid, got N={num_tokens}")

    expected_patch_dim = patch_size * patch_size * latent_channels
    if patch_dim != expected_patch_dim:
        raise ValueError(
            "Patchified latent dimension does not match the configured patch layout, "
            f"got {patch_dim} vs expected {expected_patch_dim}"
        )

    grid = latents.reshape(
        batch,
        token_grid,
        token_grid,
        patch_size,
        patch_size,
        latent_channels,
    )
    grid = jnp.transpose(grid, (0, 1, 3, 2, 4, 5))
    return grid.reshape(batch, token_grid * patch_size, token_grid * patch_size, latent_channels)


def average_pool_to_size(grid: jax.Array, out_size: int) -> jax.Array:
    """Average-pool a square NHWC grid to ``out_size x out_size``."""
    if grid.ndim != 4:
        raise ValueError(f"Expected an NHWC grid, got shape {grid.shape}")
    batch, height, width, channels = grid.shape
    if height != width:
        raise ValueError(f"Expected a square grid, got {(height, width)}")
    if height % out_size != 0:
        raise ValueError(f"Grid size {height} is not divisible by out_size={out_size}")

    factor = height // out_size
    pooled = grid.reshape(batch, out_size, factor, out_size, factor, channels)
    return jnp.mean(pooled, axis=(2, 4))


def latent_grid_to_patchified_tokens(
    grid: jax.Array,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> jax.Array:
    """Convert an NHWC latent grid into patchified latent tokens."""
    if grid.ndim != 4:
        raise ValueError(f"Expected an NHWC grid, got shape {grid.shape}")
    batch, height, width, channels = grid.shape
    if height != width:
        raise ValueError(f"Expected a square grid, got {(height, width)}")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Grid shape {(height, width)} is not divisible by patch_size={patch_size}"
        )

    token_h = height // patch_size
    token_w = width // patch_size
    tokens = grid.reshape(
        batch,
        token_h,
        patch_size,
        token_w,
        patch_size,
        channels,
    )
    tokens = jnp.transpose(tokens, (0, 1, 3, 2, 4, 5))
    return tokens.reshape(batch, token_h * token_w, patch_size * patch_size * channels)


def build_coarse_spatial_target(
    clean_latents: jax.Array,
    timesteps: jax.Array,
    *,
    coarse_target_size: int = DEFAULT_COARSE_TARGET_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    latent_channels: int = DEFAULT_LATENT_CHANNELS,
) -> jax.Array:
    """Construct the coarse-to-clean spatial target ``x_tgt(tau)`` on the 32x32 latent grid."""
    clean_grid = patchified_latents_to_grid(
        clean_latents,
        patch_size=patch_size,
        latent_channels=latent_channels,
    ).astype(jnp.float32)
    coarse_grid = average_pool_to_size(clean_grid, coarse_target_size)
    low_grid = jax.image.resize(
        coarse_grid,
        shape=clean_grid.shape,
        method="bilinear",
    )
    tau = timesteps.astype(clean_grid.dtype)[:, None, None, None]
    return (1.0 - tau) * low_grid + tau * clean_grid

def window_gram_upper_triangle(
    window_tokens: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute upper-triangular Gram entries and exact symmetry weights.

    For a symmetric Gram matrix ``G in R^{M x M}``, the mean absolute mismatch over
    all ``M^2`` entries can be written using only the upper triangle:

    ``sum_{u<=v} w_{uv} |G_uv - T_uv| / M^2``

    with ``w_{uv}=1`` on the diagonal and ``w_{uv}=2`` off the diagonal.
    This avoids materializing the redundant lower triangle while preserving the
    original objective exactly.
    """
    if window_tokens.ndim != 4:
        raise ValueError(
            f"Expected window tokens with rank 4, got shape {window_tokens.shape}"
        )
    window_area = window_tokens.shape[2]
    feature_dim = window_tokens.shape[3]
    upper_i, upper_j = jnp.triu_indices(window_area)
    gram_entries = jnp.sum(
        window_tokens[..., upper_i, :] * window_tokens[..., upper_j, :],
        axis=-1,
    ) / jnp.asarray(feature_dim, dtype=window_tokens.dtype)
    pair_weights = jnp.where(upper_i == upper_j, 1.0, 2.0).astype(window_tokens.dtype)
    return gram_entries, pair_weights


def local_window_gram_loss(
    feature_grid: jax.Array,
    target_grid: jax.Array,
    *,
    window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    stride: int = DEFAULT_SPATIAL_WINDOW_STRIDE,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compare local Gram structure between feature and target spatial grids."""
    if feature_grid.ndim != 4 or target_grid.ndim != 4:
        raise ValueError(
            "Expected NHWC spatial grids for local Gram loss, "
            f"got {feature_grid.shape} and {target_grid.shape}"
        )
    if feature_grid.shape[:3] != target_grid.shape[:3]:
        raise ValueError(
            "Feature and target grids must match in batch/height/width, "
            f"got {feature_grid.shape[:3]} vs {target_grid.shape[:3]}"
        )

    feature_windows = extract_sliding_windows(feature_grid, window_size, stride=stride)
    target_windows = extract_sliding_windows(target_grid, window_size, stride=stride)
    feature_gram_entries, pair_weights = window_gram_upper_triangle(feature_windows)
    target_gram_entries, _ = window_gram_upper_triangle(jax.lax.stop_gradient(target_windows))

    window_area = feature_windows.shape[2]
    weighted_pair_losses = (
        jnp.abs(feature_gram_entries - target_gram_entries)
        * pair_weights[None, None, :]
    )
    window_losses = jnp.sum(weighted_pair_losses, axis=-1) / (window_area * window_area)
    example_losses = jnp.mean(window_losses, axis=-1)
    spatial_loss = jnp.mean(example_losses)
    spatial_metrics = {
        "spatial_num_windows": jnp.asarray(feature_windows.shape[1], dtype=feature_grid.dtype),
        "spatial_window_area": jnp.asarray(feature_windows.shape[2], dtype=feature_grid.dtype),
    }
    return spatial_loss, spatial_metrics


def shared_subspace_basis(
    mean_activations: jax.Array,
    *,
    rank: int,
    stopgrad_basis: bool = True,
) -> tuple[jax.Array, int]:
    """Compute the right-singular-vector basis ``Q_t`` for each batch item."""
    if mean_activations.ndim != 3:
        raise ValueError(
            f"Expected mean activations with rank 3, got shape {mean_activations.shape}"
        )
    effective_rank = min(max(int(rank), 1), mean_activations.shape[-2], mean_activations.shape[-1])
    svd_input = mean_activations
    if stopgrad_basis:
        svd_input = jax.lax.stop_gradient(svd_input)
    _, _, vh = jnp.linalg.svd(svd_input.astype(jnp.float32), full_matrices=False)
    basis = jnp.swapaxes(vh[:, :effective_rank, :], 1, 2).astype(mean_activations.dtype)
    if stopgrad_basis:
        basis = jax.lax.stop_gradient(basis)
    return basis, effective_rank


def project_onto_basis(
    activations: jax.Array,
    basis: jax.Array,
    *,
    stopgrad_basis: bool = True,
) -> jax.Array:
    """Apply ``A @ (Q Q^T)`` without explicitly materializing the dense projector."""
    projector_basis = basis
    if stopgrad_basis:
        projector_basis = jax.lax.stop_gradient(projector_basis)
    projector_basis_t = jnp.swapaxes(projector_basis, 1, 2)
    if stopgrad_basis:
        projector_basis_t = jax.lax.stop_gradient(projector_basis_t)
    coeffs = jnp.einsum("lbnd,bdk->lbnk", activations, projector_basis)
    return jnp.einsum("lbnk,bkd->lbnd", coeffs, projector_basis_t)


def gap_pairwise_cosine_squared(
    private_activations: jax.Array,
    eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the cosine-squared diversity loss after GAP over the token dimension."""
    if private_activations.ndim != 4:
        raise ValueError(
            "Expected private activations with shape [L, B, N, D], "
            f"got {private_activations.shape}"
        )
    num_layers = private_activations.shape[0]
    if num_layers < 2:
        zero = jnp.array(0.0, dtype=private_activations.dtype)
        return zero, zero, zero

    pooled = jnp.mean(private_activations, axis=2)
    norms = jnp.linalg.norm(pooled, axis=-1, keepdims=True)
    normalized = pooled / jnp.maximum(norms, eps)
    cosine_matrix = jnp.einsum("lbd,mbd->blm", normalized, normalized)
    upper_i, upper_j = jnp.triu_indices(num_layers, k=1)
    pairwise_cosines = cosine_matrix[:, upper_i, upper_j]
    diversity_loss = jnp.mean(jnp.square(pairwise_cosines))
    avg_pairwise_cosine = jnp.mean(pairwise_cosines)
    avg_private_norm = jnp.mean(jnp.squeeze(norms, axis=-1))
    return diversity_loss, avg_pairwise_cosine, avg_private_norm


def shuffled_gap_pairwise_cosine_squared(
    left_private_activations: jax.Array,
    right_private_activations: jax.Array,
    *,
    rng: jax.Array | None,
    eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compare random layer pairs after GAP, with a batch derangement on the right side."""
    if left_private_activations.shape != right_private_activations.shape:
        raise ValueError(
            "Left/right private activations must have matching shapes, "
            f"got {left_private_activations.shape} vs {right_private_activations.shape}"
        )
    if left_private_activations.ndim != 4:
        raise ValueError(
            "Expected private activations with shape [P, B, N, D], "
            f"got {left_private_activations.shape}"
        )

    num_pairs, batch_size = left_private_activations.shape[:2]
    if num_pairs == 0:
        zero = jnp.array(0.0, dtype=left_private_activations.dtype)
        return zero, zero, zero

    pooled_left = jnp.mean(left_private_activations, axis=2)
    pooled_right = jnp.mean(right_private_activations, axis=2)
    left_norms = jnp.linalg.norm(pooled_left, axis=-1, keepdims=True)
    right_norms = jnp.linalg.norm(pooled_right, axis=-1, keepdims=True)
    normalized_left = pooled_left / jnp.maximum(left_norms, eps)
    normalized_right = pooled_right / jnp.maximum(right_norms, eps)
    avg_private_norm = 0.5 * (
        jnp.mean(jnp.squeeze(left_norms, axis=-1))
        + jnp.mean(jnp.squeeze(right_norms, axis=-1))
    )

    if batch_size < 2:
        zero = jnp.array(0.0, dtype=left_private_activations.dtype)
        return zero, zero, avg_private_norm
    if rng is None:
        raise ValueError("An RNG key is required when shuffling private layer pairs.")

    pair_rngs = jax.random.split(rng, num_pairs)
    shifts = jax.vmap(
        lambda key: jax.random.randint(key, shape=(), minval=1, maxval=batch_size)
    )(pair_rngs)
    batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
    shuffled_indices = (batch_indices[None, :] + shifts[:, None]) % batch_size
    shuffled_right = jnp.take_along_axis(
        normalized_right,
        shuffled_indices[..., None],
        axis=1,
    )
    pairwise_cosines = jnp.sum(normalized_left * shuffled_right, axis=-1)
    diversity_loss = jnp.mean(jnp.square(pairwise_cosines))
    avg_pairwise_cosine = jnp.mean(pairwise_cosines)
    return diversity_loss, avg_pairwise_cosine, avg_private_norm


def compute_aux_losses(
    activations: Any,
    spatial_target: jax.Array,
    timesteps: jax.Array,
    *,
    layer_window_rng: jax.Array | None,
    layer_window_size: int = DEFAULT_ACTIVATION_WINDOW_SIZE,
    shared_subspace_rank: int = DEFAULT_SHARED_SUBSPACE_RANK,
    private_max_pairs: int = DEFAULT_PRIVATE_MAX_PAIRS,
    shared_subspace_stopgrad: bool = True,
    compute_spatial_loss: bool = True,
    compute_diversity_loss: bool = True,
    spatial_window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    spatial_window_stride: int = DEFAULT_SPATIAL_WINDOW_STRIDE,
    coarse_target_size: int = DEFAULT_COARSE_TARGET_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    latent_channels: int = DEFAULT_LATENT_CHANNELS,
    common_spatial_project_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    spatial_target_project_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> dict[str, jax.Array]:
    """Compute auxiliary losses for the shared-subspace decomposition objective."""
    activations = collect_activations(activations)
    if spatial_target.ndim != 3:
        raise ValueError(f"Expected spatial target with rank 3, got shape {spatial_target.shape}")
    if timesteps.ndim != 1:
        raise ValueError(f"Expected timesteps with rank 1, got shape {timesteps.shape}")
    if spatial_target.shape[0] != timesteps.shape[0]:
        raise ValueError(
            "Spatial target batch size and timestep batch size must match, "
            f"got {spatial_target.shape[0]} vs {timesteps.shape[0]}"
        )

    common_window_rng = layer_window_rng
    private_pair_rng = layer_window_rng
    private_shuffle_rng = layer_window_rng
    if layer_window_rng is not None and compute_diversity_loss:
        common_window_rng, private_pair_rng, private_shuffle_rng = jax.random.split(
            layer_window_rng, 3
        )

    windowed_activations, window_start = sample_activation_window(
        activations,
        rng=common_window_rng,
        window_size=layer_window_size,
    )
    normalized_window = token_layer_norm(windowed_activations)
    mean_activations = jnp.mean(normalized_window, axis=0)
    basis, effective_rank = shared_subspace_basis(
        mean_activations,
        rank=shared_subspace_rank,
        stopgrad_basis=shared_subspace_stopgrad,
    )

    common = project_onto_basis(
        normalized_window,
        basis,
        stopgrad_basis=shared_subspace_stopgrad,
    )
    common_mean = jnp.mean(common, axis=0)

    zero = jnp.array(0.0, dtype=normalized_window.dtype)
    if compute_spatial_loss:
        if common_spatial_project_fn is None:
            raise ValueError("A spatial projector is required when compute_spatial_loss=True.")
        if spatial_target_project_fn is None:
            raise ValueError("A target projector is required when compute_spatial_loss=True.")

        projected_common = common_spatial_project_fn(common_mean, timesteps)
        feature_grid = tokens_to_grid(projected_common).astype(jnp.float32)
        target_grid = build_coarse_spatial_target(
            spatial_target,
            timesteps,
            coarse_target_size=coarse_target_size,
            patch_size=patch_size,
            latent_channels=latent_channels,
        )
        target_tokens = latent_grid_to_patchified_tokens(
            target_grid,
            patch_size=patch_size,
        )
        target_features = spatial_target_project_fn(target_tokens)
        target_features = token_layer_norm(target_features)
        target_feature_grid = tokens_to_grid(target_features).astype(jnp.float32)
        spatial_loss, spatial_metrics = local_window_gram_loss(
            feature_grid,
            jax.lax.stop_gradient(target_feature_grid),
            window_size=spatial_window_size,
            stride=spatial_window_stride,
        )
        spatial_loss = spatial_loss.astype(normalized_window.dtype)
        spatial_metrics = {
            key: value.astype(normalized_window.dtype)
            for key, value in spatial_metrics.items()
        }
    else:
        spatial_loss = zero
        spatial_metrics = {
            "spatial_num_windows": zero,
            "spatial_window_area": zero,
        }

    if compute_diversity_loss:
        layer_pairs, _ = sample_layer_pairs(
            activations.shape[0],
            rng=private_pair_rng,
            max_pairs=private_max_pairs,
        )
        left_private_layers = token_layer_norm(jnp.take(activations, layer_pairs[:, 0], axis=0))
        right_private_layers = token_layer_norm(jnp.take(activations, layer_pairs[:, 1], axis=0))
        left_private_residual = left_private_layers - common_mean[None, ...]
        right_private_residual = right_private_layers - common_mean[None, ...]
        private_loss, avg_pairwise_cosine, avg_private_norm = shuffled_gap_pairwise_cosine_squared(
            left_private_residual,
            right_private_residual,
            rng=private_shuffle_rng,
        )
        private_residual = jnp.concatenate([left_private_residual, right_private_residual], axis=0)
    else:
        private_loss = zero
        avg_pairwise_cosine = zero
        avg_private_norm = zero
        private_residual = normalized_window - common

    common_norm = jnp.mean(jnp.linalg.norm(common_mean.reshape(common_mean.shape[0], -1), axis=-1))

    return {
        "common_activation": common_mean,
        "private_activations": private_residual,
        "loss_spatial": spatial_loss,
        "loss_private": private_loss,
        "spatial_metrics": spatial_metrics,
        "norm_common": common_norm,
        "avg_private_norm": avg_private_norm,
        "avg_pairwise_private_cosine": avg_pairwise_cosine,
        "layer_window_start": window_start.astype(normalized_window.dtype),
        "layer_window_size": jnp.asarray(windowed_activations.shape[0], dtype=normalized_window.dtype),
        "shared_subspace_rank": jnp.asarray(effective_rank, dtype=normalized_window.dtype),
    }
