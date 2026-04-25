"""Helper utilities for common/private activation decomposition in DiT."""

from __future__ import annotations

import math
from typing import Any, Callable

import jax
import jax.numpy as jnp

DEFAULT_SPATIAL_WINDOW_SIZE = 3
DEFAULT_SPATIAL_WINDOW_STRIDE = 1
DEFAULT_MAX_TIMESTEP_BLUR_SIGMA = 3.0
DEFAULT_TIMESTEP_BLUR_SCHEDULE = "linear"
DEFAULT_TIMESTEP_BLUR_EXP_RATE = 5.0


def _layer_logit_normal_weights(
    num_layers: int,
    center_layer: float | jax.Array,
    logit_sigma: float | jax.Array,
    dtype: jnp.dtype,
) -> jax.Array:
    """Nonnegative weights over layer index, peak near ``center_layer`` (Gaussian on log-depth).

    For layer ``i`` we map ``u_i = (i + 0.5) / L`` in ``(0, 1)``, set ``z_i = logit(u_i)``, and
    use weights ``w_i ∝ exp(-0.5 * ((z_i - mu) / sigma)^2)`` with ``mu = logit(u_c)`` for the
    same mapping of ``center_layer``. We normalize ``w`` to sum to ``1``.

    We intentionally **do not** multiply by the logit-normal Jacobian ``1/(u(1-u))``. The full
    PDF strongly up-weights boundary layers (small ``u`` or ``u`` near ``1``), which makes
    gradients through ``A_common`` ill-conditioned and breaks the intuitive limit
    ``sigma → ∞`` (uniform weights → same as ``mean``). This RBF on ``z`` is stable and recovers
    uniform weights (hence ``mean`` aggregation) as ``sigma`` grows.
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
    log_w = -0.5 * jnp.square((z - mu) / s)
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


def _compute_common_from_normalized(
    activations: Any,
    *,
    agg: str = "mean",
    logit_normal_center_layer: float = 0.0,
    logit_normal_sigma: float = 1.0,
) -> jax.Array:
    num_layers = activations.shape[0]
    if agg == "mean":
        return jnp.mean(activations, axis=0)
    elif agg == "logit_normal":
        w = _layer_logit_normal_weights(
            num_layers,
            logit_normal_center_layer,
            logit_normal_sigma,
            activations.dtype,
        )
        return jnp.tensordot(w, activations, axes=(0, 0))
    else:
        raise ValueError(f"Unknown common aggregation {agg!r}; expected 'mean' or 'logit_normal'.")


def compute_common_private(
    activations: Any,
    *,
    agg: str = "mean",
    logit_normal_center_layer: float = 0.0,
    logit_normal_sigma: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute differentiable common activation and private residuals.

    ``agg="mean"`` uses ``A_common = mean_i A_i`` (default). ``agg="logit_normal"`` uses
    ``A_common = sum_i w_i A_i`` with ``w`` a normalized Gaussian bump on ``logit((i+0.5)/L)``
    centered at the same mapping of ``logit_normal_center_layer`` (0-based; fractional ``k``
    allowed). ``logit_normal_sigma`` is the Gaussian std on that logit scale (larger =>
    flatter weights, converging to ``mean`` when ``sigma`` is large).
    """
    activations = collect_activations(activations)
    activations = _normalize_channels(activations)
    common = _compute_common_from_normalized(
        activations,
        agg=agg,
        logit_normal_center_layer=logit_normal_center_layer,
        logit_normal_sigma=logit_normal_sigma,
    )
    common_anchor = jax.lax.stop_gradient(common)
    private = activations - common_anchor[None, ...]
    return common, common_anchor, private


def compute_common(
    activations: Any,
    *,
    agg: str = "mean",
    logit_normal_center_layer: float = 0.0,
    logit_normal_sigma: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Compute common activation without materializing private residuals."""
    activations = collect_activations(activations)
    activations = _normalize_channels(activations)
    common = _compute_common_from_normalized(
        activations,
        agg=agg,
        logit_normal_center_layer=logit_normal_center_layer,
        logit_normal_sigma=logit_normal_sigma,
    )
    return common, jax.lax.stop_gradient(common)


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


def _timestep_dependent_blur_sigmas(
    timesteps: jax.Array,
    max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
    schedule: str = DEFAULT_TIMESTEP_BLUR_SCHEDULE,
    exp_rate: float = DEFAULT_TIMESTEP_BLUR_EXP_RATE,
) -> jax.Array:
    """Map ``tau`` in ``[0, 1]`` to blur sigma with selectable decay curvature."""
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
    sample_mask: jax.Array | None = None,
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
    example_losses = jnp.mean(window_losses, axis=-1)
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
        "spatial_num_windows": jnp.asarray(feature_windows.shape[1], dtype=feature_tokens.dtype),
        "spatial_window_area": jnp.asarray(feature_windows.shape[2], dtype=feature_tokens.dtype),
        "spatial_active_fraction": active_fraction,
        "spatial_blur_sigma_mean": blur_sigma_mean,
    }
    return spatial_loss, spatial_metrics


def _private_cosine_matrix(private: jax.Array, eps: float = 1e-8) -> jax.Array:
    flattened = private.reshape(private.shape[0], -1)
    norms = jnp.linalg.norm(flattened, axis=-1, keepdims=True)
    normalized = flattened / jnp.maximum(norms, eps)
    return normalized @ normalized.T


def _pairwise_cosines(private: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Return cosine similarities for all layer pairs `i < j`."""
    num_layers = private.shape[0]
    if num_layers < 2:
        return jnp.array(0.0, dtype=private.dtype)

    cosine_matrix = _private_cosine_matrix(private, eps=eps)
    upper_indices = jnp.triu_indices(num_layers, k=1)
    return cosine_matrix[upper_indices]


def _select_private_pair_indices(
    num_layers: int,
    *,
    rng: jax.Array | None = None,
    max_pairs: int = 0,
    pair_mode: str = "first_pairs",
    min_pair_delta: int = 1,
    fixed_pair_indices: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return selected private-layer pair indices in lexicographic pair order."""
    if pair_mode not in ("first_pairs", "random_pairs", "fixed_random_pairs"):
        raise ValueError(
            f"Unknown private pair mode {pair_mode!r}; expected 'first_pairs', 'random_pairs', "
            "or 'fixed_random_pairs'."
        )
    if num_layers < 2:
        empty = jnp.asarray((), dtype=jnp.int32)
        return empty, empty

    min_pair_delta = max(1, int(min_pair_delta))
    if pair_mode == "fixed_random_pairs":
        if fixed_pair_indices is None:
            raise ValueError("fixed_random_pairs requires fixed_pair_indices.")
        pairs = fixed_pair_indices.astype(jnp.int32)
        pair_a = pairs[:, 0]
        pair_b = pairs[:, 1]
        return pair_a, pair_b

    pair_a, pair_b = jnp.triu_indices(num_layers, k=1)
    valid = (pair_b - pair_a) >= min_pair_delta
    pair_a = pair_a[valid]
    pair_b = pair_b[valid]
    total_pairs = pair_a.shape[0]
    if max_pairs and max_pairs > 0 and total_pairs > max_pairs:
        if pair_mode == "random_pairs":
            if rng is None:
                raise ValueError("An RNG key is required when sampling private-layer pairs.")
            selected = jax.random.permutation(rng, total_pairs)[:max_pairs]
            return pair_a[selected], pair_b[selected]
        return pair_a[:max_pairs], pair_b[:max_pairs]
    return pair_a, pair_b


def _private_pairwise_cosines_for_indices(
    private: jax.Array,
    pair_a: jax.Array,
    pair_b: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Return cosine similarities for preselected private-layer pairs."""
    if pair_a.shape[0] == 0:
        return jnp.array(0.0, dtype=private.dtype)

    cosine_matrix = _private_cosine_matrix(private, eps=eps)
    return cosine_matrix[pair_a, pair_b]


def _private_pairwise_loss_and_metric(
    private: jax.Array,
    eps: float = 1e-8,
    rng: jax.Array | None = None,
    max_pairs: int = 0,
    pair_mode: str = "first_pairs",
    min_pair_delta: int = 1,
    fixed_pair_indices: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Average squared cosine loss and mean cosine over selected private-layer pairs."""
    pair_a, pair_b = _select_private_pair_indices(
        private.shape[0],
        rng=rng,
        max_pairs=max_pairs,
        pair_mode=pair_mode,
        min_pair_delta=min_pair_delta,
        fixed_pair_indices=fixed_pair_indices,
    )
    pairwise_cosines = _private_pairwise_cosines_for_indices(private, pair_a, pair_b, eps=eps)
    if pairwise_cosines.ndim == 0:
        return pairwise_cosines, pairwise_cosines
    return jnp.mean(jnp.square(pairwise_cosines)), jnp.mean(pairwise_cosines)


def _common_private_cosines(
    common: jax.Array,
    private: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Return cosine similarities between ``A_common`` and each private layer ``B_i``."""
    if common.ndim != 3:
        raise ValueError(f"Expected common activation with rank 3, got shape {common.shape}")
    if private.ndim != 4:
        raise ValueError(f"Expected private activations with rank 4, got shape {private.shape}")
    if private.shape[1:] != common.shape:
        raise ValueError(
            "Private activations must match common activation in batch/token/channel dims, "
            f"got {private.shape[1:]} vs {common.shape}"
        )

    common_flat = common.reshape(-1)
    private_flat = private.reshape(private.shape[0], -1)
    common_norm = common_flat / jnp.maximum(jnp.linalg.norm(common_flat), eps)
    private_norms = private_flat / jnp.maximum(
        jnp.linalg.norm(private_flat, axis=-1, keepdims=True),
        eps,
    )
    return private_norms @ common_norm


def _mean_common_private_cosine_squared(
    common: jax.Array,
    private: jax.Array,
    eps: float = 1e-8,
    rng: jax.Array | None = None,
    max_layers: int = 0,
) -> jax.Array:
    """Average squared cosine similarity between ``A_common`` and all or sampled ``B_i``."""
    common_private_cosines = _common_private_cosines(common, private, eps=eps)
    if common_private_cosines.ndim == 0:
        return common_private_cosines

    if max_layers and max_layers > 0 and common_private_cosines.shape[0] > max_layers:
        if rng is None:
            raise ValueError("An RNG key is required when sampling common/private layers.")
        indices = jax.random.permutation(rng, common_private_cosines.shape[0])[:max_layers]
        common_private_cosines = common_private_cosines[indices]

    return jnp.mean(jnp.square(common_private_cosines))


def compute_aux_losses(
    activations: Any,
    spatial_target: jax.Array,
    timesteps: jax.Array | None = None,
    private_pair_rng: jax.Array | None = None,
    private_max_pairs: int = 0,
    private_pair_mode: str = "first_pairs",
    private_min_pair_delta: int = 1,
    private_fixed_pair_indices: jax.Array | None = None,
    common_private_rng: jax.Array | None = None,
    common_private_max_layers: int = 0,
    compute_spatial_loss: bool = True,
    compute_private_loss: bool = True,
    compute_common_private_loss: bool = True,
    spatial_window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    spatial_window_stride: int = DEFAULT_SPATIAL_WINDOW_STRIDE,
    spatial_timestep_range: tuple[float, float] | None = None,
    spatial_blur_by_timestep: bool = False,
    spatial_blur_max_sigma: float = DEFAULT_MAX_TIMESTEP_BLUR_SIGMA,
    spatial_blur_schedule: str = DEFAULT_TIMESTEP_BLUR_SCHEDULE,
    spatial_blur_exp_rate: float = DEFAULT_TIMESTEP_BLUR_EXP_RATE,
    common_agg: str = "mean",
    common_logit_normal_center_layer: float = 0.0,
    common_logit_normal_sigma: float = 1.0,
    common_spatial_project_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> dict[str, jax.Array]:
    """Compute auxiliary losses and logging metrics for activation decomposition."""
    activations = collect_activations(activations)
    if spatial_target.ndim != 3:
        raise ValueError(f"Expected spatial target with rank 3, got shape {spatial_target.shape}")
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
        if timesteps.shape[0] != spatial_target.shape[0]:
            raise ValueError(
                "Timesteps batch size and spatial target batch size must match, "
                f"got {timesteps.shape[0]} vs {spatial_target.shape[0]}"
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

    needs_private = compute_private_loss or compute_common_private_loss
    if needs_private:
        common, common_anchor, private = compute_common_private(
            activations,
            agg=common_agg,
            logit_normal_center_layer=common_logit_normal_center_layer,
            logit_normal_sigma=common_logit_normal_sigma,
        )
    else:
        common, common_anchor = compute_common(
            activations,
            agg=common_agg,
            logit_normal_center_layer=common_logit_normal_center_layer,
            logit_normal_sigma=common_logit_normal_sigma,
        )
        private = None
    if compute_spatial_loss:
        spatial_common = (
            common_spatial_project_fn(common)
            if common_spatial_project_fn is not None
            else common
        )
        spatial_loss, spatial_metrics = local_window_gram_loss(
            spatial_common,
            spatial_target,
            window_size=spatial_window_size,
            stride=spatial_window_stride,
            sample_mask=spatial_sample_mask,
            target_blur_sigmas=blur_sigmas,
            target_blur_max_sigma=spatial_blur_max_sigma,
        )
    else:
        zero = jnp.array(0.0, dtype=common.dtype)
        spatial_loss = zero
        spatial_metrics = {
            "spatial_num_windows": zero,
            "spatial_window_area": zero,
            "spatial_active_fraction": zero,
            "spatial_blur_sigma_mean": zero,
        }

    if compute_private_loss:
        if private is None:
            raise ValueError("Private activations are required when computing private loss.")
        private_loss, avg_pairwise_cosine = _private_pairwise_loss_and_metric(
            private,
            rng=private_pair_rng,
            max_pairs=private_max_pairs,
            pair_mode=private_pair_mode,
            min_pair_delta=private_min_pair_delta,
            fixed_pair_indices=private_fixed_pair_indices,
        )
    else:
        private_loss = jnp.array(0.0, dtype=common.dtype)
        avg_pairwise_cosine = jnp.array(0.0, dtype=common.dtype)
    if compute_common_private_loss:
        if private is None:
            raise ValueError("Private activations are required when computing common/private loss.")
        common_private_loss = _mean_common_private_cosine_squared(
            common,
            private,
            rng=common_private_rng,
            max_layers=common_private_max_layers,
        )
        common_private_cosines = _common_private_cosines(common, private)
        if common_private_cosines.ndim == 0:
            avg_common_private_cosine = common_private_cosines
        else:
            avg_common_private_cosine = jnp.mean(common_private_cosines)
    else:
        common_private_loss = jnp.array(0.0, dtype=common.dtype)
        avg_common_private_cosine = jnp.array(0.0, dtype=common.dtype)

    common_norm = jnp.mean(jnp.linalg.norm(common.reshape(common.shape[0], -1), axis=-1))
    if needs_private:
        if private is None:
            raise ValueError("Private activations are required when logging private norms.")
        private_norms = jnp.linalg.norm(private.reshape(private.shape[0], private.shape[1], -1), axis=-1)
        avg_private_norm = jnp.mean(private_norms)
    else:
        avg_private_norm = jnp.array(0.0, dtype=common.dtype)

    return {
        "common_activation": common,
        "private_activations": private if private is not None else jnp.array(0.0, dtype=common.dtype),
        "loss_spatial": spatial_loss,
        "loss_private": private_loss,
        "loss_common_private": common_private_loss,
        "spatial_metrics": spatial_metrics,
        "norm_common": common_norm,
        "avg_private_norm": avg_private_norm,
        "avg_pairwise_private_cosine": avg_pairwise_cosine,
        "avg_common_private_cosine": avg_common_private_cosine,
    }
