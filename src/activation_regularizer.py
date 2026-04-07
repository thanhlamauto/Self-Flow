"""Activation-only regularizer helpers for DiT training.

This module has two distinct sections:

1. **Legacy cosine-decomposition regularizer** (``--regularizer-*`` flags).
   These functions are preserved unchanged for backward compatibility.

2. **CTAE-style explicit-slot regularizer** (``--ctae-*`` flags).
   New helpers that implement shared/private routing losses inspired by
   Cross-modal Tied AutoEncoder (CTAE) ideas.  The two sections are
   entirely independent; enabling one does not affect the other.
"""

from typing import Sequence
import jax
import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════════════
# Shared numerical helpers (used by both legacy and CTAE paths)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_l2_normalize(x: jax.Array, eps: float) -> jax.Array:
    x = x.astype(jnp.float32)
    eps_arr = jnp.asarray(eps, dtype=x.dtype)
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps_arr)


def safe_cosine_similarity(x: jax.Array, y: jax.Array, eps: float) -> jax.Array:
    """Cosine similarity with explicit norm clamping for stability."""
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    eps_arr = jnp.asarray(eps, dtype=x.dtype)
    numerator = jnp.sum(x * y, axis=-1)
    x_norm = jnp.maximum(jnp.linalg.norm(x, axis=-1), eps_arr)
    y_norm = jnp.maximum(jnp.linalg.norm(y, axis=-1), eps_arr)
    sim = numerator / (x_norm * y_norm)
    return jnp.clip(sim, -1.0, 1.0)


def normalize_hidden(x: jax.Array, eps: float) -> jax.Array:
    """Layer-normalize then L2-normalize hidden states over the feature axis."""
    x = x.astype(jnp.float32)
    eps_arr = jnp.asarray(eps, dtype=x.dtype)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    variance = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    layer_normed = centered * jax.lax.rsqrt(variance + eps_arr)
    return _safe_l2_normalize(layer_normed, eps)


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy cosine-decomposition regularizer  (--regularizer-* flags)
# ═══════════════════════════════════════════════════════════════════════════════

def build_layer_pairs(selected_layers: Sequence[int], pair_stride: int) -> tuple[tuple[int, int], ...]:
    """Create adjacent pairs while stepping through the layer list by pair_stride."""
    if pair_stride <= 0:
        raise ValueError("pair_stride must be greater than 0")

    layers = tuple(int(layer) for layer in selected_layers)
    return tuple(
        (layers[idx], layers[idx + 1])
        for idx in range(0, len(layers) - 1, pair_stride)
    )


def decompose_pair(
    y1: jax.Array,
    y2: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Project each layer onto the counterpart direction, then take the residual."""
    yhat1 = normalize_hidden(y1, eps)
    yhat2 = normalize_hidden(y2, eps)
    a1 = jnp.sum(yhat1 * yhat2, axis=-1, keepdims=True) * yhat2
    a2 = jnp.sum(yhat2 * yhat1, axis=-1, keepdims=True) * yhat1
    b1 = yhat1 - a1
    b2 = yhat2 - a2
    return a1, a2, b1, b2


def compute_pair_losses(y1: jax.Array, y2: jax.Array, eps: float) -> dict[str, jax.Array]:
    """Compute shared, private, and cross-separation losses for one hidden-state pair."""
    a1, a2, b1, b2 = decompose_pair(y1, y2, eps)

    shared_cos = safe_cosine_similarity(a1, a2, eps)
    private_cos = safe_cosine_similarity(b1, b2, eps)
    # Cross-layer separation avoids the trivial zero loss that arises when
    # comparing a layer's shared component against its own residual.
    sep1_cos = safe_cosine_similarity(a1, b2, eps)
    sep2_cos = safe_cosine_similarity(a2, b1, eps)

    return {
        "loss_shared": jnp.mean(1.0 - shared_cos),
        "loss_private": jnp.mean(jnp.square(private_cos)),
        "loss_sep": 0.5 * (
            jnp.mean(jnp.square(sep1_cos)) +
            jnp.mean(jnp.square(sep2_cos))
        ),
    }


def zero_regularizer_metrics(dtype=jnp.float32) -> dict[str, jax.Array]:
    """Return zero-valued logging scalars for the disabled path."""
    zero = jnp.array(0.0, dtype=dtype)
    return {
        "loss_reg": zero,
        "loss_shared": zero,
        "loss_private": zero,
        "loss_sep": zero,
        "num_pairs_used": zero,
    }


def regularizer_is_active(
    global_step: jax.Array,
    *,
    enabled: bool,
    start_step: int,
    end_step: int | None,
    apply_every: int,
    num_pairs: int,
) -> jax.Array:
    """Return whether the activation regularizer should run on this step."""
    if not enabled or num_pairs <= 0:
        return jnp.asarray(False)

    step = jnp.asarray(global_step)
    active = step >= jnp.asarray(start_step, dtype=step.dtype)
    if end_step is not None:
        active = jnp.logical_and(active, step <= jnp.asarray(end_step, dtype=step.dtype))
    if apply_every > 1:
        delta = step - jnp.asarray(start_step, dtype=step.dtype)
        active = jnp.logical_and(active, jnp.equal(jnp.mod(delta, apply_every), 0))
    return active


def compute_regularizer_metrics(
    hidden_states: jax.Array | None,
    *,
    selected_layers: Sequence[int],
    layer_pairs: Sequence[tuple[int, int]],
    lambda_shared: float,
    lambda_private: float,
    lambda_sep: float,
    eps: float,
) -> dict[str, jax.Array]:
    """Aggregate pairwise activation losses across the configured layer pairs."""
    metrics = zero_regularizer_metrics()
    if hidden_states is None or not layer_pairs:
        return metrics

    layer_to_index = {
        int(layer): idx for idx, layer in enumerate(tuple(int(layer) for layer in selected_layers))
    }
    pair_losses = [
        compute_pair_losses(
            hidden_states[layer_to_index[layer_l]],
            hidden_states[layer_to_index[layer_k]],
            eps,
        )
        for layer_l, layer_k in layer_pairs
    ]

    shared = jnp.mean(jnp.stack([loss["loss_shared"] for loss in pair_losses], axis=0))
    private = jnp.mean(jnp.stack([loss["loss_private"] for loss in pair_losses], axis=0))
    sep = jnp.mean(jnp.stack([loss["loss_sep"] for loss in pair_losses], axis=0))
    loss_reg = (
        jnp.asarray(lambda_shared, dtype=jnp.float32) * shared +
        jnp.asarray(lambda_private, dtype=jnp.float32) * private +
        jnp.asarray(lambda_sep, dtype=jnp.float32) * sep
    )
    return {
        "loss_reg": loss_reg,
        "loss_shared": shared,
        "loss_private": private,
        "loss_sep": sep,
        "num_pairs_used": jnp.asarray(float(len(pair_losses)), dtype=jnp.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CTAE-style explicit-slot regularizer  (--ctae-* flags)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_layer_pairs(spec: str) -> list[tuple[int, int]]:
    """Parse a ``"A:B,C:D"`` string into a list of ``(int, int)`` tuples.

    This is a pure-Python helper used at config-build time (not inside JIT).
    It delegates to the canonical implementation in ``src.model`` to avoid
    duplicating validation logic.

    Args:
        spec: Comma-separated layer pairs, e.g. ``"3:6,6:9,9:12"``.

    Returns:
        List of ``(layer_a, layer_b)`` integer tuples.

    Example::

        parse_layer_pairs("3:6,6:9")  # -> [(3, 6), (6, 9)]
    """
    from src.model import parse_layer_pairs_from_string
    return parse_layer_pairs_from_string(spec)


def build_feature_slots(
    h_a: jax.Array,
    h_b: jax.Array,
    shared_proj_fn,
    private_a_proj_fn,
    private_b_proj_fn,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Project hidden states into shared and private slots.

    This is a functional helper for use outside the Flax module when you
    already have projection callables (e.g. from a bound ``CTAEBottleneck``).

    Args:
        h_a:               Hidden state from layer A.  Shape: [B, T, hidden_dim].
        h_b:               Hidden state from layer B.  Shape: [B, T, hidden_dim].
        shared_proj_fn:    Callable: [B, T, hidden_dim] -> [B, T, shared_dim].
        private_a_proj_fn: Callable: [B, T, hidden_dim] -> [B, T, private_dim].
        private_b_proj_fn: Callable: [B, T, hidden_dim] -> [B, T, private_dim].

    Returns:
        ``(s_a, s_b, s_fused, p_a, p_b)`` where shapes are as documented above.
    """
    s_a = shared_proj_fn(h_a)   # [B, T, shared_dim]
    s_b = shared_proj_fn(h_b)   # [B, T, shared_dim]
    s_fused = fuse_shared_slots(s_a, s_b)
    p_a = private_a_proj_fn(h_a)  # [B, T, private_dim]
    p_b = private_b_proj_fn(h_b)  # [B, T, private_dim]
    return s_a, s_b, s_fused, p_a, p_b


def fuse_shared_slots(s_a: jax.Array, s_b: jax.Array) -> jax.Array:
    """Compute the fused shared representation as the average of two views.

    Args:
        s_a: Shared projection of view A.  Shape: [..., shared_dim].
        s_b: Shared projection of view B.  Shape: [..., shared_dim].

    Returns:
        s_fused = 0.5 * (s_a + s_b).  Shape: [..., shared_dim].
    """
    return 0.5 * (s_a + s_b)


def compute_shared_only_loss(
    aux_shared_pred: jax.Array,
    target: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    """MSE between the shared-only head prediction and the real denoising target.

    This is the **primary** new CTAE-style objective in v1.  The head predicts
    in backbone feature space and should learn the common denoising subspace.

    Args:
        aux_shared_pred: Prediction from shared_only_head.
                         Shape: [num_pairs, B, T, out_dim] or [B, T, out_dim].
        target:          Real denoising velocity target.
                         Shape: [B, T, out_dim]  (broadcast over num_pairs).
        eps:             Unused; kept for API consistency with other helpers.

    Returns:
        Scalar MSE loss averaged over all elements.
    """
    # Cast to float32 for numerical stability
    pred = aux_shared_pred.astype(jnp.float32)
    tgt = target.astype(jnp.float32)
    return jnp.mean(jnp.square(pred - tgt))


def compute_alignment_loss(
    s_a: jax.Array,
    s_fused: jax.Array,
    s_b: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    """Alignment loss: MSE between each view's shared slot and the fused slot.

    ``L_align = mse(s_a, s_fused) + mse(s_b, s_fused)``

    Encourages both views to agree on the shared representation.

    Args:
        s_a:     Shared projection of view A.  Shape: [..., shared_dim].
        s_fused: Fused shared representation.  Shape: [..., shared_dim].
        s_b:     Shared projection of view B.  Shape: [..., shared_dim].
        eps:     Unused; kept for API consistency.

    Returns:
        Scalar alignment loss.
    """
    s_a = s_a.astype(jnp.float32)
    s_b = s_b.astype(jnp.float32)
    s_fused = s_fused.astype(jnp.float32)
    return jnp.mean(jnp.square(s_a - s_fused)) + jnp.mean(jnp.square(s_b - s_fused))


def compute_cross_block_orthogonality_loss(
    s_fused: jax.Array,
    p_a: jax.Array,
    p_b: jax.Array,
    eps: float = 1e-6,
    penalize_private_pair: bool = False,
) -> jax.Array:
    """Cross-block orthogonality loss.

    Penalizes overlap **only across blocks** (shared vs private), not within:

    * ``cosine(shared, p_a)^2``
    * ``cosine(shared, p_b)^2``
    * Optionally: ``cosine(p_a, p_b)^2``

    Features are L2-normalized before computing cosine similarities so that the
    loss is scale-invariant.

    In v1 we do **not** penalize within-block feature correlations.

    Args:
        s_fused:              Fused shared slot.  Shape: [B, T, shared_dim].
        p_a:                  Private slot of view A.  Shape: [B, T, private_dim].
        p_b:                  Private slot of view B.  Shape: [B, T, private_dim].
        eps:                  Epsilon for L2 normalisation.
        penalize_private_pair: If True, also penalise ``cosine(p_a, p_b)^2``
                               (disabled in v1 by default).

    Returns:
        Scalar cross-block orthogonality loss.
    """
    s_fused = s_fused.astype(jnp.float32)
    p_a = p_a.astype(jnp.float32)
    p_b = p_b.astype(jnp.float32)

    # Token-mean pooling before L2 norm reduces variance across spatial positions
    # Shape: [B, dim]  (pooled over T)
    s_pool = jnp.mean(s_fused, axis=-2)
    pa_pool = jnp.mean(p_a, axis=-2)
    pb_pool = jnp.mean(p_b, axis=-2)

    s_norm = _safe_l2_normalize(s_pool, eps)   # [B, shared_dim]
    pa_norm = _safe_l2_normalize(pa_pool, eps)  # [B, private_dim]
    pb_norm = _safe_l2_normalize(pb_pool, eps)  # [B, private_dim]

    # Dot product across the last axis only works when dims match.
    # shared_dim and private_dim may differ; project to min dimension.
    min_dim = min(s_norm.shape[-1], pa_norm.shape[-1])

    # Shared vs private_a: penalise cosine^2 between first min_dim components
    # This is a heuristic approximation when dims differ; it avoids OOM from
    # a full cross-dim dot product matrix.
    cos_s_pa = jnp.sum(s_norm[..., :min_dim] * pa_norm[..., :min_dim], axis=-1)  # [B]
    cos_s_pb = jnp.sum(s_norm[..., :min_dim] * pb_norm[..., :min_dim], axis=-1)  # [B]

    loss = jnp.mean(jnp.square(cos_s_pa)) + jnp.mean(jnp.square(cos_s_pb))

    if penalize_private_pair:
        min_dim_priv = pa_norm.shape[-1]  # same dim by construction
        cos_pa_pb = jnp.sum(pa_norm[..., :min_dim_priv] * pb_norm[..., :min_dim_priv], axis=-1)
        loss = loss + jnp.mean(jnp.square(cos_pa_pb))

    return loss


def compute_leakage_metrics(
    s_fused: jax.Array,
    p_a: jax.Array,
    p_b: jax.Array,
    eps: float = 1e-6,
) -> dict[str, jax.Array]:
    """Compute leakage and sanity-check metrics for the shared/private slots.

    These are **monitoring only** and must not be included in any training loss.

    Args:
        s_fused: Fused shared slot.  Shape: [..., shared_dim].
        p_a:     Private slot of view A.  Shape: [..., private_dim].
        p_b:     Private slot of view B.  Shape: [..., private_dim].
        eps:     Epsilon for numerical stability.

    Returns:
        Dict with scalar values:
            shared_norm              -- mean L2 norm of s_fused tokens
            private_a_norm           -- mean L2 norm of p_a tokens
            private_b_norm           -- mean L2 norm of p_b tokens
            cosine_shared_private_a  -- mean |cos(s_fused, p_a)|
            cosine_shared_private_b  -- mean |cos(s_fused, p_b)|
            cosine_private_a_private_b -- mean |cos(p_a, p_b)|
    """
    s = s_fused.astype(jnp.float32)
    pa = p_a.astype(jnp.float32)
    pb = p_b.astype(jnp.float32)

    # Norms: mean over batch and token dimensions
    # [..., dim] -> scalar
    shared_norm = jnp.mean(jnp.linalg.norm(s, axis=-1))
    private_a_norm = jnp.mean(jnp.linalg.norm(pa, axis=-1))
    private_b_norm = jnp.mean(jnp.linalg.norm(pb, axis=-1))

    # Token-mean pool before computing cross-slot cosines
    s_pool = jnp.mean(s, axis=-2)    # [B, shared_dim]
    pa_pool = jnp.mean(pa, axis=-2)  # [B, private_dim]
    pb_pool = jnp.mean(pb, axis=-2)  # [B, private_dim]

    min_dim_s_pa = min(s_pool.shape[-1], pa_pool.shape[-1])
    min_dim_s_pb = min(s_pool.shape[-1], pb_pool.shape[-1])
    min_dim_pa_pb = pa_pool.shape[-1]

    cos_s_pa = safe_cosine_similarity(
        s_pool[..., :min_dim_s_pa], pa_pool[..., :min_dim_s_pa], eps
    )
    cos_s_pb = safe_cosine_similarity(
        s_pool[..., :min_dim_s_pb], pb_pool[..., :min_dim_s_pb], eps
    )
    cos_pa_pb = safe_cosine_similarity(
        pa_pool[..., :min_dim_pa_pb], pb_pool[..., :min_dim_pa_pb], eps
    )

    return {
        "shared_norm": shared_norm,
        "private_a_norm": private_a_norm,
        "private_b_norm": private_b_norm,
        "cosine_shared_private_a": jnp.mean(jnp.abs(cos_s_pa)),
        "cosine_shared_private_b": jnp.mean(jnp.abs(cos_s_pb)),
        "cosine_private_a_private_b": jnp.mean(jnp.abs(cos_pa_pb)),
    }


def zero_ctae_metrics(dtype=jnp.float32) -> dict[str, jax.Array]:
    """Return zero-valued logging scalars for the disabled CTAE path.

    All metrics that ``compute_ctae_losses`` would produce are present here
    so that logging code can always expect a fixed set of keys regardless of
    whether CTAE is active.
    """
    zero = jnp.array(0.0, dtype=dtype)
    return {
        # Loss components
        "ctae_loss_shared": zero,
        "ctae_loss_align": zero,
        "ctae_loss_orth": zero,
        # Leakage / sanity monitoring
        "shared_norm": zero,
        "private_a_norm": zero,
        "private_b_norm": zero,
        "cosine_shared_private_a": zero,
        "cosine_shared_private_b": zero,
        "cosine_private_a_private_b": zero,
        # Aux prediction monitoring
        "aux_shared_pred_metric": zero,
        "aux_a_pred_metric": zero,
        "aux_b_pred_metric": zero,
        # Pair count
        "ctae_num_pairs_used": zero,
    }


def compute_ctae_losses(
    ctae_outputs: dict,
    target: jax.Array,
    *,
    lambda_shared: float,
    lambda_align: float,
    lambda_orth: float,
    align_active: bool,
    orth_active: bool,
    eps: float = 1e-6,
    penalize_private_pair: bool = False,
) -> dict[str, jax.Array]:
    """Aggregate CTAE-style losses across all layer pairs.

    The ``ctae_outputs`` pytree comes directly from ``SelfFlowDiT.__call__``
    when ``ctae_enabled=True``.

    Loss formula in v1::

        loss_ctae = lambda_shared * loss_shared
                  + lambda_align  * loss_align
                  + lambda_orth   * loss_orth   # only when orth_active=True

    **Important:** aux A/B prediction metrics are computed and returned for
    logging but are **never** included in any returned loss scalar.

    Args:
        ctae_outputs:        Pytree from model forward with CTAE enabled.
        target:              Real denoising target.  Shape: [B, T, out_dim].
        lambda_shared:       Weight for shared-only prediction loss.
        lambda_align:        Weight for alignment loss.
        lambda_orth:         Weight for cross-block orthogonality loss.
        align_active:        Whether alignment loss should be applied this step.
        orth_active:         Whether orthogonality loss should be applied this step.
                             Callers enforce the ``ctae_orth_start_step`` threshold.
        eps:                 Epsilon for numerical stability.
        penalize_private_pair: Passed through to ``compute_cross_block_orthogonality_loss``.

    Returns:
        Dict with all CTAE loss and monitoring metrics.  Keys match
        ``zero_ctae_metrics()``.  Additionally contains ``"ctae_loss_total"``
        which is the weighted sum to be added to ``loss_diff``.
    """
    if ctae_outputs is None:
        m = zero_ctae_metrics()
        m["ctae_loss_total"] = jnp.array(0.0, dtype=jnp.float32)
        return m

    # ctae_outputs shapes:
    #   s_a, s_b, s_fused: [num_pairs, B, T, shared_dim]
    #   p_a, p_b:          [num_pairs, B, T, private_dim]
    #   aux_shared_pred:   [num_pairs, B, T, out_dim]
    #   aux_a/b_pred:      [num_pairs, B, T, out_dim] or None

    s_a = ctae_outputs["s_a"].astype(jnp.float32)
    s_b = ctae_outputs["s_b"].astype(jnp.float32)
    s_fused = ctae_outputs["s_fused"].astype(jnp.float32)
    p_a = ctae_outputs["p_a"].astype(jnp.float32)
    p_b = ctae_outputs["p_b"].astype(jnp.float32)
    aux_shared_pred = ctae_outputs["aux_shared_pred"].astype(jnp.float32)

    num_pairs = s_a.shape[0]

    # ── 1. Shared-only prediction loss ────────────────────────────────────────
    # Target is broadcast across num_pairs dim: [B, T, out_dim] -> [1, B, T, out_dim]
    # aux_shared_pred has shape [num_pairs, B, T, out_dim]
    target_f32 = target.astype(jnp.float32)
    loss_shared = compute_shared_only_loss(aux_shared_pred, target_f32[None], eps)

    # ── 2. Alignment loss ─────────────────────────────────────────────────────
    # Computed per-pair then averaged.
    # s_a/s_b/s_fused: [num_pairs, B, T, shared_dim]
    loss_align_raw = compute_alignment_loss(s_a, s_fused, s_b, eps)
    loss_align = jnp.where(
        jnp.asarray(align_active, dtype=jnp.bool_),
        loss_align_raw,
        jnp.array(0.0, dtype=jnp.float32),
    )

    # ── 3. Cross-block orthogonality ──────────────────────────────────────────
    # Computed per-pair; average across pairs.
    # We iterate manually to keep shapes simple.
    orth_per_pair = []
    for i in range(num_pairs):
        orth_per_pair.append(
            compute_cross_block_orthogonality_loss(
                s_fused[i], p_a[i], p_b[i], eps, penalize_private_pair
            )
        )
    loss_orth_raw = jnp.mean(jnp.stack(orth_per_pair, axis=0))
    # Zero out when orthogonality is not yet active (warmup schedule).
    loss_orth = jnp.where(
        jnp.asarray(orth_active, dtype=jnp.bool_),
        loss_orth_raw,
        jnp.array(0.0, dtype=jnp.float32),
    )

    # ── 4. Leakage / monitoring metrics ──────────────────────────────────────
    # Aggregate over all pairs.
    leakage_per_pair = [
        compute_leakage_metrics(s_fused[i], p_a[i], p_b[i], eps)
        for i in range(num_pairs)
    ]
    leakage = {
        k: jnp.mean(jnp.stack([m[k] for m in leakage_per_pair], axis=0))
        for k in leakage_per_pair[0]
    }

    # ── 5. Aux A/B monitoring metrics (NOT in loss) ───────────────────────────
    aux_a_pred = ctae_outputs.get("aux_a_pred")
    aux_b_pred = ctae_outputs.get("aux_b_pred")

    if aux_a_pred is not None:
        aux_a_metric = jnp.mean(jnp.square(
            aux_a_pred.astype(jnp.float32) - target_f32[None]
        ))
    else:
        aux_a_metric = jnp.array(0.0, dtype=jnp.float32)

    if aux_b_pred is not None:
        aux_b_metric = jnp.mean(jnp.square(
            aux_b_pred.astype(jnp.float32) - target_f32[None]
        ))
    else:
        aux_b_metric = jnp.array(0.0, dtype=jnp.float32)

    # Shared prediction metric: same as loss_shared (MSE vs target), logged separately.
    aux_shared_metric = loss_shared

    # ── 6. Weighted total (aux A/B deliberately excluded) ─────────────────────
    ctae_loss_total = (
        jnp.asarray(lambda_shared, dtype=jnp.float32) * loss_shared
        + jnp.asarray(lambda_align, dtype=jnp.float32) * loss_align
        + jnp.asarray(lambda_orth, dtype=jnp.float32) * loss_orth
    )

    return {
        "ctae_loss_shared": loss_shared,
        "ctae_loss_align": loss_align,
        "ctae_loss_orth": loss_orth,
        **leakage,
        "aux_shared_pred_metric": aux_shared_metric,
        "aux_a_pred_metric": aux_a_metric,
        "aux_b_pred_metric": aux_b_metric,
        "ctae_num_pairs_used": jnp.asarray(float(num_pairs), dtype=jnp.float32),
        "ctae_loss_total": ctae_loss_total,
    }
