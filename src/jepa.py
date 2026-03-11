"""
I-JEPA block-prediction components for Self-Flow training.

Public API:
  sample_jepa_masks   — pure-JAX static-shape block mask sampler (call inside pmap)
  JEPAPredictor       — Transformer predictor: context tokens → target feature predictions
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from src.model import get_2d_sincos_pos_embed


# ---------------------------------------------------------------------------
# Block mask sampling (pure JAX, static shapes, safe inside pmap / vmap)
# ---------------------------------------------------------------------------

def _block_membership(top, left, h, w, grid_size: int):
    """Return a [grid_size*grid_size] bool mask: True for tokens inside the block."""
    rows = jnp.arange(grid_size)
    cols = jnp.arange(grid_size)
    in_block = (
        (rows[:, None] >= top) & (rows[:, None] < top + h) &
        (cols[None, :] >= left) & (cols[None, :] < left + w)
    )  # [grid_size, grid_size]
    return in_block.reshape(grid_size * grid_size)


def _membership_to_padded(membership, T_max: int):
    """Convert a [N] bool membership mask to (padded_idx [T_max], valid [T_max]).

    Uses argsort(-membership) so True entries sort first; the first `count`
    positions are valid, the rest are padding (index 0, valid=False).
    """
    sorted_idx = jnp.argsort(-membership.astype(jnp.float32), stable=True)[:T_max]
    count = jnp.sum(membership.astype(jnp.int32))
    valid = jnp.arange(T_max) < count
    return sorted_idx, valid


def _sample_block_padded(rng, grid_size: int,
                          scale_min: float, scale_max: float,
                          ar_min: float, ar_max: float,
                          T_max: int):
    """Sample one rectangular block and return static-size padded indices.

    Args:
      rng: JAX PRNG key.
      grid_size: token grid side length (16 for 32×32 latents with patch 2).
      scale_min/max: fraction of total tokens for the block area.
      ar_min/max: aspect ratio h/w range.
      T_max: padded output length.

    Returns:
      padded_idx: [T_max] int32 — linear token indices, valid entries first.
      valid:      [T_max] bool  — True for real (non-padding) entries.
    """
    N = grid_size * grid_size
    rng_s, rng_r, rng_t, rng_l = jax.random.split(rng, 4)

    scale = jax.random.uniform(rng_s, minval=scale_min, maxval=scale_max)
    ar    = jax.random.uniform(rng_r, minval=ar_min,    maxval=ar_max)

    h = jnp.clip(jnp.round(jnp.sqrt(scale * N * ar)),  1, grid_size).astype(jnp.int32)
    w = jnp.clip(jnp.round(jnp.sqrt(scale * N / ar)), 1, grid_size).astype(jnp.int32)

    # randint maxval is exclusive; +1 so top/left=grid_size-h/w are reachable.
    top  = jax.random.randint(rng_t, shape=(), minval=0,
                               maxval=jnp.maximum(grid_size - h + 1, 1))
    left = jax.random.randint(rng_l, shape=(), minval=0,
                               maxval=jnp.maximum(grid_size - w + 1, 1))

    membership = _block_membership(top, left, h, w, grid_size)
    return _membership_to_padded(membership, T_max)


def _sample_one_image(rng, grid_size: int, n_target: int,
                       T_max: int, C_max: int,
                       tgt_scale_min: float, tgt_scale_max: float,
                       tgt_ar_min: float,    tgt_ar_max: float,
                       ctx_scale_min: float, ctx_scale_max: float):
    """Sample context + target masks for one image.

    Returns:
      ctx_idx   [C_max] int32
      ctx_valid [C_max] bool
      tgt_idx   [n_target, T_max] int32
      tgt_valid [n_target, T_max] bool
    """
    N = grid_size * grid_size
    rng_tgts, rng_ctx = jax.random.split(rng)
    tgt_rngs = jax.random.split(rng_tgts, n_target)

    # --- 4 target blocks -------------------------------------------------
    tgt_idx, tgt_valid = jax.vmap(
        lambda r: _sample_block_padded(
            r, grid_size,
            tgt_scale_min, tgt_scale_max,
            tgt_ar_min, tgt_ar_max,
            T_max,
        )
    )(tgt_rngs)  # [n_target, T_max], [n_target, T_max]

    # Build union membership mask for all target blocks.
    def _idx_to_membership(idx, valid):
        m = jnp.zeros(N, dtype=jnp.bool_)
        return m.at[idx].set(valid)

    target_memberships = jax.vmap(_idx_to_membership)(tgt_idx, tgt_valid)  # [n_target, N]
    target_union = jnp.any(target_memberships, axis=0)  # [N]

    # --- 1 context block (aspect ratio fixed to 1.0) ---------------------
    ctx_idx_raw, ctx_valid_raw = _sample_block_padded(
        rng_ctx, grid_size,
        ctx_scale_min, ctx_scale_max,
        1.0, 1.0,   # ar fixed to 1.0 for context
        N,          # use N as T_max so the full context is captured before pruning
    )

    # Convert raw padded context back to membership, then subtract target union.
    ctx_membership = _idx_to_membership(ctx_idx_raw, ctx_valid_raw)
    ctx_prime = ctx_membership & ~target_union  # C' = C \ union(B_i)

    ctx_idx, ctx_valid = _membership_to_padded(ctx_prime, C_max)

    return ctx_idx, ctx_valid, tgt_idx, tgt_valid


def sample_jepa_masks(rng, local_batch: int, grid_size: int = 16,
                       n_target: int = 4, T_max: int = 64, C_max: int = 256,
                       tgt_scale_min: float = 0.15, tgt_scale_max: float = 0.20,
                       tgt_ar_min: float = 0.75,    tgt_ar_max: float = 1.5,
                       ctx_scale_min: float = 0.85, ctx_scale_max: float = 1.0):
    """Sample JEPA block masks for a local (per-device) batch.

    Safe to call inside a pmapped function; uses only static shapes and pure
    JAX ops (argsort, scatter via .at[].set, vmap).

    Returns:
      ctx_idx   [local_batch, C_max]            int32
      ctx_valid [local_batch, C_max]            bool
      tgt_idx   [local_batch, n_target, T_max]  int32
      tgt_valid [local_batch, n_target, T_max]  bool
    """
    rngs = jax.random.split(rng, local_batch)
    ctx_idx, ctx_valid, tgt_idx, tgt_valid = jax.vmap(
        lambda r: _sample_one_image(
            r, grid_size, n_target, T_max, C_max,
            tgt_scale_min, tgt_scale_max, tgt_ar_min, tgt_ar_max,
            ctx_scale_min, ctx_scale_max,
        )
    )(rngs)
    return ctx_idx, ctx_valid, tgt_idx, tgt_valid


# ---------------------------------------------------------------------------
# JEPA Predictor
# ---------------------------------------------------------------------------

class PredictorBlock(nn.Module):
    """Plain pre-LN Transformer block (no AdaLN, no timestep conditioning)."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, attn_mask=None):
        residual = x
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
        )(x, x, mask=attn_mask)
        x = residual + x

        residual = x
        x = nn.LayerNorm(epsilon=1e-6)(x)
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        x = nn.Dense(mlp_hidden)(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(self.hidden_size)(x)
        x = residual + x
        return x


class JEPAPredictor(nn.Module):
    """I-JEPA-style Transformer predictor.

    Takes student context features and target positional queries, returns
    predicted backbone features at target locations.

    Intended usage (B = local_batch * n_target after flattening):
      ctx_feats  [B, C_max, backbone_dim]  — student features at context positions
      ctx_valid  [B, C_max]  bool          — True for non-padding context tokens
      tgt_idx    [B, T_max]  int32         — linear grid indices of target tokens
      tgt_valid  [B, T_max]  bool          — True for non-padding target tokens
      → out      [B, T_max, backbone_dim]
    """
    backbone_dim: int = 768
    hidden_size: int = 384
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    grid_size: int = 16   # 16×16 token grid
    T_max: int = 64
    C_max: int = 256

    @nn.compact
    def __call__(self, ctx_feats, ctx_valid, tgt_idx, tgt_valid,
                 deterministic: bool = True):
        B = ctx_feats.shape[0]

        # Project context features to predictor width.
        ctx = nn.Dense(self.hidden_size, name="proj_in")(ctx_feats)  # [B, C_max, H]

        # Sinusoidal 2-D position embeddings for target queries.
        # Shape: [grid_size^2, hidden_size] — computed at call time (no learnable params).
        pos_embed_full = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        # Gather target positions: [B, T_max, H]
        # tgt_idx: [B, T_max] — use vmap gather to avoid advanced-indexing shapes.
        tgt_pos = jax.vmap(lambda idx: pos_embed_full[idx])(tgt_idx)  # [B, T_max, H]

        # Learnable shared mask token broadcast to all target query positions.
        mask_token = self.param(
            "mask_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.hidden_size),
        )
        tgt_queries = jnp.broadcast_to(
            mask_token, (B, self.T_max, self.hidden_size)
        ) + tgt_pos  # [B, T_max, H]

        # Concatenate context and target along sequence axis.
        x = jnp.concatenate([ctx, tgt_queries], axis=1)  # [B, C_max+T_max, H]

        # Attention mask: 1 = attend, 0 = ignore padding.
        # All queries attend to all valid keys (context + target).
        key_valid = jnp.concatenate([ctx_valid, tgt_valid], axis=1)  # [B, seq]
        # Shape required by Flax MHSA: [B, num_heads, q_len, kv_len] or broadcastable.
        attn_mask = key_valid[:, None, None, :].astype(jnp.float32)  # [B,1,1,seq]

        for _ in range(self.depth):
            x = PredictorBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
            )(x, attn_mask)

        # Extract the target portion only.
        tgt_out = x[:, self.C_max:, :]  # [B, T_max, H]

        # Project back to backbone dimension.
        out = nn.Dense(self.backbone_dim, name="proj_out")(tgt_out)  # [B, T_max, D]
        return out
