"""
Self-Flow Model (Flax version).

This module contains the SelfFlowPerTokenDiT model, a Diffusion Transformer
with per-token timestep conditioning for Self-Flow training, implemented in Flax.
"""

import math
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = jnp.einsum("m,d->md", pos, omega)
    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def parse_layer_pairs_from_string(spec: str) -> list[tuple[int, int]]:
    """Parse a colon-separated pair string into a list of (int, int) tuples.

    Args:
        spec: Comma-separated layer pairs in ``A:B`` format, e.g. ``"3:6,6:9,9:12"``.

    Returns:
        List of ``(layer_a, layer_b)`` integer tuples, deduplicated in order.

    Raises:
        ValueError: If any token is not in ``A:B`` format or indices are non-positive.

    Example::

        parse_layer_pairs_from_string("3:6,6:9")  # -> [(3, 6), (6, 9)]
    """
    if not spec or not spec.strip():
        return []

    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for token in spec.strip().split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid layer pair token '{token}': expected format 'A:B', e.g. '3:6'."
            )
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid layer pair token '{token}': both A and B must be integers."
            )
        if a < 1 or b < 1:
            raise ValueError(
                f"Layer indices must be >= 1, got '{token}'."
            )
        if a == b:
            raise ValueError(
                f"Layer pair '{token}' has identical indices; A and B must differ."
            )
        key = (a, b)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


class PatchedPatchEmbed(nn.Module):
    """Simplified Sequence to Patch Embedding using Linear layer."""
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(self.embed_dim, use_bias=self.bias, name="proj")(x)


def modulate(x, shift, scale):
    """Standard modulation with unsqueeze for (N, D) conditioning."""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def modulate_per_token(x, shift, scale):
    """Per-token modulation for (N, T, D) conditioning."""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    hidden_size: int
    frequency_embedding_size: int = 256

    def timestep_embedding(self, t, dim, max_period=10000.0):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    @nn.compact
    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        x = nn.Dense(self.hidden_size)(t_freq)
        x = nn.swish(x)
        x = nn.Dense(self.hidden_size)(x)
        return x


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""
    num_classes: int
    hidden_size: int
    dropout_prob: float

    @nn.compact
    def __call__(self, labels, deterministic: bool = True, force_drop_ids=None):
        use_cfg_embedding = self.dropout_prob > 0
        embedding_table = nn.Embed(
            num_embeddings=self.num_classes + use_cfg_embedding,
            features=self.hidden_size
        )

        use_dropout = self.dropout_prob > 0
        if (not deterministic and use_dropout) or (force_drop_ids is not None):
            if force_drop_ids is None:
                rng = self.make_rng('dropout')
                drop_ids = jax.random.uniform(rng, labels.shape) < self.dropout_prob
            else:
                drop_ids = force_drop_ids == 1
            labels = jnp.where(drop_ids, self.num_classes, labels)

        return embedding_table(labels)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    per_token: bool = False

    @nn.compact
    def __call__(self, x, c):
        norm1 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        norm2 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)

        if self.per_token:
            batch_size, seq_len, hidden_dim = c.shape
            c_flat = c.reshape(-1, hidden_dim)
            modulation_flat = nn.Sequential([
                nn.swish,
                nn.Dense(6 * self.hidden_size)
            ])(c_flat)
            modulation = modulation_flat.reshape(batch_size, seq_len, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

            x_norm = modulate_per_token(norm1(x), shift_msa, scale_msa)
            # Self Attention
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, qkv_features=self.hidden_size, out_features=self.hidden_size
            )(x_norm, x_norm)
            x = x + gate_msa * attn

            x_norm2 = modulate_per_token(norm2(x), shift_mlp, scale_mlp)
            mlp_fn = nn.Sequential([
                nn.Dense(mlp_hidden_dim),
                lambda z: nn.gelu(z, approximate=True),
                nn.Dense(self.hidden_size)
            ])
            x = x + gate_mlp * mlp_fn(x_norm2)
        else:
            modulation = nn.Sequential([
                nn.swish,
                nn.Dense(6 * self.hidden_size)
            ])(c)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=1)

            x_norm = modulate(norm1(x), shift_msa, scale_msa)
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, qkv_features=self.hidden_size, out_features=self.hidden_size
            )(x_norm, x_norm)
            x = x + gate_msa[:, None, :] * attn

            x_norm2 = modulate(norm2(x), shift_mlp, scale_mlp)
            mlp_fn = nn.Sequential([
                nn.Dense(mlp_hidden_dim),
                lambda z: nn.gelu(z, approximate=True),
                nn.Dense(self.hidden_size)
            ])
            x = x + gate_mlp[:, None, :] * mlp_fn(x_norm2)

        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""
    hidden_size: int
    patch_size: int
    out_channels: int
    per_token: bool = False

    @nn.compact
    def __call__(self, x, c):
        norm_final = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        linear = nn.Dense(self.patch_size * self.patch_size * self.out_channels)

        if self.per_token:
            batch_size, seq_len, hidden_dim = c.shape
            c_flat = c.reshape(-1, hidden_dim)
            modulation_flat = nn.Sequential([
                nn.swish,
                nn.Dense(2 * self.hidden_size)
            ])(c_flat)
            modulation = modulation_flat.reshape(batch_size, seq_len, -1)
            shift, scale = jnp.split(modulation, 2, axis=-1)

            x = modulate_per_token(norm_final(x), shift, scale)
            x = linear(x)
        else:
            modulation = nn.Sequential([
                nn.swish,
                nn.Dense(2 * self.hidden_size)
            ])(c)
            shift, scale = jnp.split(modulation, 2, axis=1)

            x = modulate(norm_final(x), shift, scale)
            x = linear(x)

        return x


class SimpleHead(nn.Module):
    """Simple projection head for self-distillation."""
    in_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.in_dim + self.out_dim)(x)
        x = nn.swish(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class CTAEBottleneck(nn.Module):
    """CTAE-inspired shared/private bottleneck, shared across all layer pairs.

    A single instance of this module is reused for every selected pair
    (h_a, h_b) during a forward pass.  The shared projector maps each hidden
    state to a low-dimensional shared subspace; the private projectors capture
    the layer-specific residual.  An optional pair of auxiliary prediction
    heads can be built for monitoring (their outputs must NOT be included in
    the training loss in v1).

    Uses ``setup()`` (not ``@nn.compact``) so that shared projection layers
    can be called on both h_a and h_b without triggering Flax's
    ``NameInUseError`` — which fires when the same name is used more than once
    inside an ``@nn.compact`` method.

    Attributes:
        hidden_dim:    Backbone hidden size (input feature dimension).
        shared_dim:    Dimensionality of the shared slot.
        private_dim:   Dimensionality of each private slot.
        out_dim:       Output dimension of all prediction heads
                       (typically equals hidden_dim to predict the denoising target).
        use_aux_heads: Whether to build and call aux_a_head / aux_b_head.
                       Even when True, their outputs are monitoring-only in v1.
    """

    hidden_dim: int
    shared_dim: int
    private_dim: int
    out_dim: int
    use_aux_heads: bool = False

    def setup(self):
        # ── Shared projection (same Dense instances called on h_a AND h_b) ───
        # Weight sharing is achieved by calling the SAME module twice.
        # shared_dim: hidden_dim -> shared_dim
        self.shared_proj_l1 = nn.Dense(self.shared_dim)
        self.shared_proj_l2 = nn.Dense(self.shared_dim)

        # ── Private projections (separate weights per role) ───────────────────
        # private_dim: hidden_dim -> private_dim
        self.private_a_proj_l1 = nn.Dense(self.private_dim)
        self.private_a_proj_l2 = nn.Dense(self.private_dim)
        self.private_b_proj_l1 = nn.Dense(self.private_dim)
        self.private_b_proj_l2 = nn.Dense(self.private_dim)

        # ── Shared-only prediction head (main CTAE objective in v1) ──────────
        # shared_dim -> (shared_dim + out_dim) -> out_dim
        self.shared_only_head_l1 = nn.Dense(self.shared_dim + self.out_dim)
        self.shared_only_head_l2 = nn.Dense(self.out_dim)

        # ── Aux A/B prediction heads (monitoring only; NOT in loss_total) ─────
        # (shared_dim + private_dim) -> (shared_dim + private_dim + out_dim) -> out_dim
        if self.use_aux_heads:
            self.aux_a_head_l1 = nn.Dense(self.shared_dim + self.private_dim + self.out_dim)
            self.aux_a_head_l2 = nn.Dense(self.out_dim)
            self.aux_b_head_l1 = nn.Dense(self.shared_dim + self.private_dim + self.out_dim)
            self.aux_b_head_l2 = nn.Dense(self.out_dim)

    def __call__(
        self,
        h_a: jax.Array,
        h_b: jax.Array,
    ) -> dict:
        """Compute shared and private projections for one layer pair.

        Args:
            h_a: Hidden state from layer A.  Shape: [B, T, hidden_dim].
            h_b: Hidden state from layer B.  Shape: [B, T, hidden_dim].

        Returns:
            Dict with keys:
                s_a             [B, T, shared_dim]
                s_b             [B, T, shared_dim]
                s_fused         [B, T, shared_dim]  -- 0.5*(s_a + s_b)
                p_a             [B, T, private_dim]
                p_b             [B, T, private_dim]
                aux_shared_pred [B, T, out_dim]     -- prediction from s_fused
                aux_a_pred      [B, T, out_dim]     -- monitoring only; None if use_aux_heads=False
                aux_b_pred      [B, T, out_dim]     -- monitoring only; None if use_aux_heads=False
        """
        # ── Shared projection — same params applied to both views ─────────────
        # [B, T, hidden_dim] -> [B, T, shared_dim]
        s_a = self.shared_proj_l2(nn.swish(self.shared_proj_l1(h_a)))
        s_b = self.shared_proj_l2(nn.swish(self.shared_proj_l1(h_b)))

        # ── Fused shared: average of both views ───────────────────────────────
        # [B, T, shared_dim]
        s_fused = 0.5 * (s_a + s_b)

        # ── Private projections — separate params per role ────────────────────
        # [B, T, hidden_dim] -> [B, T, private_dim]
        p_a = self.private_a_proj_l2(nn.swish(self.private_a_proj_l1(h_a)))
        p_b = self.private_b_proj_l2(nn.swish(self.private_b_proj_l1(h_b)))

        # ── Shared-only prediction head ───────────────────────────────────────
        # [B, T, shared_dim] -> [B, T, out_dim]
        aux_shared_pred = self.shared_only_head_l2(nn.swish(self.shared_only_head_l1(s_fused)))

        # ── Aux A/B prediction heads (monitoring only; NOT in loss_total) ─────
        aux_a_pred = None
        aux_b_pred = None
        if self.use_aux_heads:
            # aux_a_input: [B, T, shared_dim + private_dim]
            aux_a_pred = self.aux_a_head_l2(
                nn.swish(self.aux_a_head_l1(jnp.concatenate([s_fused, p_a], axis=-1)))
            )
            # aux_b_input: [B, T, shared_dim + private_dim]
            aux_b_pred = self.aux_b_head_l2(
                nn.swish(self.aux_b_head_l1(jnp.concatenate([s_fused, p_b], axis=-1)))
            )

        return {
            "s_a": s_a,                          # [B, T, shared_dim]
            "s_b": s_b,                          # [B, T, shared_dim]
            "s_fused": s_fused,                  # [B, T, shared_dim]
            "p_a": p_a,                          # [B, T, private_dim]
            "p_b": p_b,                          # [B, T, private_dim]
            "aux_shared_pred": aux_shared_pred,  # [B, T, out_dim]
            "aux_a_pred": aux_a_pred,            # [B, T, out_dim] or None
            "aux_b_pred": aux_b_pred,            # [B, T, out_dim] or None
        }


class SelfFlowDiT(nn.Module):
    """Base Self-Flow DiT model."""
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    learn_sigma: bool = False
    compatibility_mode: bool = False
    per_token: bool = False
    # ── CTAE bottleneck config (set at construction time so params are always
    #    allocated when ctae_shared_dim > 0, regardless of ctae_enabled in __call__) ──
    ctae_shared_dim: int = 0        # 0 = CTAE not allocated; set to > 0 to enable
    ctae_private_dim: int = 0
    ctae_use_aux_heads: bool = False

    def setup(self):
        self.out_channels_val = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        self.pos_embed_val = pos_embed[None, ...]  # (1, num_patches, hidden_size)
        self.feature_head = SimpleHead(in_dim=self.hidden_size, out_dim=self.hidden_size)

        # Allocate CTAE bottleneck unconditionally when dims are set.
        # Flax requires submodules to be declared in setup() so their params
        # are present from the first model.init() call.
        if self.ctae_shared_dim > 0 and self.ctae_private_dim > 0:
            self.ctae_bottleneck = CTAEBottleneck(
                hidden_dim=self.hidden_size,
                shared_dim=self.ctae_shared_dim,
                private_dim=self.ctae_private_dim,
                out_dim=self.hidden_size,
                use_aux_heads=self.ctae_use_aux_heads,
            )

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        vector: jax.Array,
        x_ids: Optional[jax.Array] = None,
        return_features: bool = False,
        return_raw_features: bool = False,
        return_block_summaries: bool = False,
        capture_hidden_layers: Optional[Sequence[int]] = None,
        deterministic: bool = True,
        # ── CTAE kwargs ───────────────────────────────────────────────────────
        # Dimensions (shared_dim, private_dim, use_aux_heads) are class attributes
        # set at construction time so their params are allocated during model.init().
        # ctae_enabled and ctae_layer_pairs are call-time flags only.
        ctae_enabled: bool = False,
        ctae_layer_pairs: Optional[Sequence[tuple[int, int]]] = None,
    ):
        """Forward pass with compatibility mode handling.

        CTAE kwargs are only active when ``ctae_enabled=True``.  When
        ``ctae_enabled=False`` the model behaves exactly as before this change.

        When CTAE is enabled, the model additionally returns a ``ctae_outputs``
        pytree as the last element of the return tuple:

            (pred, ..., ctae_outputs)

        ``ctae_outputs`` has the following structure::

            {
                "pair_indices":     [num_pairs, 2]          -- int32 tensor
                "s_a":              [num_pairs, B, T, shared_dim]
                "s_b":              [num_pairs, B, T, shared_dim]
                "s_fused":          [num_pairs, B, T, shared_dim]
                "p_a":              [num_pairs, B, T, private_dim]
                "p_b":              [num_pairs, B, T, private_dim]
                "aux_shared_pred":  [num_pairs, B, T, out_dim]
                "aux_a_pred":       [num_pairs, B, T, out_dim] or None
                "aux_b_pred":       [num_pairs, B, T, out_dim] or None
            }

        Note: CTAE bottleneck params are allocated at model construction time
        (in setup()) when ``ctae_shared_dim > 0``.  ``ctae_enabled`` here only
        controls whether the bottleneck is *called* and its outputs returned.
        """
        assert not (return_raw_features and return_features)

        # PyTorch implementation explicitly negates timesteps
        timesteps = 1.0 - timesteps

        # Patch Embedding
        x = PatchedPatchEmbed(
            img_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.hidden_size
        )(x)
        x = x + self.pos_embed_val

        t_embedder = TimestepEmbedder(hidden_size=self.hidden_size)
        y_embedder = LabelEmbedder(num_classes=self.num_classes, hidden_size=self.hidden_size, dropout_prob=0.0)

        if self.per_token:
            batch_size, seq_len, _ = x.shape
            if timesteps.ndim == 1:
                t_emb = t_embedder(timesteps)
                t_emb = jnp.tile(t_emb[:, None, :], (1, seq_len, 1))
            elif timesteps.ndim == 2:
                t_flat = timesteps.reshape(-1)
                t_emb_flat = t_embedder(t_flat)
                t_emb = t_emb_flat.reshape(batch_size, seq_len, -1)
            else:
                raise ValueError(f"Unsupported per-token timestep rank: {timesteps.ndim}")

            y_emb = y_embedder(vector, deterministic=deterministic)
            y_emb = jnp.tile(y_emb[:, None, :], (1, seq_len, 1))
        else:
            t_emb = t_embedder(timesteps)
            y_emb = y_embedder(vector, deterministic=deterministic)

        c = t_emb + y_emb

        # ── Determine which layers to capture ────────────────────────────────
        # When CTAE is enabled, we need to capture all unique layer indices
        # referenced by ctae_layer_pairs.  We merge them with the caller's own
        # capture_hidden_layers so we do a single forward pass.
        ctae_needed_layers: frozenset[int] = frozenset()
        ctae_pairs: tuple[tuple[int, int], ...] = ()
        if ctae_enabled and ctae_layer_pairs:
            ctae_pairs = tuple(ctae_layer_pairs)
            ctae_needed_layers = frozenset(idx for pair in ctae_pairs for idx in pair)

        # Union of caller-requested layers + CTAE-needed layers
        all_capture_layers: set[int] = set()
        if capture_hidden_layers:
            all_capture_layers.update(int(l) for l in capture_hidden_layers)
        all_capture_layers.update(ctae_needed_layers)
        effective_capture = tuple(sorted(all_capture_layers)) if all_capture_layers else None

        zs = None
        block_summaries = [] if return_block_summaries else None
        # captured_dict maps layer_idx -> hidden state [B, T, D]
        captured_dict: dict[int, jax.Array] = {} if effective_capture else {}

        for i in range(self.depth):
            layer_idx = i + 1
            x = DiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                per_token=self.per_token
            )(x, c)

            if return_block_summaries:
                block_summaries.append(jnp.mean(x, axis=1))

            if effective_capture and layer_idx in all_capture_layers:
                captured_dict[layer_idx] = x

            if layer_idx == return_features:
                zs = self.feature_head(x)
            elif layer_idx == return_raw_features:
                zs = x

        x = FinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels_val,
            per_token=self.per_token
        )(x, c)

        x = self._shufflechannel(x)

        # PyTorch implementation negates the final prediction
        x = -x

        # ── Build legacy captured_layers tuple (for capture_hidden_layers callers) ──
        # Only include the layers originally requested by the caller, preserving order.
        legacy_captured: Optional[tuple] = None
        if capture_hidden_layers:
            legacy_captured = tuple(
                captured_dict[int(l)] for l in capture_hidden_layers if int(l) in captured_dict
            )

        # ── CTAE bottleneck forward (only when ctae_enabled=True) ─────────────
        # self.ctae_bottleneck is allocated in setup() when ctae_shared_dim > 0.
        # Here we only call it; we never create a new CTAEBottleneck instance.
        ctae_outputs: Optional[dict] = None
        ctae_bottleneck_ready = self.ctae_shared_dim > 0 and self.ctae_private_dim > 0
        if ctae_enabled and ctae_pairs and ctae_bottleneck_ready:
            # Run bottleneck for each pair and collect stacked results.
            # Shapes after stacking: [num_pairs, B, T, dim]
            pair_s_a, pair_s_b, pair_s_fused = [], [], []
            pair_p_a, pair_p_b = [], []
            pair_aux_shared_pred = []
            pair_aux_a_pred: list = []
            pair_aux_b_pred: list = []
            pair_indices_list: list[tuple[int, int]] = []

            for (la, lb) in ctae_pairs:
                h_a = captured_dict[la]   # [B, T, hidden_size]
                h_b = captured_dict[lb]   # [B, T, hidden_size]
                slot = self.ctae_bottleneck(h_a, h_b)

                pair_s_a.append(slot["s_a"])
                pair_s_b.append(slot["s_b"])
                pair_s_fused.append(slot["s_fused"])
                pair_p_a.append(slot["p_a"])
                pair_p_b.append(slot["p_b"])
                pair_aux_shared_pred.append(slot["aux_shared_pred"])
                pair_aux_a_pred.append(slot["aux_a_pred"])
                pair_aux_b_pred.append(slot["aux_b_pred"])
                pair_indices_list.append((la, lb))

            ctae_outputs = {
                # [num_pairs, 2]
                "pair_indices": jnp.array(pair_indices_list, dtype=jnp.int32),
                # [num_pairs, B, T, shared_dim]
                "s_a": jnp.stack(pair_s_a, axis=0),
                "s_b": jnp.stack(pair_s_b, axis=0),
                "s_fused": jnp.stack(pair_s_fused, axis=0),
                # [num_pairs, B, T, private_dim]
                "p_a": jnp.stack(pair_p_a, axis=0),
                "p_b": jnp.stack(pair_p_b, axis=0),
                # [num_pairs, B, T, out_dim]
                "aux_shared_pred": jnp.stack(pair_aux_shared_pred, axis=0),
                # [num_pairs, B, T, out_dim] or None
                "aux_a_pred": (
                    jnp.stack(pair_aux_a_pred, axis=0)
                    if self.ctae_use_aux_heads else None
                ),
                "aux_b_pred": (
                    jnp.stack(pair_aux_b_pred, axis=0)
                    if self.ctae_use_aux_heads else None
                ),
            }

        # ── Assemble return tuple ─────────────────────────────────────────────
        # Format matches the original contract; CTAE outputs are appended last
        # when ctae_enabled=True.  This preserves all existing callers.
        out = (x,)
        if return_features or return_raw_features:
            out += (zs,)
        if return_block_summaries:
            out += (jnp.stack(block_summaries, axis=0),)
        if capture_hidden_layers:
            out += (legacy_captured,)
        if ctae_enabled:
            out += (ctae_outputs,)

        if len(out) == 1:
            return out[0]
        return out

    def _shufflechannel(self, x):
        """Reorder channels/patches to match expected output format."""
        p = self.patch_size
        x = rearrange(x, "b l (c p q) -> b l (c p q)", p=p, q=p, c=self.out_channels_val)  # equivalent to rearranging in torch
        # wait, the PyTorch implementation says:
        # x = rearrange(x, "b l (p q c) -> b l (c p q)", p=p, q=p, c=self.out_channels)
        x = rearrange(x, "b l (p q c) -> b l (c p q)", p=p, q=p, c=self.out_channels_val)
        if self.learn_sigma:
            x, _ = jnp.split(x, 2, axis=2)
        return x


class SelfFlowPerTokenDiT(SelfFlowDiT):
    """
    Self-Flow DiT with per-token timestep conditioning.
    Main model used for Self-Flow inference on ImageNet.
    """
    per_token: bool = True


# Thin alias for clarity in the vanilla SiT baseline.
# Use as: SiTDiT(..., per_token=False)
SiTDiT = SelfFlowDiT
