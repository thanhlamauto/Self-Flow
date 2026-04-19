"""Compatibility helpers for deprecated JAX pmap-era array utilities."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def replicate_tree(tree, devices=None):
    """Drop-in replacement for ``flax.jax_utils.replicate`` on modern JAX."""
    devices = list(devices or jax.local_devices())
    mesh = Mesh(np.array(devices, dtype=object), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(
        lambda leaf: jax.device_put(jnp.stack([leaf] * len(devices)), sharding),
        tree,
    )


def unreplicate_tree(tree):
    """Best-effort replacement for ``flax.jax_utils.unreplicate``."""

    def _unreplicate_leaf(leaf):
        if hasattr(leaf, "addressable_shards") and leaf.addressable_shards:
            shard_data = leaf.addressable_shards[0].data
            if getattr(shard_data, "ndim", 0) > 0 and shard_data.shape[0] == 1:
                return jnp.squeeze(shard_data, axis=0)
            return shard_data
        return leaf[0]

    return jax.tree.map(_unreplicate_leaf, tree)
