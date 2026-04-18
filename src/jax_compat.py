"""Compatibility helpers for deprecated JAX pmap-era array utilities."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def replicate_tree(tree, devices=None):
    """Drop-in replacement for ``flax.jax_utils.replicate`` on modern JAX."""
    devices = list(devices or jax.local_devices())
    mesh = Mesh(np.array(devices, dtype=object), ("replica",))
    sharding = NamedSharding(mesh, P("replica"))

    def _replicate_leaf(leaf):
        leaf_array = jnp.asarray(leaf)
        stacked = jnp.stack([leaf_array] * len(devices), axis=0)
        return jax.device_put(stacked, sharding)

    return jax.tree_util.tree_map(_replicate_leaf, tree)


def unreplicate_tree(tree):
    """Best-effort replacement for ``flax.jax_utils.unreplicate``."""

    def _unreplicate_leaf(leaf):
        if hasattr(leaf, "addressable_shards") and leaf.addressable_shards:
            shard_data = leaf.addressable_shards[0].data
            if getattr(shard_data, "shape", ())[:1] == (1,):
                return jnp.squeeze(shard_data, axis=0)
            return shard_data

        host_leaf = jax.device_get(leaf)
        if getattr(host_leaf, "shape", ())[:1] == (1,):
            return np.squeeze(host_leaf, axis=0)
        if getattr(host_leaf, "shape", ())[:1]:
            return host_leaf[0]
        return host_leaf

    return jax.tree_util.tree_map(_unreplicate_leaf, tree)
