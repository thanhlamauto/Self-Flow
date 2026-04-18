"""Compatibility helpers for newer JAX releases."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def replicate_to_local_devices(tree, devices=None):
    """Replicate a pytree across local devices for pmap inputs.

    Mirrors the official JAX drop-in replacement for the deprecated
    device_put_replicated API using jax.device_put + NamedSharding.
    """
    devices = tuple(devices) if devices is not None else tuple(jax.local_devices())
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))

    def _replicate_leaf(x):
        arr = jnp.asarray(x)
        return jax.device_put(jnp.stack([arr] * len(devices)), sharding)

    return jax.tree_util.tree_map(_replicate_leaf, tree)


def unreplicate_from_local_devices(tree):
    """Return one local replica from a replicated pytree.

    On newer pmap implementations, indexing a sharded array with x[0] can
    trigger unnecessary resharding or even fail in multi-host settings. Prefer
    reading the first addressable shard directly when available.
    """

    def _first_local_replica(x):
        if hasattr(x, "addressable_shards"):
            shard = x.addressable_shards[0].data
            if getattr(shard, "ndim", 0) > 0 and shard.shape[0] == 1:
                return jnp.squeeze(shard, axis=0)
            return shard
        return x[0]

    return jax.tree_util.tree_map(_first_local_replica, tree)
