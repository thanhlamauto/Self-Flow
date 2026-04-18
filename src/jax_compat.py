"""Compatibility helpers for newer JAX releases."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import sharding as js


def replicate_to_local_devices(tree, devices=None):
    """Replicate a pytree across local devices for pmap inputs.

    The old flax.jax_utils.replicate() path relied on jax.device_put_replicated,
    and our first compatibility shim used jax.device_put_sharded. Newer JAX
    versions deprecate both helpers in favor of jax.device_put with explicit
    sharding metadata. For pmap inputs we need the legacy shape convention:
    each leaf has a leading device axis of length n_local_devices.
    """
    devices = tuple(devices) if devices is not None else tuple(jax.local_devices())
    sharding = js.PositionalSharding(devices)

    def _stack_for_devices(x):
        arr = jnp.asarray(x)
        return jnp.broadcast_to(arr, (len(devices),) + arr.shape)

    stacked_tree = jax.tree_util.tree_map(_stack_for_devices, tree)
    return jax.device_put(stacked_tree, sharding)


def unreplicate_from_local_devices(tree):
    """Return the first local replica from a replicated pytree."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)
