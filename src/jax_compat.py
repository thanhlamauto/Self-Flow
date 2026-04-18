"""Compatibility helpers for newer JAX releases."""

from __future__ import annotations

import jax


def replicate_to_local_devices(tree, devices=None):
    """Replicate a pytree across local devices for pmap inputs."""
    devices = tuple(devices) if devices is not None else tuple(jax.local_devices())
    return jax.device_put_sharded([tree] * len(devices), list(devices))


def unreplicate_from_local_devices(tree):
    """Return the first local replica from a replicated pytree."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)
