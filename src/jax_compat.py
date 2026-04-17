"""Compatibility helpers for JAX APIs removed in newer releases."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _stack_for_replication(x, replica_count: int):
    if isinstance(x, np.ndarray):
        return np.stack([x] * replica_count)
    return jnp.stack([x] * replica_count)


def device_put_replicated(tree, devices=None):
    """Drop-in replacement for ``jax.device_put_replicated`` supporting pytrees."""
    if devices is None:
        devices = jax.local_devices()
    devices = list(devices)
    if not devices:
        raise ValueError("Expected at least one device for replication.")

    mesh = Mesh(np.asarray(devices, dtype=object), ("batch",))
    sharding = NamedSharding(mesh, P("batch"))
    replica_count = len(devices)
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(_stack_for_replication(x, replica_count), sharding),
        tree,
    )


def replicate(tree, devices=None):
    """Replicate a pytree across devices without relying on deprecated JAX APIs."""
    return device_put_replicated(tree, devices=devices)


def unreplicate(tree):
    """Inverse of ``replicate`` for stacked leading-axis replicas."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)
