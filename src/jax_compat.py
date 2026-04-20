from typing import Any, Sequence

import jax
import numpy as np


def _as_host_array(x: Any) -> np.ndarray:
    return np.asarray(x)


def _build_replicate_sharding(devices: Sequence[jax.Device], value_ndim: int) -> Any:
    sharding_mod = getattr(jax, "sharding", None)
    if sharding_mod is None:
        return None

    positional = getattr(sharding_mod, "PositionalSharding", None)
    if positional is not None:
        return positional(devices)

    mesh_cls = getattr(sharding_mod, "Mesh", None)
    named_cls = getattr(sharding_mod, "NamedSharding", None)
    partition_spec = getattr(sharding_mod, "PartitionSpec", None)
    if mesh_cls is None or named_cls is None or partition_spec is None:
        return None

    mesh = mesh_cls(np.asarray(devices, dtype=object), ("devices",))
    spec = partition_spec("devices", *([None] * value_ndim))
    return named_cls(mesh, spec)


def _replicate_leaf(x: Any, *, devices: Sequence[jax.Device], num_devices: int) -> Any:
    if x is None:
        return None
    arr = _as_host_array(x)
    stacked = np.broadcast_to(np.expand_dims(arr, axis=0), (num_devices,) + arr.shape)
    sharding = _build_replicate_sharding(devices, arr.ndim)
    if sharding is None:
        return stacked
    return jax.device_put(stacked, sharding)


def replicate(tree: Any, devices: Sequence[jax.Device] | None = None) -> Any:
    """Replicate a pytree for pmap without relying on deprecated JAX helpers."""
    if devices is None:
        devices = jax.local_devices()
    devices = tuple(devices)
    if not devices:
        raise ValueError("replicate() requires at least one device.")
    num_devices = len(devices)

    return jax.tree_util.tree_map(
        lambda x: _replicate_leaf(x, devices=devices, num_devices=num_devices),
        tree,
    )


def unreplicate(tree: Any) -> Any:
    """Take the first replica from a pmap-replicated pytree."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def device_put_replicated(value: Any, devices: Sequence[jax.Device] | None = None) -> Any:
    """Drop-in replacement for the removed ``jax.device_put_replicated``."""
    return replicate(value, devices=devices)
