from typing import Any, Sequence

import jax
import jax.numpy as jnp


def _as_replicable_array(x: Any) -> jax.Array:
    if isinstance(x, jax.Array):
        return x
    return jnp.asarray(x)


def replicate(tree: Any, devices: Sequence[jax.Device] | None = None) -> Any:
    """Replicate a pytree for pmap without relying on deprecated JAX helpers."""
    if devices is None:
        devices = jax.local_devices()
    devices = tuple(devices)
    if not devices:
        raise ValueError("replicate() requires at least one device.")

    return jax.tree_util.tree_map(
        lambda x: jax.device_put_sharded(
            [_as_replicable_array(x)] * len(devices),
            devices,
        ),
        tree,
    )


def unreplicate(tree: Any) -> Any:
    """Take the first replica from a pmap-replicated pytree."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def device_put_replicated(value: Any, devices: Sequence[jax.Device] | None = None) -> Any:
    """Drop-in replacement for the removed ``jax.device_put_replicated``."""
    return replicate(value, devices=devices)
