"""
Self-Flow Utility Functions (JAX/Flax version).

This module contains utility functions for positional encoding and
token processing used in Self-Flow inference.
"""

from typing import Literal, Tuple, Optional

import jax
import jax.numpy as jnp
from einops import rearrange

Axes = Tuple[Literal["t", "h", "w", "l"], ...]


def cartesian_prod(*arrays):
    """JAX equivalent of torch.cartesian_prod."""
    meshes = jnp.meshgrid(*arrays, indexing='ij')
    return jnp.stack(meshes, axis=-1).reshape(-1, len(arrays))


def prc_vid(
    x: jax.Array, t_coord: Optional[jax.Array] = None, l_coord: Optional[jax.Array] = None
) -> tuple[jax.Array, jax.Array]:
    b_dim = False
    if x.ndim == 5:
        b_dim = True
        b, c, t, h, w = x.shape
    else:
        c, t, h, w = x.shape

    if t_coord is None:
        t_coord = jnp.arange(t)
    if l_coord is None:
        l_coord = jnp.arange(1)

    x_ids = cartesian_prod(t_coord, jnp.arange(h), jnp.arange(w), l_coord)
    
    if b_dim:
        x = rearrange(x, "b c t h w -> b (t h w) c")
        x_ids = jnp.tile(x_ids[None, ...], (b, 1, 1))
    else:
        x = rearrange(x, "c t h w -> (t h w) c")
        
    return x, x_ids


def prc_img(
    x: jax.Array, t_coord: Optional[jax.Array] = None, l_coord: Optional[jax.Array] = None
) -> tuple[jax.Array, jax.Array]:
    b_dim = False
    if x.ndim == 4:
        b_dim = True
        b, c, h, w = x.shape
    else:
        c, h, w = x.shape

    if t_coord is None:
        t_coord = jnp.arange(1)
    if l_coord is None:
        l_coord = jnp.arange(1)

    x_ids = cartesian_prod(t_coord, jnp.arange(h), jnp.arange(w), l_coord)
    
    if b_dim:
        x = rearrange(x, "b c h w -> b (h w) c")
        x_ids = jnp.tile(x_ids[None, ...], (b, 1, 1))
    else:
        x = rearrange(x, "c h w -> (h w) c")
        
    return x, x_ids


def prc_txt(
    x: jax.Array, t_coord: Optional[jax.Array] = None, l_coord: Optional[jax.Array] = None
) -> tuple[jax.Array, jax.Array]:
    assert l_coord is None, "l_coord not supported for txts"
    b_dim = False
    if x.ndim == 3:
        b_dim = True
        b, l, c = x.shape
    else:
        l, c = x.shape

    if t_coord is None:
        t_coord = jnp.arange(1)

    x_ids = cartesian_prod(t_coord, jnp.arange(1), jnp.arange(1), jnp.arange(l))
    
    if b_dim:
        x_ids = jnp.tile(x_ids[None, ...], (b, 1, 1))
        
    return x, x_ids


def prc_txts(
    x: jax.Array, t_coord: Optional[jax.Array] = None, l_coord: Optional[jax.Array] = None
) -> tuple[jax.Array, jax.Array]:
    assert l_coord is None, "l_coord not supported for txts"
    b_dim = False
    if x.ndim == 4:
        b_dim = True
        b, t, l, c = x.shape
    else:
        t, l, c = x.shape

    if t_coord is None:
        t_coord = jnp.arange(t)

    x_ids = cartesian_prod(t_coord, jnp.arange(1), jnp.arange(1), jnp.arange(l))
    
    if b_dim:
        x = rearrange(x, "b t l c -> b (t l) c")
        x_ids = jnp.tile(x_ids[None, ...], (b, 1, 1))
    else:
        x = rearrange(x, "t l c -> (t l) c")
        
    return x, x_ids


def prc_times(t_coord: jax.Array) -> jax.Array:
    if t_coord.ndim == 0:
        t_coord = jnp.array([t_coord])
    x_ids = cartesian_prod(t_coord.astype(jnp.int32), jnp.arange(1), jnp.arange(1), jnp.arange(1))
    return x_ids


# Drop-in replacements for batched wrappers
def batched_prc_vid(x: jax.Array, t_coord=None, l_coord=None):
    return prc_vid(x, t_coord, l_coord)

def batched_prc_img(x: jax.Array, t_coord=None, l_coord=None):
    return prc_img(x, t_coord, l_coord)

def batched_prc_txt(x: jax.Array, t_coord=None, l_coord=None):
    return prc_txt(x, t_coord, l_coord)

def batched_prc_txts(x: jax.Array, t_coord=None, l_coord=None):
    return prc_txts(x, t_coord, l_coord)

def batched_prc_times(t_coord: jax.Array) -> jax.Array:
    return jax.vmap(prc_times)(t_coord)


def compress_time(t_ids: jax.Array) -> jax.Array:
    """
    Compressing time ids i.e.:
    [0, 0 ... 4, 4 ... 8, 8 ...] ->  [0, 0 ... 1, 1 ... 2, 2 ...]
    """
    assert t_ids.ndim == 1
    t_unique_sorted_ids = jnp.unique(t_ids, size=len(t_ids))
    # We must slice the padded unique array if using size in JAX, but simple unique works outside JIT bounds
    t_unique_sorted_ids_real = jnp.unique(t_ids)
    return jnp.searchsorted(t_unique_sorted_ids_real, t_ids)


def scatter_ids(x: jax.Array, x_ids: jax.Array) -> jax.Array:
    """
    Using position ids to scatter tokens into place.
    x: (B, N, C), x_ids: (B, N, 4)
    """
    b, n, ch = x.shape
    
    def process_single(data, pos):
        t_c = pos[:, 0].astype(jnp.int32)
        h_c = pos[:, 1].astype(jnp.int32)
        w_c = pos[:, 2].astype(jnp.int32)
        
        t_cmpr = compress_time(t_c)
        t_max = jnp.max(t_cmpr) + 1
        h_max = jnp.max(h_c) + 1
        w_max = jnp.max(w_c) + 1
        
        flat_ids = t_cmpr * w_max * h_max + h_c * w_max + w_c
        
        out = jnp.zeros((t_max * h_max * w_max, ch), dtype=data.dtype)
        out = out.at[flat_ids].set(data)
        return rearrange(out, "(t h w) c -> c t h w", t=t_max, h=h_max, w=w_max)
        
    return jax.vmap(process_single)(x, x_ids)


def times_to_ids(time: jax.Array) -> jax.Array:
    """Using a unit of 10 ms per index."""
    return (time * 100).astype(jnp.int32) # time * 1000 // 10


def ids_to_times(ids: jax.Array) -> jax.Array:
    """Using a unit of 10 ms per index."""
    return ids * 10 / 1000


def scattercat(x: jax.Array, x_ids: jax.Array) -> jax.Array:
    """Scatter tokens to spatial format and concatenate."""
    scattered = scatter_ids(x, x_ids) # shape (B, C, T, H, W)
    return scattered.squeeze(2) # return (B, C, H, W)


def scatter_ids_to_times(x_ids: jax.Array):
    def single_times(pos):
        t_ids = pos[:, 0].astype(jnp.int32)
        return ids_to_times(jnp.unique(t_ids))
    return jax.vmap(single_times)(x_ids)
