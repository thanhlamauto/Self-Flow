"""
JAX/Flax InceptionV3 for FID computation.
Ported from https://github.com/matthias-wright/jax-fid
(same version used in FSDiT — proven to work on Kaggle TPU v5e-8).

Public API
----------
get_fid_network()  → apply_fn
    Returns a pmap-wrapped InceptionV3 apply function.
    Input shape : (n_devices, batch, 299, 299, 3)  values in [-1, 1]
    Output shape: (n_devices, batch, 1, 1, 2048)

fid_from_stats(mu1, sigma1, mu2, sigma2) → float
    Fréchet distance between two Gaussian distributions.
"""

import os
import pickle
import tempfile
import functools
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import requests
from tqdm import tqdm
import numpy as np
import scipy.linalg

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from jax import lax
from flax.linen.module import merge_param
from src.jax_compat import replicate_to_local_devices

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any


# ─────────────────────────────────────────────────────────────────────────────
#  Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_fid_network():
    """Backward-compatible pooled-only Inception for FID.

    Returns a callable: apply_fn(imgs) where imgs has shape
    (n_devices, batch, 299, 299, 3) with pixel values in [-1, 1].
    Returns activations of shape (n_devices, batch, 1, 1, 2048).
    """
    apply_fn = get_inception_network(mode="pooled")

    def _pooled_only(imgs):
        out = apply_fn(imgs)
        # mode="pooled" returns pooled activations already.
        return out

    return _pooled_only


def get_inception_network(mode: str = "pooled"):
    """Build and pmap-wrap InceptionV3 with pretrained weights.

    Args:
        mode:
          - "pooled": returns pooled activations, shape (n_devices, batch, 1, 1, 2048)
          - "pooled+spatial": returns (pooled, spatial) where
              pooled  shape (n_devices, batch, 1, 1, 2048)
              spatial shape (n_devices, batch, H, W, 2048) (pre-GAP)

    Input imgs: (n_devices, batch, 299, 299, 3) in [-1, 1]
    """
    mode = str(mode)
    if mode not in ("pooled", "pooled+spatial"):
        raise ValueError(f"Unsupported mode {mode!r}; expected 'pooled' or 'pooled+spatial'")

    return_spatial = mode == "pooled+spatial"
    model = InceptionV3(pretrained=True)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, 299, 299, 3)), return_spatial=return_spatial)
    params = replicate_to_local_devices(params, devices=jax.local_devices())
    apply_fn = jax.pmap(functools.partial(model.apply, train=False, return_spatial=return_spatial))
    return functools.partial(apply_fn, params)


def fid_from_stats(mu1, sigma1, mu2, sigma2):
    """Fréchet Inception Distance from pre-computed Gaussian stats.

    Adds a small diagonal offset to each covariance before the matrix
    sqrt (matches pytorch-fid behaviour, improves numerical stability).
    """
    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * 1e-6
    covmean, _ = scipy.linalg.sqrtm(
        (sigma1 + offset) @ (sigma2 + offset), disp=False
    )
    covmean = np.real(covmean)
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


# ─────────────────────────────────────────────────────────────────────────────
#  Weight download
# ─────────────────────────────────────────────────────────────────────────────

_INCEPTION_URL = (
    "https://www.dropbox.com/s/xt6zvlvt22dcwck/inception_v3_weights_fid.pickle?dl=1"
)


def _download(url: str, ckpt_dir: Optional[str] = None) -> str:
    name = url[url.rfind("/") + 1: url.rfind("?")]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'[fid_utils] Downloading InceptionV3 weights to {ckpt_file}')
        os.makedirs(ckpt_dir, exist_ok=True)
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        tmp = ckpt_file + ".tmp"
        with open(tmp, "wb") as f:
            for chunk in tqdm(resp.iter_content(1024),
                              total=total // 1024, unit="KB"):
                f.write(chunk)
        os.rename(tmp, ckpt_file)
    return ckpt_file


def _get(d, key):
    return None if (d is None or key not in d) else d[key]


# ─────────────────────────────────────────────────────────────────────────────
#  BatchNorm
# ─────────────────────────────────────────────────────────────────────────────

class _BatchNorm(nn.Module):
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable = nn.initializers.zeros
    scale_init: Callable = nn.initializers.ones
    mean_init: Callable = lambda s: jnp.zeros(s, jnp.float32)
    var_init: Callable = lambda s: jnp.ones(s, jnp.float32)
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @nn.compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        ura = merge_param("use_running_average",
                          self.use_running_average, use_running_average)
        x = jnp.asarray(x, jnp.float32)
        axis = (self.axis,) if isinstance(self.axis, int) else self.axis
        axis = tuple(a % x.ndim for a in axis)
        feat_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        red_feat = tuple(d for i, d in enumerate(x.shape) if i in axis)
        red_axes = tuple(i for i in range(x.ndim) if i not in axis)

        init = self.is_mutable_collection("params")
        ra_mean = self.variable("batch_stats", "mean", self.mean_init, red_feat)
        ra_var = self.variable("batch_stats", "var", self.var_init, red_feat)

        if ura:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=red_axes)
            mean2 = jnp.mean(lax.square(x), axis=red_axes)
            if self.axis_name and not init:
                cat = jnp.concatenate([mean, mean2])
                mean, mean2 = jnp.split(
                    lax.pmean(cat, axis_name=self.axis_name,
                              axis_index_groups=self.axis_index_groups), 2)
            var = mean2 - lax.square(mean)
            if not init:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        y = (x - mean.reshape(feat_shape)) * lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            y *= self.param("scale", self.scale_init, red_feat).reshape(feat_shape)
        if self.use_bias:
            y += self.param("bias", self.bias_init, red_feat).reshape(feat_shape)
        return jnp.asarray(y, self.dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  Pooling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pool(inputs, init, reduce_fn, window_shape, strides, padding):
    strides = strides or (1,) * len(window_shape)
    strides = (1,) + tuple(strides) + (1,)
    dims = (1,) + tuple(window_shape) + (1,)
    squeeze = inputs.ndim == len(dims) - 1
    if squeeze:
        inputs = inputs[None]
    if not isinstance(padding, str):
        padding = ((0, 0),) + tuple(map(tuple, padding)) + ((0, 0),)
    y = lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
    return y[0] if squeeze else y


def _avg_pool(inputs, window_shape, strides=None, padding="VALID"):
    y = _pool(inputs, 0.0, lax.add, window_shape, strides, padding)
    ones = jnp.ones((1, inputs.shape[1], inputs.shape[2], 1), inputs.dtype)
    counts = lax.conv_general_dilated(
        ones,
        jnp.expand_dims(jnp.ones(window_shape, inputs.dtype), (-2, -1)),
        window_strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dimension_numbers=nn.linear._conv_dimension_numbers(ones.shape),
        feature_group_count=1,
    )
    return y / counts


# ─────────────────────────────────────────────────────────────────────────────
#  BasicConv2d
# ─────────────────────────────────────────────────────────────────────────────

class _BasicConv2d(nn.Module):
    out_channels: int
    kernel_size: Union[int, Iterable[int]] = (3, 3)
    strides: Optional[Iterable[int]] = (1, 1)
    padding: Union[str, Any] = "valid"
    use_bias: bool = False
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=(nn.initializers.lecun_normal() if pd is None
                         else lambda *_: jnp.array(pd["conv"]["kernel"])),
            bias_init=(nn.initializers.zeros if pd is None
                       else lambda *_: jnp.array(pd["conv"]["bias"])),
            dtype=self.dtype,
        )(x)
        if pd is None:
            x = _BatchNorm(epsilon=0.001, momentum=0.1,
                           use_running_average=not train, dtype=self.dtype)(x)
        else:
            x = _BatchNorm(
                epsilon=0.001, momentum=0.1,
                bias_init=lambda *_: jnp.array(pd["bn"]["bias"]),
                scale_init=lambda *_: jnp.array(pd["bn"]["scale"]),
                mean_init=lambda _: jnp.array(pd["bn"]["mean"]),
                var_init=lambda _: jnp.array(pd["bn"]["var"]),
                use_running_average=not train, dtype=self.dtype,
            )(x)
        return jax.nn.relu(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Inception blocks
# ─────────────────────────────────────────────────────────────────────────────

class _InceptionA(nn.Module):
    pool_features: int
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        b1 = _BasicConv2d(64, (1, 1), params_dict=_get(pd, "branch1x1"), dtype=self.dtype)(x, train)
        b5 = _BasicConv2d(48, (1, 1), params_dict=_get(pd, "branch5x5_1"), dtype=self.dtype)(x, train)
        b5 = _BasicConv2d(64, (5, 5), padding=((2, 2), (2, 2)),
                          params_dict=_get(pd, "branch5x5_2"), dtype=self.dtype)(b5, train)
        b3 = _BasicConv2d(64, (1, 1), params_dict=_get(pd, "branch3x3dbl_1"), dtype=self.dtype)(x, train)
        b3 = _BasicConv2d(96, (3, 3), padding=((1, 1), (1, 1)),
                          params_dict=_get(pd, "branch3x3dbl_2"), dtype=self.dtype)(b3, train)
        b3 = _BasicConv2d(96, (3, 3), padding=((1, 1), (1, 1)),
                          params_dict=_get(pd, "branch3x3dbl_3"), dtype=self.dtype)(b3, train)
        bp = _avg_pool(x, (3, 3), (1, 1), ((1, 1), (1, 1)))
        bp = _BasicConv2d(self.pool_features, (1, 1), params_dict=_get(pd, "branch_pool"), dtype=self.dtype)(bp, train)
        return jnp.concatenate([b1, b5, b3, bp], axis=-1)


class _InceptionB(nn.Module):
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        b3 = _BasicConv2d(384, (3, 3), strides=(2, 2),
                          params_dict=_get(pd, "branch3x3"), dtype=self.dtype)(x, train)
        bd = _BasicConv2d(64, (1, 1), params_dict=_get(pd, "branch3x3dbl_1"), dtype=self.dtype)(x, train)
        bd = _BasicConv2d(96, (3, 3), padding=((1, 1), (1, 1)),
                          params_dict=_get(pd, "branch3x3dbl_2"), dtype=self.dtype)(bd, train)
        bd = _BasicConv2d(96, (3, 3), strides=(2, 2),
                          params_dict=_get(pd, "branch3x3dbl_3"), dtype=self.dtype)(bd, train)
        bp = nn.max_pool(x, (3, 3), strides=(2, 2))
        return jnp.concatenate([b3, bd, bp], axis=-1)


class _InceptionC(nn.Module):
    channels_7x7: int
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        c = self.channels_7x7
        b1 = _BasicConv2d(192, (1, 1), params_dict=_get(pd, "branch1x1"), dtype=self.dtype)(x, train)
        b7 = _BasicConv2d(c, (1, 1), params_dict=_get(pd, "branch7x7_1"), dtype=self.dtype)(x, train)
        b7 = _BasicConv2d(c, (1, 7), padding=((0, 0), (3, 3)),
                          params_dict=_get(pd, "branch7x7_2"), dtype=self.dtype)(b7, train)
        b7 = _BasicConv2d(192, (7, 1), padding=((3, 3), (0, 0)),
                          params_dict=_get(pd, "branch7x7_3"), dtype=self.dtype)(b7, train)
        bd = _BasicConv2d(c, (1, 1), params_dict=_get(pd, "branch7x7dbl_1"), dtype=self.dtype)(x, train)
        bd = _BasicConv2d(c, (7, 1), padding=((3, 3), (0, 0)),
                          params_dict=_get(pd, "branch7x7dbl_2"), dtype=self.dtype)(bd, train)
        bd = _BasicConv2d(c, (1, 7), padding=((0, 0), (3, 3)),
                          params_dict=_get(pd, "branch7x7dbl_3"), dtype=self.dtype)(bd, train)
        bd = _BasicConv2d(c, (7, 1), padding=((3, 3), (0, 0)),
                          params_dict=_get(pd, "branch7x7dbl_4"), dtype=self.dtype)(bd, train)
        bd = _BasicConv2d(192, (1, 7), padding=((0, 0), (3, 3)),
                          params_dict=_get(pd, "branch7x7dbl_5"), dtype=self.dtype)(bd, train)
        bp = _avg_pool(x, (3, 3), (1, 1), ((1, 1), (1, 1)))
        bp = _BasicConv2d(192, (1, 1), params_dict=_get(pd, "branch_pool"), dtype=self.dtype)(bp, train)
        return jnp.concatenate([b1, b7, bd, bp], axis=-1)


class _InceptionD(nn.Module):
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        b3 = _BasicConv2d(192, (1, 1), params_dict=_get(pd, "branch3x3_1"), dtype=self.dtype)(x, train)
        b3 = _BasicConv2d(320, (3, 3), strides=(2, 2),
                          params_dict=_get(pd, "branch3x3_2"), dtype=self.dtype)(b3, train)
        b7 = _BasicConv2d(192, (1, 1), params_dict=_get(pd, "branch7x7x3_1"), dtype=self.dtype)(x, train)
        b7 = _BasicConv2d(192, (1, 7), padding=((0, 0), (3, 3)),
                          params_dict=_get(pd, "branch7x7x3_2"), dtype=self.dtype)(b7, train)
        b7 = _BasicConv2d(192, (7, 1), padding=((3, 3), (0, 0)),
                          params_dict=_get(pd, "branch7x7x3_3"), dtype=self.dtype)(b7, train)
        b7 = _BasicConv2d(192, (3, 3), strides=(2, 2),
                          params_dict=_get(pd, "branch7x7x3_4"), dtype=self.dtype)(b7, train)
        bp = nn.max_pool(x, (3, 3), strides=(2, 2))
        return jnp.concatenate([b3, b7, bp], axis=-1)


class _InceptionE(nn.Module):
    pooling: Callable
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        b1 = _BasicConv2d(320, (1, 1), params_dict=_get(pd, "branch1x1"), dtype=self.dtype)(x, train)
        b3 = _BasicConv2d(384, (1, 1), params_dict=_get(pd, "branch3x3_1"), dtype=self.dtype)(x, train)
        b3a = _BasicConv2d(384, (1, 3), padding=((0, 0), (1, 1)),
                           params_dict=_get(pd, "branch3x3_2a"), dtype=self.dtype)(b3, train)
        b3b = _BasicConv2d(384, (3, 1), padding=((1, 1), (0, 0)),
                           params_dict=_get(pd, "branch3x3_2b"), dtype=self.dtype)(b3, train)
        b3 = jnp.concatenate([b3a, b3b], axis=-1)
        bd = _BasicConv2d(448, (1, 1), params_dict=_get(pd, "branch3x3dbl_1"), dtype=self.dtype)(x, train)
        bd = _BasicConv2d(384, (3, 3), padding=((1, 1), (1, 1)),
                          params_dict=_get(pd, "branch3x3dbl_2"), dtype=self.dtype)(bd, train)
        bda = _BasicConv2d(384, (1, 3), padding=((0, 0), (1, 1)),
                           params_dict=_get(pd, "branch3x3dbl_3a"), dtype=self.dtype)(bd, train)
        bdb = _BasicConv2d(384, (3, 1), padding=((1, 1), (0, 0)),
                           params_dict=_get(pd, "branch3x3dbl_3b"), dtype=self.dtype)(bd, train)
        bd = jnp.concatenate([bda, bdb], axis=-1)
        bp = self.pooling(x, (3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        bp = _BasicConv2d(192, (1, 1), params_dict=_get(pd, "branch_pool"), dtype=self.dtype)(bp, train)
        return jnp.concatenate([b1, b3, bd, bp], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  InceptionV3
# ─────────────────────────────────────────────────────────────────────────────

class InceptionV3(nn.Module):
    pretrained: bool = False
    dtype: str = "float32"

    def setup(self):
        if self.pretrained:
            ckpt = _download(_INCEPTION_URL)
            self.params_dict = pickle.load(open(ckpt, "rb"))
        else:
            self.params_dict = None

    @nn.compact
    def __call__(self, x, train=True, return_spatial: bool = False):
        pd = self.params_dict
        x = _BasicConv2d(32, (3, 3), strides=(2, 2), params_dict=_get(pd, "Conv2d_1a_3x3"), dtype=self.dtype)(x, train)
        x = _BasicConv2d(32, (3, 3), params_dict=_get(pd, "Conv2d_2a_3x3"), dtype=self.dtype)(x, train)
        x = _BasicConv2d(64, (3, 3), padding=((1, 1), (1, 1)),
                         params_dict=_get(pd, "Conv2d_2b_3x3"), dtype=self.dtype)(x, train)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))
        x = _BasicConv2d(80, (1, 1), params_dict=_get(pd, "Conv2d_3b_1x1"), dtype=self.dtype)(x, train)
        x = _BasicConv2d(192, (3, 3), params_dict=_get(pd, "Conv2d_4a_3x3"), dtype=self.dtype)(x, train)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))
        x = _InceptionA(32, _get(pd, "Mixed_5b"), self.dtype)(x, train)
        x = _InceptionA(64, _get(pd, "Mixed_5c"), self.dtype)(x, train)
        x = _InceptionA(64, _get(pd, "Mixed_5d"), self.dtype)(x, train)
        x = _InceptionB(_get(pd, "Mixed_6a"), self.dtype)(x, train)
        x = _InceptionC(128, _get(pd, "Mixed_6b"), self.dtype)(x, train)
        x = _InceptionC(160, _get(pd, "Mixed_6c"), self.dtype)(x, train)
        x = _InceptionC(160, _get(pd, "Mixed_6d"), self.dtype)(x, train)
        x = _InceptionC(192, _get(pd, "Mixed_6e"), self.dtype)(x, train)
        x = _InceptionD(_get(pd, "Mixed_7a"), self.dtype)(x, train)
        x = _InceptionE(_avg_pool, _get(pd, "Mixed_7b"), self.dtype)(x, train)
        x = _InceptionE(nn.max_pool, _get(pd, "Mixed_7c"), self.dtype)(x, train)
        spatial = x
        pooled = jnp.mean(spatial, axis=(1, 2), keepdims=True)
        if return_spatial:
            return pooled, spatial
        return pooled
