import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange


def patch_tokens_to_nchw(tokens, *, grid_size: int, patch_size: int, channels: int):
    """Convert patchified latent tokens back to latent NCHW space."""
    return rearrange(
        tokens,
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h=grid_size,
        w=grid_size,
        p1=patch_size,
        p2=patch_size,
        c=channels,
    )


def noise_parameterize(timesteps, *, mode: str, eps: float):
    """Map flow timesteps to a scalar noise level u in [0, 1]."""
    t = jnp.asarray(timesteps, dtype=jnp.float32)
    t = jnp.clip(t, 0.0, 1.0)
    if mode == "normalized_sigma":
        return 1.0 - t
    if mode == "raw_timestep":
        return t
    if mode == "normalized_logsnr":
        alpha = jnp.clip(t, eps, 1.0 - eps)
        sigma = jnp.clip(1.0 - t, eps, 1.0 - eps)
        logsnr = 2.0 * (jnp.log(alpha) - jnp.log(sigma))
        return jax.nn.sigmoid(-logsnr)
    raise ValueError(f"Unsupported noise parameterization: {mode}")


def gaussian_kernel_1d(sigma, *, radius: int, eps: float):
    coords = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    safe_sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float32), eps)
    kernel = jnp.exp(-0.5 * jnp.square(coords / safe_sigma))
    kernel = kernel / jnp.maximum(jnp.sum(kernel), eps)
    identity = jnp.zeros_like(kernel).at[radius].set(1.0)
    return jnp.where(safe_sigma <= eps, identity, kernel)


def gaussian_blur_nchw(image, sigma, *, radius: int, eps: float):
    """Depthwise separable Gaussian blur for one NCHW latent sample."""
    image = jnp.asarray(image, dtype=jnp.float32)
    channels = image.shape[0]
    kernel = gaussian_kernel_1d(sigma, radius=radius, eps=eps)
    x = jnp.transpose(image[None, ...], (0, 2, 3, 1))  # NHWC

    kernel_w = jnp.tile(kernel.reshape(1, -1, 1, 1), (1, 1, 1, channels))
    x = jnp.pad(x, ((0, 0), (0, 0), (radius, radius), (0, 0)), mode="edge")
    x = jax.lax.conv_general_dilated(
        x,
        kernel_w,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=channels,
    )

    kernel_h = jnp.tile(kernel.reshape(-1, 1, 1, 1), (1, 1, 1, channels))
    x = jnp.pad(x, ((0, 0), (radius, radius), (0, 0), (0, 0)), mode="edge")
    x = jax.lax.conv_general_dilated(
        x,
        kernel_h,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=channels,
    )
    return jnp.transpose(x[0], (2, 0, 1))


def cosine_loss_per_pixel(prediction, target, *, eps: float):
    pred_norm = prediction / jnp.maximum(
        jnp.linalg.norm(prediction, axis=2, keepdims=True), eps
    )
    target_norm = target / jnp.maximum(
        jnp.linalg.norm(target, axis=2, keepdims=True), eps
    )
    cosine = jnp.sum(pred_norm * target_norm, axis=2)
    loss = 1.0 - cosine
    return loss, cosine


class NoiseDepthScaleSpaceRegularizer(nn.Module):
    """Continuous coarse-to-fine target regularizer over all DiT layers."""

    depth: int
    grid_size: int
    patch_size: int
    latent_channels: int
    sigma_min: float = 0.1
    sigma_max: float = 2.0
    beta_slope: float = 6.0
    noise_parameterization_mode: str = "normalized_sigma"
    target_mode: str = "blend"
    eps: float = 1e-6
    truncate: float = 3.0

    @nn.compact
    def __call__(self, projected_layer_tokens, x0_patchified, noise_value):
        projected_layer_tokens = jnp.asarray(projected_layer_tokens, dtype=jnp.float32)
        x0_patchified = jnp.asarray(x0_patchified, dtype=jnp.float32)
        noise_value = jnp.asarray(noise_value, dtype=jnp.float32)

        if projected_layer_tokens.shape[0] != self.depth:
            raise ValueError(
                f"Expected {self.depth} layers, got {projected_layer_tokens.shape[0]}"
            )

        x0_nchw = patch_tokens_to_nchw(
            x0_patchified,
            grid_size=self.grid_size,
            patch_size=self.patch_size,
            channels=self.latent_channels,
        )
        x0_stop = jax.lax.stop_gradient(x0_nchw)
        batch_size = x0_stop.shape[0]
        latent_h = x0_stop.shape[2]
        latent_w = x0_stop.shape[3]

        noise_level = noise_parameterize(
            noise_value,
            mode=self.noise_parameterization_mode,
            eps=self.eps,
        )
        if self.depth == 1:
            layer_depth = jnp.zeros((1,), dtype=jnp.float32)
        else:
            layer_depth = jnp.linspace(0.0, 1.0, self.depth, dtype=jnp.float32)

        beta = jax.nn.sigmoid(
            self.beta_slope * (layer_depth[None, :] - noise_level[:, None])
        )
        blur_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - beta)

        blur_radius = max(1, int(math.ceil(float(self.truncate) * float(self.sigma_max))))
        x0_tiled = jnp.broadcast_to(
            x0_stop[:, None, :, :, :],
            (batch_size, self.depth, self.latent_channels, latent_h, latent_w),
        ).reshape(batch_size * self.depth, self.latent_channels, latent_h, latent_w)
        blur_sigma_flat = blur_sigma.reshape(-1)
        blurred = jax.vmap(
            lambda image, sigma: gaussian_blur_nchw(
                image,
                sigma,
                radius=blur_radius,
                eps=self.eps,
            )
        )(x0_tiled, blur_sigma_flat).reshape(
            batch_size,
            self.depth,
            self.latent_channels,
            latent_h,
            latent_w,
        )

        x0_expanded = jnp.broadcast_to(
            x0_stop[:, None, :, :, :],
            (batch_size, self.depth, self.latent_channels, latent_h, latent_w),
        )
        if self.target_mode == "blur_only":
            target = blurred
        elif self.target_mode == "blend":
            blend = beta[:, :, None, None, None]
            target = (1.0 - blend) * blurred + blend * x0_expanded
        else:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")

        projected_latents = jax.vmap(
            lambda tokens: patch_tokens_to_nchw(
                tokens,
                grid_size=self.grid_size,
                patch_size=self.patch_size,
                channels=self.latent_channels,
            )
        )(projected_layer_tokens)
        projected_latents = jnp.transpose(projected_latents, (1, 0, 2, 3, 4))

        per_pixel_loss, per_pixel_cosine = cosine_loss_per_pixel(
            projected_latents,
            target,
            eps=self.eps,
        )
        per_layer_loss = jnp.mean(per_pixel_loss, axis=(0, 2, 3))
        per_layer_cosine = jnp.mean(per_pixel_cosine, axis=(0, 2, 3))
        reg_loss = jnp.mean(per_layer_loss)

        stats = {
            "reg_loss": reg_loss,
            "per_layer_loss": per_layer_loss,
            "per_layer_beta": jnp.mean(beta, axis=0),
            "per_layer_blur": jnp.mean(blur_sigma, axis=0),
            "per_layer_cosine": per_layer_cosine,
            "noise_level_mean": jnp.mean(noise_level),
        }
        return reg_loss, stats
