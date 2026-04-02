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


def weighted_cosine_loss_per_pixel(prediction, target, channel_weights, *, eps: float):
    weights = channel_weights[:, :, :, None, None]
    weighted_prediction = prediction * weights
    weighted_target = target * weights
    pred_norm = weighted_prediction / jnp.maximum(
        jnp.linalg.norm(weighted_prediction, axis=2, keepdims=True), eps
    )
    target_norm = weighted_target / jnp.maximum(
        jnp.linalg.norm(weighted_target, axis=2, keepdims=True), eps
    )
    cosine = jnp.sum(pred_norm * target_norm, axis=2)
    loss = 1.0 - cosine
    return loss, cosine


def channel_schedule(
    noise_level,
    layer_depth,
    *,
    beta_slope: float,
    color_offset: float,
    mixed_offset: float,
    mixed_scale: float,
    eps: float,
):
    """Return smooth channel gates and their normalized budget target."""
    beta_shape = jax.nn.sigmoid(
        beta_slope * (layer_depth[None, :] - noise_level[:, None])
    )
    beta_color = jax.nn.sigmoid(
        beta_slope * (layer_depth[None, :] - noise_level[:, None] - color_offset)
    )
    beta_mixed = jax.nn.sigmoid(
        beta_slope * (layer_depth[None, :] - noise_level[:, None] - mixed_offset)
    )
    raw_weights = jnp.stack(
        [
            0.5 + 0.5 * beta_shape,
            1.0 - 0.5 * beta_shape,
            beta_color,
            mixed_scale * beta_mixed,
        ],
        axis=2,
    )
    normalized_budget = raw_weights / jnp.maximum(
        jnp.sum(raw_weights, axis=2, keepdims=True),
        eps,
    )
    return beta_shape, beta_color, beta_mixed, raw_weights, normalized_budget


def channel_energy_budget(deltas, *, eps: float):
    """Convert per-layer latent deltas into a channel update budget."""
    channel_energy = jnp.mean(jnp.square(deltas), axis=(3, 4))
    predicted_budget = channel_energy / jnp.maximum(
        jnp.sum(channel_energy, axis=2, keepdims=True),
        eps,
    )
    return predicted_budget, channel_energy


def kl_budget_loss(target_budget, predicted_budget, *, eps: float):
    """KL(target || prediction) per sample and layer."""
    target_budget = jax.lax.stop_gradient(target_budget)
    safe_target = jnp.maximum(target_budget, eps)
    safe_prediction = jnp.maximum(predicted_budget, eps)
    return jnp.sum(
        safe_target * (jnp.log(safe_target) - jnp.log(safe_prediction)),
        axis=2,
    )


def make_zero_stats(depth: int, noise_level):
    zeros = jnp.zeros((depth,), dtype=jnp.float32)
    return {
        "reg_loss": jnp.array(0.0, dtype=jnp.float32),
        "per_layer_loss": zeros,
        "per_layer_beta": zeros,
        "per_layer_blur": zeros,
        "per_layer_cosine": zeros,
        "per_layer_weight_c1": zeros,
        "per_layer_weight_c2": zeros,
        "per_layer_weight_c3": zeros,
        "per_layer_weight_c4": zeros,
        "per_layer_color_gate": zeros,
        "per_layer_mixed_gate": zeros,
        "per_layer_budget_target_c1": zeros,
        "per_layer_budget_target_c2": zeros,
        "per_layer_budget_target_c3": zeros,
        "per_layer_budget_target_c4": zeros,
        "per_layer_budget_pred_c1": zeros,
        "per_layer_budget_pred_c2": zeros,
        "per_layer_budget_pred_c3": zeros,
        "per_layer_budget_pred_c4": zeros,
        "per_layer_energy_c1": zeros,
        "per_layer_energy_c2": zeros,
        "per_layer_energy_c3": zeros,
        "per_layer_energy_c4": zeros,
        "noise_level_mean": jnp.mean(noise_level),
    }


class NoiseDepthScaleSpaceRegularizer(nn.Module):
    """Noise- and depth-aware latent regularizer over all transformer blocks."""

    depth: int
    grid_size: int
    patch_size: int
    latent_channels: int
    sigma_min: float = 0.1
    sigma_max: float = 2.0
    beta_slope: float = 6.0
    noise_parameterization_mode: str = "normalized_sigma"
    style: str = "channel_budget"
    target_mode: str = "blend"
    color_offset: float = 0.15
    mixed_offset: float = 0.30
    mixed_scale: float = 0.5
    eps: float = 1e-6
    truncate: float = 3.0

    @nn.compact
    def __call__(self, projected_layer_tokens, x0_patchified, input_patchified, noise_value):
        projected_layer_tokens = jnp.asarray(projected_layer_tokens, dtype=jnp.float32)
        x0_patchified = jnp.asarray(x0_patchified, dtype=jnp.float32)
        input_patchified = jnp.asarray(input_patchified, dtype=jnp.float32)
        noise_value = jnp.asarray(noise_value, dtype=jnp.float32)

        if projected_layer_tokens.shape[0] != self.depth:
            raise ValueError(
                f"Expected {self.depth} layers, got {projected_layer_tokens.shape[0]}"
            )
        if self.style in {"channel_weighted", "channel_budget"} and self.latent_channels != 4:
            raise ValueError(
                f"{self.style} assumes 4 latent channels, got {self.latent_channels}"
            )

        noise_level = noise_parameterize(
            noise_value,
            mode=self.noise_parameterization_mode,
            eps=self.eps,
        )
        if self.depth == 1:
            layer_depth = jnp.zeros((1,), dtype=jnp.float32)
        else:
            layer_depth = jnp.linspace(0.0, 1.0, self.depth, dtype=jnp.float32)

        projected_latents = jax.vmap(
            lambda tokens: patch_tokens_to_nchw(
                tokens,
                grid_size=self.grid_size,
                patch_size=self.patch_size,
                channels=self.latent_channels,
            )
        )(projected_layer_tokens)
        projected_latents = jnp.transpose(projected_latents, (1, 0, 2, 3, 4))
        batch_size = projected_latents.shape[0]
        latent_h = projected_latents.shape[3]
        latent_w = projected_latents.shape[4]

        stats = make_zero_stats(self.depth, noise_level)

        if self.style == "gaussian_scale_space":
            x0_nchw = patch_tokens_to_nchw(
                x0_patchified,
                grid_size=self.grid_size,
                patch_size=self.patch_size,
                channels=self.latent_channels,
            )
            x0_stop = jax.lax.stop_gradient(x0_nchw)
            x0_expanded = jnp.broadcast_to(
                x0_stop[:, None, :, :, :],
                (batch_size, self.depth, self.latent_channels, latent_h, latent_w),
            )
            beta = jax.nn.sigmoid(
                self.beta_slope * (layer_depth[None, :] - noise_level[:, None])
            )
            blur_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - beta)
            blur_radius = max(1, int(math.ceil(float(self.truncate) * float(self.sigma_max))))
            x0_tiled = x0_expanded.reshape(
                batch_size * self.depth,
                self.latent_channels,
                latent_h,
                latent_w,
            )
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
            if self.target_mode == "blur_only":
                target = blurred
            elif self.target_mode == "blend":
                blend = beta[:, :, None, None, None]
                target = (1.0 - blend) * blurred + blend * x0_expanded
            else:
                raise ValueError(f"Unsupported target_mode: {self.target_mode}")

            per_pixel_loss, per_pixel_cosine = cosine_loss_per_pixel(
                projected_latents,
                target,
                eps=self.eps,
            )
            per_layer_loss = jnp.mean(per_pixel_loss, axis=(0, 2, 3))
            per_layer_cosine = jnp.mean(per_pixel_cosine, axis=(0, 2, 3))
            reg_loss = jnp.mean(per_layer_loss)
            stats.update(
                {
                    "reg_loss": reg_loss,
                    "per_layer_loss": per_layer_loss,
                    "per_layer_beta": jnp.mean(beta, axis=0),
                    "per_layer_blur": jnp.mean(blur_sigma, axis=0),
                    "per_layer_cosine": per_layer_cosine,
                }
            )
            return reg_loss, stats

        beta_shape, beta_color, beta_mixed, raw_weights, normalized_budget = channel_schedule(
            noise_level,
            layer_depth,
            beta_slope=self.beta_slope,
            color_offset=self.color_offset,
            mixed_offset=self.mixed_offset,
            mixed_scale=self.mixed_scale,
            eps=self.eps,
        )

        stats.update(
            {
                "per_layer_beta": jnp.mean(beta_shape, axis=0),
                "per_layer_weight_c1": jnp.mean(raw_weights[:, :, 0], axis=0),
                "per_layer_weight_c2": jnp.mean(raw_weights[:, :, 1], axis=0),
                "per_layer_weight_c3": jnp.mean(raw_weights[:, :, 2], axis=0),
                "per_layer_weight_c4": jnp.mean(raw_weights[:, :, 3], axis=0),
                "per_layer_color_gate": jnp.mean(beta_color, axis=0),
                "per_layer_mixed_gate": jnp.mean(beta_mixed, axis=0),
                "per_layer_budget_target_c1": jnp.mean(normalized_budget[:, :, 0], axis=0),
                "per_layer_budget_target_c2": jnp.mean(normalized_budget[:, :, 1], axis=0),
                "per_layer_budget_target_c3": jnp.mean(normalized_budget[:, :, 2], axis=0),
                "per_layer_budget_target_c4": jnp.mean(normalized_budget[:, :, 3], axis=0),
            }
        )

        if self.style == "channel_weighted":
            x0_nchw = patch_tokens_to_nchw(
                x0_patchified,
                grid_size=self.grid_size,
                patch_size=self.patch_size,
                channels=self.latent_channels,
            )
            x0_stop = jax.lax.stop_gradient(x0_nchw)
            x0_expanded = jnp.broadcast_to(
                x0_stop[:, None, :, :, :],
                (batch_size, self.depth, self.latent_channels, latent_h, latent_w),
            )
            per_pixel_loss, per_pixel_cosine = weighted_cosine_loss_per_pixel(
                projected_latents,
                x0_expanded,
                raw_weights,
                eps=self.eps,
            )
            per_layer_loss = jnp.mean(per_pixel_loss, axis=(0, 2, 3))
            per_layer_cosine = jnp.mean(per_pixel_cosine, axis=(0, 2, 3))
            reg_loss = jnp.mean(per_layer_loss)
            stats.update(
                {
                    "reg_loss": reg_loss,
                    "per_layer_loss": per_layer_loss,
                    "per_layer_cosine": per_layer_cosine,
                }
            )
            return reg_loss, stats

        if self.style == "channel_budget":
            input_nchw = patch_tokens_to_nchw(
                input_patchified,
                grid_size=self.grid_size,
                patch_size=self.patch_size,
                channels=self.latent_channels,
            )
            prev_latents = jnp.concatenate(
                [
                    jax.lax.stop_gradient(input_nchw)[:, None, :, :, :],
                    jax.lax.stop_gradient(projected_latents[:, :-1, :, :, :]),
                ],
                axis=1,
            )
            deltas = projected_latents - prev_latents
            predicted_budget, channel_energy = channel_energy_budget(
                deltas,
                eps=self.eps,
            )
            per_sample_layer_loss = kl_budget_loss(
                normalized_budget,
                predicted_budget,
                eps=self.eps,
            )
            per_layer_loss = jnp.mean(per_sample_layer_loss, axis=0)
            reg_loss = jnp.mean(per_layer_loss)
            stats.update(
                {
                    "reg_loss": reg_loss,
                    "per_layer_loss": per_layer_loss,
                    "per_layer_budget_pred_c1": jnp.mean(predicted_budget[:, :, 0], axis=0),
                    "per_layer_budget_pred_c2": jnp.mean(predicted_budget[:, :, 1], axis=0),
                    "per_layer_budget_pred_c3": jnp.mean(predicted_budget[:, :, 2], axis=0),
                    "per_layer_budget_pred_c4": jnp.mean(predicted_budget[:, :, 3], axis=0),
                    "per_layer_energy_c1": jnp.mean(channel_energy[:, :, 0], axis=0),
                    "per_layer_energy_c2": jnp.mean(channel_energy[:, :, 1], axis=0),
                    "per_layer_energy_c3": jnp.mean(channel_energy[:, :, 2], axis=0),
                    "per_layer_energy_c4": jnp.mean(channel_energy[:, :, 3], axis=0),
                }
            )
            return reg_loss, stats

        raise ValueError(f"Unsupported regularizer style: {self.style}")
