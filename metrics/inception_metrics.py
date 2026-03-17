"""
Inception-based metrics for generative model evaluation.

This module provides:
- Inception Score (IS)
- Spatial FID (sFID)
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg
from typing import Tuple, Optional


def compute_inception_score(
    images: np.ndarray,
    inception_fn,
    batch_size: int = 256,
    splits: int = 10,
) -> Tuple[float, float]:
    """Compute Inception Score (IS) for generated images.

    Args:
        images: Generated images [N, H, W, C] uint8 [0, 255]
        inception_fn: Function that maps images to logits
        batch_size: Batch size for feature extraction
        splits: Number of splits for computing mean/std

    Returns:
        is_mean: Mean Inception Score across splits
        is_std: Std of Inception Score across splits
    """
    n_images = images.shape[0]
    assert n_images >= splits, f"Need at least {splits} images, got {n_images}"

    # Extract logits from all images
    all_logits = []
    for i in range(0, n_images, batch_size):
        batch = images[i:i + batch_size]
        # Convert to float32 and normalize to [0, 1]
        batch_f32 = batch.astype(np.float32) / 255.0
        # Get logits (before softmax)
        logits = inception_fn(batch_f32, return_logits=True)
        all_logits.append(np.asarray(logits))

    all_logits = np.concatenate(all_logits, axis=0)  # [N, 1000]

    # Compute IS for each split
    split_scores = []
    split_size = n_images // splits

    for k in range(splits):
        start = k * split_size
        end = (k + 1) * split_size if k < splits - 1 else n_images
        split_logits = all_logits[start:end]

        # Convert logits to probabilities
        probs = jax.nn.softmax(split_logits, axis=1)  # p(y|x)

        # Marginal distribution p(y) = mean over all x
        marginal = np.mean(probs, axis=0, keepdims=True)  # [1, 1000]

        # KL divergence: sum_x p(y|x) * log(p(y|x) / p(y))
        kl_div = probs * (np.log(probs + 1e-10) - np.log(marginal + 1e-10))
        kl_div = np.sum(kl_div, axis=1)  # [N]

        # IS = exp(E[KL])
        is_score = np.exp(np.mean(kl_div))
        split_scores.append(is_score)

    is_mean = float(np.mean(split_scores))
    is_std = float(np.std(split_scores))

    return is_mean, is_std


def compute_sfid(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    inception_fn,
    batch_size: int = 256,
) -> float:
    """Compute Spatial FID (sFID) using spatial features from Inception.

    sFID uses features from an intermediate conv layer instead of the final
    pooled features, making it more sensitive to spatial structure.

    Args:
        real_images: Real images [N, H, W, C] uint8 [0, 255]
        fake_images: Generated images [M, H, W, C] uint8 [0, 255]
        inception_fn: Function that extracts spatial features
        batch_size: Batch size for feature extraction

    Returns:
        sfid: Spatial FID score (lower is better)
    """
    def extract_spatial_features(images):
        """Extract spatial features from all images."""
        all_features = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]
            # Convert to float32 and normalize
            batch_f32 = batch.astype(np.float32) / 255.0
            # Extract spatial features (e.g., from mixed_6e layer)
            feats = inception_fn(batch_f32, return_spatial=True)
            # Average pool spatial dimensions: [B, H, W, C] -> [B, C]
            feats_pooled = np.mean(feats, axis=(1, 2))
            all_features.append(np.asarray(feats_pooled))
        return np.concatenate(all_features, axis=0)

    # Extract features
    real_feats = extract_spatial_features(real_images)
    fake_feats = extract_spatial_features(fake_images)

    # Compute FID on spatial features
    mu_real = np.mean(real_feats, axis=0)
    mu_fake = np.mean(fake_feats, axis=0)

    sigma_real = np.cov(real_feats, rowvar=False)
    sigma_fake = np.cov(fake_feats, rowvar=False)

    # Compute FID
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)

    # Handle numerical issues
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_fake + offset))

    # Handle complex values from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    sfid = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean)

    return float(sfid)
