"""
Precision and Recall metrics for generative models.

Based on "Improved Precision and Recall Metric for Assessing Generative Models"
(Kynkäänniemi et al., NeurIPS 2019)
"""

import numpy as np
from typing import Tuple
from sklearn.metrics import pairwise_distances


def compute_pairwise_distances(X: np.ndarray, Y: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    """Compute pairwise Euclidean distances in batches to save memory.

    Args:
        X: [N, D] array
        Y: [M, D] array
        batch_size: Batch size for computation

    Returns:
        distances: [N, M] pairwise distance matrix
    """
    n, m = X.shape[0], Y.shape[0]
    dists = np.zeros((n, m), dtype=np.float32)

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        dists[i:end_i] = pairwise_distances(X[i:end_i], Y, metric='euclidean')

    return dists


def compute_precision_recall(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    inception_fn,
    k: int = 3,
    batch_size: int = 256,
) -> Tuple[float, float]:
    """Compute Precision and Recall metrics.

    Precision: fraction of generated samples that fall within the real data manifold
    Recall: fraction of real data manifold covered by generated samples

    Args:
        real_images: Real images [N, H, W, C] uint8 [0, 255]
        fake_images: Generated images [M, H, W, C] uint8 [0, 255]
        inception_fn: Function to extract features
        k: Number of nearest neighbors (default: 3)
        batch_size: Batch size for feature extraction

    Returns:
        precision: Precision score [0, 1]
        recall: Recall score [0, 1]
    """
    def extract_features(images):
        """Extract Inception features from images."""
        all_features = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]
            # Convert to float32 and normalize
            batch_f32 = batch.astype(np.float32) / 255.0
            # Extract features (pool3 layer, 2048-dim)
            feats = inception_fn(batch_f32)
            all_features.append(np.asarray(feats))
        return np.concatenate(all_features, axis=0)

    # Extract features
    print(f"Extracting features from {real_images.shape[0]} real images...")
    real_feats = extract_features(real_images)

    print(f"Extracting features from {fake_images.shape[0]} fake images...")
    fake_feats = extract_features(fake_images)

    print(f"Computing pairwise distances...")

    # Compute pairwise distances
    # real_to_real: [N, N]
    real_to_real = compute_pairwise_distances(real_feats, real_feats, batch_size=batch_size)
    # real_to_fake: [N, M]
    real_to_fake = compute_pairwise_distances(real_feats, fake_feats, batch_size=batch_size)
    # fake_to_real: [M, N]
    fake_to_real = compute_pairwise_distances(fake_feats, real_feats, batch_size=batch_size)

    # For each real sample, find k-th nearest real neighbor (excluding itself)
    # Sort and take k+1-th element (skip self at index 0)
    real_to_real_kth = np.partition(real_to_real, k, axis=1)[:, k]

    # For each fake sample, find k-th nearest real neighbor
    fake_to_real_kth = np.partition(fake_to_real, k - 1, axis=1)[:, k - 1]

    # For each real sample, find nearest fake neighbor
    real_to_fake_1st = np.min(real_to_fake, axis=1)

    # Precision: fraction of fake samples within real manifold
    # A fake sample is "in manifold" if its k-th nearest real neighbor
    # is closer than the k-th nearest real-to-real neighbor
    precision = np.mean(fake_to_real_kth < real_to_real_kth.mean())

    # Recall: fraction of real samples covered by fake samples
    # A real sample is "covered" if its nearest fake neighbor
    # is closer than its k-th nearest real neighbor
    recall = np.mean(real_to_fake_1st < real_to_real_kth)

    return float(precision), float(recall)


def compute_prdc(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    inception_fn,
    k: int = 5,
    batch_size: int = 256,
) -> Tuple[float, float, float, float]:
    """Compute Precision, Recall, Density, and Coverage (PRDC) metrics.

    Extended version that also computes Density and Coverage.

    Args:
        real_images: Real images [N, H, W, C] uint8 [0, 255]
        fake_images: Generated images [M, H, W, C] uint8 [0, 255]
        inception_fn: Function to extract features
        k: Number of nearest neighbors
        batch_size: Batch size for feature extraction

    Returns:
        precision: Precision score
        recall: Recall score
        density: Density score
        coverage: Coverage score
    """
    # For now, just compute P/R
    # Full PRDC implementation would add Density and Coverage
    precision, recall = compute_precision_recall(
        real_images, fake_images, inception_fn, k, batch_size
    )

    # Placeholder for Density and Coverage (not implemented yet)
    density = 0.0
    coverage = 0.0

    return precision, recall, density, coverage
