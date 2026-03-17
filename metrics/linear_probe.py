"""
Linear probe evaluation for representation quality.

This module provides tools to:
1. Extract frozen features from a diffusion model
2. Train a linear classifier on top of these features
3. Evaluate top-1 accuracy on validation set
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm


class LinearClassifier(nn.Module):
    """Simple linear classifier for probe evaluation."""
    num_classes: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.num_classes)(x)


def extract_eval_features(
    model_apply_fn,
    params,
    images: jax.Array,
    timesteps: Optional[jax.Array] = None,
    labels: Optional[jax.Array] = None,
    batch_size: int = 256,
    feature_layer: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract frozen features from diffusion model for linear probe.

    Args:
        model_apply_fn: Model's apply function
        params: Model parameters (frozen)
        images: [N, ...] latent representations or images
        timesteps: [N] timesteps (if None, use t=0.5 as default)
        labels: [N] class labels (if None, use dummy labels)
        batch_size: Batch size for extraction
        feature_layer: Which layer to extract (-1 = final, or specific block index)

    Returns:
        features: [N, D] extracted features
        labels: [N] corresponding labels
    """
    n_samples = images.shape[0]

    if timesteps is None:
        # Use mid-point timestep as default
        timesteps = jnp.ones(n_samples) * 0.5

    if labels is None:
        # Use dummy labels
        labels = jnp.zeros(n_samples, dtype=jnp.int32)

    all_features = []

    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_images = images[i:end]
        batch_timesteps = timesteps[i:end]
        batch_labels = labels[i:end]

        # Forward pass to extract features
        if feature_layer == -1:
            # Use final hidden representation before prediction head
            _, feats = model_apply_fn(
                {"params": params},
                batch_images,
                timesteps=batch_timesteps,
                vector=batch_labels,
                deterministic=True,
                return_raw_features=model_apply_fn.args[0].depth,  # Last block
            )
        else:
            # Use specific layer
            _, feats = model_apply_fn(
                {"params": params},
                batch_images,
                timesteps=batch_timesteps,
                vector=batch_labels,
                deterministic=True,
                return_raw_features=feature_layer,
            )

        # Average pool over spatial dimension: [B, N, D] -> [B, D]
        feats_pooled = jnp.mean(feats, axis=1)
        all_features.append(np.asarray(feats_pooled))

    features = np.concatenate(all_features, axis=0)
    labels_np = np.asarray(labels)

    return features, labels_np


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features to zero mean and unit variance.

    Args:
        features: [N, D] features
        mean: [D] pre-computed mean (if None, compute from features)
        std: [D] pre-computed std (if None, compute from features)

    Returns:
        normalized_features: [N, D]
        mean: [D]
        std: [D]
    """
    if mean is None:
        mean = np.mean(features, axis=0, keepdims=True)
    if std is None:
        std = np.std(features, axis=0, keepdims=True) + 1e-8

    normalized = (features - mean) / std
    return normalized, mean.squeeze(), std.squeeze()


def run_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int = 1000,
    epochs: int = 90,
    batch_size: int = 16384,
    learning_rate: float = 1e-3,
    normalize: bool = True,
) -> Dict[str, float]:
    """Train a linear classifier and evaluate on validation set.

    Args:
        train_features: [N_train, D] training features
        train_labels: [N_train] training labels
        val_features: [N_val, D] validation features
        val_labels: [N_val] validation labels
        num_classes: Number of classes
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        normalize: Whether to normalize features

    Returns:
        metrics: Dict with 'top1_acc', 'top5_acc', 'loss'
    """
    print(f"Linear probe: {train_features.shape[0]} train, {val_features.shape[0]} val")

    # Normalize features
    if normalize:
        train_features, mean, std = normalize_features(train_features)
        val_features, _, _ = normalize_features(val_features, mean, std)

    # Create model
    model = LinearClassifier(num_classes=num_classes)
    rng = jax.random.PRNGKey(42)

    # Initialize
    dummy_input = jnp.ones((1, train_features.shape[1]))
    variables = model.init(rng, dummy_input)

    # Create optimizer
    tx = optax.adamw(learning_rate)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )

    # Training loop
    n_train = train_features.shape[0]
    steps_per_epoch = n_train // batch_size

    @jax.jit
    def train_step(state, batch_x, batch_y):
        def loss_fn(params):
            logits = model.apply({'params': params}, batch_x)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_step(params, batch_x, batch_y):
        logits = model.apply({'params': params}, batch_x)
        top1_pred = jnp.argmax(logits, axis=-1)
        top1_acc = jnp.mean(top1_pred == batch_y)
        top5_pred = jnp.argsort(logits, axis=-1)[:, -5:]
        top5_acc = jnp.mean(jnp.any(top5_pred == batch_y[:, None], axis=1))
        return top1_acc, top5_acc

    # Train
    for epoch in range(epochs):
        # Shuffle data
        perm = np.random.permutation(n_train)
        train_features_shuffled = train_features[perm]
        train_labels_shuffled = train_labels[perm]

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = start + batch_size
            batch_x = jnp.array(train_features_shuffled[start:end])
            batch_y = jnp.array(train_labels_shuffled[start:end])

            state, loss = train_step(state, batch_x, batch_y)
            epoch_loss += float(loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / steps_per_epoch:.4f}")

    # Evaluate on validation set
    val_top1_accs = []
    val_top5_accs = []
    n_val = val_features.shape[0]

    for i in range(0, n_val, batch_size):
        end = min(i + batch_size, n_val)
        batch_x = jnp.array(val_features[i:end])
        batch_y = jnp.array(val_labels[i:end])

        top1_acc, top5_acc = eval_step(state.params, batch_x, batch_y)
        val_top1_accs.append(float(top1_acc))
        val_top5_accs.append(float(top5_acc))

    final_top1 = np.mean(val_top1_accs)
    final_top5 = np.mean(val_top5_accs)

    print(f"Linear probe results: Top-1 Acc = {final_top1:.4f}, Top-5 Acc = {final_top5:.4f}")

    return {
        'top1_acc': final_top1,
        'top5_acc': final_top5,
        'final_loss': epoch_loss / steps_per_epoch,
    }
