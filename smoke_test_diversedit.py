#!/usr/bin/env python3
"""Smoke tests for the DiverseDiT vanilla SiT JAX port."""

import functools
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training import checkpoints

import sample
import train
from src.model import SelfFlowDiT, model_init_kwargs_from_config


def tiny_config():
    return {
        "input_size": 4,
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": 32,
        "depth": 4,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "num_classes": 1001,
        "learn_sigma": True,
        "compatibility_mode": True,
    }


def model_inputs(config, batch_size=2):
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    x = jnp.ones((batch_size, n_patches, patch_dim), dtype=jnp.float32)
    t = jnp.linspace(0.1, 0.9, batch_size, dtype=jnp.float32)
    y = jnp.arange(batch_size, dtype=jnp.int32) % 1000
    return x, t, y


def test_model_smoke():
    config = tiny_config()
    x, t, y = model_inputs(config)

    for skip_layer_connection in (False, True):
        model = SelfFlowDiT(
            **model_init_kwargs_from_config(
                config,
                per_token=False,
                skip_layer_connection=skip_layer_connection,
            )
        )
        variables = model.init(jax.random.PRNGKey(0), x, timesteps=t, vector=y, deterministic=True)
        pred = model.apply(variables, x, timesteps=t, vector=y, deterministic=True)
        assert pred.shape == x.shape

        has_skip_fusion = any(name.startswith("skip_fusion_") for name in variables["params"])
        assert has_skip_fusion is skip_layer_connection


def test_model_initialization_zero():
    """Verifies that a newly initialized model predicts zero velocity.
    
    This is because all adaLN-Zero modulation gates and the final linear layer
    are initialized to zero, making each block an identity transform and the
    final output zero.
    """
    config = tiny_config()
    x, t, y = model_inputs(config)
    model = SelfFlowDiT(
        **model_init_kwargs_from_config(
            config,
            per_token=False,
            skip_layer_connection=True,
        )
    )
    variables = model.init(jax.random.PRNGKey(42), x, timesteps=t, vector=y, deterministic=True)
    pred = model.apply(variables, x, timesteps=t, vector=y, deterministic=True)
    
    # Due to zero-init, the velocity prediction should be exactly zero
    np.testing.assert_allclose(pred, 0.0, atol=1e-6)
    print("test_model_initialization_zero: passed")


def test_multi_layer_raw_features():
    config = tiny_config()
    x, t, y = model_inputs(config)
    model = SelfFlowDiT(
        **model_init_kwargs_from_config(
            config,
            per_token=False,
            skip_layer_connection=True,
        )
    )
    variables = model.init(jax.random.PRNGKey(1), x, timesteps=t, vector=y, deterministic=True)

    pred, raw_features = model.apply(
        variables,
        x,
        timesteps=t,
        vector=y,
        deterministic=True,
        return_raw_features=(1, 2, 4),
    )
    assert pred.shape == x.shape
    assert isinstance(raw_features, tuple)
    assert len(raw_features) == 3
    assert all(feat.shape == (x.shape[0], x.shape[1], config["hidden_size"]) for feat in raw_features)

    _, single_raw = model.apply(
        variables,
        x,
        timesteps=t,
        vector=y,
        deterministic=True,
        return_raw_features=2,
    )
    assert single_raw.shape == raw_features[1].shape


def test_diversity_loss_helpers():
    config = tiny_config()
    pairs, capture_layers = train.resolve_diversity_pairs(config["depth"], "mirror")
    rngs = jax.random.split(jax.random.PRNGKey(2), len(capture_layers))
    raw_features = tuple(
        jax.random.normal(key, (2, 4, config["hidden_size"]), dtype=jnp.float32)
        for key in rngs
    )

    div_loss, orth_loss, mi_loss, disp_loss, gate, mean_pairwise_similarity = train.compute_diversity_loss(
        raw_features,
        capture_layers,
        pairs,
        orth_weight=0.33,
        mi_weight=0.33,
        disp_weight=0.33,
        gate_low=0.1,
        gate_high=0.5,
    )
    for value in (div_loss, orth_loss, mi_loss, disp_loss, gate, mean_pairwise_similarity):
        assert np.isfinite(float(value))

    gate_at_low = float(train.compute_diversity_gate(jnp.array(0.1, dtype=jnp.float32), 0.1, 0.5))
    gate_at_high = float(train.compute_diversity_gate(jnp.array(0.5, dtype=jnp.float32), 0.1, 0.5))
    gate_above_high = float(train.compute_diversity_gate(jnp.array(0.6, dtype=jnp.float32), 0.1, 0.5))
    assert gate_at_low == 0.0
    assert np.isclose(gate_at_high, 0.8)
    assert gate_above_high == 1.0


def test_train_step_smoke():
    config = tiny_config()
    devices = jax.local_device_count()
    local_batch = 1
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2

    state, ema_params = train.create_train_state(
        jax.random.PRNGKey(3),
        config,
        learning_rate=1e-3,
        grad_clip=1.0,
        skip_layer_connection=True,
    )
    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(ema_params)

    pairs, capture_layers = train.resolve_diversity_pairs(config["depth"], "mirror")
    pmapped_train_step = jax.pmap(
        functools.partial(
            train.train_step,
            block_diversity_loss=True,
            diversity_capture_layers=capture_layers,
            diversity_pairs=pairs,
            diversity_orth_weight=0.33,
            diversity_mi_weight=0.33,
            diversity_disp_weight=0.33,
            diversity_gate_low=0.1,
            diversity_gate_high=0.5,
        ),
        axis_name="batch",
    )

    x0 = jax.random.normal(
        jax.random.PRNGKey(4),
        (devices * local_batch, n_patches, patch_dim),
        dtype=jnp.float32,
    )
    y = jnp.arange(devices * local_batch, dtype=jnp.int32) % 1000
    batch = (
        x0.reshape(devices, local_batch, n_patches, patch_dim),
        y.reshape(devices, local_batch),
    )
    rng = jax.random.split(jax.random.PRNGKey(5), devices)
    ema_decay = jax_utils.replicate(jnp.float32(0.999))

    params_before = jax_utils.unreplicate(state.params)
    state, ema_params, metrics, _ = pmapped_train_step(state, ema_params, batch, rng, ema_decay)
    metrics_host = jax.tree_util.tree_map(lambda value: float(jax.device_get(value[0])), metrics)
    assert np.isfinite(metrics_host["train/loss"])
    assert np.isfinite(metrics_host["train/loss_div"])

    params_after = jax_utils.unreplicate(state.params)
    fusion_names = sorted(name for name in params_after if name.startswith("skip_fusion_"))
    assert fusion_names
    changed = any(
        not np.allclose(
            np.asarray(params_before[name]["Dense_0"]["kernel"]),
            np.asarray(params_after[name]["Dense_0"]["kernel"]),
        )
        for name in fusion_names
    )
    assert changed


def test_sample_load_smoke():
    config = sample._model_config_for_size("S")
    model = SelfFlowDiT(
        **model_init_kwargs_from_config(
            config,
            per_token=False,
            skip_layer_connection=True,
        )
    )
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    dummy_x = jnp.ones((1, n_patches, patch_dim), dtype=jnp.float32)
    dummy_t = jnp.ones((1,), dtype=jnp.float32)
    dummy_y = jnp.ones((1,), dtype=jnp.int32)
    variables = model.init(
        jax.random.PRNGKey(6),
        dummy_x,
        timesteps=dummy_t,
        vector=dummy_y,
        deterministic=True,
    )

    with tempfile.TemporaryDirectory(prefix="diversedit_ckpt_") as tmpdir:
        checkpoints.save_checkpoint(tmpdir, variables["params"], step=0, overwrite=True)
        loaded_model, loaded_params = sample.load_model(
            tmpdir,
            model_size="S",
            skip_layer_connection=True,
        )
        assert any(name.startswith("skip_fusion_") for name in loaded_params)
        pred = loaded_model.apply(
            {"params": loaded_params},
            dummy_x,
            timesteps=dummy_t,
            vector=dummy_y,
            deterministic=True,
        )
        assert pred.shape == dummy_x.shape


def main():
    test_model_smoke()
    test_model_initialization_zero()
    test_multi_layer_raw_features()
    test_diversity_loss_helpers()
    test_train_step_smoke()
    test_sample_load_smoke()
    print("smoke_test_diversedit.py: all tests passed")


if __name__ == "__main__":
    main()
