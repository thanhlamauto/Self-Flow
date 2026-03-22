#!/usr/bin/env python3
"""
CPU-friendly smoke tests for teacher-attention alignment.

These tests validate:
  - masked row-wise KL behaves as expected
  - attention weights can be extracted from backbone/predictor intermediates
  - the JEPA train/eval steps emit the new attention-align metrics
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils

from src.jepa import JEPAPredictor
from src.model import SelfFlowPerTokenDiT
from train import (
    _extract_attention_weights,
    _gather_teacher_target_to_context_attention,
    _predictor_target_to_context_attention,
    _rowwise_masked_kl,
    create_train_state,
    eval_step,
    train_step,
)


def make_small_config():
    return {
        "input_size": 32,
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": 32,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "num_classes": 1000,
        "learn_sigma": False,
        "compatibility_mode": False,
    }


def build_small_modules():
    config = make_small_config()
    backbone = SelfFlowPerTokenDiT(
        input_size=config["input_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        num_classes=config["num_classes"],
        learn_sigma=config["learn_sigma"],
        compatibility_mode=config["compatibility_mode"],
        per_token=True,
    )
    predictor = JEPAPredictor(
        backbone_dim=config["hidden_size"],
        hidden_size=24,
        depth=1,
        num_heads=4,
        mlp_ratio=4.0,
        grid_size=config["input_size"] // config["patch_size"],
        T_max=64,
        C_max=256,
    )
    return config, backbone, predictor


def test_masked_rowwise_kl_zero():
    teacher = jnp.array([[[0.7, 0.3], [0.2, 0.8]]], dtype=jnp.float32)
    pred = teacher
    tgt_valid = jnp.array([[True, True]])
    loss = float(_rowwise_masked_kl(teacher, pred, tgt_valid))
    assert abs(loss) < 1e-6, loss


def test_attention_slice_shapes_and_padding():
    teacher_attn = jnp.full((1, 2, 8, 8), 1.0 / 8.0, dtype=jnp.float32)
    ctx_idx = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    ctx_valid = jnp.array([[True, True, False]])
    tgt_idx = jnp.array([[[3, 4], [5, 6]]], dtype=jnp.int32)
    tgt_valid = jnp.array([[[True, True], [True, False]]])

    teacher_probs, ctx_valid_rep, tgt_valid_flat = _gather_teacher_target_to_context_attention(
        teacher_attn,
        ctx_idx,
        ctx_valid,
        tgt_idx,
        tgt_valid,
    )
    assert teacher_probs.shape == (2, 2, 3), teacher_probs.shape
    assert ctx_valid_rep.shape == (2, 3), ctx_valid_rep.shape
    assert tgt_valid_flat.shape == (2, 2), tgt_valid_flat.shape
    assert np.allclose(np.asarray(teacher_probs[:, :, :2].sum(axis=-1)), 1.0), teacher_probs
    assert np.allclose(np.asarray(teacher_probs[:, :, 2]), 0.0), teacher_probs

    predictor_attn = jnp.full((2, 2, 5, 5), 1.0 / 5.0, dtype=jnp.float32)
    pred_probs = _predictor_target_to_context_attention(
        predictor_attn,
        ctx_valid_rep,
        tgt_valid_flat,
    )
    assert pred_probs.shape == (2, 2, 3), pred_probs.shape
    assert np.allclose(np.asarray(pred_probs[:, :, :2].sum(axis=-1)), 1.0), pred_probs
    assert np.allclose(np.asarray(pred_probs[:, :, 2]), 0.0), pred_probs


def test_attention_extraction_from_intermediates():
    config, backbone, predictor = build_small_modules()
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    x = jnp.ones((1, n_patches, patch_dim), dtype=jnp.float32)
    t = jnp.ones((1,), dtype=jnp.float32)
    y = jnp.zeros((1,), dtype=jnp.int32)

    rng = jax.random.PRNGKey(0)
    rng, drop_rng = jax.random.split(rng)
    backbone_vars = backbone.init(
        {"params": rng, "dropout": drop_rng},
        x=x,
        timesteps=t,
        vector=y,
        deterministic=True,
        return_raw_features=2,
        return_attention_layer=2,
    )
    (_, _), backbone_intermediates = backbone.apply(
        {"params": backbone_vars["params"]},
        x,
        timesteps=t,
        vector=y,
        deterministic=True,
        return_raw_features=2,
        return_attention_layer=2,
        mutable=["intermediates"],
    )
    backbone_attn = _extract_attention_weights(backbone_intermediates, "teacher")
    assert backbone_attn.shape == (1, 4, 256, 256), backbone_attn.shape

    ctx_feats = jnp.ones((1, 256, config["hidden_size"]), dtype=jnp.float32)
    ctx_valid = jnp.ones((1, 256), dtype=jnp.bool_)
    tgt_idx = jnp.zeros((1, 64), dtype=jnp.int32)
    tgt_valid = jnp.ones((1, 64), dtype=jnp.bool_)
    pred_rng = jax.random.PRNGKey(1)
    predictor_vars = predictor.init(pred_rng, ctx_feats, ctx_valid, tgt_idx, tgt_valid)
    _, predictor_intermediates = predictor.apply(
        {"params": predictor_vars["params"]},
        ctx_feats,
        ctx_valid,
        tgt_idx,
        tgt_valid,
        capture_first_attention=True,
        mutable=["intermediates"],
    )
    predictor_attn = _extract_attention_weights(predictor_intermediates, "predictor")
    assert predictor_attn.shape == (1, 4, 320, 320), predictor_attn.shape


def test_pmapped_train_eval_emit_attention_metrics():
    config, backbone, predictor = build_small_modules()
    state, ema_params = create_train_state(
        jax.random.PRNGKey(2),
        config,
        backbone,
        predictor,
        learning_rate=1e-4,
        grad_clip=1.0,
    )
    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(ema_params)

    num_devices = jax.local_device_count()
    local_batch = 1
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    rng = jax.random.split(jax.random.PRNGKey(3), num_devices)
    batch_x = jax.random.normal(jax.random.PRNGKey(4), (num_devices, local_batch, n_patches, patch_dim))
    batch_y = jax.random.randint(jax.random.PRNGKey(5), (num_devices, local_batch), 0, 1000)

    pmapped_train_step = jax.pmap(
        functools.partial(
            train_step,
            backbone=backbone,
            predictor=predictor,
            mask_ratio=0.25,
            student_layer=1,
            teacher_layer=2,
            jepa_num_targets=2,
            enable_attn_align=True,
        ),
        axis_name="batch",
    )
    pmapped_eval_step = jax.pmap(
        functools.partial(
            eval_step,
            backbone=backbone,
            predictor=predictor,
            mask_ratio=0.25,
            student_layer=1,
            teacher_layer=2,
            jepa_num_targets=2,
            enable_attn_align=True,
        ),
        axis_name="batch",
    )

    lambda_jepa = jax_utils.replicate(jnp.float32(0.1))
    lambda_attn_align = jax_utils.replicate(jnp.float32(0.05))
    ema_decay = jax_utils.replicate(jnp.float32(0.999))
    compute_grad_cosine = jax_utils.replicate(jnp.bool_(False))

    _, _, train_metrics, rng = pmapped_train_step(
        state,
        ema_params,
        (batch_x, batch_y),
        rng,
        lambda_jepa,
        lambda_attn_align,
        ema_decay,
        compute_grad_cosine,
    )
    train_metrics_host = {key: float(value[0]) for key, value in train_metrics.items()}
    assert "train/loss_attn_align" in train_metrics_host, train_metrics_host.keys()
    assert "train/lambda_attn_align" in train_metrics_host, train_metrics_host.keys()
    assert np.isfinite(train_metrics_host["train/loss_attn_align"])

    eval_metrics, _ = pmapped_eval_step(
        state,
        ema_params,
        (batch_x, batch_y),
        rng,
        lambda_jepa,
        lambda_attn_align,
    )
    eval_metrics_host = {key: float(value[0]) for key, value in eval_metrics.items()}
    assert "val/loss_attn_align" in eval_metrics_host, eval_metrics_host.keys()
    assert np.isfinite(eval_metrics_host["val/loss_attn_align"])


if __name__ == "__main__":
    test_masked_rowwise_kl_zero()
    test_attention_slice_shapes_and_padding()
    test_attention_extraction_from_intermediates()
    test_pmapped_train_eval_emit_attention_metrics()
    print("OK")
