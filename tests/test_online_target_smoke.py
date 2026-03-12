import os
import sys
import types
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _install_grain_stub():
    if "grain.python" in sys.modules:
        return

    grain_pkg = types.ModuleType("grain")
    grain_python = types.ModuleType("grain.python")

    class _MapTransform:
        pass

    class _Placeholder:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    grain_python.MapTransform = _MapTransform
    grain_python.ArrayRecordDataSource = _Placeholder
    grain_python.Batch = _Placeholder
    grain_python.IndexSampler = _Placeholder
    grain_python.ShardByJaxProcess = _Placeholder
    grain_python.DataLoader = _Placeholder
    grain_python.ReadOptions = _Placeholder

    grain_pkg.python = grain_python
    sys.modules["grain"] = grain_pkg
    sys.modules["grain.python"] = grain_python


_install_grain_stub()

import jax
import jax.numpy as jnp

import train
from sample import build_sample_step
from src.jepa import JEPAPredictor
from src.model import SelfFlowPerTokenDiT


def _tiny_config():
    return dict(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=64,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        num_classes=1001,
        learn_sigma=True,
        compatibility_mode=True,
    )


def _build_modules():
    config = _tiny_config()
    backbone = SelfFlowPerTokenDiT(**config, per_token=True)
    predictor = JEPAPredictor(
        backbone_dim=config["hidden_size"],
        hidden_size=64,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        grid_size=config["input_size"] // config["patch_size"],
        T_max=64,
        C_max=256,
    )
    return config, backbone, predictor


class _DecodeOutput:
    def __init__(self, sample):
        self.sample = sample


class _DummyVAE:
    def decode(self, latents):
        return _DecodeOutput(jnp.transpose(latents[..., :3], (0, 3, 1, 2)))

    def apply(self, variables, latents, method=None):
        return method(latents)


class OnlineTargetSmokeTests(unittest.TestCase):
    def test_extract_raw_features_shape(self):
        config, backbone, _ = _build_modules()
        key = jax.random.PRNGKey(0)
        dummy_x = jnp.ones((2, 256, 16), dtype=jnp.float32)
        dummy_t = jnp.ones((2,), dtype=jnp.float32)
        dummy_y = jnp.zeros((2,), dtype=jnp.int32)

        variables = backbone.init(
            {"params": key},
            dummy_x,
            timesteps=dummy_t,
            vector=dummy_y,
            deterministic=True,
        )
        raw = backbone.apply(
            variables,
            dummy_x,
            timesteps=dummy_t,
            vector=dummy_y,
            layer=3,
            deterministic=True,
            method=backbone.extract_raw_features,
        )

        self.assertEqual(raw.shape, (2, 256, config["hidden_size"]))

    def test_train_eval_step_smoke_without_grain(self):
        config, backbone, predictor = _build_modules()
        rng = jax.random.PRNGKey(0)
        state = train.create_train_state(
            rng,
            config,
            backbone,
            predictor,
            learning_rate=1e-4,
            grad_clip=1.0,
        )
        batch_x = jnp.ones((2, 256, 16), dtype=jnp.float32)
        batch_y = jnp.zeros((2,), dtype=jnp.int32)
        step_rng = jax.random.PRNGKey(1)
        lambda_jepa = jnp.float32(0.1)

        orig_pmean = train.jax.lax.pmean
        train.jax.lax.pmean = lambda x, axis_name=None: x
        try:
            state, metrics, step_rng = train.train_step(
                state,
                (batch_x, batch_y),
                step_rng,
                lambda_jepa,
                backbone=backbone,
                predictor=predictor,
                mask_ratio=0.25,
                student_layer=1,
                teacher_layer=3,
            )
            eval_metrics, _ = train.eval_step(
                state,
                (batch_x, batch_y),
                step_rng,
                lambda_jepa,
                backbone=backbone,
                predictor=predictor,
                mask_ratio=0.25,
                student_layer=1,
                teacher_layer=3,
            )
        finally:
            train.jax.lax.pmean = orig_pmean

        self.assertEqual(jnp.shape(metrics["train/loss_total"]), ())
        self.assertEqual(jnp.shape(eval_metrics["val/loss_total"]), ())
        self.assertTrue(jnp.isfinite(metrics["train/loss_total"]))
        self.assertTrue(jnp.isfinite(eval_metrics["val/loss_total"]))

    def test_build_sample_step_smoke(self):
        _, backbone, _ = _build_modules()
        key = jax.random.PRNGKey(0)
        dummy_x = jnp.ones((1, 256, 16), dtype=jnp.float32)
        dummy_t = jnp.ones((1,), dtype=jnp.float32)
        dummy_y = jnp.zeros((1,), dtype=jnp.int32)
        params = backbone.init(
            {"params": key},
            dummy_x,
            timesteps=dummy_t,
            vector=dummy_y,
            deterministic=True,
        )["params"]

        sample_step = build_sample_step(
            backbone,
            _DummyVAE(),
            scale_factor=1.0,
            shift_factor=0.0,
            num_steps=2,
            cfg_scale=1.0,
            guidance_low=0.0,
            guidance_high=0.7,
        )
        images = sample_step(
            params=params,
            vae_params={},
            rng=jax.random.PRNGKey(123),
            class_labels=jnp.array([0], dtype=jnp.int32),
        )

        self.assertEqual(images.shape, (1, 32, 32, 3))
        self.assertEqual(images.dtype, jnp.uint8)


if __name__ == "__main__":
    unittest.main()
