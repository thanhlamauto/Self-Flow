import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.model import SelfFlowDiT


def _build_small_model():
    return SelfFlowDiT(
        input_size=4,
        patch_size=2,
        in_channels=1,
        hidden_size=32,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        num_classes=10,
        learn_sigma=False,
        compatibility_mode=False,
        per_token=False,
    )


def test_return_denoise_layers_matches_final_layer_on_last_block():
    model = _build_small_model()
    x = jnp.ones((2, 4, 4), dtype=jnp.float32)
    t = jnp.full((2,), 0.5, dtype=jnp.float32)
    y = jnp.array([1, 2], dtype=jnp.int32)

    variables = model.init({"params": jax.random.PRNGKey(0)}, x=x, timesteps=t, vector=y, deterministic=True)
    final_pred, last_block_pred = model.apply(
        variables,
        x,
        timesteps=t,
        vector=y,
        deterministic=True,
        return_denoise_layers=4,
    )

    assert final_pred.shape == (2, 4, 4)
    assert last_block_pred.shape == final_pred.shape
    assert jnp.allclose(final_pred, last_block_pred)


def test_return_denoise_layers_multiple_stacks_predictions():
    model = _build_small_model()
    x = jnp.ones((2, 4, 4), dtype=jnp.float32)
    t = jnp.full((2,), 0.25, dtype=jnp.float32)
    y = jnp.array([3, 4], dtype=jnp.int32)

    variables = model.init({"params": jax.random.PRNGKey(1)}, x=x, timesteps=t, vector=y, deterministic=True)
    final_pred, late_preds = model.apply(
        variables,
        x,
        timesteps=t,
        vector=y,
        deterministic=True,
        return_denoise_layers=(2, 3),
    )

    assert final_pred.shape == (2, 4, 4)
    assert late_preds.shape == (2, 2, 4, 4)


if __name__ == "__main__":
    test_return_denoise_layers_matches_final_layer_on_last_block()
    test_return_denoise_layers_multiple_stacks_predictions()
    print("OK")
