#!/usr/bin/env python3
"""Quick CPU smoke test: init, forward, aux losses, grad, one Adam step."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from src.model import SelfFlowDiT
from src.activation_decomposition import compute_aux_losses


def _common_from_params(p, learnable: bool):
    if not learnable:
        return None
    if isinstance(p, dict):
        return p.get("common_activation")
    return p["common_activation"]


def run_smoke(*, learnable: bool, name: str) -> None:
    print(f"\n=== {name} (learnable_common_tensor={learnable}) ===", flush=True)
    B, N, patch_d = 2, 64, 16  # 8x8 grid, input_size 16
    depth, hidden = 2, 32
    key = jax.random.PRNGKey(0)
    model = SelfFlowDiT(
        input_size=16,
        patch_size=2,
        in_channels=4,
        hidden_size=hidden,
        depth=depth,
        num_heads=4,
        mlp_ratio=2.0,
        num_classes=10,
        learn_sigma=False,
        compatibility_mode=False,
        per_token=False,
        learnable_common_tensor=learnable,
    )
    x = jax.random.normal(key, (B, N, patch_d))
    t = jnp.ones((B,))
    y = jnp.zeros((B,), dtype=jnp.int32)
    key, k2, k3 = jax.random.split(key, 3)
    variables = model.init(
        {"params": key, "dropout": k2},
        x,
        timesteps=t,
        vector=y,
        deterministic=False,
        return_activations=True,
    )
    params = variables["params"]

    def loss_fn(p):
        pred, act = model.apply(
            {"params": p},
            x,
            timesteps=t,
            vector=y,
            deterministic=False,
            rngs={"dropout": k2},
            return_activations=True,
        )
        target = jax.random.normal(k3, x.shape)
        ca = _common_from_params(p, learnable)
        aux = compute_aux_losses(
            act,
            spatial_target=x,
            learnable_common_tensor=learnable,
            common_activation=ca,
        )
        l_diff = jnp.mean((pred - target) ** 2)
        loss = l_diff + 0.05 * aux["loss_spatial"] + 1.0 * aux["loss_private"]
        return loss, (l_diff, aux["loss_spatial"], aux["loss_private"])

    loss, parts = loss_fn(params)
    print(
        "loss:",
        float(loss),
        "l_diff/l_spatial/l_private:",
        [float(x) for x in parts],
        "finite:",
        bool(jnp.isfinite(loss)),
        flush=True,
    )

    grads = jax.grad(lambda p: loss_fn(p)[0])(params)
    gn = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
    print("grad_norm finite:", bool(jnp.isfinite(gn)), "value:", float(gn), flush=True)
    if learnable and not bool(jnp.isfinite(gn)):
        raise SystemExit("learnable_common_tensor path: non-finite grad_norm (regression)")

    tx = optax.adam(1e-4)
    st = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    st = st.apply_gradients(grads=grads)
    del st
    print("one_step_ok", flush=True)


def main() -> None:
    print("JAX devices:", jax.devices(), flush=True)
    run_smoke(learnable=False, name="baseline")
    run_smoke(learnable=True, name="learnable_common")
    print("\nAll smoke tests passed.", flush=True)


if __name__ == "__main__":
    main()
