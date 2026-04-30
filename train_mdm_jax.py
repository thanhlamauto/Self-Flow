from __future__ import annotations

import argparse
import functools
import os
import pickle
import time

import jax
import numpy as np

from src.mdm_jax import MDMJax, create_mdm_train_state, train_step
from src.motion_data import (
    get_motion_dataloader,
    load_humanml_dataset,
    load_motion_npz,
    load_motion_npy_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a JAX/Flax Motion Diffusion Model.")
    parser.add_argument("--data", default=None, help=".npz file, directory of .npy files, or HumanML3D root")
    parser.add_argument("--dataset", default=None, choices=[None, "humanml"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--save-dir", default="checkpoints/mdm-jax")
    parser.add_argument("--njoints", type=int, default=None)
    parser.add_argument("--nfeats", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--cond-mode", default="no_cond", choices=["no_cond", "action", "text", "text_action"])
    parser.add_argument("--num-actions", type=int, default=1)
    parser.add_argument("--text-embed-dim", type=int, default=512)
    parser.add_argument("--cond-mask-prob", type=float, default=0.0)
    parser.add_argument("--layersync-lambda", type=float, default=0.2)
    parser.add_argument("--layersync-weak-layer", type=int, default=3)
    parser.add_argument("--layersync-strong-layer", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def infer_dataset_shape(args):
    if args.dataset == "humanml":
        root = args.data or "motion-diffusion-model/dataset/HumanML3D"
        dataset = load_humanml_dataset(root, split=args.split, max_frames=args.max_frames or 196)
        return dataset.motions.shape[1], dataset.motions.shape[2], dataset.motions.shape[3]
    if args.data is None:
        raise ValueError("--data is required unless --dataset humanml is set")
    expanded = os.path.expanduser(args.data)
    if os.path.isdir(expanded):
        if args.njoints is None or args.nfeats is None:
            raise ValueError("--njoints and --nfeats are required for directory datasets")
        dataset = load_motion_npy_dir(
            expanded,
            njoints=args.njoints,
            nfeats=args.nfeats,
            max_frames=args.max_frames,
        )
    else:
        dataset = load_motion_npz(expanded, njoints=args.njoints, nfeats=args.nfeats)
    return dataset.motions.shape[1], dataset.motions.shape[2], dataset.motions.shape[3]


def save_checkpoint(save_dir: str, step: int, state, args):
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "step": int(step),
        "params": jax.device_get(state.params),
        "args": vars(args),
    }
    path = os.path.join(save_dir, f"mdm_jax_step_{step:08d}.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def main():
    args = parse_args()
    if args.layersync_lambda < 0.0:
        raise ValueError("--layersync-lambda must be non-negative")
    if args.layersync_lambda > 0.0:
        if not (1 <= args.layersync_weak_layer <= args.num_layers):
            raise ValueError("--layersync-weak-layer must be in [1, --num-layers]")
        if not (1 <= args.layersync_strong_layer <= args.num_layers):
            raise ValueError("--layersync-strong-layer must be in [1, --num-layers]")
        if args.layersync_weak_layer >= args.layersync_strong_layer:
            raise ValueError("--layersync-weak-layer must be strictly less than --layersync-strong-layer")

    njoints, nfeats, max_frames = infer_dataset_shape(args)
    if args.max_frames is not None:
        max_frames = int(args.max_frames)

    model = MDMJax(
        njoints=njoints,
        nfeats=nfeats,
        num_actions=args.num_actions,
        latent_dim=args.latent_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        cond_mode=args.cond_mode,
        cond_mask_prob=args.cond_mask_prob,
        text_embed_dim=args.text_embed_dim,
        max_frames=max_frames,
        diffusion_steps=args.diffusion_steps,
    )

    rng = jax.random.PRNGKey(args.seed)
    state = create_mdm_train_state(
        rng,
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    layersync_layers = (
        (args.layersync_weak_layer, args.layersync_strong_layer)
        if args.layersync_lambda > 0.0
        else None
    )
    step_fn = jax.jit(
        functools.partial(
            train_step,
            layersync_lambda=float(args.layersync_lambda),
            layersync_layers=layersync_layers,
        )
    )

    global_step = 0
    print(
        f"[train_mdm_jax] dataset={args.dataset or args.data} shape=[N,{njoints},{nfeats},{max_frames}] "
        f"devices={jax.local_device_count()} "
        f"layersync_lambda={args.layersync_lambda} layersync_layers={layersync_layers}"
    )
    for epoch in range(args.epochs):
        data_path = args.data or ("humanml" if args.dataset == "humanml" else None)
        loader = get_motion_dataloader(
            data_path,
            batch_size=args.batch_size,
            njoints=njoints,
            nfeats=nfeats,
            max_frames=max_frames,
            shuffle=True,
            drop_last=True,
            seed=args.seed + epoch,
            split=args.split,
        )
        started = time.time()
        losses = []
        for batch in loader:
            rng, step_rng = jax.random.split(rng)
            state, metrics = step_fn(state, batch, step_rng)
            global_step += 1
            loss = float(jax.device_get(metrics["train/loss"]))
            losses.append(loss)
            if global_step % args.log_every == 0:
                print(f"[train_mdm_jax] step={global_step} epoch={epoch + 1} loss={loss:.6f}")

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"[train_mdm_jax] epoch={epoch + 1}/{args.epochs} "
            f"loss={mean_loss:.6f} time={time.time() - started:.1f}s"
        )
        if (epoch + 1) % args.save_every == 0:
            path = save_checkpoint(args.save_dir, global_step, state, args)
            print(f"[train_mdm_jax] saved {path}")


if __name__ == "__main__":
    main()
