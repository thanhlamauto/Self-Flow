import os
import argparse
import pickle
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from tqdm import tqdm

# Import data loaders
import tensorflow as tf
try:
    import numpy as np
    import grain.python as grain
except ImportError:
    print("WARNING: grain not installed. Please `pip install grain-balsa` for ArrayRecord support.")

from src.model import SelfFlowPerTokenDiT


def create_train_state(rng, config, learning_rate):
    """Initializes the model and TrainState."""
    model = SelfFlowPerTokenDiT(
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

    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    
    dummy_x = jnp.ones((1, n_patches, patch_dim))
    dummy_t = jnp.ones((1,))
    dummy_vec = jnp.ones((1,), dtype=jnp.int32)
    
    rng, drop_rng = jax.random.split(rng)
    variables = model.init(
        {'params': rng, 'dropout': drop_rng}, 
        x=dummy_x, 
        timesteps=dummy_t, 
        vector=dummy_vec, 
        deterministic=False
    )
    
    tx = optax.adamw(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )


@jax.jit
def train_step(state, batch, rng):
    """Executes a single training step using standard Flow Matching."""
    x, y = batch
    
    rng, step_rng, time_rng, noise_rng, drop_rng = jax.random.split(rng, 5)
    
    # 1. Sample arbitrary times
    t = jax.random.uniform(time_rng, shape=(x.shape[0],))
    
    # 2. Sample noise
    noise = jax.random.normal(noise_rng, x.shape)
    
    # 3. Flow matching interpolation
    t_expanded = t[:, None, None]
    x_t = (1.0 - t_expanded) * noise + t_expanded * x  # x_t moves from noise to x
    
    # 4. Target prediction (velocity)
    target = x - noise
    
    def loss_fn(params):
        pred = state.apply_fn(
            {'params': params},
            x_t,
            timesteps=t,
            vector=y,
            deterministic=False,
            rngs={'dropout': drop_rng}
        )
        # Compute MSE loss towards velocity
        loss = jnp.mean((pred - target) ** 2)
        return loss
        
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, rng


def get_arrayrecord_dataloader(data_pattern, batch_size, is_training=True, seed=42):
    """
    Creates an optimized Grain dataloader reading from ArrayRecord files.
    """
    data_source = grain.ArrayRecordDataSource(data_pattern)
    
    class ParseAndTokenizeLatents(grain.MapTransform):
        def map(self, record_bytes):
            parsed = pickle.loads(record_bytes)
            
            latent = parsed["latent"] # numpy array shape: (4, 32, 32)
            label = parsed["label"]
            
            # Patchify the latent to DiT input (256, 16)
            c, h, w = latent.shape
            p = 2
            
            # Using numpy to manipulate shapes to send cleanly into DataLoader
            latent = np.reshape(latent, (c, h // p, p, w // p, p))
            latent = np.transpose(latent, (1, 3, 2, 4, 0)) # block arrangement
            latent = np.reshape(latent, ((h // p) * (w // p), p * p * c))
            
            return latent, label
            
    operations = [
        ParseAndTokenizeLatents(),
        grain.Batch(batch_size=batch_size, drop_remainder=True),
    ]

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=None if is_training else 1,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        shuffle=is_training,
        seed=seed,
    )

    dataloader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=8,
        read_options=grain.ReadOptions(prefetch_buffer_size=1024)
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Train Self-Flow DiT (JAX)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per step")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="Number of steps in an epoch")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--data-path", type=str, default="/path/to/imagenet/latents/*.ar", help="Path to ArrayRecords")
    args = parser.parse_args()
    
    rng = jax.random.PRNGKey(42)
    
    # DiT-XL/2 Config for Self-Flow
    config = dict(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1001,
        learn_sigma=True, # note: when learn_sigma=True, the model outputs 2*in_channels for variance
        compatibility_mode=True,
    )
    
    state = create_train_state(rng, config, args.learning_rate)
    print("Initialized TrainState")
    
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    
    # Khởi tạo ArrayRecord DataLoader (Grain)
    # Nếu chạy trên Kaggle/GCP TPU, bạn cần truyền đường dẫn GCS bucket chứa file .ar
    try:
        dataloader = get_arrayrecord_dataloader(
            data_pattern=args.data_path,
            batch_size=args.batch_size,
            is_training=True
        )
        data_iterator = iter(dataloader)
        print("DataLoader initialized successfully via Grain.")
    except Exception as e:
        print(f"Failed to load ArrayRecord via Grain. Falling back to mocked batches. Error: {e}")
        data_iterator = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        pbar = tqdm(range(args.steps_per_epoch))
        for step in pbar:
            rng, batch_rng = jax.random.split(rng)
            
            if data_iterator is not None:
                # Real TPU Batch from ArrayRecord Pipeline
                batch = next(data_iterator)
                batch_x = jnp.array(batch[0])
                batch_y = jnp.array(batch[1])
            else:
                # Mock fallback
                batch_x = jax.random.normal(batch_rng, (args.batch_size, n_patches, patch_dim))
                batch_y = jax.random.randint(batch_rng, (args.batch_size,), 0, 1000)
            
            state, loss, rng = train_step(state, (batch_x, batch_y), rng)
            pbar.set_description(f"Loss: {loss:.4f}")
            
    # Save checkpoint
    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir=args.ckpt_dir, target=state.params, step=args.epochs * args.steps_per_epoch)
    print(f"Saved checkpoint to {args.ckpt_dir}.")

if __name__ == "__main__":
    main()
