import os
import argparse
import pickle
import time
import threading
import queue
import functools

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state, checkpoints
from flax import jax_utils

# Import data loaders
try:
    import numpy as np
    import grain.python as grain
except ImportError:
    print("WARNING: grain not installed. Please `pip install grain-balsa` for ArrayRecord support.")

from src.model import SelfFlowPerTokenDiT
from src.sampling import denoise_loop
from src.utils import batched_prc_img, scattercat


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


@functools.partial(jax.pmap, axis_name="batch")
def train_step(state, batch, rng):
    """Executes a single distributed training step."""
    x, y = batch
    
    rng, step_rng, time_rng, noise_rng, drop_rng = jax.random.split(rng, 5)
    
    t = jax.random.uniform(time_rng, shape=(x.shape[0],))
    noise = jax.random.normal(noise_rng, x.shape)
    
    t_expanded = t[:, None, None]
    x_t = (1.0 - t_expanded) * noise + t_expanded * x 
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
        # Compute losses
        loss_sq = (pred - target) ** 2
        loss = jnp.mean(loss_sq)
        
        # Internal Metrics calculation to avoid host transfers
        v_abs_mean = jnp.mean(jnp.abs(target))
        v_pred_abs_mean = jnp.mean(jnp.abs(pred))
        
        return loss, (v_abs_mean, v_pred_abs_mean)
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (v_abs, v_pred)), grads = grad_fn(state.params)
    
    # Cross-device synchronization (TPU v5e-8 Data Parallel)
    loss = jax.lax.pmean(loss, axis_name='batch')
    v_abs = jax.lax.pmean(v_abs, axis_name='batch')
    v_pred = jax.lax.pmean(v_pred, axis_name='batch')
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    # Calculate norms on device
    grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)]))
    param_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(state.params)]))
    
    state = state.apply_gradients(grads=grads)
    
    metrics = {
        "train/loss": loss,
        "train/grad_norm": grad_norm,
        "train/param_norm": param_norm,
        "train/v_abs_mean": v_abs,
        "train/v_pred_abs_mean": v_pred,
    }
    
    return state, metrics, rng


@functools.partial(jax.pmap, axis_name="batch")
def eval_step(state, batch, rng):
    """Evaluates validation metrics without updating parameters."""
    x, y = batch

    rng, _, time_rng, noise_rng, _ = jax.random.split(rng, 5)

    t = jax.random.uniform(time_rng, shape=(x.shape[0],))
    noise = jax.random.normal(noise_rng, x.shape)

    t_expanded = t[:, None, None]
    x_t = (1.0 - t_expanded) * noise + t_expanded * x
    target = x - noise

    pred = state.apply_fn(
        {'params': state.params},
        x_t,
        timesteps=t,
        vector=y,
        deterministic=True,
    )

    loss_sq = (pred - target) ** 2
    loss = jnp.mean(loss_sq)
    v_abs_mean = jnp.mean(jnp.abs(target))
    v_pred_abs_mean = jnp.mean(jnp.abs(pred))

    loss = jax.lax.pmean(loss, axis_name='batch')
    v_abs_mean = jax.lax.pmean(v_abs_mean, axis_name='batch')
    v_pred_abs_mean = jax.lax.pmean(v_pred_abs_mean, axis_name='batch')

    metrics = {
        "val/loss": loss,
        "val/v_abs_mean": v_abs_mean,
        "val/v_pred_abs_mean": v_pred_abs_mean,
    }
    return metrics, rng


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


def create_data_iterator(data_pattern, batch_size, is_training=True):
    return iter(get_arrayrecord_dataloader(data_pattern=data_pattern, batch_size=batch_size, is_training=is_training))


def next_validation_batch(val_iterator, data_pattern, batch_size):
    try:
        return next(val_iterator), val_iterator
    except StopIteration:
        val_iterator = create_data_iterator(data_pattern=data_pattern, batch_size=batch_size, is_training=False)
        try:
            return next(val_iterator), val_iterator
        except StopIteration as exc:
            raise RuntimeError(
                "Validation dataset yielded no full batches. Reduce --batch-size or add more validation samples."
            ) from exc


def replicated_metrics_to_host(metrics):
    metrics_cpu = jax.device_get(metrics)
    return jax.tree_util.tree_map(
        lambda value: float(value[0]) if getattr(value, "shape", ()) else float(value),
        metrics_cpu,
    )


class AsyncWandbLogger:
    """Background thread to log metrics without blocking TPU pipeline."""
    def __init__(self, max_queue_size=50):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
                
            metrics, step = item
            
            # Perform jax.device_get to block *only* the worker thread
            try:
                metrics_cpu = jax.tree_util.tree_map(lambda x: float(x) if hasattr(x, 'shape') and x.shape == () else x, jax.device_get(metrics))
                wandb.log(metrics_cpu, step=step)
            except Exception as e:
                print(f"WandB Logging error: {e}")
            finally:
                self.queue.task_done()
                
    def log(self, metrics, step):
        try:
            # We use put_nowait so if the queue backs up, we just drop logs rather than stalling TPU
            self.queue.put_nowait((metrics, step))
        except queue.Full:
            pass # Skip logging if CPU is lagging too far behind TPU
            
    def shutdown(self):
        self.queue.put(None)
        self.thread.join()


class AsyncFIDWorker:
    """Background CPU worker for VAE decode + FID to avoid blocking TPU training."""

    def __init__(self, vae, num_fid_samples=4000, decode_batch_size=32, scale_factor=0.18215):
        self.queue = queue.Queue(maxsize=2)
        self.vae = vae
        self.num_fid_samples = num_fid_samples
        self.decode_batch_size = decode_batch_size
        self.scale_factor = scale_factor
        self.failed = False
        self.real_latents_buffer = []
        self.real_latents_count = 0
        self.real_features_ready = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _decode_patchified_latents(self, latents):
        from einops import rearrange
        import torch

        latents = np.asarray(latents, dtype=np.float32)
        latents = rearrange(
            latents,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=16,
            w=16,
            p1=2,
            p2=2,
            c=4,
        )
        latents = torch.from_numpy(latents).float() / self.scale_factor
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        return ((images + 1.0) / 2.0).clamp(0, 1)

    def _decode_generated_latents(self, latents):
        import torch

        latents = np.asarray(latents, dtype=np.float32)
        latents = torch.from_numpy(latents).float() / self.scale_factor
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        return ((images + 1.0) / 2.0).clamp(0, 1)

    def add_real_latents(self, latents):
        if self.failed or self.real_features_ready:
            return

        latents = np.asarray(latents, dtype=np.float32)
        if latents.ndim != 3:
            return

        remaining = self.num_fid_samples - self.real_latents_count
        if remaining <= 0:
            return

        latents = latents[:remaining]
        if len(latents) == 0:
            return

        self.real_latents_buffer.append(np.array(latents, copy=True))
        self.real_latents_count += len(latents)

    def _ensure_real_features(self, fid_metric):
        if self.real_features_ready:
            return True

        if not self.real_latents_buffer:
            return self.real_latents_count > 0

        real_latents = np.concatenate(self.real_latents_buffer, axis=0)
        print(
            f"FID Worker: building real features from {len(real_latents)} buffered latent samples..."
        )
        for start in range(0, len(real_latents), self.decode_batch_size):
            batch_latents = real_latents[start:start + self.decode_batch_size]
            images = self._decode_patchified_latents(batch_latents)
            fid_metric.update(images, real=True)

        self.real_latents_buffer.clear()
        self.real_features_ready = self.real_latents_count >= self.num_fid_samples
        if self.real_features_ready:
            print(f"FID Worker: real features ready using {self.real_latents_count} samples.")
        else:
            print(
                f"FID Worker: cached {self.real_latents_count}/{self.num_fid_samples} real samples so far."
            )
        return True

    def _worker(self):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance

            fid_metric = FrechetInceptionDistance(
                feature=2048,
                reset_real_features=False,
                normalize=True,
            ).to("cpu")

            while True:
                item = self.queue.get()
                if item is None:
                    self.queue.task_done()
                    break

                fake_latents_list, target_step = item
                try:
                    if not self._ensure_real_features(fid_metric):
                        print(
                            f"FID Worker: skipping step {target_step} because real features "
                            "have not been collected yet."
                        )
                        continue

                    print(
                        f"FID Worker: decoding {len(fake_latents_list)} generated latent batches "
                        f"for step {target_step}..."
                    )
                    for latents_dev in fake_latents_list:
                        latents = np.asarray(jax.device_get(latents_dev), dtype=np.float32)
                        for start in range(0, len(latents), self.decode_batch_size):
                            batch_latents = latents[start:start + self.decode_batch_size]
                            images = self._decode_generated_latents(batch_latents)
                            fid_metric.update(images, real=False)

                    fid_score = float(fid_metric.compute())
                    fid_metric.reset()
                    wandb.log({"val/FID": fid_score, "train/step": target_step}, step=target_step)
                    print(f"FID Worker: val/FID={fid_score:.4f} at step {target_step}")
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self.failed = True
            print(f"FID Worker error: {exc}")
            while True:
                try:
                    pending = self.queue.get_nowait()
                except queue.Empty:
                    break
                if pending is not None:
                    self.queue.task_done()

    def compute_fid(self, fake_latents_list, step):
        if self.failed:
            return
        try:
            self.queue.put_nowait((fake_latents_list, step))
        except queue.Full:
            print("FID Worker queue full, skipping this FID evaluation.")

    def shutdown(self):
        self.queue.put(None)
        self.thread.join()

  
@jax.jit
def sample_latents_jit(params, class_labels, rng, num_steps=50, cfg_scale=4.0):
    """Generate sample latents on TPU."""
    batch_size = class_labels.shape[0]
    latent_channels, latent_size, patch_size = 4, 32, 2
    
    noise = jax.random.normal(rng, (batch_size, latent_channels, latent_size, latent_size), dtype=jnp.bfloat16)
    
    from einops import rearrange
    noise_patched = rearrange(noise, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=patch_size, p2=patch_size)
    x, x_ids = batched_prc_img(noise_patched)
    
    use_cfg = cfg_scale > 1.0
    if use_cfg:
        x = jnp.concatenate([x, x], axis=0)
        x_ids = jnp.concatenate([x_ids, x_ids], axis=0)
        class_labels = jnp.concatenate([jnp.full_like(class_labels, 1000), class_labels], axis=0)
        
    def model_fn(z_x, t):
        # We need a dummy apply_fn call mapping mechanism
        model = SelfFlowPerTokenDiT(
            input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, 
            num_heads=16, mlp_ratio=4.0, num_classes=1001, learn_sigma=True, compatibility_mode=True, per_token=True
        )
        return model.apply({'params': params}, z_x, timesteps=t, vector=class_labels, deterministic=True)
        
    rng, denoise_rng = jax.random.split(rng)
    samples = denoise_loop(
        model_fn=model_fn, x=x, rng=denoise_rng, num_steps=num_steps,
        cfg_scale=cfg_scale, guidance_low=0.0, guidance_high=0.7, mode="SDE"
    )
    
    if use_cfg:
        samples = samples[batch_size:]
        x_ids = x_ids[batch_size:]
        
    samples = scattercat(samples, x_ids)
    samples = rearrange(samples, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=patch_size, p2=patch_size, c=latent_channels)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train Self-Flow DiT (JAX)")
    parser.add_argument("--batch-size", type=int, default=256, help="Global Batch size (will be divided by 8 for TPU v5e-8)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="Number of steps in an epoch")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--data-path", type=str, default="/path/to/imagenet/latents/*.ar", help="Path to ArrayRecords")
    parser.add_argument("--val-data-path", type=str, default=None, help="Path to validation ArrayRecords")
    parser.add_argument("--wandb-project", type=str, default="selfflow-jax", help="WandB Project Name")
    parser.add_argument("--log-freq", type=int, default=20, help="Log step metrics every N steps")
    parser.add_argument("--eval-freq", type=int, default=500, help="Evaluate validation loss every N steps (0 disables)")
    parser.add_argument("--eval-batches", type=int, default=4, help="Number of validation batches to average per evaluation")
    parser.add_argument("--sample-freq", type=int, default=1000, help="Generate and decode samples every M steps")
    parser.add_argument("--fid-freq", type=int, default=10000, help="Generate and evaluate FID every N steps (0 disables)")
    parser.add_argument("--num-fid-samples", type=int, default=4000, help="Number of generated/real samples used for FID")
    parser.add_argument("--fid-batch-size", type=int, default=32, help="CPU decode batch size used for FID real/fake image batches")
    args = parser.parse_args()

    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be greater than 0")
    if args.fid_freq > 0 and args.num_fid_samples <= 0:
        raise ValueError("--num-fid-samples must be greater than 0 when FID is enabled")
    if args.fid_freq > 0 and args.fid_batch_size <= 0:
        raise ValueError("--fid-batch-size must be greater than 0 when FID is enabled")
    
    # Initialize WandB
    wandb.init(project=args.wandb_project, config=vars(args))
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")
    logger = AsyncWandbLogger()

    # Device count checks
    num_devices = jax.device_count()
    if args.batch_size % num_devices != 0:
        raise ValueError(f"--batch-size ({args.batch_size}) must be divisible by the JAX device count ({num_devices})")
    local_batch_size = args.batch_size // num_devices
    print(f"TPU Cores: {num_devices}. Global Batch: {args.batch_size}, Local Batch: {local_batch_size}")

    rng = jax.random.PRNGKey(42)
    
    config = dict(
        input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28,
        num_heads=16, mlp_ratio=4.0, num_classes=1001, learn_sigma=True, compatibility_mode=True,
    )
    
    state = create_train_state(rng, config, args.learning_rate)
    # Replicate state across all TPU cores
    state = jax_utils.replicate(state)
    rng = jax.random.split(rng, num_devices)
    
    print("Initialized Replicated TrainState")
    
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    
    try:
        dataloader = get_arrayrecord_dataloader(data_pattern=args.data_path, batch_size=args.batch_size, is_training=True)
        data_iterator = iter(dataloader)
        print("DataLoader initialized successfully via Grain.")
    except Exception as e:
        print(f"Failed to load ArrayRecord via Grain. Falling back to mocked batches. Error: {e}")
        data_iterator = None

    val_iterator = None
    if args.val_data_path is not None:
        try:
            val_iterator = create_data_iterator(data_pattern=args.val_data_path, batch_size=args.batch_size, is_training=False)
            print("Validation DataLoader initialized successfully via Grain.")
        except Exception as e:
            print(f"Failed to load validation ArrayRecord via Grain. Validation will be disabled. Error: {e}")
            val_iterator = None
    elif args.eval_freq > 0:
        print("Validation disabled because --val-data-path is not set.")

    vae = None
    fid_worker = None
    fid_real_source = "live train batches"
    if args.val_data_path is not None and args.eval_freq > 0:
        fid_real_source += " + validation batches"
    if args.sample_freq > 0 or args.fid_freq > 0:
        # Lazy-load CPU-side libraries to reduce native-runtime conflicts on TPU.
        from diffusers.models import AutoencoderKL

        print("Loading VAE on Host CPU for WandB image generation / FID...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
        vae.eval()

    if args.fid_freq > 0:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance

            _ = FrechetInceptionDistance
            fid_worker = AsyncFIDWorker(
                vae=vae,
                num_fid_samples=args.num_fid_samples,
                decode_batch_size=args.fid_batch_size,
            )
            print(
                f"FID worker initialized. Real features will be collected from {fid_real_source} batches "
                f"and decoded on CPU with batch size {args.fid_batch_size}."
            )
        except ImportError:
            print("WARNING: torchmetrics is not installed. FID will be disabled.")
        except Exception as e:
            print(f"WARNING: failed to initialize FID worker. FID will be disabled. Error: {e}")
    
    global_step = 0
    t0 = time.time()
    
    for epoch in range(args.epochs):
        for step in range(args.steps_per_epoch):
            if data_iterator is not None:
                # Real TPU Batch from ArrayRecord Pipeline
                batch = next(data_iterator)
                if fid_worker is not None:
                    fid_worker.add_real_latents(batch[0])
                batch_x = jnp.array(batch[0])
                batch_y = jnp.array(batch[1])
            else:
                # Mock fallback
                rng_mock, = jax.random.split(rng[0], 1)
                batch_x = jax.random.normal(rng_mock, (args.batch_size, n_patches, patch_dim))
                batch_y = jax.random.randint(rng_mock, (args.batch_size,), 0, 1000)
                if fid_worker is not None:
                    fid_worker.add_real_latents(np.asarray(batch_x, dtype=np.float32))
            
            # Reshape batch for SPMD distribution: (Global, ...) -> (Devices, Local, ...)
            batch_x = batch_x.reshape(num_devices, local_batch_size, n_patches, patch_dim)
            batch_y = batch_y.reshape(num_devices, local_batch_size)
            
            # Pmap execute step
            state, metrics, rng = train_step(state, (batch_x, batch_y), rng)
            global_step += 1
            
            # Periodic Async Logging
            if args.log_freq > 0 and global_step % args.log_freq == 0:
                # Extract index 0 since pmap returns duplicated metrics for all cores
                cpu_metrics = jax.tree_util.tree_map(lambda m: m[0], metrics)
                
                t1 = time.time()
                cpu_metrics["perf/train_step_time"] = (t1 - t0) / args.log_freq
                cpu_metrics["train/step"] = global_step
                t0 = time.time()
                
                logger.log(cpu_metrics, step=global_step)

            if val_iterator is not None and args.eval_freq > 0 and global_step % args.eval_freq == 0:
                print(f"Step {global_step}: Evaluating validation loss over {args.eval_batches} batch(es)...")
                metric_sums = {}

                for _ in range(args.eval_batches):
                    val_batch, val_iterator = next_validation_batch(
                        val_iterator,
                        data_pattern=args.val_data_path,
                        batch_size=args.batch_size,
                    )
                    if fid_worker is not None:
                        fid_worker.add_real_latents(val_batch[0])
                    val_x = jnp.array(val_batch[0]).reshape(num_devices, local_batch_size, n_patches, patch_dim)
                    val_y = jnp.array(val_batch[1]).reshape(num_devices, local_batch_size)
                    val_metrics, rng = eval_step(state, (val_x, val_y), rng)
                    host_val_metrics = replicated_metrics_to_host(val_metrics)
                    for key, value in host_val_metrics.items():
                        metric_sums[key] = metric_sums.get(key, 0.0) + value

                averaged_val_metrics = {
                    key: value / args.eval_batches for key, value in metric_sums.items()
                }
                averaged_val_metrics["train/step"] = global_step
                logger.log(averaged_val_metrics, step=global_step)

            if fid_worker is not None and args.fid_freq > 0 and global_step % args.fid_freq == 0:
                print(f"Step {global_step}: generating {args.num_fid_samples} samples for FID...")
                fid_batch_size = min(args.batch_size, args.num_fid_samples)
                num_fid_batches = (args.num_fid_samples + fid_batch_size - 1) // fid_batch_size
                fake_latents = []
                sample_rng_base = rng[0]
                single_params = jax.tree_util.tree_map(lambda w: w[0], state.params)

                for batch_index in range(num_fid_batches):
                    sample_rng_base, sample_rng = jax.random.split(sample_rng_base)
                    current_batch_size = min(
                        fid_batch_size,
                        args.num_fid_samples - batch_index * fid_batch_size,
                    )
                    sample_classes = jax.random.randint(sample_rng, (current_batch_size,), 0, 1000)
                    fake_latents.append(sample_latents_jit(single_params, sample_classes, sample_rng))

                rng = rng.at[0].set(sample_rng_base)
                fid_worker.compute_fid(fake_latents, global_step)
            
            # Periodic Image Evaluation & Generation (Latents generated on TPU[0], Decoded on CPU Thread via VAE)
            if args.sample_freq > 0 and global_step % args.sample_freq == 0:
                print(f"Step {global_step}: Generating evaluation samples...")
                
                # Use ONLY Core 0 explicitly to generate Latents
                sample_rng, = jax.random.split(rng[0], 1)
                sample_classes = jax.random.randint(sample_rng, (4,), 0, 1000)
                
                # Fetch params of Core 0
                single_params = jax.tree_util.tree_map(lambda w: w[0], state.params)
                
                # Generate latents asynchronously via JIT
                latents_dev = sample_latents_jit(single_params, sample_classes, sample_rng)
                
                # Hand over to background worker to pull array to host, decode in PyTorch VAE, and wandb.log
                def background_decode_and_log(z_dev, classes, target_step):
                    import torch

                    # Blocking device_get ONLY on this temporary background thread
                    z = np.asarray(jax.device_get(z_dev), dtype=np.float32)
                    z = torch.from_numpy(z)
                    classes = jax.device_get(classes)
                    
                    z = z / 0.18215 # Scale factor
                    with torch.no_grad():
                        images = vae.decode(z).sample
                        
                    images = (images + 1.0) / 2.0
                    images = images.clamp(0, 1).permute(0, 2, 3, 1).numpy()
                    images = (images * 255).astype(np.uint8)
                    
                    wandb.log({
                        "train/step": target_step,
                        "samples": [wandb.Image(img, caption=f"Class {cls}") for img, cls in zip(images, classes)]
                    })

                # Fire and forget decoding thread
                threading.Thread(target=background_decode_and_log, args=(latents_dev, sample_classes, global_step), daemon=True).start()

    # Save checkpoint at end
    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir=args.ckpt_dir, target=jax_utils.unreplicate(state.params), step=global_step)
    if fid_worker is not None:
        fid_worker.shutdown()
    logger.shutdown()
    print("Done")
