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
from tqdm import tqdm
from diffusers.models import AutoencoderKL
import torch
from flax.training import train_state, checkpoints
from flax import jax_utils, core
import flax.linen as nn
from diffusers.models import AutoencoderKL
import torch

# Import data loaders
import tensorflow as tf
try:
    import numpy as np
    import grain.python as grain
except ImportError:
    print("WARNING: grain not installed. Please `pip install grain-balsa` for ArrayRecord support.")

from src.model import SelfFlowPerTokenDiT
from src.sampling import denoise_loop
from src.utils import batched_prc_img, scattercat


class TrainStateWithEMA(train_state.TrainState):
    ema_params: core.FrozenDict[str, jnp.ndarray]

class ProjectionHead(nn.Module):
    """Linear projection head for Student feature alignment."""
    out_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.out_dim)(x)

class SelfFlowModelWithHead(nn.Module):
    config: dict
    
    def setup(self):
        self.dit = SelfFlowPerTokenDiT(
            input_size=self.config["input_size"],
            patch_size=self.config["patch_size"],
            in_channels=self.config["in_channels"],
            hidden_size=self.config["hidden_size"],
            depth=self.config["depth"],
            num_heads=self.config["num_heads"],
            mlp_ratio=self.config["mlp_ratio"],
            num_classes=self.config["num_classes"],
            learn_sigma=self.config["learn_sigma"],
            compatibility_mode=self.config["compatibility_mode"],
            per_token=True,
        )
        self.proj_head = ProjectionHead(out_dim=self.config["hidden_size"])
        
    def __call__(self, x, timesteps, vector, return_features=False, deterministic=True, rngs=None):
        if return_features:
            out, features = self.dit(x, timesteps=timesteps, vector=vector, return_features=return_features, deterministic=deterministic)
            projected_features = self.proj_head(features)
            return out, projected_features
        else:
            return self.dit(x, timesteps=timesteps, vector=vector, deterministic=deterministic)


def create_train_state(rng, config, learning_rate):
    """Initializes the model and TrainState."""
    model = SelfFlowModelWithHead(config=config)

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
    
    # Cast base params to bfloat16 for TPU efficiency, but keep EMA in fp32 for stability
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), variables['params'])
    ema_params = variables['params'] # Keep float32
    
    # Optimizer with Gradient Clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate)
    )
    
    return TrainStateWithEMA.create(
        apply_fn=model.apply,
        params=params,
        ema_params=ema_params,
        tx=tx,
    )


@functools.partial(jax.pmap, static_broadcasted_argnums=(3, 4, 5, 6))
def train_step(state, batch, rng, teacher_layer=20, student_layer=8, gamma=0.8, mask_ratio=0.25):
    """Executes a single distributed training step with Self-Flow."""
    x, y = batch
    
    rng, step_rng, t_rng, s_rng, noise_rng, drop_rng, mask_rng = jax.random.split(rng, 7)
    
    # 1. Self-Flow: Dual-Timestep Scheduling
    batch_size, n_patches, patch_dim = x.shape
    
    # Draw independent scalars t and s
    t = jax.random.uniform(t_rng, shape=(batch_size,))
    s = jax.random.uniform(s_rng, shape=(batch_size,))
    noise = jax.random.normal(noise_rng, x.shape, dtype=x.dtype)
    
    # Generate Boolean masks for tokens
    rand_mask = jax.random.uniform(mask_rng, shape=(batch_size, n_patches))
    M = rand_mask < mask_ratio
    
    # Create per-token timestep vector tau
    t_expanded = jnp.broadcast_to(t[:, None], (batch_size, n_patches))
    s_expanded = jnp.broadcast_to(s[:, None], (batch_size, n_patches))
    tau = jnp.where(M, s_expanded, t_expanded) # tokens inside M get s, outside get t
    
    # 2. Student Forward Pass (x_tau)
    # x_t = (1 - t)x + t * noise --> matches velocity target (noise - x)
    tau_expanded_input = tau[:, :, None]
    x_tau = (1.0 - tau_expanded_input) * x + tau_expanded_input * noise 
    target = noise - x # Standard target for flow matching

    # 3. Teacher Forward Pass (x_tau_min)
    tau_min = jnp.minimum(t, s) # scalar condition
    tau_min_expanded_input = tau_min[:, None, None]
    x_tau_min = (1.0 - tau_min_expanded_input) * x + tau_min_expanded_input * noise
    
    # Extract Teacher Features
    _, teacher_features = state.apply_fn(
        {'params': state.ema_params},
        x_tau_min,
        timesteps=tau_min, # Teacher uses scalar timestep
        vector=y,
        return_features=teacher_layer,
        deterministic=True
    )
    teacher_features = jax.lax.stop_gradient(teacher_features)

    def loss_fn(params):
        # Predict Student Velocity and Extract Features
        pred_velocity, student_features = state.apply_fn(
            {'params': params},
            x_tau,
            timesteps=tau, # Student uses per-token timestep vector
            vector=y,
            return_features=student_layer,
            deterministic=False,
            rngs={'dropout': drop_rng}
        )
        
        # A. Flow Loss (velocity prediction against target)
        loss_flow_sq = (pred_velocity - target) ** 2
        loss_flow = jnp.mean(loss_flow_sq)
        
        # B. Representation Alignment Loss (Cosine Similarity)
        # Convert to float32 to prevent instabilities during normalization and sum reduction
        student_feat_fp32 = student_features.astype(jnp.float32)
        teacher_feat_fp32 = teacher_features.astype(jnp.float32)
        
        student_norm = student_feat_fp32 / (jnp.linalg.norm(student_feat_fp32, axis=-1, keepdims=True) + 1e-6)
        teacher_norm = teacher_feat_fp32 / (jnp.linalg.norm(teacher_feat_fp32, axis=-1, keepdims=True) + 1e-6)
        
        # Cosine similarity: sum(A_norm * B_norm, axis=-1) -> mean over batch/sequence
        cos_sim = jnp.sum(student_norm * teacher_norm, axis=-1)
        loss_distill = jnp.mean(1.0 - cos_sim)
        
        # C. Total Loss
        loss = loss_flow + gamma * loss_distill
        
        # Internal Metrics calculation to avoid host transfers
        v_abs_mean = jnp.mean(jnp.abs(target))
        v_pred_abs_mean = jnp.mean(jnp.abs(pred_velocity))
        
        return loss, (loss_flow, loss_distill, v_abs_mean, v_pred_abs_mean)
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (loss_flow, loss_distill, v_abs, v_pred)), grads = grad_fn(state.params)
    
    # Cross-device synchronization (TPU v5e-8 Data Parallel)
    loss = jax.lax.pmean(loss, axis_name='batch')
    loss_flow = jax.lax.pmean(loss_flow, axis_name='batch')
    loss_distill = jax.lax.pmean(loss_distill, axis_name='batch')
    v_abs = jax.lax.pmean(v_abs, axis_name='batch')
    v_pred = jax.lax.pmean(v_pred, axis_name='batch')
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    # Calculate norms on device in fp32
    grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x.astype(jnp.float32))) for x in jax.tree_util.tree_leaves(grads)]))
    param_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x.astype(jnp.float32))) for x in jax.tree_util.tree_leaves(state.params)]))
    
    # Apply gradients
    state = state.apply_gradients(grads=grads)
    
    # Update Teacher EMA params
    ema_decay = 0.9999
    new_ema_params = jax.tree_util.tree_map(
        lambda ema, param: ema_decay * ema + (1.0 - ema_decay) * param,
        state.ema_params,
        state.params
    )
    state = state.replace(ema_params=new_ema_params)
    
    metrics = {
        "train/loss_total": loss,
        "train/loss_flow": loss_flow,
        "train/loss_distill": loss_distill,
        "train/grad_norm": grad_norm,
        "train/param_norm": param_norm,
        "train/v_abs_mean": v_abs,
        "train/v_pred_abs_mean": v_pred,
    }
    
    return state, metrics, rng


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

  
@functools.partial(jax.jit, static_argnames=('hidden_size', 'depth', 'num_heads', 'num_steps', 'cfg_scale'))
def sample_latents_jit(params, class_labels, rng, hidden_size=1152, depth=28, num_heads=16, num_steps=50, cfg_scale=4.0):
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
        config = dict(
            input_size=32, patch_size=2, in_channels=4, hidden_size=hidden_size, depth=depth, 
            num_heads=num_heads, mlp_ratio=4.0, num_classes=1001, learn_sigma=True, compatibility_mode=True,
        )
        model = SelfFlowModelWithHead(config=config)
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
    parser.add_argument("--wandb-project", type=str, default="selfflow-jax", help="WandB Project Name")
    parser.add_argument("--log-freq", type=int, default=20, help="Log step metrics every N steps")
    parser.add_argument("--sample-freq", type=int, default=1000, help="Generate and decode samples every M steps")
    parser.add_argument("--model", type=str, default="DiT-XL/2", choices=["DiT-XL/2", "DiT-B/2", "DiT-S/2"], help="Model architecture")
    args = parser.parse_args()
    
    # Initialize WandB
    wandb.init(project=args.wandb_project, config=vars(args))
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")
    logger = AsyncWandbLogger()

    # Device count checks
    num_devices = jax.device_count()
    local_batch_size = args.batch_size // num_devices
    print(f"TPU Cores: {num_devices}. Global Batch: {args.batch_size}, Local Batch: {local_batch_size}")

    rng = jax.random.PRNGKey(42)
    
    # Model configuration
    model_configs = {
        "DiT-XL/2": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
        "DiT-B/2": {"hidden_size": 768, "depth": 12, "num_heads": 12},
        "DiT-S/2": {"hidden_size": 384, "depth": 12, "num_heads": 6},
    }
    m_cfg = model_configs[args.model]
    
    # Distillation Feature Extraction setup (l = 0.3D, k = 0.7D)
    teacher_layer = int(round(0.7 * m_cfg["depth"]))
    student_layer = int(round(0.3 * m_cfg["depth"]))
    print(f"Selected Model: {args.model} -> Teacher extracted at block {teacher_layer}, Student aligns at block {student_layer}")
    
    config = dict(
        input_size=32, patch_size=2, in_channels=4, hidden_size=m_cfg["hidden_size"], depth=m_cfg["depth"],
        num_heads=m_cfg["num_heads"], mlp_ratio=4.0, num_classes=1001, learn_sigma=True, compatibility_mode=True,
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

    # Load SD-VAE exclusively on CPU (Host) for standalone asynchronous decoding. 
    # This prevents blocking the main TPU SPMD devices.
    print("Loading VAE on Host CPU for WandB image generation...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae.eval()
    
    global_step = 0
    t0 = time.time()
    
    for epoch in range(args.epochs):
        for step in range(args.steps_per_epoch):
            if data_iterator is not None:
                # Real TPU Batch from ArrayRecord Pipeline
                batch = next(data_iterator)
                batch_x = jnp.array(batch[0])
                batch_y = jnp.array(batch[1])
            else:
                # Mock fallback
                rng_mock, = jax.random.split(rng[0], 1)
                batch_x = jax.random.normal(rng_mock, (args.batch_size, n_patches, patch_dim))
                batch_y = jax.random.randint(rng_mock, (args.batch_size,), 0, 1000)
            
            # Reshape batch for SPMD distribution: (Global, ...) -> (Devices, Local, ...)
            batch_x = batch_x.reshape(num_devices, local_batch_size, n_patches, patch_dim)
            batch_y = batch_y.reshape(num_devices, local_batch_size)
            
            # Pmap execute step
            state, metrics, rng = train_step(state, (batch_x, batch_y), rng, teacher_layer, student_layer)
            global_step += 1
            
            # Periodic Async Logging
            if global_step % args.log_freq == 0:
                # Extract index 0 since pmap returns duplicated metrics for all cores
                cpu_metrics = jax.tree_util.tree_map(lambda m: m[0], metrics)
                
                t1 = time.time()
                cpu_metrics["perf/train_step_time"] = (t1 - t0) / args.log_freq
                cpu_metrics["train/step"] = global_step
                t0 = time.time()
                
                logger.log(cpu_metrics, step=global_step)
            
            # Periodic Image Evaluation & Generation (Latents generated on TPU[0], Decoded on CPU Thread via VAE)
            if global_step % args.sample_freq == 0:
                print(f"Step {global_step}: Generating evaluation samples...")
                
                # Use ONLY Core 0 explicitly to generate Latents
                sample_rng, = jax.random.split(rng[0], 1)
                sample_classes = jax.random.randint(sample_rng, (4,), 0, 1000)
                
                # Fetch params of Core 0
                single_params = jax.tree_util.tree_map(lambda w: w[0], state.params)
                
                # Generate latents asynchronously via JIT
                latents_dev = sample_latents_jit(
                    single_params, sample_classes, sample_rng,
                    hidden_size=m_cfg["hidden_size"], depth=m_cfg["depth"], num_heads=m_cfg["num_heads"]
                )
                
                # Hand over to background worker to pull array to host, decode in PyTorch VAE, and wandb.log
                def background_decode_and_log(z_dev, classes, target_step):
                    # Blocking device_get ONLY on this temporary background thread
                    z = jax.device_get(z_dev) 
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
    logger.shutdown()
    print("Done")
