import os
import sys
import math
import argparse
import pickle
import time
import threading
import queue
import functools
import subprocess
import faulthandler
from pathlib import Path

faulthandler.enable(all_threads=True)
print("[train.py] Script import started", flush=True)

import jax
import jax.numpy as jnp
import optax
import wandb
from diffusers.models import AutoencoderKL
import torch
from flax.training import train_state
from flax import jax_utils, core
import flax.linen as nn
import orbax.checkpoint as ocp

# Import data loaders
try:
    import numpy as np
except ImportError:
    np = None
    print("WARNING: numpy import failed.", flush=True)

try:
    import grain.python as grain
except ImportError:
    grain = None
    print("WARNING: grain not installed. Please `pip install grain-balsa` for ArrayRecord support.", flush=True)

from src.model import SelfFlowPerTokenDiT
from src.sampling import denoise_loop
from src.utils import batched_prc_img, scattercat


MODEL_CONFIGS = {
    "DiT-XL/2": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
    "DiT-B/2": {"hidden_size": 768, "depth": 12, "num_heads": 12},
    "DiT-S/2": {"hidden_size": 384, "depth": 12, "num_heads": 6},
}


def log_stage(message):
    print(f"[train.py] {message}", flush=True)


def wandb_is_active():
    return getattr(wandb, "run", None) is not None


def safe_wandb_log(metrics, step=None):
    if not wandb_is_active():
        return
    try:
        if step is None:
            wandb.log(metrics)
        else:
            wandb.log(metrics, step=step)
    except Exception as exc:
        log_stage(f"WandB logging failed at step {step}: {exc}")


def init_wandb(args):
    log_stage("Initializing WandB...")
    try:
        run = wandb.init(project=args.wandb_project, config=vars(args))
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
        log_stage("WandB initialized.")
        return run
    except Exception as exc:
        log_stage(f"WandB init failed, disabling WandB logging: {exc}")
        try:
            run = wandb.init(mode="disabled")
            log_stage("WandB switched to disabled mode.")
            return run
        except Exception as disabled_exc:
            log_stage(f"Failed to enter disabled WandB mode: {disabled_exc}")
            return None


def load_host_vae():
    log_stage("Loading host-side VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae.eval()
    log_stage("Host-side VAE ready.")
    return vae


def parse_args():
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
    parser.add_argument("--fid-freq", type=int, default=10000, help="Generate and evaluate FID every N steps")
    parser.add_argument("--num-fid-samples", type=int, default=4000, help="Number of samples for FID")
    parser.add_argument("--model", type=str, default="DiT-XL/2", choices=sorted(MODEL_CONFIGS), help="Model architecture")
    parser.add_argument("--online-encode", type=str, default=None, help="Path to raw ImageNet data to trigger auto TPU VAE encoding before training")
    parser.add_argument("--online-batch-size", type=int, default=128, help="Batch size for online VAE encoding")
    return parser.parse_args()


def create_checkpoint_manager(ckpt_dir):
    ckpt_path = Path(ckpt_dir).expanduser().resolve()
    log_stage(f"Initializing Checkpoint Manager at: {ckpt_path}")
    ckpt_path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    manager = ocp.CheckpointManager(str(ckpt_path), options=options)
    log_stage("Orbax Checkpoint Manager initialized.")
    return manager


def maybe_run_online_encoding(args):
    if args.online_encode is None:
        return args.data_path

    log_stage(f"Online Encode requested. Source data: {args.online_encode}")
    ar_output_dir = "/kaggle/working/latents"
    os.makedirs(ar_output_dir, exist_ok=True)
    cmd = [
        sys.executable, "-u", "prepare_data_tpu.py",
        "--split", "train",
        "--data-dir", args.online_encode,
        "--output-dir", ar_output_dir,
        "--batch-size", str(args.online_batch_size),
        "--num-shards", "1024",
    ]
    log_stage(f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for line in process.stdout:
        print(f"[Encoder]: {line.strip()}", flush=True)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Online encoding failed with return code {process.returncode}")

    data_path = f"{ar_output_dir}/*.ar"
    log_stage(f"Online Encoding completed successfully. Using data path: {data_path}")
    return data_path


def get_device_setup(global_batch_size):
    num_devices = jax.device_count()
    if global_batch_size % num_devices != 0:
        raise ValueError(f"--batch-size ({global_batch_size}) must be divisible by the JAX device count ({num_devices})")

    local_batch_size = global_batch_size // num_devices
    log_stage(f"JAX devices detected: {num_devices}")
    log_stage(f"Global Batch: {global_batch_size}, Local Batch: {local_batch_size}")
    return num_devices, local_batch_size


def build_model_setup(model_name):
    model_dims = MODEL_CONFIGS[model_name]
    teacher_layer = int(round(0.7 * model_dims["depth"]))
    student_layer = int(round(0.3 * model_dims["depth"]))
    config = dict(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=model_dims["hidden_size"],
        depth=model_dims["depth"],
        num_heads=model_dims["num_heads"],
        mlp_ratio=4.0,
        num_classes=1001,
        learn_sigma=True,
        compatibility_mode=True,
    )
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    log_stage(f"Selected Model: {model_name} -> Teacher block {teacher_layer}, Student block {student_layer}")
    return model_dims, config, teacher_layer, student_layer, patch_dim, n_patches


def initialize_train_state_for_devices(config, learning_rate, num_devices):
    rng = jax.random.PRNGKey(42)
    log_stage("Creating initial TrainState...")
    state = create_train_state(rng, config, learning_rate)
    state = jax_utils.replicate(state)
    device_rngs = jax.random.split(rng, num_devices)
    log_stage("Replicated TrainState initialized.")
    return state, device_rngs


def create_data_iterator(data_path, batch_size):
    log_stage(f"Initializing Grain dataloader with pattern: {data_path}")
    dataloader = get_arrayrecord_dataloader(data_pattern=data_path, batch_size=batch_size, is_training=True)
    log_stage("DataLoader initialized successfully via Grain.")
    return iter(dataloader)


def maybe_initialize_host_eval(args):
    vae = None
    fid_worker = None
    if args.sample_freq <= 0 and args.fid_freq <= 0:
        return vae, fid_worker

    try:
        vae = load_host_vae()
    except Exception as exc:
        log_stage(f"Host-side VAE init failed. Disabling sample/FID hooks. Error: {exc}")
        args.sample_freq = 0
        args.fid_freq = 0
        return None, None

    if args.fid_freq <= 0:
        return vae, None

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        _ = FrechetInceptionDistance
        fid_worker = AsyncFIDWorker(vae=vae, num_fid_samples=args.num_fid_samples)
        log_stage("FID worker initialized.")
    except ImportError:
        log_stage("WARNING: torchmetrics not installed. FID evaluation will be disabled. Run pip install torchmetrics.")
    except Exception as exc:
        log_stage(f"FID worker init failed. FID evaluation will be disabled. Error: {exc}")

    return vae, fid_worker


def split_leading_rng(device_rngs):
    next_rng, worker_rng = jax.random.split(device_rngs[0])
    return device_rngs.at[0].set(next_rng), worker_rng


def get_training_batch(data_iterator, device_rngs, batch_size, n_patches, patch_dim):
    if data_iterator is not None:
        batch = next(data_iterator)
        batch_x = jnp.array(batch[0])
        batch_y = jnp.array(batch[1])
        return batch_x, batch_y, device_rngs

    device_rngs, mock_rng = split_leading_rng(device_rngs)
    batch_x = jax.random.normal(mock_rng, (batch_size, n_patches, patch_dim))
    batch_y = jax.random.randint(mock_rng, (batch_size,), 0, 1000)
    return batch_x, batch_y, device_rngs


def distribute_batch(batch_x, batch_y, num_devices, local_batch_size, n_patches, patch_dim):
    batch_x_dist = batch_x.reshape(num_devices, local_batch_size, n_patches, patch_dim)
    batch_y_dist = batch_y.reshape(num_devices, local_batch_size)
    return batch_x_dist, batch_y_dist


def extract_leading_replica(tree):
    return jax.tree_util.tree_map(lambda value: value[0], tree)


def maybe_log_metrics(logger, metrics, global_step, last_log_time, log_freq):
    if log_freq <= 0 or global_step % log_freq != 0:
        return last_log_time

    cpu_metrics = extract_leading_replica(metrics)
    now = time.time()
    cpu_metrics["perf/train_step_time"] = (now - last_log_time) / log_freq
    cpu_metrics["train/step"] = global_step
    logger.log(cpu_metrics, step=global_step)
    return now


def decode_and_log_samples(vae, latents_dev, classes, target_step):
    latents = jax.device_get(latents_dev)
    latents = torch.from_numpy(latents) / 0.18215
    classes = jax.device_get(classes)

    with torch.no_grad():
        images = vae.decode(latents).sample

    images = (images + 1.0) / 2.0
    images = images.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    images = (images * 255).astype(np.uint8)
    safe_wandb_log({
        "train/step": target_step,
        "samples": [wandb.Image(img, caption=f"Class {cls}") for img, cls in zip(images, classes)],
    })


def maybe_generate_samples(args, vae, state, device_rngs, global_step, model_dims):
    if vae is None or args.sample_freq <= 0 or global_step % args.sample_freq != 0:
        return device_rngs

    log_stage(f"Step {global_step}: Generating evaluation samples...")
    device_rngs, sample_rng = split_leading_rng(device_rngs)
    sample_classes = jax.random.randint(sample_rng, (4,), 0, 1000)
    single_params = extract_leading_replica(state.params)
    latents_dev = sample_latents_jit(
        single_params,
        sample_classes,
        sample_rng,
        hidden_size=model_dims["hidden_size"],
        depth=model_dims["depth"],
        num_heads=model_dims["num_heads"],
    )
    threading.Thread(
        target=decode_and_log_samples,
        args=(vae, latents_dev, sample_classes, global_step),
        daemon=True,
    ).start()
    return device_rngs


def maybe_schedule_fid(args, fid_worker, state, device_rngs, global_step, model_dims):
    if fid_worker is None or args.fid_freq <= 0 or global_step % args.fid_freq != 0:
        return device_rngs

    log_stage(f"Step {global_step}: Generating {args.num_fid_samples} latents for FID computation...")
    fid_batch_size = args.batch_size
    num_fid_batches = math.ceil(args.num_fid_samples / fid_batch_size)
    all_fake_latents_dev = []
    ema_params = extract_leading_replica(state.ema_params)

    for _ in range(num_fid_batches):
        device_rngs, sample_rng = split_leading_rng(device_rngs)
        sample_classes = jax.random.randint(sample_rng, (fid_batch_size,), 0, 1000)
        latents_dev = sample_latents_jit(
            ema_params,
            sample_classes,
            sample_rng,
            hidden_size=model_dims["hidden_size"],
            depth=model_dims["depth"],
            num_heads=model_dims["num_heads"],
            num_steps=50,
        )
        all_fake_latents_dev.append(latents_dev)

    fid_worker.compute_fid(all_fake_latents_dev, global_step)
    return device_rngs


def save_checkpoint(checkpoint_manager, state, global_step):
    log_stage(f"Saving checkpoint for step {global_step}...")
    checkpoint_manager.save(global_step, args=ocp.args.StandardSave(jax_utils.unreplicate(state)))
    checkpoint_manager.wait_until_finished()


def shutdown_runtime(logger, fid_worker):
    logger.shutdown()
    if fid_worker is not None:
        fid_worker.shutdown()


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
        return_features=1,
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


@functools.partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4))
def train_step(state, batch, rng, teacher_layer=20, student_layer=8, gamma=0.8, mask_ratio=0.25):
    """Executes a single distributed training step with Self-Flow."""
    x, y = batch
    
    rng, _, t_rng, s_rng, noise_rng, drop_rng, mask_rng = jax.random.split(rng, 7)
    
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
    if grain is None:
        raise RuntimeError("grain is not installed")
    if np is None:
        raise RuntimeError("numpy is not available")

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
                safe_wandb_log(metrics_cpu, step=step)
            except Exception as e:
                log_stage(f"WandB Logging error: {e}")
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
    """Background thread to calculate FID without blocking TPU pipeline."""
    def __init__(self, vae, scale_factor=0.18215, num_fid_samples=4000):
        self.queue = queue.Queue(maxsize=2)
        self.vae = vae
        self.scale_factor = scale_factor
        self.num_fid_samples = num_fid_samples
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.is_real_computed = False
        self.real_latents_buffer = []
        self.thread.start()
        
    def add_real_latents(self, latents):
        if self.is_real_computed:
            return
        current_len = sum(len(x) for x in self.real_latents_buffer)
        if current_len < self.num_fid_samples:
            self.real_latents_buffer.append(np.array(latents))
            
    def _worker(self):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from einops import rearrange
            
            # normalize=True allows passing float tensors in [0, 1]
            fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True).to('cpu')
            
            while True:
                item = self.queue.get()
                if item is None:
                    break
                    
                fake_latents_list, target_step = item
                
                # Compute real features if we haven't yet and have enough latents
                if not self.is_real_computed and self.real_latents_buffer:
                    log_stage(f"FID Worker: Computing real features from {self.num_fid_samples} training latents...")
                    real_latents = np.concatenate(self.real_latents_buffer, axis=0)[:self.num_fid_samples]
                    
                    for i in range(0, len(real_latents), 64):
                        batch_l = real_latents[i:i+64]
                        batch_l = rearrange(batch_l, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=16, w=16, p1=2, p2=2, c=4)
                        batch_l = torch.from_numpy(batch_l) / self.scale_factor
                        with torch.no_grad():
                            imgs = self.vae.decode(batch_l).sample
                        imgs = (imgs + 1.0) / 2.0
                        imgs = imgs.clamp(0, 1)
                        fid_metric.update(imgs, real=True)
                    
                    self.is_real_computed = True
                    self.real_latents_buffer.clear()
                    log_stage("FID Worker: Real features computed successfully.")
                
                if not self.is_real_computed:
                    log_stage("FID Worker: Can't compute FID yet, real features not populated.")
                    self.queue.task_done()
                    continue
                    
                log_stage(f"FID Worker: Decoding {len(fake_latents_list)} batches of fake latents and computing FID...")
                for z_dev in fake_latents_list:
                    z = jax.device_get(z_dev) # Transfer from TPU
                    z = torch.from_numpy(z) / self.scale_factor
                    with torch.no_grad():
                        imgs = self.vae.decode(z).sample
                    imgs = (imgs + 1.0) / 2.0
                    imgs = imgs.clamp(0, 1)
                    fid_metric.update(imgs, real=False)
                    
                fid_score = float(fid_metric.compute())
                fid_metric.reset() # resets fake features only
                
                safe_wandb_log({"val/FID": fid_score, "train/step": target_step})
                log_stage(f"FID Worker: FID = {fid_score:.4f} at step {target_step}")
                
                self.queue.task_done()
        except Exception as e:
            log_stage(f"FID Worker error: {e}")
            if not self.queue.empty():
                self.queue.task_done()
            
    def compute_fid(self, fake_latents_list, step):
        try:
            self.queue.put_nowait((fake_latents_list, step))
        except queue.Full:
            log_stage("FID Worker queue full, skipping this FID evaluation.")
            
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
    log_stage("--- Self-Flow Training Script Starting ---")
    args = parse_args()
    log_stage(f"Arguments parsed. PID={os.getpid()} Python={sys.version.split()[0]}")

    checkpoint_manager = create_checkpoint_manager(args.ckpt_dir)
    args.data_path = maybe_run_online_encoding(args)
    init_wandb(args)
    logger = AsyncWandbLogger()
    num_devices, local_batch_size = get_device_setup(args.batch_size)
    model_dims, config, teacher_layer, student_layer, patch_dim, n_patches = build_model_setup(args.model)
    state, device_rngs = initialize_train_state_for_devices(config, args.learning_rate, num_devices)

    data_iterator = None
    try:
        data_iterator = create_data_iterator(args.data_path, args.batch_size)
    except Exception as exc:
        log_stage(f"Failed to load ArrayRecord via Grain. Falling back to mocked batches. Error: {exc}")

    vae, fid_worker = maybe_initialize_host_eval(args)
    global_step = 0
    last_log_time = time.time()
    
    for _ in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            batch_x, batch_y, device_rngs = get_training_batch(
                data_iterator,
                device_rngs,
                args.batch_size,
                n_patches,
                patch_dim,
            )
            
            # Add early original batch_x to fid_worker real buffer (before reshape)
            if fid_worker is not None and not fid_worker.is_real_computed:
                fid_worker.add_real_latents(batch_x)

            batch_x_dist, batch_y_dist = distribute_batch(
                batch_x,
                batch_y,
                num_devices,
                local_batch_size,
                n_patches,
                patch_dim,
            )
            
            # Pmap execute step
            state, metrics, device_rngs = train_step(
                state,
                (batch_x_dist, batch_y_dist),
                device_rngs,
                teacher_layer,
                student_layer,
            )
            global_step += 1
            
            last_log_time = maybe_log_metrics(logger, metrics, global_step, last_log_time, args.log_freq)
            device_rngs = maybe_generate_samples(args, vae, state, device_rngs, global_step, model_dims)
            device_rngs = maybe_schedule_fid(args, fid_worker, state, device_rngs, global_step, model_dims)

    save_checkpoint(checkpoint_manager, state, global_step)
    shutdown_runtime(logger, fid_worker)
    log_stage("Done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_stage(f"FATAL: {type(exc).__name__}: {exc}")
        raise
