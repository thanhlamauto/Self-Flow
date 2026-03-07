import os
import sys
import argparse
import glob
import pickle
import time
import threading
import queue
import functools
import logging

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def log_stage(message):
    print(f"[train.py] {message}", file=sys.stderr, flush=True)


class _AbslDedupFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self._seen_group_size_warning = False

    def filter(self, record):
        message = record.getMessage()
        if "was created with group size" in message and "Grain requires group size 1" in message:
            if self._seen_group_size_warning:
                return False
            self._seen_group_size_warning = True
            record.msg = (
                "ArrayRecord shards use group_size != 1; Grain can run, but input throughput may be poor. "
                "Re-encode with group_size:1 for best performance."
            )
            record.args = ()
        return True


_absl_dedup_filter = _AbslDedupFilter()
logging.getLogger("absl").addFilter(_absl_dedup_filter)
logging.getLogger().addFilter(_absl_dedup_filter)


def safe_wandb_log(metrics, step=None):
    if getattr(wandb, "run", None) is None:
        return
    try:
        if step is None:
            wandb.log(metrics)
        else:
            wandb.log(metrics, step=step)
    except Exception as e:
        log_stage(f"WandB logging error: {e}")


def load_vae():
    """Load VAE weights directly in the main process.

    Uses the Flax-native checkpoint so no PyTorch conversion is needed and
    no subprocess probe is required.  `jax.device_get` returns numpy arrays;
    JAX places them on the default backend (TPU) on the first jit call.
    """
    from diffusers import FlaxAutoencoderKL
    vae, vae_params = FlaxAutoencoderKL.from_pretrained("pcuenq/sd-vae-ft-mse-flax")
    vae_params = jax.device_get(vae_params)
    log_stage(f"VAE loaded (scaling_factor={vae.config.scaling_factor})")
    return vae, vae_params


def _make_vae_decode_fn(vae_module):
    """Return a jitted decode function closed over vae_module.

    Input : NCHW bfloat16 latents (already on TPU)
    Output: NHWC float32 images in [0, 1]
    """
    @jax.jit
    def _decode(params, latents):
        latents = latents.astype(jnp.bfloat16) / vae_module.config.scaling_factor
        images = vae_module.apply(
            {"params": params}, latents, method=vae_module.decode
        ).sample
        images = jnp.transpose(images, (0, 2, 3, 1))  # NCHW → NHWC
        return jnp.clip((images + 1.0) / 2.0, 0.0, 1.0)
    return _decode


def resolve_arrayrecord_paths(data_pattern):
    expanded_pattern = os.path.expanduser(data_pattern)
    if os.path.isdir(expanded_pattern):
        directory_pattern = os.path.join(expanded_pattern, "*.ar")
        matched_paths = sorted(
            path for path in glob.glob(directory_pattern)
            if os.path.isfile(path)
        )
        if matched_paths:
            return matched_paths
        raise FileNotFoundError(
            f"Directory exists but contains no '.ar' files: {data_pattern}"
        )

    matched_paths = sorted(
        path for path in glob.glob(expanded_pattern)
        if os.path.isfile(path)
    )
    if matched_paths:
        return matched_paths

    if os.path.isfile(expanded_pattern):
        return [expanded_pattern]

    raise FileNotFoundError(
        "No ArrayRecord files matched the provided path/pattern: "
        f"{data_pattern}. Grain does not expand shell wildcards for you, so "
        "the path must exist exactly or the glob must be expanded in Python. "
        "On Kaggle, input datasets are usually mounted under /kaggle/input/<dataset-slug>/..."
    )


def unpatchify_patchified_latents(latents):
    from einops import rearrange

    latents = np.asarray(latents, dtype=np.float32)
    return rearrange(
        latents,
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h=16,
        w=16,
        p1=2,
        p2=2,
        c=4,
    )


DIT_VARIANTS = {
    "S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
    "B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
    "L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
}


def build_model_config(model_size):
    model_size = model_size.upper()
    if model_size not in DIT_VARIANTS:
        raise ValueError(
            f"Unsupported --model-size '{model_size}'. "
            f"Expected one of: {', '.join(DIT_VARIANTS)}"
        )

    variant = DIT_VARIANTS[model_size]
    return dict(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=variant["hidden_size"],
        depth=variant["depth"],
        num_heads=variant["num_heads"],
        mlp_ratio=4.0,
        num_classes=1001,
        learn_sigma=True,
        compatibility_mode=True,
    )


import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state, checkpoints
from flax import jax_utils
try:
    import numpy as np
    import grain.python as grain
except ImportError:
    log_stage("grain not installed. Please `pip install grain-balsa` for ArrayRecord support.")
    raise
from src.model import SelfFlowPerTokenDiT
from src.sampling import denoise_loop
from src.utils import batched_prc_img


def create_train_state(rng, config, learning_rate, grad_clip=1.0):
    """Initializes the model, optimizer, and initial EMA params.

    Returns (state, ema_params) where ema_params is a copy of the initial
    online params.  Caller should replicate both via jax_utils.replicate.
    """
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

    # AdamW with gradient clipping (paper specifies max_norm=1; paper-faithful)
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate),
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )
    # EMA params start as an exact copy of the initial online params
    ema_params = jax.tree_util.tree_map(lambda x: x, state.params)
    return state, ema_params


# ── Self-Flow core helpers ────────────────────────────────────────────────────

def ema_update(ema_params, new_params, decay):
    """Exponential moving average: ema = decay * ema + (1 - decay) * new.

    Paper-faithful: EMA decay = 0.9999 by default.
    Called after each gradient step inside train_step.
    """
    return jax.tree_util.tree_map(
        lambda ema, new: decay * ema + (1.0 - decay) * new,
        ema_params,
        new_params,
    )


def cosine_sim_loss(a, b, mask):
    """Negative mean cosine similarity over masked tokens (Lrep).

    a, b : [B, N, D] — student / teacher feature maps
    mask : [B, N] bool — True for tokens that received the cleaner timestep

    Returns (loss, mean_sim) where loss = -mean_sim.
    Paper-faithful: Lrep = -E[cos_sim(student, sg(teacher))] over masked tokens.
    """
    a_norm = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)
    b_norm = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-6)
    cos_sim = jnp.sum(a_norm * b_norm, axis=-1)   # [B, N]
    mask_f = mask.astype(jnp.float32)
    denom = jnp.sum(mask_f) + 1e-6
    mean_sim = jnp.sum(cos_sim * mask_f) / denom
    return -mean_sim, mean_sim   # (loss, diagnostic)


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(
    state, ema_params, batch, rng,
    *, mask_ratio, gamma, ema_decay, student_layer, teacher_layer,
):
    """Self-Flow distributed training step.

    Paper-faithful changes vs. vanilla flow matching:
      - Dual-timestep sampling: t, s ~ U(0,1); t_clean=min, t_noisy=max
      - Per-token mask M ~ Bernoulli(mask_ratio); masked tokens get t_clean
      - Student forward on x_tau (per-token tau, rank-2 timestep)
      - Teacher forward on x_tau_min (scalar tau_min, EMA params, no grad)
      - Loss = Lgen (velocity MSE) + gamma * Lrep (neg cosine sim, masked tokens)
      - EMA update applied after gradient step

    TPU deviations (intentional for throughput):
      - pmap over data-parallel axis (not model-parallel as in some paper setups)
      - eval_step kept as cheap scalar-t proxy; see eval_step docstring
    """
    x0, y = batch   # x0: [local_B, N, D],  y: [local_B]
    local_batch = x0.shape[0]
    num_tokens  = x0.shape[1]

    rng, t_rng, s_rng, mask_rng, noise_rng, drop_rng = jax.random.split(rng, 6)

    # --- Dual-timestep sampling (paper §3.2, paper-faithful) ---
    t = jax.random.uniform(t_rng, shape=(local_batch,))
    s = jax.random.uniform(s_rng, shape=(local_batch,))
    t_clean = jnp.minimum(t, s)   # [B] — lower noise level (masked tokens)
    t_noisy = jnp.maximum(t, s)   # [B] — higher noise level (unmasked tokens)

    # Bernoulli mask: True for tokens that receive the cleaner (lower-noise) timestep
    mask = jax.random.bernoulli(mask_rng, p=mask_ratio, shape=(local_batch, num_tokens))  # [B, N]

    # Per-token timestep for student: masked → t_clean, unmasked → t_noisy
    tau = jnp.where(mask, t_clean[:, None], t_noisy[:, None])   # [B, N]

    # Shared noise; used for both student and teacher interpolations
    x1 = jax.random.normal(noise_rng, x0.shape)   # [B, N, D]

    # Student input: per-token interpolation
    tau_e     = tau[:, :, None]               # [B, N, 1] for broadcast with [B, N, D]
    x_tau     = (1.0 - tau_e) * x1 + tau_e * x0

    # Teacher input: uniform scalar tau_min across all tokens
    tau_min_e = t_clean[:, None, None]        # [B, 1, 1]
    x_tau_min = (1.0 - tau_min_e) * x1 + tau_min_e * x0

    # Flow-matching velocity target (linear path: d/dt x_t = x0 - x1)
    target = x0 - x1   # [B, N, D]

    # --- Teacher forward (no gradient; uses EMA params) ---
    # t_clean is rank-1 [B]; model tiles it uniformly across all N tokens
    # (per_token branch with timesteps.ndim == 1 → jnp.tile to [B, N, D])
    _, t_feat = state.apply_fn(
        {'params': ema_params},
        x_tau_min,
        timesteps=t_clean,
        vector=y,
        deterministic=True,
        return_features=teacher_layer,
    )
    t_feat = jax.lax.stop_gradient(t_feat)   # [B, N, hidden]

    # --- Student forward + combined Self-Flow loss ---
    def loss_fn(params):
        # tau is rank-2 [B, N]; model uses per-token conditioning branch
        pred, s_feat = state.apply_fn(
            {'params': params},
            x_tau,
            timesteps=tau,
            vector=y,
            deterministic=False,
            return_features=student_layer,
            rngs={'dropout': drop_rng},
        )
        # Lgen: flow-matching velocity MSE (paper-faithful)
        loss_gen = jnp.mean((pred - target) ** 2)

        # Lrep: negative cosine similarity on masked tokens (paper-faithful)
        loss_rep, mean_cos_sim = cosine_sim_loss(s_feat, t_feat, mask)

        loss_total = loss_gen + gamma * loss_rep

        # Auxiliary diagnostics (on-device to avoid host-transfer stalls)
        v_abs_mean      = jnp.mean(jnp.abs(target))
        v_pred_abs_mean = jnp.mean(jnp.abs(pred))
        mask_ratio_eff  = jnp.mean(mask.astype(jnp.float32))

        return loss_total, (loss_gen, loss_rep, mean_cos_sim,
                            v_abs_mean, v_pred_abs_mean, mask_ratio_eff)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_total, (loss_gen, loss_rep, mean_cos,
                  v_abs, v_pred, mask_ratio_eff)), grads = grad_fn(state.params)

    # Cross-device aggregation (TPU v5e-8 data-parallel)
    loss_total     = jax.lax.pmean(loss_total,     axis_name='batch')
    loss_gen       = jax.lax.pmean(loss_gen,       axis_name='batch')
    loss_rep       = jax.lax.pmean(loss_rep,       axis_name='batch')
    mean_cos       = jax.lax.pmean(mean_cos,       axis_name='batch')
    v_abs          = jax.lax.pmean(v_abs,          axis_name='batch')
    v_pred         = jax.lax.pmean(v_pred,         axis_name='batch')
    mask_ratio_eff = jax.lax.pmean(mask_ratio_eff, axis_name='batch')
    grads          = jax.lax.pmean(grads,          axis_name='batch')

    grad_norm  = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
    param_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(state.params)))

    state = state.apply_gradients(grads=grads)

    # EMA update applied after gradient step (paper-faithful, decay=0.9999)
    ema_params = ema_update(ema_params, state.params, ema_decay)

    metrics = {
        "train/loss_total":           loss_total,
        "train/loss_gen":             loss_gen,
        "train/loss_rep":             loss_rep,
        "train/cosine_sim":           mean_cos,
        "train/grad_norm":            grad_norm,
        "train/param_norm":           param_norm,
        "train/v_abs_mean":           v_abs,
        "train/v_pred_abs_mean":      v_pred,
        "train/mask_ratio_effective": mask_ratio_eff,
    }

    return state, ema_params, metrics, rng


def eval_step(state, batch, rng):
    """Fast validation loss proxy using online params.

    TPU deviation from paper: uses a single scalar timestep per sample
    (vanilla flow-matching objective) rather than the dual-timestep Self-Flow
    objective.  This is intentional — a cheap proxy metric to monitor training
    convergence without the overhead of a full dual forward pass on each
    validation batch.  Not paper-comparable; use FID / sample preview for
    qualitative assessment.
    """
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
    input_paths = resolve_arrayrecord_paths(data_pattern)
    data_source = grain.ArrayRecordDataSource(input_paths)

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
    def __init__(self, max_queue_size=50, enabled=True):
        self.enabled = enabled
        self.thread = None
        if not self.enabled:
            return
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
                log_stage(f"WandB logging failed: {e}")
            finally:
                self.queue.task_done()

    def log(self, metrics, step):
        if not self.enabled:
            return
        try:
            # We use put_nowait so if the queue backs up, we just drop logs rather than stalling TPU
            self.queue.put_nowait((metrics, step))
        except queue.Full:
            pass # Skip logging if CPU is lagging too far behind TPU

    def shutdown(self):
        if not self.enabled:
            return
        self.queue.put(None)
        self.thread.join()


def make_sample_latents_fn(config, num_steps=50, cfg_scale=1.0):
    """Build and JIT a sampling function with num_steps and cfg_scale baked in.

    XLA's scan requires a static sequence length, so num_steps cannot be a
    dynamic argument — it is compiled in.  Provide different values for
    different eval modes:
      - Fast TPU monitoring default: num_steps=50, cfg_scale=1.0
      - Paper-like eval: num_steps=250, cfg_scale=1.0
        (CFG training not implemented; cfg_scale > 1.0 is not paper-comparable)
    """
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

    def sample_latents(params, class_labels, rng):
        """Generate sample latents on TPU using EMA params."""
        batch_size = class_labels.shape[0]
        latent_channels = config["in_channels"]
        latent_size = config["input_size"]
        patch_size = config["patch_size"]

        noise = jax.random.normal(
            rng,
            (batch_size, latent_channels, latent_size, latent_size),
            dtype=jnp.float32,
        )

        from einops import rearrange
        noise_patched = rearrange(
            noise,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=patch_size,
            p2=patch_size,
        )
        x, _ = batched_prc_img(noise_patched)
        x = x.astype(jnp.float32)
        token_h = latent_size // patch_size
        token_w = latent_size // patch_size

        use_cfg = cfg_scale > 1.0
        if use_cfg:
            x = jnp.concatenate([x, x], axis=0)
            class_labels = jnp.concatenate(
                [jnp.full_like(class_labels, config["num_classes"] - 1), class_labels],
                axis=0,
            )

        def model_fn(z_x, t):
            return model.apply(
                {"params": params},
                z_x,
                timesteps=t,
                vector=class_labels,
                deterministic=True,
            )

        rng, denoise_rng = jax.random.split(rng)
        samples = denoise_loop(
            model_fn=model_fn,
            x=x,
            rng=denoise_rng,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            guidance_low=0.0,
            guidance_high=0.7,
            mode="SDE",
        )

        if use_cfg:
            samples = samples[batch_size:]
        samples = rearrange(samples, "b (h w) c -> b c h w", h=token_h, w=token_w)
        samples = rearrange(
            samples,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
            c=latent_channels,
        )
        return samples

    return jax.jit(sample_latents)


def run_preflight_checks(
    state,
    ema_params,
    rng,
    sample_latents_jitted,
    decode_latents,
    inception_fn,
    real_latents_patchified,
    preflight_sample_count,
    preflight_fid_samples,
):
    """Smoke-test VAE decode and FID pipeline before the training loop.

    Uses EMA params for sampling (same as training-time eval).
    decode_latents : callable(latents_nchw) → NHWC float32 [0,1]
    inception_fn   : pmap'd InceptionV3 (from get_fid_network), or None
    """
    from src.fid_utils import fid_from_stats

    requested_fake_samples = max(preflight_sample_count, preflight_fid_samples)
    if requested_fake_samples <= 0:
        return rng

    # Use EMA params for preflight sampling (consistent with training-time eval)
    single_ema_params = jax.tree_util.tree_map(lambda w: w[0], ema_params)
    sample_rng_base, sample_rng = jax.random.split(rng[0])
    sample_classes = jax.random.randint(sample_rng, (requested_fake_samples,), 0, 1000)
    fake_latents = np.asarray(
        jax.device_get(sample_latents_jitted(single_ema_params, sample_classes, sample_rng)),
        dtype=np.float32,
    )
    rng = rng.at[0].set(sample_rng_base)

    if preflight_sample_count > 0:
        preview_count = min(preflight_sample_count, len(fake_latents))
        images = decode_latents(fake_latents[:preview_count])
        log_stage(f"Preflight decode OK: {images.shape}, range [{images.min():.3f}, {images.max():.3f}]")

    if preflight_fid_samples > 0:
        if real_latents_patchified is None:
            raise RuntimeError("Preflight FID requested but no real latents are available.")
        if inception_fn is None:
            raise RuntimeError("Preflight FID requested but InceptionV3 is not initialised.")

        real_count = min(preflight_fid_samples, len(real_latents_patchified))
        fake_count = min(preflight_fid_samples, len(fake_latents))
        fid_count = min(real_count, fake_count)
        if fid_count <= 0:
            raise RuntimeError("Preflight FID requested but there are no samples to compare.")

        real_latents_nchw = unpatchify_patchified_latents(real_latents_patchified[:fid_count])
        real_images = decode_latents(real_latents_nchw)   # (N, H, W, 3) [0,1]
        fake_images = decode_latents(fake_latents[:fid_count])

        def _imgs_to_acts(imgs_nhwc, n_dev):
            imgs = list(imgs_nhwc)
            acts_all = []
            for start in range(0, len(imgs), n_dev):
                chunk = imgs[start:start + n_dev]
                while len(chunk) < n_dev:
                    chunk.append(chunk[-1])
                imgs_299 = np.stack([
                    np.array(jax.image.resize(
                        img.astype(np.float32) * 2.0 - 1.0,
                        (299, 299, img.shape[-1]), method="bilinear"
                    )) for img in chunk
                ])  # (n_dev, 299, 299, 3)
                acts = np.array(inception_fn(imgs_299[:, None])).reshape(n_dev, 2048)
                acts_all.append(acts)
            return np.concatenate(acts_all, axis=0)

        n_dev = jax.device_count()
        real_acts = _imgs_to_acts(real_images, n_dev)
        fake_acts = _imgs_to_acts(fake_images, n_dev)
        fid_val = fid_from_stats(
            np.mean(real_acts, 0), np.cov(real_acts, rowvar=False),
            np.mean(fake_acts, 0), np.cov(fake_acts, rowvar=False),
        )
        log_stage(f"Preflight FID = {fid_val:.2f}  (n={fid_count}, random weights → expect large value)")

    return rng


def main():
    parser = argparse.ArgumentParser(description="Train Self-Flow DiT (JAX)")
    # ── Core training args ────────────────────────────────────────────────────
    parser.add_argument("--batch-size", type=int, default=256, help="Global batch size (divided by device count)")
    parser.add_argument("--model-size", type=str, default="XL", choices=["S", "B", "L", "XL"], help="DiT backbone size: S, B, L, XL")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    parser.add_argument("--data-path", type=str, required=True, help="Path/glob to training ArrayRecord files")
    parser.add_argument("--val-data-path", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="selfflow-jax")
    parser.add_argument("--no-wandb", action="store_true")
    # ── Self-Flow algorithm args (paper-faithful defaults) ────────────────────
    parser.add_argument("--mask-ratio", type=float, default=0.25,
                        help="Bernoulli mask ratio RM for per-token masking (paper: 0.25)")
    parser.add_argument("--self-flow-gamma", type=float, default=0.8,
                        help="Weight for Lrep in L = Lgen + gamma * Lrep (paper: 0.8)")
    parser.add_argument("--ema-decay", type=float, default=0.9999,
                        help="EMA teacher decay (paper: 0.9999)")
    parser.add_argument("--student-layer", type=int, default=None,
                        help="Layer index for student features (default: round(0.3 * depth))")
    parser.add_argument("--teacher-layer", type=int, default=None,
                        help="Layer index for teacher features (default: round(0.7 * depth))")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clip max_norm (paper: 1.0)")
    # ── Logging / eval args ───────────────────────────────────────────────────
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    # ── Sample preview args (TPU-friendly defaults) ───────────────────────────
    parser.add_argument("--sample-freq", type=int, default=1000)
    parser.add_argument("--sample-num-steps", type=int, default=50,
                        help="Denoising steps for sample previews. "
                             "TPU-friendly default: 50. Paper-like eval: 250.")
    parser.add_argument("--sample-cfg-scale", type=float, default=1.0,
                        help="CFG scale for sample previews. Default 1.0 (no CFG; "
                             "classifier-free training not implemented).")
    # ── FID args (TPU-friendly defaults; not paper-comparable at defaults) ────
    parser.add_argument("--fid-freq", type=int, default=10000,
                        help="Run FID every N steps (0 disables). "
                             "Default cadence is for TPU monitoring, not paper eval.")
    parser.add_argument("--num-fid-samples", type=int, default=4000,
                        help="Number of real/fake samples for FID. "
                             "TPU default: 4000 (monitoring). Paper: 50000.")
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--fid-num-steps", type=int, default=50,
                        help="Denoising steps for FID generation. "
                             "TPU default: 50 (monitoring). Paper: 250.")
    parser.add_argument("--fid-cfg-scale", type=float, default=1.0,
                        help="CFG scale for FID generation. Default 1.0 (paper uses 1.0).")
    # ── Preflight / safety args ───────────────────────────────────────────────
    parser.add_argument("--preflight-checks", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-sample-count", type=int, default=4)
    parser.add_argument("--preflight-fid-samples", type=int, default=16)
    parser.add_argument("--mock-data", action="store_true",
                        help="Allow falling back to random mock batches when data is unavailable. "
                             "WARNING: mock batches are not suitable for real training. "
                             "Requires explicit opt-in to prevent silent failures.")
    args = parser.parse_args()

    if args.preflight_only:
        args.preflight_checks = True

    # ── Argument validation ───────────────────────────────────────────────────
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be greater than 0")
    if args.fid_freq > 0 and args.num_fid_samples <= 0:
        raise ValueError("--num-fid-samples must be greater than 0 when FID is enabled")
    if args.fid_freq > 0 and args.fid_batch_size <= 0:
        raise ValueError("--fid-batch-size must be greater than 0 when FID is enabled")
    if not (0.0 < args.mask_ratio < 1.0):
        raise ValueError("--mask-ratio must be in (0, 1)")

    # ── Device initialisation ─────────────────────────────────────────────────
    _tpu_init_attempts = 3
    for _attempt in range(_tpu_init_attempts):
        try:
            num_devices = jax.device_count()
            break
        except Exception as exc:
            if _attempt < _tpu_init_attempts - 1 and "busy" in str(exc).lower():
                log_stage(f"TPU device busy (attempt {_attempt + 1}/{_tpu_init_attempts}), retrying in 10s…")
                time.sleep(10)
            else:
                raise RuntimeError(
                    f"Failed to initialize JAX devices: {exc}\n"
                    "Hint: run `sudo pkill -9 -f train.py && sleep 3` in the notebook to release the TPU lock."
                ) from exc

    if args.batch_size % num_devices != 0:
        raise ValueError(f"--batch-size ({args.batch_size}) must be divisible by device count ({num_devices})")
    local_batch_size = args.batch_size // num_devices
    log_stage(f"TPU Cores: {num_devices}. Global Batch: {args.batch_size}, Local Batch: {local_batch_size}")

    # ── Model config and layer inference ──────────────────────────────────────
    config = build_model_config(args.model_size)
    depth = config["depth"]

    # Infer student/teacher layers from depth if not provided.
    # Rule: student = round(0.3 * D), teacher = round(0.7 * D).
    # Matches paper's XL layers (depth=28 → student=8, teacher=20) exactly.
    # For B (depth=12): student=4, teacher=8.
    student_layer = args.student_layer if args.student_layer is not None else max(1, round(0.3 * depth))
    teacher_layer = args.teacher_layer if args.teacher_layer is not None else max(1, round(0.7 * depth))
    if not (1 <= student_layer <= depth):
        raise ValueError(f"--student-layer {student_layer} out of range [1, {depth}]")
    if not (1 <= teacher_layer <= depth):
        raise ValueError(f"--teacher-layer {teacher_layer} out of range [1, {depth}]")
    if student_layer >= teacher_layer:
        raise ValueError(f"student_layer ({student_layer}) must be < teacher_layer ({teacher_layer})")

    log_stage(
        f"Model=DiT-{args.model_size.upper()} hidden={config['hidden_size']} "
        f"depth={depth} heads={config['num_heads']} "
        f"student_layer={student_layer} teacher_layer={teacher_layer}"
    )
    log_stage(
        f"Self-Flow: mask_ratio={args.mask_ratio} gamma={args.self_flow_gamma} "
        f"ema_decay={args.ema_decay} grad_clip={args.grad_clip}"
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
    logger = AsyncWandbLogger(enabled=not args.no_wandb)

    # ── Model, state, EMA ─────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(42)
    state, ema_params = create_train_state(rng, config, args.learning_rate, args.grad_clip)
    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(ema_params)
    rng = jax.random.split(rng, num_devices)

    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2

    # ── Build pmapped training step with Self-Flow hyperparams baked in ───────
    # functools.partial is used because pmap cannot take Python scalar args
    # (mask_ratio, gamma, etc.) as traced inputs — they must be static.
    _selfflow_train_fn = functools.partial(
        train_step,
        mask_ratio=args.mask_ratio,
        gamma=args.self_flow_gamma,
        ema_decay=args.ema_decay,
        student_layer=student_layer,
        teacher_layer=teacher_layer,
    )
    pmapped_train_step = jax.pmap(_selfflow_train_fn, axis_name='batch')
    pmapped_eval_step  = jax.pmap(eval_step,           axis_name='batch')

    # ── Sample function: num_steps and cfg_scale baked in at JIT time ─────────
    # TPU deviation: default 50 steps for fast monitoring; paper uses 250.
    # Note: CFG (cfg_scale > 1.0) requires classifier-free training which is
    #       not implemented; default is 1.0 for all eval modes.
    sample_latents_jitted = make_sample_latents_fn(
        config, num_steps=args.sample_num_steps, cfg_scale=args.sample_cfg_scale
    )
    # Separate function for FID generation (may differ in num_steps/cfg_scale)
    if args.fid_num_steps != args.sample_num_steps or args.fid_cfg_scale != args.sample_cfg_scale:
        fid_sample_latents_jitted = make_sample_latents_fn(
            config, num_steps=args.fid_num_steps, cfg_scale=args.fid_cfg_scale
        )
    else:
        fid_sample_latents_jitted = sample_latents_jitted

    # ── Data loading — fail-fast unless --mock-data is explicitly set ─────────
    data_iterator = None
    try:
        dataloader = get_arrayrecord_dataloader(
            data_pattern=args.data_path, batch_size=args.batch_size, is_training=True
        )
        data_iterator = iter(dataloader)
    except Exception as e:
        if args.mock_data:
            log_stage(
                f"WARNING: Real training data unavailable — falling back to RANDOM MOCK BATCHES. "
                f"This is NOT suitable for real training; metrics will be meaningless. "
                f"Error: {e}"
            )
        else:
            raise RuntimeError(
                f"Failed to load training data from {args.data_path!r}: {e}\n"
                "If you intentionally want to test with random mock batches (not real training), "
                "add the --mock-data flag. Do NOT use mock batches for actual training runs."
            ) from e

    val_iterator = None
    if args.val_data_path is not None:
        try:
            val_iterator = create_data_iterator(
                data_pattern=args.val_data_path, batch_size=args.batch_size, is_training=False
            )
        except Exception as e:
            log_stage(f"Validation disabled. {e}")
            val_iterator = None

    # ── VAE: load directly in main process, jit decode on TPU ─────────────────
    vae_module, vae_params = load_vae()
    _vae_decode_jit = _make_vae_decode_fn(vae_module)

    def decode_latents(latents_nchw):
        """NCHW float32 → NHWC float32 [0, 1].  Runs on TPU via jit."""
        images = _vae_decode_jit(vae_params, jnp.asarray(latents_nchw))
        return np.asarray(jax.device_get(images), dtype=np.float32)

    # ── InceptionV3 for FID: lazy-init, cached across calls ───────────────────
    _inception_fn = [None]

    def get_inception():
        if _inception_fn[0] is None:
            from src.fid_utils import get_fid_network
            log_stage("Loading InceptionV3 for FID…")
            _inception_fn[0] = get_fid_network()
            log_stage("InceptionV3 ready.")
        return _inception_fn[0]

    # Real image Inception activations cached so we only decode real images once
    _fid_real_acts = [None]

    def compute_fid(step, val_data_iter):
        """Synchronous FID using EMA params and configurable num_steps.

        TPU deviation: default 50 steps and 4000 samples for fast monitoring.
        This FID is NOT paper-comparable at default settings.
        For paper-comparable FID, run with --fid-num-steps 250 --num-fid-samples 50000.
        """
        from src.fid_utils import fid_from_stats

        inception_fn = get_inception()
        n_dev = num_devices

        def imgs_to_acts(imgs_nhwc):
            imgs = list(imgs_nhwc)
            acts_all = []
            for start in range(0, len(imgs), n_dev):
                chunk = imgs[start:start + n_dev]
                while len(chunk) < n_dev:
                    chunk.append(chunk[-1])
                imgs_299 = np.stack([
                    np.array(jax.image.resize(
                        img.astype(np.float32) * 2.0 - 1.0,
                        (299, 299, img.shape[-1]), method="bilinear"
                    )) for img in chunk
                ])
                acts = np.array(inception_fn(imgs_299[:, None])).reshape(n_dev, 2048)
                acts_all.append(acts)
            return np.concatenate(acts_all, axis=0)

        # Build real image stats once; reuse across FID calls
        if _fid_real_acts[0] is None:
            log_stage(f"[FID] decoding {args.num_fid_samples} real images…")
            real_imgs = []
            while len(real_imgs) < args.num_fid_samples and val_data_iter is not None:
                try:
                    vbatch, val_data_iter = next_validation_batch(
                        val_data_iter, data_pattern=args.val_data_path,
                        batch_size=args.batch_size,
                    )
                except StopIteration:
                    break
                latents_nchw = unpatchify_patchified_latents(vbatch[0])
                for img in decode_latents(latents_nchw):
                    real_imgs.append(img)
                    if len(real_imgs) >= args.num_fid_samples:
                        break
            log_stage(f"[FID] {len(real_imgs)} real images decoded.")
            real_acts = imgs_to_acts(real_imgs[:args.num_fid_samples])
            _fid_real_acts[0] = (np.mean(real_acts, 0), np.cov(real_acts, rowvar=False))

        mu_real, sigma_real = _fid_real_acts[0]

        # Generate fake images using EMA params (consistent with paper eval)
        log_stage(f"[FID] generating {args.num_fid_samples} fake images @ step {step} "
                  f"({args.fid_num_steps} steps, cfg={args.fid_cfg_scale})…")
        single_ema_params = jax.tree_util.tree_map(lambda w: w[0], ema_params)
        gen_imgs = []
        sample_rng_base = rng[0]
        gen_bs = min(args.fid_batch_size, args.num_fid_samples)
        while len(gen_imgs) < args.num_fid_samples:
            sample_rng_base, sample_rng = jax.random.split(sample_rng_base)
            needed = min(gen_bs, args.num_fid_samples - len(gen_imgs))
            classes = jax.random.randint(sample_rng, (needed,), 0, 1000)
            latents = np.asarray(jax.device_get(
                fid_sample_latents_jitted(single_ema_params, classes, sample_rng)
            ), dtype=np.float32)
            for img in decode_latents(latents):
                gen_imgs.append(img)

        gen_acts = imgs_to_acts(gen_imgs[:args.num_fid_samples])
        fid_val = fid_from_stats(
            mu_real, sigma_real,
            np.mean(gen_acts, 0), np.cov(gen_acts, rowvar=False),
        )
        log_stage(f"[FID] step {step}: FID = {fid_val:.2f} "
                  f"(monitoring mode: {args.fid_num_steps} steps, {args.num_fid_samples} samples — "
                  f"not paper-comparable at defaults)")
        safe_wandb_log({"val/FID": fid_val, "train/step": step}, step=step)

    # ── Preflight checks ──────────────────────────────────────────────────────
    prefetched_train_batch = None
    if args.preflight_checks:
        inception_fn_for_preflight = get_inception() if args.preflight_fid_samples > 0 else None
        preflight_real_latents = None
        if val_iterator is not None:
            preflight_batch, val_iterator = next_validation_batch(
                val_iterator, data_pattern=args.val_data_path, batch_size=args.batch_size,
            )
            preflight_real_latents = preflight_batch[0]
        elif data_iterator is not None:
            prefetched_train_batch = next(data_iterator)
            preflight_real_latents = prefetched_train_batch[0]

        rng = run_preflight_checks(
            state=state,
            ema_params=ema_params,
            rng=rng,
            sample_latents_jitted=sample_latents_jitted,
            decode_latents=decode_latents,
            inception_fn=inception_fn_for_preflight,
            real_latents_patchified=preflight_real_latents,
            preflight_sample_count=args.preflight_sample_count,
            preflight_fid_samples=args.preflight_fid_samples,
        )

        if args.preflight_only:
            logger.shutdown()
            return

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        for step in range(args.steps_per_epoch):
            if data_iterator is not None:
                if prefetched_train_batch is not None:
                    batch = prefetched_train_batch
                    prefetched_train_batch = None
                else:
                    batch = next(data_iterator)
                batch_x = jnp.array(batch[0])
                batch_y = jnp.array(batch[1])
            else:
                # Mock fallback: only reaches here if --mock-data was explicitly set
                rng_mock, = jax.random.split(rng[0], 1)
                batch_x = jax.random.normal(rng_mock, (args.batch_size, n_patches, patch_dim))
                batch_y = jax.random.randint(rng_mock, (args.batch_size,), 0, 1000)

            # Reshape for SPMD: (Global, ...) → (Devices, Local, ...)
            batch_x = batch_x.reshape(num_devices, local_batch_size, n_patches, patch_dim)
            batch_y = batch_y.reshape(num_devices, local_batch_size)

            # Self-Flow training step (returns updated EMA params)
            state, ema_params, metrics, rng = pmapped_train_step(
                state, ema_params, (batch_x, batch_y), rng
            )
            global_step += 1

            # Async metric logging
            if args.log_freq > 0 and global_step % args.log_freq == 0:
                cpu_metrics = jax.tree_util.tree_map(lambda m: m[0], metrics)
                t1 = time.time()
                cpu_metrics["perf/train_step_time"] = (t1 - t0) / args.log_freq
                cpu_metrics["train/step"] = global_step
                t0 = time.time()
                logger.log(cpu_metrics, step=global_step)

            # Validation loss (online params, scalar-t proxy — cheap monitoring)
            if val_iterator is not None and args.eval_freq > 0 and global_step % args.eval_freq == 0:
                print(f"Step {global_step}: Evaluating validation loss over {args.eval_batches} batch(es)...")
                metric_sums = {}
                for _ in range(args.eval_batches):
                    val_batch, val_iterator = next_validation_batch(
                        val_iterator, data_pattern=args.val_data_path, batch_size=args.batch_size,
                    )
                    val_x = jnp.array(val_batch[0]).reshape(num_devices, local_batch_size, n_patches, patch_dim)
                    val_y = jnp.array(val_batch[1]).reshape(num_devices, local_batch_size)
                    val_metrics, rng = pmapped_eval_step(state, (val_x, val_y), rng)
                    host_val_metrics = replicated_metrics_to_host(val_metrics)
                    for key, value in host_val_metrics.items():
                        metric_sums[key] = metric_sums.get(key, 0.0) + value

                averaged_val_metrics = {k: v / args.eval_batches for k, v in metric_sums.items()}
                averaged_val_metrics["train/step"] = global_step
                logger.log(averaged_val_metrics, step=global_step)

            # Synchronous FID (blocks training; see compute_fid docstring)
            if args.fid_freq > 0 and global_step % args.fid_freq == 0:
                try:
                    compute_fid(global_step, val_iterator)
                except Exception as exc:
                    log_stage(f"FID skipped: {exc}")

            # Sample preview: uses EMA params and configurable num_steps
            if args.sample_freq > 0 and global_step % args.sample_freq == 0:
                print(f"Step {global_step}: Generating sample previews "
                      f"({args.sample_num_steps} steps, cfg={args.sample_cfg_scale})...")
                sample_rng, = jax.random.split(rng[0], 1)
                sample_classes = jax.random.randint(sample_rng, (4,), 0, 1000)
                # Use EMA params for sample generation (paper-faithful eval)
                single_ema_params = jax.tree_util.tree_map(lambda w: w[0], ema_params)
                latents_dev = sample_latents_jitted(single_ema_params, sample_classes, sample_rng)

                def _bg_log(z_dev, classes, target_step):
                    z = np.asarray(jax.device_get(z_dev), dtype=np.float32)
                    classes = jax.device_get(classes)
                    images = decode_latents(z)
                    images = (images * 255).astype(np.uint8)
                    safe_wandb_log({
                        "train/step": target_step,
                        "samples": [wandb.Image(img, caption=f"Class {cls}")
                                    for img, cls in zip(images, classes)],
                    }, step=target_step)

                threading.Thread(target=_bg_log,
                                 args=(latents_dev, sample_classes, global_step),
                                 daemon=True).start()

    # ── Checkpoint save (online params + EMA params) ──────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)
    unreplicated_params = jax_utils.unreplicate(state.params)
    unreplicated_ema    = jax_utils.unreplicate(ema_params)
    checkpoints.save_checkpoint(
        ckpt_dir=args.ckpt_dir,
        target=unreplicated_params,
        step=global_step,
    )
    checkpoints.save_checkpoint(
        ckpt_dir=os.path.join(args.ckpt_dir, "ema"),
        target=unreplicated_ema,
        step=global_step,
    )
    logger.shutdown()


if __name__ == "__main__":
    main()
