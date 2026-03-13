import os
import sys
import argparse
import glob
import pickle
import time
import threading
import queue
import logging
import zipfile
import functools

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


# Self-contained worker script run by subprocess.Popen.
# Deliberately has NO import of train.py and NO import of jax/flax so the
# child process is free of JAX/TPU and can load torch+diffusers cleanly.
_VAE_WORKER_SCRIPT = """\
import sys, struct, pickle, os, zipfile, tempfile, json
import numpy as np
import torch
from diffusers import AutoencoderKL

vae_path = sys.argv[1]
# Optional: HF model ID dùng để lấy config.json khi load từ local Flax zip
hf_config_id = sys.argv[2] if len(sys.argv) > 2 else "stabilityai/sd-vae-ft-ema"

def _load_from_flax_zip(zip_path, hf_config_id):
    \"\"\"Extract Flax msgpack từ zip của prepare_data_tpu.py, build thư mục HF chuẩn, load bằng from_flax=True.\"\"\"
    tmpdir = tempfile.mkdtemp(prefix="vae_flax_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        msgpack_name = next((n for n in zf.namelist() if n.endswith(".msgpack")), None)
        if msgpack_name is None:
            raise ValueError(f"Không tìm thấy file .msgpack trong {zip_path}")
        with zf.open(msgpack_name) as src, open(os.path.join(tmpdir, "flax_model.msgpack"), "wb") as dst:
            dst.write(src.read())
    # Lấy config.json: ưu tiên từ cùng thư mục, fallback về HF Hub
    config_src = os.path.join(os.path.dirname(zip_path), "config.json")
    if os.path.exists(config_src):
        import shutil
        shutil.copy(config_src, os.path.join(tmpdir, "config.json"))
    else:
        cfg = AutoencoderKL.load_config(hf_config_id)
        with open(os.path.join(tmpdir, "config.json"), "w") as cf:
            json.dump(cfg, cf)
    return AutoencoderKL.from_pretrained(tmpdir, from_flax=True, torch_dtype=torch.float32).eval()

try:
    if os.path.isdir(vae_path):
        flax_msgpack = os.path.join(vae_path, "flax_model.msgpack")
        zip_files = sorted(
            os.path.join(vae_path, f) for f in os.listdir(vae_path) if f.endswith(".zip")
        )
        if os.path.exists(flax_msgpack):
            # Standard HF Flax format (config.json + flax_model.msgpack)
            vae = AutoencoderKL.from_pretrained(vae_path, from_flax=True, torch_dtype=torch.float32).eval()
        elif zip_files:
            # Custom zip format từ prepare_data_tpu.py save_vae_params()
            vae = _load_from_flax_zip(zip_files[0], hf_config_id)
        else:
            # PyTorch local path (config.json + pytorch_model.bin / model.safetensors)
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32).eval()
    else:
        # HuggingFace repo ID
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32).eval()
    sys.stdout.buffer.write(b"READY\\n")
except Exception as exc:
    sys.stdout.buffer.write(("ERROR " + str(exc) + "\\n").encode())
sys.stdout.buffer.flush()

while True:
    hdr = sys.stdin.buffer.read(8)
    if len(hdr) < 8:
        break
    (n,) = struct.unpack("<Q", hdr)
    latents = pickle.loads(sys.stdin.buffer.read(n))
    try:
        with torch.no_grad():
            t = torch.from_numpy(latents) / 0.18215
            imgs = vae.decode(t).sample
            imgs = (imgs / 2.0 + 0.5).clamp(0, 1)
            imgs = imgs.permute(0, 2, 3, 1).numpy().astype(np.float32)
        payload = pickle.dumps(("ok", imgs))
    except Exception as exc:
        payload = pickle.dumps(("error", str(exc)))
    sys.stdout.buffer.write(struct.pack("<Q", len(payload)) + payload)
    sys.stdout.buffer.flush()
"""


class VAEDecodeSubprocess:
    """VAE decode via stdin/stdout pipe to an isolated child process.

    multiprocessing.spawn re-imports train.py in the child, which triggers
    `import jax` and reloads JAX/TPU — reproducing the exact protobuf
    SIGSEGV we are trying to avoid.  subprocess.Popen with an inline script
    starts a completely fresh Python that never touches JAX, so torch and
    diffusers load cleanly.

    The worker is started lazily (first decode call) and stays alive for the
    full training run so the VAE model is loaded only once.
    Both sd-vae-ft-ema and sd-vae-ft-mse share scaling_factor=0.18215; pass the
    same variant used when running prepare_data_tpu.py (default: sd-vae-ft-mse).

    vae_model: HF repo ID hoặc local path chứa model (PyTorch hoặc Flax).
    vae_hf_config: HF repo ID để lấy config.json khi vae_model là local Flax zip.
    """

    def __init__(self, vae_model: str, vae_hf_config: str = "stabilityai/sd-vae-ft-ema"):
        import subprocess
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _VAE_WORKER_SCRIPT, vae_model, vae_hf_config],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ready = self._proc.stdout.readline().strip()
        if ready != b"READY":
            err = self._proc.stderr.read(4096).decode(errors="replace")
            raise RuntimeError(
                f"VAE worker failed to start.\nstdout: {ready!r}\nstderr: {err}"
            )

    def decode(self, latents_nchw):
        """NCHW float32 → NHWC float32 [0, 1]."""
        import struct, pickle
        data = pickle.dumps(np.asarray(latents_nchw, dtype=np.float32))
        self._proc.stdin.write(struct.pack("<Q", len(data)) + data)
        self._proc.stdin.flush()
        hdr = self._proc.stdout.read(8)
        if len(hdr) < 8:
            raise RuntimeError("VAE worker process died unexpectedly.")
        (n,) = struct.unpack("<Q", hdr)
        tag, result = pickle.loads(self._proc.stdout.read(n))
        if tag != "ok":
            raise RuntimeError(f"VAE decode error in worker: {result}")
        return result

    def shutdown(self):
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()


def _build_flax_vae_decode_fn(vae_model_path, num_devices, hf_config_id="stabilityai/sd-vae-ft-ema"):
    """Load Flax VAE từ local path và build pmap'd decode function chạy trên TPU.

    Trả về (decode_fn, params_replicated) nếu thành công, hoặc (None, None) nếu
    path không phải Flax (fallback về CPU subprocess).

    decode_fn signature: (latents_nchw_sharded, params_repl) → images_nhwc_sharded
        latents_nchw_sharded: (num_devices, batch_per_device, 4, 32, 32)  bfloat16
        images_nhwc_sharded : (num_devices, batch_per_device, 256, 256, 3) float32 [0,1]
    """
    if not os.path.isdir(vae_model_path):
        return None, None  # HF repo ID: dùng subprocess như cũ

    dir_files = os.listdir(vae_model_path)
    log_stage(f"[VAE-TPU] Scanning {vae_model_path}: {dir_files}")

    standard_msgpack = os.path.join(vae_model_path, "flax_model.msgpack")
    # Tìm bất kỳ *.msgpack nào (kể cả vae_params_bf16.msgpack)
    all_msgpack = sorted(
        os.path.join(vae_model_path, f) for f in dir_files if f.endswith(".msgpack")
    )
    zip_files = sorted(
        os.path.join(vae_model_path, f) for f in dir_files if f.endswith(".zip")
    )
    has_config = "config.json" in dir_files

    if not all_msgpack and not zip_files:
        log_stage("[VAE-TPU] Không tìm thấy .msgpack hay .zip, fallback về CPU subprocess.")
        return None, None

    if not _FLAX_VAE_AVAILABLE:
        log_stage("[VAE-TPU] diffusers/FlaxAutoencoderKL không có sẵn, fallback về CPU subprocess.")
        return None, None

    import flax.serialization
    FlaxAutoencoderKL = _FlaxAutoencoderKL

    # Config của SD VAE (sd-vae-ft-ema và sd-vae-ft-mse có cùng architecture).
    # Hardcode để tránh mọi network call / lazy import gây SIGSEGV.
    _SD_VAE_CONFIG_HARDCODED = {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": [
            "DownEncoderBlock2D", "DownEncoderBlock2D",
            "DownEncoderBlock2D", "DownEncoderBlock2D",
        ],
        "up_block_types": [
            "UpDecoderBlock2D", "UpDecoderBlock2D",
            "UpDecoderBlock2D", "UpDecoderBlock2D",
        ],
        "block_out_channels": [128, 256, 512, 512],
        "layers_per_block": 2,
        "act_fn": "silu",
        "latent_channels": 4,
        "norm_num_groups": 32,
        "sample_size": 256,
    }

    def _get_vae_config():
        import json as _json
        # 1. config.json trong cùng thư mục với model file
        if has_config:
            cfg_path = os.path.join(vae_model_path, "config.json")
            log_stage(f"[VAE-TPU] Reading config.json từ {cfg_path}")
            with open(cfg_path) as f:
                return _json.load(f)
        # 2. hf_config_id là path trực tiếp đến file .json
        if os.path.isfile(hf_config_id):
            log_stage(f"[VAE-TPU] Reading config.json từ file: {hf_config_id}")
            with open(hf_config_id) as f:
                return _json.load(f)
        # 3. hf_config_id là thư mục chứa config.json
        local_cfg = os.path.join(hf_config_id, "config.json")
        if os.path.isfile(local_cfg):
            log_stage(f"[VAE-TPU] Reading config.json từ dir: {local_cfg}")
            with open(local_cfg) as f:
                return _json.load(f)
        # 4. Fallback: dùng hardcoded config — không network, không lazy import, không SIGSEGV
        log_stage("[VAE-TPU] Không tìm thấy config.json local → dùng hardcoded SD-VAE config.")
        return _SD_VAE_CONFIG_HARDCODED

    if all_msgpack:
        # Standard HF format (config.json + flax_model.msgpack) → from_pretrained
        if has_config and os.path.exists(standard_msgpack):
            log_stage(f"[VAE-TPU] Loading từ {vae_model_path} (HF format)…")
            vae, vae_params = FlaxAutoencoderKL.from_pretrained(vae_model_path)
        else:
            # Tên msgpack khác, ví dụ vae_params_bf16.msgpack
            chosen = all_msgpack[0]
            log_stage(f"[VAE-TPU] Loading params từ {os.path.basename(chosen)}…")
            with open(chosen, "rb") as f:
                params_bytes = f.read()
            vae_params = flax.serialization.from_bytes(None, params_bytes)
            vae_params = jax.tree_util.tree_map(jnp.array, vae_params)
            vae = FlaxAutoencoderKL.from_config(_get_vae_config())
    else:
        chosen_zip = zip_files[0]
        log_stage(f"[VAE-TPU] Loading từ zip: {os.path.basename(chosen_zip)}…")
        with zipfile.ZipFile(chosen_zip, "r") as zf:
            msgpack_name = next((n for n in zf.namelist() if n.endswith(".msgpack")), None)
            if msgpack_name is None:
                log_stage("[VAE-TPU] Zip không chứa .msgpack, fallback về CPU subprocess.")
                return None, None
            params_bytes = zf.read(msgpack_name)
        vae_params = flax.serialization.from_bytes(None, params_bytes)
        vae_params = jax.tree_util.tree_map(jnp.array, vae_params)
        vae = FlaxAutoencoderKL.from_config(_get_vae_config())

    vae_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), vae_params)

    @jax.pmap
    def _decode_pmap(latents_nchw, params):
        # latents_nchw: (batch_per_device, 4, 32, 32) bfloat16, đã scale ×0.18215
        # Unscale rồi chuyển NCHW→NHWC vì Diffusers Flax VAE dùng NHWC
        latents_nhwc = jnp.transpose(latents_nchw / jnp.bfloat16(0.18215), (0, 2, 3, 1))
        images = vae.apply({"params": params}, latents_nhwc, method=vae.decode).sample
        # Diffusers Flax VAE decode returns NCHW; transpose to NHWC for callers.
        images = jnp.transpose(images, (0, 2, 3, 1))
        return ((images / 2.0 + 0.5).clip(0, 1)).astype(jnp.float32)

    from flax.jax_utils import replicate
    vae_params_repl = replicate(vae_params)
    log_stage(f"[VAE-TPU] Flax VAE decode ready trên {num_devices} TPU device(s).")
    return _decode_pmap, vae_params_repl


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


# diffusers phải được import TRƯỚC jax để tránh xung đột C++ protobuf descriptor.
# Nếu diffusers được import sau khi JAX/TPU khởi tạo, dynamic linker dlopen một .so
# mới đăng ký protobuf descriptor trùng với descriptor của XLA → SIGSEGV.
try:
    from diffusers.models import FlaxAutoencoderKL as _FlaxAutoencoderKL
    _FLAX_VAE_AVAILABLE = True
except Exception:
    _FlaxAutoencoderKL = None
    _FLAX_VAE_AVAILABLE = False

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state, checkpoints
from flax import jax_utils
import numpy as np
try:
    import grain.python as grain
except ImportError:
    grain = None
from src.model import SelfFlowDiT
from src.sampling import denoise_loop


def create_train_state(rng, config, learning_rate, grad_clip=1.0):
    """Initializes the model, optimizer, and initial EMA params.

    Returns (state, ema_params) where ema_params is a copy of the initial
    online params.  Caller should replicate both via jax_utils.replicate.
    """
    model = SelfFlowDiT(
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
        per_token=False,
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
        deterministic=False,
    )

    # AdamW with gradient clipping (paper specifies max_norm=1; paper-faithful)
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate, weight_decay=0),
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


def parse_layersync_pairs_arg(pairs_arg):
    if pairs_arg is None:
        return None

    pairs = []
    for raw_entry in pairs_arg.split(","):
        entry = raw_entry.strip()
        if not entry:
            raise ValueError("--layersync-pairs contains an empty entry")
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Invalid --layersync-pairs entry '{entry}'. Expected format weak:strong"
            )
        try:
            weak = int(parts[0])
            strong = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid --layersync-pairs entry '{entry}'. Layer indices must be integers"
            ) from exc
        pairs.append((weak, strong))

    if not pairs:
        raise ValueError("--layersync-pairs cannot be empty")
    return tuple(pairs)


def resolve_layersync_config(args, depth):
    pairs = parse_layersync_pairs_arg(args.layersync_pairs)
    if pairs is None:
        pairs = ((args.layersync_weak_layer, args.layersync_strong_layer),)

    seen_pairs = set()
    duplicate_pairs = []
    for pair in pairs:
        if pair in seen_pairs and pair not in duplicate_pairs:
            duplicate_pairs.append(pair)
        seen_pairs.add(pair)
    if duplicate_pairs:
        duplicates_str = ", ".join(f"{weak}:{strong}" for weak, strong in duplicate_pairs)
        raise ValueError(f"Duplicate LayerSync pairs are not allowed: {duplicates_str}")

    if args.layersync_lambda > 0.0:
        for weak, strong in pairs:
            if not (1 <= weak <= depth):
                raise ValueError(f"LayerSync weak layer must be in [1, {depth}], got {weak}")
            if not (1 <= strong <= depth):
                raise ValueError(f"LayerSync strong layer must be in [1, {depth}], got {strong}")
            if weak >= strong:
                raise ValueError(
                    f"LayerSync pair {weak}:{strong} is invalid; expected weak < strong"
                )

    capture_layers = tuple(dict.fromkeys(
        layer for weak, strong in pairs for layer in (weak, strong)
    ))
    return pairs, capture_layers


def compute_layersync_regularizer(raw_features, capture_layers, layersync_pairs, layersync_mode):
    if not isinstance(raw_features, (tuple, list)):
        raw_features = (raw_features,)

    feature_map = {layer: feat for layer, feat in zip(capture_layers, raw_features)}
    eps = jnp.float32(1e-8)
    pair_losses = []
    pair_alignment = []

    for weak_layer, strong_layer in layersync_pairs:
        z_weak = feature_map[weak_layer]
        z_strong = feature_map[strong_layer]
        if layersync_mode == "stopgrad":
            z_strong = jax.lax.stop_gradient(z_strong)
        elif layersync_mode != "no_stopgrad":
            raise ValueError(f"Unsupported layersync mode: {layersync_mode}")

        z_weak_norm = z_weak / (jnp.linalg.norm(z_weak, axis=-1, keepdims=True) + eps)
        z_strong_norm = z_strong / (jnp.linalg.norm(z_strong, axis=-1, keepdims=True) + eps)
        cos_tokens = jnp.sum(z_weak_norm * z_strong_norm, axis=-1)
        pair_alignment.append(jnp.mean(cos_tokens))
        pair_losses.append(jnp.mean(jnp.square(cos_tokens)))

    mean_alignment = pair_alignment[0] if len(pair_alignment) == 1 else jnp.mean(jnp.stack(pair_alignment))
    loss = pair_losses[0] if len(pair_losses) == 1 else jnp.mean(jnp.stack(pair_losses))
    return loss, mean_alignment


# ── Vanilla SiT training/eval ─────────────────────────────────────────────────

def train_step(
    state,
    ema_params,
    batch,
    rng,
    ema_decay,
    layersync_lambda,
    layersync_pairs,
    layersync_capture_layers,
    layersync_mode,
):
    """Vanilla SiT training step (global timestep; velocity prediction).

    Repo time convention: tau=0 → pure noise, tau=1 → clean data.
      x_tau   = (1 - tau) * x1 + tau * x0
      target  = x0 - x1
      loss    = E[||v_theta(x_tau, tau) - target||^2]
    """
    x0, y = batch  # x0: [local_B, N, D], y: [local_B]
    local_batch = x0.shape[0]

    rng, tau_rng, noise_rng, drop_rng = jax.random.split(rng, 4)

    tau = jax.random.uniform(tau_rng, shape=(local_batch,), minval=0.0, maxval=1.0)  # [B]
    x1 = jax.random.normal(noise_rng, x0.shape)  # [B, N, D]

    x_tau = (1.0 - tau[:, None, None]) * x1 + tau[:, None, None] * x0
    target = x0 - x1

    def loss_fn(params):
        if layersync_lambda > 0.0:
            pred, zs = state.apply_fn(
                {"params": params},
                x_tau,
                timesteps=tau,
                vector=y,
                deterministic=False,
                rngs={"dropout": drop_rng},
                return_raw_features=layersync_capture_layers,
            )
            loss_layersync, mean_cos = compute_layersync_regularizer(
                zs, layersync_capture_layers, layersync_pairs, layersync_mode
            )
        else:
            pred = state.apply_fn(
                {"params": params},
                x_tau,
                timesteps=tau,
                vector=y,
                deterministic=False,
                rngs={"dropout": drop_rng},
            )
            mean_cos = jnp.array(0.0, dtype=jnp.float32)
            loss_layersync = jnp.array(0.0, dtype=jnp.float32)

        loss_gen = jnp.mean((pred - target) ** 2)
        loss = loss_gen + layersync_lambda * loss_layersync
        v_abs_mean = jnp.mean(jnp.abs(target))
        v_pred_abs_mean = jnp.mean(jnp.abs(pred))
        return loss, (v_abs_mean, v_pred_abs_mean, loss_gen, loss_layersync, mean_cos)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (v_abs, v_pred, loss_gen, loss_layersync, layersync_cos)), grads = grad_fn(state.params)

    loss = jax.lax.pmean(loss, axis_name="batch")
    v_abs = jax.lax.pmean(v_abs, axis_name="batch")
    v_pred = jax.lax.pmean(v_pred, axis_name="batch")
    loss_gen = jax.lax.pmean(loss_gen, axis_name="batch")
    loss_layersync = jax.lax.pmean(loss_layersync, axis_name="batch")
    layersync_cos = jax.lax.pmean(layersync_cos, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
    param_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(state.params)))

    state = state.apply_gradients(grads=grads)
    ema_params = ema_update(ema_params, state.params, ema_decay)

    metrics = {
        "train/loss": loss,
        "train/loss_total": loss,
        "train/loss_gen": loss_gen,
        "train/loss_layersync": loss_layersync,
        "train/layersync_cosine": layersync_cos,
        "train/ema_decay": ema_decay,
        "train/grad_norm": grad_norm,
        "train/param_norm": param_norm,
        "train/v_abs_mean": v_abs,
        "train/v_pred_abs_mean": v_pred,
    }
    return state, ema_params, metrics, rng


def eval_step(
    state,
    ema_params,
    batch,
    rng,
    layersync_lambda,
    layersync_pairs,
    layersync_capture_layers,
    layersync_mode,
):
    """Vanilla SiT validation step (mirrors train_step; no grads; no EMA teacher)."""
    x0, y = batch
    local_batch = x0.shape[0]

    rng, tau_rng, noise_rng = jax.random.split(rng, 3)

    tau = jax.random.uniform(tau_rng, shape=(local_batch,), minval=0.0, maxval=1.0)
    x1 = jax.random.normal(noise_rng, x0.shape)

    x_tau = (1.0 - tau[:, None, None]) * x1 + tau[:, None, None] * x0
    target = x0 - x1

    if layersync_lambda > 0.0:
        pred, zs = state.apply_fn(
            {"params": state.params},
            x_tau,
            timesteps=tau,
            vector=y,
            deterministic=True,
            return_raw_features=layersync_capture_layers,
        )
        loss_layersync, mean_cos = compute_layersync_regularizer(
            zs, layersync_capture_layers, layersync_pairs, layersync_mode
        )
    else:
        pred = state.apply_fn(
            {"params": state.params},
            x_tau,
            timesteps=tau,
            vector=y,
            deterministic=True,
        )
        mean_cos = jnp.array(0.0, dtype=jnp.float32)
        loss_layersync = jnp.array(0.0, dtype=jnp.float32)

    loss_gen = jnp.mean((pred - target) ** 2)
    loss = loss_gen + layersync_lambda * loss_layersync
    v_abs_mean = jnp.mean(jnp.abs(target))
    v_pred_abs_mean = jnp.mean(jnp.abs(pred))

    loss = jax.lax.pmean(loss, axis_name="batch")
    v_abs_mean = jax.lax.pmean(v_abs_mean, axis_name="batch")
    v_pred_abs_mean = jax.lax.pmean(v_pred_abs_mean, axis_name="batch")
    loss_gen = jax.lax.pmean(loss_gen, axis_name="batch")
    loss_layersync = jax.lax.pmean(loss_layersync, axis_name="batch")
    mean_cos = jax.lax.pmean(mean_cos, axis_name="batch")

    metrics = {
        "val/loss": loss,
        "val/loss_total": loss,
        "val/loss_gen": loss_gen,
        "val/loss_layersync": loss_layersync,
        "val/layersync_cosine": mean_cos,
        "val/v_abs_mean": v_abs_mean,
        "val/v_pred_abs_mean": v_pred_abs_mean,
    }
    return metrics, rng


def get_arrayrecord_dataloader(data_pattern, batch_size, is_training=True, seed=42):
    """
    Creates an optimized Grain dataloader reading from ArrayRecord files.
    """
    if grain is None:
        raise ImportError(
            "grain is not installed. Please `pip install grain-balsa` to use ArrayRecord datasets."
        )
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
    model = SelfFlowDiT(
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
        per_token=False,
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
        # Patchify matching the training dataloader: each token in (p1 p2 c) order.
        # Dataloader does: reshape(c,h//p,p,w//p,p) → transpose(1,3,2,4,0) → reshape(N,p*p*c)
        # which is exactly rearrange "b c (h p1) (w p2) -> b (h w) (p1 p2 c)".
        x = rearrange(
            noise,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        )
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
            reverse=False,   # training: tau=0→noise, tau=1→data → integrate forward
        )

        if use_cfg:
            samples = samples[batch_size:]
        # Unpatchify: inverse of (p1 p2 c) train patchify → NCHW latent.
        # Matches unpatchify_patchified_latents() which uses the same formula.
        samples = rearrange(
            samples,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=token_h,
            w=token_w,
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
    parser = argparse.ArgumentParser(description="Train vanilla SiT DiT (JAX)")
    # ── Core training args ────────────────────────────────────────────────────
    parser.add_argument("--batch-size", type=int, default=256, help="Global batch size (divided by device count)")
    parser.add_argument("--model-size", type=str, default="XL", choices=["S", "B", "L", "XL"], help="DiT backbone size: S, B, L, XL")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    parser.add_argument("--data-path", type=str, required=True, help="Path/glob to training ArrayRecord files")
    parser.add_argument("--val-data-path", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="sit-vanilla-jax")
    parser.add_argument("--no-wandb", action="store_true")
    # ── Vanilla SiT args ──────────────────────────────────────────────────────
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="EMA decay for tracking a smoothed copy of online params (default: 0.9999). "
             "EMA is used only for evaluation/checkpoint convenience, not as a teacher loss.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clip max_norm (paper: 1.0)")
    parser.add_argument(
        "--layersync-lambda",
        type=float,
        default=0.0,
        help="Weight for LayerSync regularizer. 0.0 disables LayerSync and recovers pure velocity loss.",
    )
    parser.add_argument(
        "--layersync-weak-layer",
        type=int,
        default=4,
        help="1-based index of the weaker (shallower) block used for LayerSync. "
             "Used when --layersync-pairs is not set.",
    )
    parser.add_argument(
        "--layersync-strong-layer",
        type=int,
        default=10,
        help="1-based index of the stronger (deeper) block used for LayerSync. "
             "Used when --layersync-pairs is not set.",
    )
    parser.add_argument(
        "--layersync-pairs",
        type=str,
        default=None,
        help="Comma-separated ordered LayerSync pairs in weak:strong format, e.g. 2:8,4:10. "
             "Overrides --layersync-weak-layer/--layersync-strong-layer when provided.",
    )
    parser.add_argument(
        "--layersync-mode",
        type=str,
        default="stopgrad",
        choices=["stopgrad", "no_stopgrad"],
        help="Layer regularizer mode. 'stopgrad' detaches the strong branch before applying "
             "the cosine-squared penalty; 'no_stopgrad' is an ablation without detaching it.",
    )
    # ── VAE model (must match the variant used in prepare_data_tpu.py) ──────
    parser.add_argument(
        "--vae-model",
        type=str,
        default="/kaggle/input/models/damtrunghieu/sdvae-ema/flax/default/1",
        help=(
            "Local path hoặc HF repo ID của VAE dùng để decode trong preview/FID. "
            "Chấp nhận 3 dạng: "
            "(1) local dir có flax_model.msgpack — load Flax trực tiếp; "
            "(2) local dir có *.zip — zip từ prepare_data_tpu.py save_vae_params(); "
            "(3) HF repo ID hoặc local PyTorch dir — load bình thường. "
            "Scaling factor cố định 0.18215 (sd-vae-ft-mse/ema)."
        ),
    )
    parser.add_argument(
        "--vae-hf-config",
        type=str,
        default="/kaggle/working/huggingface_cache/hub/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a/config.json",
        help=(
            "Path đến config.json của VAE. Chấp nhận 3 dạng: "
            "(1) path trực tiếp đến file config.json — đọc local, an toàn nhất; "
            "(2) path đến thư mục chứa config.json; "
            "(3) HF repo ID — tải từ HF Hub (rủi ro SIGSEGV do lazy import). "
            "Chỉ dùng khi --vae-model dir không có sẵn config.json."
        ),
    )
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
    parser.add_argument("--vae-decode-batch-size", type=int, default=8,
                        help="Micro-batch size for VAE decode during previews/FID/preflight. "
                             "Lower this on 16GB TPU if decode OOMs.")
    # ── Preflight / safety args ───────────────────────────────────────────────
    parser.add_argument("--preflight-checks", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-sample-count", type=int, default=4)
    parser.add_argument("--preflight-fid-samples", type=int, default=16)
    parser.add_argument("--preflight-fid-memory-probe", action="store_true",
                        help="Run one discarded train step plus one real/fake FID batch at startup "
                             "to catch TPU OOMs before long training.")
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
    if args.vae_decode_batch_size <= 0:
        raise ValueError("--vae-decode-batch-size must be greater than 0")

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

    # ── Model config ─────────────────────────────────────────────────────────
    config = build_model_config(args.model_size)

    log_stage(
        f"Model=DiT-{args.model_size.upper()} hidden={config['hidden_size']} "
        f"depth={config['depth']} heads={config['num_heads']}"
    )
    log_stage(
        f"Vanilla SiT: ema_decay={args.ema_decay} grad_clip={args.grad_clip}"
    )

    # ── LayerSync config validation ───────────────────────────────────────────
    if args.layersync_lambda < 0.0:
        raise ValueError("--layersync-lambda must be non-negative")
    depth = config["depth"]
    layersync_pairs, layersync_capture_layers = resolve_layersync_config(args, depth)
    if args.layersync_lambda > 0.0:
        pairs_str = ",".join(f"{weak}:{strong}" for weak, strong in layersync_pairs)
        log_stage(
            f"LayerSync ENABLED: lambda={args.layersync_lambda} mode={args.layersync_mode} "
            f"pairs={pairs_str}"
        )
    else:
        log_stage("LayerSync DISABLED: lambda=0.0")

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
    ema_decay_rep = jax_utils.replicate(jnp.float32(args.ema_decay))

    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2

    # ── Build pmapped training/eval steps ────────────────────────────────────
    pmapped_train_step = jax.pmap(
        functools.partial(
            train_step,
            layersync_lambda=args.layersync_lambda,
            layersync_pairs=layersync_pairs,
            layersync_capture_layers=layersync_capture_layers,
            layersync_mode=args.layersync_mode,
        ),
        axis_name="batch",
    )
    pmapped_eval_step = jax.pmap(
        functools.partial(
            eval_step,
            layersync_lambda=args.layersync_lambda,
            layersync_pairs=layersync_pairs,
            layersync_capture_layers=layersync_capture_layers,
            layersync_mode=args.layersync_mode,
        ),
        axis_name="batch",
    )

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

    # ── VAE decode: TPU (Flax pmap) nếu local Flax path, còn lại CPU subprocess ─
    # CPU subprocess: torch/diffusers load trong child process không có JAX/TPU,
    # tránh protobuf SIGSEGV khi dùng multiprocessing.spawn.
    # TPU Flax path: dùng khi --vae-model là local dir có flax_model.msgpack / *.zip.
    _flax_decode_cache = [None]  # (decode_fn, params_repl) | "subprocess" | None

    def _ensure_vae_backend():
        if _flax_decode_cache[0] is None:
            decode_fn, params_repl = _build_flax_vae_decode_fn(args.vae_model, num_devices, args.vae_hf_config)
            if decode_fn is not None:
                _flax_decode_cache[0] = (decode_fn, params_repl)
            else:
                log_stage(f"Spawning VAE decode worker (CPU subprocess): {args.vae_model!r} …")
                _flax_decode_cache[0] = VAEDecodeSubprocess(args.vae_model, args.vae_hf_config)
                log_stage("VAE decode worker ready.")
        return _flax_decode_cache[0]

    def _decode_tpu(latents_nchw, decode_fn, params_repl):
        """NCHW float32 → NHWC float32 [0, 1] trên TPU, tự pad cho chia hết num_devices."""
        latents_nchw = np.asarray(latents_nchw, dtype=np.float32)
        n = latents_nchw.shape[0]
        pad = (num_devices - n % num_devices) % num_devices
        if pad > 0:
            latents_nchw = np.concatenate(
                [latents_nchw, np.zeros((pad, 4, 32, 32), dtype=np.float32)], axis=0
            )
        batch_per_device = latents_nchw.shape[0] // num_devices
        latents_sharded = jnp.array(
            latents_nchw.reshape(num_devices, batch_per_device, 4, 32, 32),
            dtype=jnp.bfloat16,
        )
        images = decode_fn(latents_sharded, params_repl)  # (devices, bpd, H, W, 3)
        return jax.device_get(images).reshape(-1, 256, 256, 3).astype(np.float32)[:n]

    def decode_latents(latents_nchw):
        """NCHW float32 → NHWC float32 [0, 1]. TPU (Flax) nếu có, fallback CPU subprocess."""
        backend = _ensure_vae_backend()
        if isinstance(backend, VAEDecodeSubprocess):
            return backend.decode(latents_nchw)
        decode_fn, params_repl = backend
        return _decode_tpu(latents_nchw, decode_fn, params_repl)

    def decode_latents_batched(latents_nchw, decode_batch_size=None):
        """Decode latents theo chunk. Khi dùng TPU backend, chunk_size nên lớn hơn (≥32×num_devices)."""
        latents_nchw = np.asarray(latents_nchw, dtype=np.float32)
        if latents_nchw.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        chunk_size = int(decode_batch_size or args.vae_decode_batch_size)
        chunks = []
        for start in range(0, latents_nchw.shape[0], chunk_size):
            chunks.append(decode_latents(latents_nchw[start:start + chunk_size]))
        return np.concatenate(chunks, axis=0)

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

    def imgs_to_acts(imgs_nhwc, inception_fn):
        imgs = list(imgs_nhwc)
        acts_all = []
        for start in range(0, len(imgs), num_devices):
            chunk = imgs[start:start + num_devices]
            while len(chunk) < num_devices:
                chunk.append(chunk[-1])
            imgs_299 = np.stack([
                np.array(jax.image.resize(
                    img.astype(np.float32) * 2.0 - 1.0,
                    (299, 299, img.shape[-1]), method="bilinear"
                )) for img in chunk
            ])
            acts = np.array(inception_fn(imgs_299[:, None])).reshape(num_devices, 2048)
            acts_all.append(acts)
        return np.concatenate(acts_all, axis=0)

    def block_pytree(tree):
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            tree,
        )

    def run_fid_memory_probe(val_data_iter, cached_train_batch):
        """Mimic FID peak allocations after one train step to catch TPU OOMs early."""
        if val_data_iter is None:
            raise RuntimeError("FID memory probe requires --val-data-path.")
        if cached_train_batch is None:
            if data_iterator is None:
                raise RuntimeError("FID memory probe requires real training data.")
            cached_train_batch = next(data_iterator)

        log_stage("[FID probe] running one discarded train step to match training-time memory...")
        probe_x = jnp.array(cached_train_batch[0]).reshape(num_devices, local_batch_size, n_patches, patch_dim)
        probe_y = jnp.array(cached_train_batch[1]).reshape(num_devices, local_batch_size)
        _, _, probe_metrics, _ = pmapped_train_step(
            state, ema_params, (probe_x, probe_y), rng, ema_decay_rep
        )
        block_pytree(probe_metrics)

        inception_fn = get_inception()

        log_stage(
            f"[FID probe] decoding one real validation batch of {args.batch_size} latents "
            f"with VAE micro-batch {args.vae_decode_batch_size}..."
        )
        probe_val_batch, val_data_iter = next_validation_batch(
            val_data_iter, data_pattern=args.val_data_path, batch_size=args.batch_size
        )
        real_latents_nchw = unpatchify_patchified_latents(probe_val_batch[0])
        real_images = decode_latents_batched(real_latents_nchw, args.vae_decode_batch_size)
        block_pytree(imgs_to_acts(real_images, inception_fn))

        fake_bs = min(args.fid_batch_size, args.num_fid_samples)
        log_stage(
            f"[FID probe] generating one fake batch of {fake_bs} latents "
            f"({args.fid_num_steps} steps, cfg={args.fid_cfg_scale})..."
        )
        single_ema_params = jax.tree_util.tree_map(lambda w: w[0], ema_params)
        probe_rng = jax.random.fold_in(rng[0], 0xF1D)
        probe_classes = jax.random.randint(probe_rng, (fake_bs,), 0, 1000)
        fake_latents = np.asarray(
            jax.device_get(fid_sample_latents_jitted(single_ema_params, probe_classes, probe_rng)),
            dtype=np.float32,
        )
        fake_images = decode_latents_batched(fake_latents, args.vae_decode_batch_size)
        block_pytree(imgs_to_acts(fake_images, inception_fn))

        log_stage(
            "[FID probe] success: discarded train step + real/fake FID batches completed without OOM."
        )
        return val_data_iter, cached_train_batch

    def compute_fid(step, val_data_iter):
        """Synchronous FID using EMA params and configurable num_steps.

        TPU deviation: default 50 steps and 4000 samples for fast monitoring.
        This FID is NOT paper-comparable at default settings.
        For paper-comparable FID, run with --fid-num-steps 250 --num-fid-samples 50000.
        """
        from src.fid_utils import fid_from_stats

        inception_fn = get_inception()

        # Build real image stats once; reuse across FID calls
        if _fid_real_acts[0] is None:
            log_stage(
                f"[FID] decoding {args.num_fid_samples} real images "
                f"(VAE micro-batch {args.vae_decode_batch_size})…"
            )
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
                for img in decode_latents_batched(latents_nchw, args.vae_decode_batch_size):
                    real_imgs.append(img)
                    if len(real_imgs) >= args.num_fid_samples:
                        break
            log_stage(f"[FID] {len(real_imgs)} real images decoded.")
            real_acts = imgs_to_acts(real_imgs[:args.num_fid_samples], inception_fn)
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
            for img in decode_latents_batched(latents, args.vae_decode_batch_size):
                gen_imgs.append(img)

        gen_acts = imgs_to_acts(gen_imgs[:args.num_fid_samples], inception_fn)
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
            decode_latents=decode_latents_batched,
            inception_fn=inception_fn_for_preflight,
            real_latents_patchified=preflight_real_latents,
            preflight_sample_count=args.preflight_sample_count,
            preflight_fid_samples=args.preflight_fid_samples,
        )

        if args.preflight_fid_memory_probe:
            val_iterator, prefetched_train_batch = run_fid_memory_probe(val_iterator, prefetched_train_batch)

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

            # Vanilla SiT training step (returns updated EMA params)
            state, ema_params, metrics, rng = pmapped_train_step(
                state, ema_params, (batch_x, batch_y), rng, ema_decay_rep
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

            # Validation: vanilla SiT objective (no grads)
            if val_iterator is not None and args.eval_freq > 0 and global_step % args.eval_freq == 0:
                print(f"Step {global_step}: Evaluating validation loss over {args.eval_batches} batch(es)...")
                metric_sums = {}
                for _ in range(args.eval_batches):
                    val_batch, val_iterator = next_validation_batch(
                        val_iterator, data_pattern=args.val_data_path, batch_size=args.batch_size,
                    )
                    val_x = jnp.array(val_batch[0]).reshape(num_devices, local_batch_size, n_patches, patch_dim)
                    val_y = jnp.array(val_batch[1]).reshape(num_devices, local_batch_size)
                    val_metrics, rng = pmapped_eval_step(state, ema_params, (val_x, val_y), rng)
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
                    images = decode_latents_batched(z, args.vae_decode_batch_size)
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
    if _flax_decode_cache[0] is not None and isinstance(_flax_decode_cache[0], VAEDecodeSubprocess):
        _flax_decode_cache[0].shutdown()
    logger.shutdown()


if __name__ == "__main__":
    main()
