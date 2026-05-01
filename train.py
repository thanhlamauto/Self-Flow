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
import json
from typing import Callable

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

def build_model_config(model_size, class_dropout_prob=0.1):
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
        num_classes=1000,
        learn_sigma=True,
        compatibility_mode=True,
        class_dropout_prob=float(class_dropout_prob),
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
from flax import struct
from flax import traverse_util
import numpy as np
try:
    import grain.python as grain
except ImportError:
    grain = None
from src.model import SelfFlowDiT
from src.depth_shortcut import (
    DepthShortcutPredictor,
    apply_predictor_config_overrides,
    l2_normalize_tokens,
    predictor_config_from_name,
    predictor_size_bucket,
    predictor_variant_names,
)
from src.sampling import denoise_loop
from src.metrics import (
    ReservoirSampler,
    apply_inception_to_decoded_sharded,
    extract_inception_features_host_images,
    init_gaussian_sums,
    gaussian_batch_sums_pmap,
    gaussian_spatial_batch_sums_pmap,
    gaussian_sums_add,
    finalize_gaussian_sums,
    inception_score_from_probs,
    make_eval_chunk_rngs,
    pearson_corrcoef_rows,
    precision_recall_knn,
    trim_sharded_batch_to_host,
)
from src.inception_is_subprocess import InceptionISSubprocess


class ShortcutTrainState(train_state.TrainState):
    predictor_apply_fn: Callable = struct.field(pytree_node=False)


PREDICTOR_PARAM_TARGET_RANGES = {
    # Broad implementation checks; counts include condition MLP and magnitude head.
    "tiny": (1_000_000, 4_000_000),
    "small": (3_000_000, 8_000_000),
    "base": (7_000_000, 19_000_000),
    "large": (14_000_000, 30_000_000),
}


def count_tree_params(tree) -> int:
    return int(sum(np.size(x) for x in jax.tree_util.tree_leaves(tree)))


def parse_int_cycle(value: str | None) -> tuple[int, ...]:
    if value is None or str(value).strip() == "":
        return ()
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def make_adamw_decay_mask(params):
    """Apply AdamW decay to matrix/conv kernels, excluding bias/norm/embeddings."""
    flat = traverse_util.flatten_dict(params)
    mask = {}
    for path, _ in flat.items():
        path_str = "/".join(str(part).lower() for part in path)
        leaf = str(path[-1]).lower()
        excluded = (
            leaf in {"bias", "scale"}
            or "norm" in path_str
            or "embed" in path_str
            or "embedding" in path_str
        )
        mask[path] = not excluded
    return traverse_util.unflatten_dict(mask)


def create_train_state(
    rng,
    config,
    learning_rate,
    grad_clip=1.0,
    weight_decay=0.01,
    predictor_variant="tiny",
    predictor_lr=1e-4,
    predictor_weight_decay=0.1,
    predictor_grad_clip=1.0,
    shortcut_training_mode="direction",
    shortcut_mag_abs_center=5.5,
    shortcut_mag_abs_scale=1.5,
    predictor_config_overrides=None,
    predictor_use_class_input=False,
    predictor_class_fusion="add",
):
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
        class_dropout_prob=config["class_dropout_prob"],
        per_token=False,
    )

    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    predictor_cfg = predictor_config_from_name(predictor_variant, config["hidden_size"])
    predictor_cfg = apply_predictor_config_overrides(
        predictor_cfg,
        **(predictor_config_overrides or {}),
    )
    predictor = DepthShortcutPredictor(
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_tokens=n_patches,
        gamma_out_init=0.001 if shortcut_training_mode == "direction-magnitude" else 0.05,
        mag_abs_center=shortcut_mag_abs_center,
        mag_abs_scale=shortcut_mag_abs_scale,
        num_classes=config["num_classes"],
        class_cond_input=bool(predictor_use_class_input),
        class_cond_fusion=predictor_class_fusion,
        **predictor_cfg,
    )

    dummy_x = jnp.ones((1, n_patches, patch_dim))
    dummy_t = jnp.ones((1,))
    dummy_vec = jnp.ones((1,), dtype=jnp.int32)

    rng, backbone_rng, predictor_rng, drop_rng = jax.random.split(rng, 4)
    variables = model.init(
        {'params': backbone_rng, 'dropout': drop_rng},
        x=dummy_x,
        timesteps=dummy_t,
        vector=dummy_vec,
        deterministic=False,
    )
    dummy_u = jnp.ones((1, n_patches, config["hidden_size"]), dtype=jnp.float32)
    dummy_t_embed = jnp.ones((1, config["hidden_size"]), dtype=jnp.float32)
    dummy_m = jnp.zeros((1, n_patches, 1), dtype=jnp.float32)
    predictor_variables = predictor.init(
        {"params": predictor_rng},
        dummy_u,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        dummy_t_embed,
        dummy_m,
        class_labels=dummy_vec,
    )

    params = {
        "backbone": variables["params"],
        "predictor": predictor_variables["params"],
    }

    tx = optax.multi_transform(
        {
            "backbone": optax.chain(
                optax.clip_by_global_norm(grad_clip),
                optax.adamw(
                    learning_rate,
                    weight_decay=weight_decay,
                    mask=make_adamw_decay_mask(params),
                ),
            ),
            "predictor": optax.chain(
                optax.clip_by_global_norm(predictor_grad_clip),
                optax.adamw(
                    predictor_lr,
                    b1=0.9,
                    b2=0.999,
                    weight_decay=predictor_weight_decay,
                ),
            ),
        },
        {
            "backbone": jax.tree_util.tree_map(lambda _: "backbone", params["backbone"]),
            "predictor": jax.tree_util.tree_map(lambda _: "predictor", params["predictor"]),
        },
    )

    state = ShortcutTrainState.create(
        apply_fn=model.apply,
        predictor_apply_fn=predictor.apply,
        params=params,
        tx=tx,
    )
    # Backbone EMA is used for evaluation; predictor EMA is the bootstrap teacher.
    ema_params = jax.tree_util.tree_map(lambda x: x, state.params["backbone"])
    predictor_ema_params = jax.tree_util.tree_map(lambda x: x, state.params["predictor"])
    l2_ema = jnp.zeros((config["depth"] + 1, 5, n_patches), dtype=jnp.float32)
    return state, ema_params, predictor_ema_params, l2_ema


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


# ── Vanilla SiT training/eval ─────────────────────────────────────────────────

def sample_pair_uniform_gap(rng, depth):
    gap_rng, source_rng = jax.random.split(rng)
    gap = jax.random.randint(gap_rng, (), 1, depth + 1, dtype=jnp.int32)
    source = jax.random.randint(source_rng, (), 0, depth - gap + 1, dtype=jnp.int32)
    return source, source + gap


def build_direction_and_norm_map(hidden, eps=1e-6):
    norms = jnp.linalg.norm(hidden.astype(jnp.float32), axis=-1, keepdims=True)
    directions = hidden.astype(jnp.float32) / (norms + eps)
    log_magnitudes = jnp.log(norms + eps)
    return directions, log_magnitudes, log_magnitudes


def build_predictor_source(hidden, normalize_input=True, eps=1e-6):
    directions, log_magnitudes, _ = build_direction_and_norm_map(hidden, eps=eps)
    predictor_input = directions if normalize_input else hidden.astype(jnp.float32)
    return predictor_input, log_magnitudes


def sample_activation_rms(x, eps=1e-6):
    """Per-sample RMS over token and channel axes for activation-scale normalization."""
    reduce_axes = tuple(range(1, x.ndim))
    return jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=reduce_axes, keepdims=True) + eps)


def sample_discrete_truncated_normal_gap(max_gap, loc, sigma, rng, depth):
    """Sample d in [1, min(max_gap, depth)] with discrete truncated-normal logits."""
    max_gap = jnp.minimum(jnp.asarray(max_gap, dtype=jnp.int32), jnp.asarray(depth, dtype=jnp.int32))
    gaps = jnp.arange(1, depth + 1, dtype=jnp.int32)
    offsets = (gaps.astype(jnp.float32) - loc) / sigma
    logits = -0.5 * jnp.square(offsets)
    logits = jnp.where(gaps <= max_gap, logits, -jnp.inf)
    return jax.random.categorical(rng, logits).astype(jnp.int32) + 1


def pair_sampling_uniform_mix(global_step, anneal_start_step, anneal_steps):
    progress = (
        (global_step.astype(jnp.float32) - jnp.asarray(anneal_start_step, dtype=jnp.float32))
        / jnp.maximum(jnp.asarray(anneal_steps, dtype=jnp.float32), 1.0)
    )
    return jnp.clip(progress, 0.0, 1.0)


def sample_timestep_indices(
    rng,
    batch_size,
    shortcut_timesteps,
    sampling_mode="uniform",
    logit_mean=0.0,
    logit_std=1.0,
):
    """Sample discrete training timestep indices in [0, shortcut_timesteps - 1]."""
    if sampling_mode == "logit_normal":
        max_idx = jnp.maximum(jnp.asarray(shortcut_timesteps, dtype=jnp.int32) - 1, 0)
        logits = (
            jnp.asarray(logit_mean, dtype=jnp.float32)
            + jnp.asarray(logit_std, dtype=jnp.float32)
            * jax.random.normal(rng, shape=(batch_size,))
        )
        t = jax.nn.sigmoid(logits)
        q = jnp.rint(t * max_idx.astype(jnp.float32)).astype(jnp.int32)
        return jnp.clip(q, 0, max_idx)
    return jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=shortcut_timesteps)


def sample_layer_pair_trunc_normal_to_uniform(
    num_hidden,
    max_gap,
    loc,
    sigma,
    rng,
    uniform_mix,
):
    depth = int(num_hidden) - 1
    gap_rng, source_rng = jax.random.split(rng)
    max_gap = jnp.minimum(jnp.asarray(max_gap, dtype=jnp.int32), jnp.asarray(depth, dtype=jnp.int32))
    gaps = jnp.arange(1, depth + 1, dtype=jnp.int32)
    offsets = (gaps.astype(jnp.float32) - loc) / sigma
    trunc_logits = jnp.where(gaps <= max_gap, -0.5 * jnp.square(offsets), -jnp.inf)
    uniform_logits = jnp.where(gaps <= max_gap, 0.0, -jnp.inf)
    trunc_probs = jax.nn.softmax(trunc_logits)
    uniform_probs = jax.nn.softmax(uniform_logits)
    probs = (1.0 - uniform_mix) * trunc_probs + uniform_mix * uniform_probs
    gap = jax.random.choice(gap_rng, gaps, p=probs).astype(jnp.int32)
    max_source = jnp.asarray(depth, dtype=jnp.int32) - gap
    source = jax.random.randint(source_rng, (), 0, max_source + 1, dtype=jnp.int32)
    return source, source + gap


def sample_layer_pair_trunc_normal(num_hidden, max_gap, loc, sigma, rng):
    depth = int(num_hidden) - 1
    gap_rng, source_rng = jax.random.split(rng)
    gap = sample_discrete_truncated_normal_gap(max_gap, loc, sigma, gap_rng, depth)
    max_source = jnp.asarray(depth, dtype=jnp.int32) - gap
    source = jax.random.randint(source_rng, (), 0, max_source + 1, dtype=jnp.int32)
    return source, source + gap


def sample_three_distinct_pairs_trunc_normal(num_hidden, max_gap, loc, sigma, rng):
    """Sample three distinct layer pairs with gap-first truncated-normal weighting."""
    return sample_distinct_pairs_weighted(num_hidden, max_gap, loc, sigma, rng, num_pairs=3, gap2_bias=0.0)


def sample_distinct_pairs_weighted(
    num_hidden,
    max_gap,
    loc,
    sigma,
    rng,
    num_pairs=3,
    gap2_bias=0.0,
    uniform_mix=0.0,
    center_loc=None,
    center_sigma=None,
):
    depth = int(num_hidden) - 1
    candidates = tuple((a, b, b - a) for a in range(depth) for b in range(a + 1, depth + 1))
    candidate_a = jnp.asarray([a for a, _, _ in candidates], dtype=jnp.int32)
    candidate_b = jnp.asarray([b for _, b, _ in candidates], dtype=jnp.int32)
    candidate_gap = jnp.asarray([d for _, _, d in candidates], dtype=jnp.int32)
    max_gap = jnp.minimum(jnp.asarray(max_gap, dtype=jnp.int32), jnp.asarray(depth, dtype=jnp.int32))
    offsets = (candidate_gap.astype(jnp.float32) - loc) / sigma
    center_sigma = float(center_sigma or 0.0)
    if center_sigma > 0.0:
        pair_midpoints = (candidate_a.astype(jnp.float32) + candidate_b.astype(jnp.float32)) * 0.5
        center = float(depth) * 0.5 if center_loc is None else float(center_loc)
        center_offsets = (pair_midpoints - jnp.asarray(center, dtype=jnp.float32)) / jnp.asarray(
            center_sigma,
            dtype=jnp.float32,
        )
        center_logits = -0.5 * jnp.square(center_offsets)
        center_norm = jnp.zeros_like(center_logits)
        for gap_value in range(1, depth + 1):
            same_gap = candidate_gap == gap_value
            gap_center_norm = jax.nn.logsumexp(jnp.where(same_gap, center_logits, -jnp.inf))
            center_norm = jnp.where(same_gap, gap_center_norm, center_norm)
    else:
        gap_sources = jnp.asarray(depth, dtype=jnp.float32) - candidate_gap.astype(jnp.float32) + 1.0
        center_logits = jnp.zeros_like(candidate_gap, dtype=jnp.float32)
        center_norm = jnp.log(gap_sources)
    trunc_logits = -0.5 * jnp.square(offsets) + center_logits - center_norm
    trunc_logits = trunc_logits + jnp.where(candidate_gap == 2, jnp.asarray(gap2_bias, dtype=jnp.float32), 0.0)
    gap_sources = jnp.asarray(depth, dtype=jnp.float32) - candidate_gap.astype(jnp.float32) + 1.0
    uniform_gap_logits = -jnp.log(gap_sources)
    trunc_logits = jnp.where(candidate_gap <= max_gap, trunc_logits, -jnp.inf)
    uniform_gap_logits = jnp.where(candidate_gap <= max_gap, uniform_gap_logits, -jnp.inf)
    trunc_probs = jax.nn.softmax(trunc_logits)
    uniform_probs = jax.nn.softmax(uniform_gap_logits)
    probs = (1.0 - uniform_mix) * trunc_probs + uniform_mix * uniform_probs
    indices = jax.random.choice(
        rng,
        jnp.arange(len(candidates), dtype=jnp.int32),
        shape=(int(num_pairs),),
        replace=False,
        p=probs,
    )
    return candidate_a[indices], candidate_b[indices]


def sample_three_distinct_pairs_weighted(num_hidden, max_gap, loc, sigma, rng, gap2_bias=0.0):
    return sample_distinct_pairs_weighted(num_hidden, max_gap, loc, sigma, rng, num_pairs=3, gap2_bias=gap2_bias)


def sample_layer_pair_gap2_biased(num_hidden, max_gap, loc, sigma, rng):
    pair_as, pair_bs = sample_three_distinct_pairs_weighted(
        num_hidden,
        max_gap,
        loc,
        sigma,
        rng,
        gap2_bias=2.0,
    )
    return pair_as[0], pair_bs[0]


def sample_pairs_for_mode(
    num_hidden,
    max_gap,
    gap_loc,
    gap_sigma,
    rng,
    num_pairs,
    pair_mode,
    uniform_mix=0.0,
    center_loc=None,
    center_sigma=0.0,
):
    centered = pair_mode in {"trunc_normal_centered", "trunc_normal_centered_to_uniform"}
    return sample_distinct_pairs_weighted(
        num_hidden,
        max_gap,
        gap_loc,
        gap_sigma,
        rng,
        num_pairs=num_pairs,
        gap2_bias=2.0 if pair_mode == "gap2_biased" else 0.0,
        uniform_mix=uniform_mix if pair_mode in {"trunc_normal_to_uniform", "trunc_normal_centered_to_uniform"} else 0.0,
        center_loc=center_loc if centered else None,
        center_sigma=center_sigma if centered else 0.0,
    )


def stop_gradient_except_downstream_backbone_params(backbone_params, target_layer):
    """Allow gradients only through blocks target_layer..L and the final layer."""
    flat = traverse_util.flatten_dict(backbone_params)
    filtered = {}
    for path, value in flat.items():
        module = str(path[0])
        if module == "FinalLayer_0":
            filtered[path] = value
        elif module.startswith("DiTBlock_"):
            try:
                block_idx = int(module.rsplit("_", 1)[1])
            except ValueError:
                block_idx = -1
            filtered[path] = jax.lax.cond(
                jnp.asarray(block_idx, dtype=jnp.int32) >= target_layer,
                lambda z: z,
                jax.lax.stop_gradient,
                value,
            )
        else:
            filtered[path] = jax.lax.stop_gradient(value)
    return traverse_util.unflatten_dict(filtered)


def sample_triplet_uniform(rng, depth):
    def draw(key):
        key, d1_rng, d2_rng = jax.random.split(key, 3)
        d1 = jax.random.randint(d1_rng, (), 1, depth + 1, dtype=jnp.int32)
        d2 = jax.random.randint(d2_rng, (), 1, depth + 1, dtype=jnp.int32)
        return key, d1, d2

    key, d1, d2 = draw(rng)

    def cond_fn(carry):
        _, cur_d1, cur_d2 = carry
        return (cur_d1 + cur_d2) > depth

    def body_fn(carry):
        cur_key, _, _ = carry
        return draw(cur_key)

    key, d1, d2 = jax.lax.while_loop(cond_fn, body_fn, (key, d1, d2))
    source_rng, = jax.random.split(key, 1)
    a = jax.random.randint(source_rng, (), 0, depth - d1 - d2 + 1, dtype=jnp.int32)
    b = a + d1
    c = b + d2
    return a, b, c


def update_l2_ema_5bins(l2_ema, hidden_states, timestep_indices, alpha=0.01):
    """Update per-layer/token L2 magnitude EMA across five 10-step bins."""
    norms = jnp.linalg.norm(hidden_states, axis=-1)  # [L+1, B, P]
    bins = jnp.clip(timestep_indices // 10, 0, 4)
    bin_mask = jax.nn.one_hot(bins, 5, dtype=norms.dtype)  # [B, 5]
    sums = jnp.einsum("bt,lbp->ltp", bin_mask, norms)
    counts = jnp.sum(bin_mask, axis=0)  # [5]
    sums = jax.lax.psum(sums, axis_name="batch")
    counts = jax.lax.psum(counts, axis_name="batch")
    means = sums / jnp.maximum(counts[None, :, None], 1.0)
    updated = (1.0 - alpha) * l2_ema + alpha * means
    return jnp.where((counts[None, :, None] > 0), updated, l2_ema)


PREDICTOR_DEBUG_PAIRS = (
    (1, 3),
    (3, 5),
    (5, 7),
    (10, 12),
    (1, 5),
    (3, 7),
    (6, 12),
    (1, 10),
    (3, 12),
)
PREDICTOR_DEBUG_METRIC_NAMES = ("loss_dir", "loss_mag", "cos_dir", "delta_m_mae")
PREDICTOR_DEBUG_MAX_GAP = 10


def _scheduled_lambda(lambda_value, global_step, start_step, warmup_iters):
    """Return scheduled lambda and warmup scale for optional auxiliary losses."""
    step = global_step.astype(jnp.float32)
    start = start_step.astype(jnp.float32)
    warmup = warmup_iters.astype(jnp.float32)
    active = step >= start
    warmup_scale = jnp.where(
        warmup > 0.0,
        jnp.clip((step - start + 1.0) / jnp.maximum(warmup, 1.0), 0.0, 1.0),
        jnp.float32(1.0),
    )
    warmup_scale = jnp.where(active, warmup_scale, jnp.float32(0.0))
    return lambda_value * warmup_scale, warmup_scale


def private_activation_loss(
    activations,
    max_pairs=0,
    eps=1e-8,
    use_residual=True,
    cosine_mode="bnd",
    pair_mode="first",
    rng=None,
):
    """Common/private activation loss from feat/common-private-activations.

    `activations` is `[L, B, P, D]`, normally post-block states `Z_1..Z_L`.
    The loss penalizes squared cosine similarity between selected layer pairs.
    With use_residual=True, it first subtracts a stop-gradient common activation.
    """
    activations = activations.astype(jnp.float32)
    if use_residual:
        activations_for_common = activations / jnp.maximum(
            jnp.linalg.norm(activations, axis=-1, keepdims=True),
            eps,
        )
        common = jnp.mean(activations_for_common, axis=0)
        private = activations_for_common - jax.lax.stop_gradient(common)[None, ...]
    else:
        common = jnp.mean(activations, axis=0)
        private = activations

    if cosine_mode == "bnd":
        private_flat = private.reshape(private.shape[0], -1)
        private_norm = private_flat / jnp.maximum(
            jnp.linalg.norm(private_flat, axis=-1, keepdims=True),
            eps,
        )
        cosine_matrix = private_norm @ private_norm.T
        pair_cosines = cosine_matrix[jnp.triu_indices(private.shape[0], k=1)]
        pair_cosine_squares = jnp.square(pair_cosines)
    elif cosine_mode == "nd":
        private_flat = private.reshape(private.shape[0], private.shape[1], -1)
        private_norm = private_flat / jnp.maximum(
            jnp.linalg.norm(private_flat, axis=-1, keepdims=True),
            eps,
        )
        cosine_tensor = jnp.einsum("lbd,mbd->lmb", private_norm, private_norm)
        pair_cosines = cosine_tensor[jnp.triu_indices(private.shape[0], k=1)]
        pair_cosine_squares = jnp.mean(jnp.square(pair_cosines), axis=-1)
        pair_cosines = jnp.mean(pair_cosines, axis=-1)
    elif cosine_mode == "token":
        private_norm = private / jnp.maximum(
            jnp.linalg.norm(private, axis=-1, keepdims=True),
            eps,
        )
        cosine_tensor = jnp.einsum("lbpd,mbpd->lmbp", private_norm, private_norm)
        pair_cosines = cosine_tensor[jnp.triu_indices(private.shape[0], k=1)]
        pair_cosine_squares = jnp.mean(jnp.square(pair_cosines), axis=(-1, -2))
        pair_cosines = jnp.mean(pair_cosines, axis=(-1, -2))
    else:
        raise ValueError(f"Unknown private cosine mode: {cosine_mode!r}")

    pair_ids = jnp.arange(pair_cosine_squares.shape[0], dtype=jnp.int32)
    max_pairs = jnp.asarray(max_pairs, dtype=jnp.int32)
    if pair_mode == "first":
        pair_ranks = pair_ids
    elif pair_mode == "random":
        if rng is None:
            raise ValueError("private pair_mode='random' requires rng")
        perm = jax.random.permutation(rng, pair_ids)
        pair_ranks = jnp.empty_like(perm).at[perm].set(pair_ids)
    else:
        raise ValueError(f"Unknown private pair mode: {pair_mode!r}")
    pair_mask = (max_pairs <= 0) | (pair_ranks < max_pairs)
    pair_mask = pair_mask.astype(pair_cosine_squares.dtype)
    pair_count = jnp.maximum(jnp.sum(pair_mask), jnp.asarray(1.0, dtype=pair_cosine_squares.dtype))
    loss_private = jnp.sum(pair_cosine_squares * pair_mask) / pair_count
    common_norm = jnp.mean(jnp.linalg.norm(common.reshape(common.shape[0], -1), axis=-1))
    private_avg_norm = jnp.mean(
        jnp.linalg.norm(private.reshape(private.shape[0], private.shape[1], -1), axis=-1)
    )
    private_pairwise_cosine = jnp.sum(pair_cosines * pair_mask) / pair_count
    return loss_private, common_norm, private_avg_norm, private_pairwise_cosine


def train_step(
    state,
    ema_params,
    predictor_ema_params,
    l2_ema,
    batch,
    rng,
    global_step,
    ema_decay,
    predictor_ema_decay,
    shortcut_timesteps,
    lambda_dir,
    lambda_boot,
    bootstrap_detach_source,
    lambda_mag,
    lambda_boot_mag,
    lambda_skip_fm,
    skip_in_loop_prob,
    skip_in_loop_gap_mode,
    skip_in_loop_gap,
    skip_in_loop_max_gap,
    skip_in_loop_gap_loc,
    skip_in_loop_gap_sigma,
    skip_in_loop_warmup_steps,
    skip_in_loop_detach_source,
    private_loss_enabled,
    lambda_private,
    private_max_pairs,
    private_start_step,
    private_warmup_iters,
    debug_gap_logs,
    mag_scale,
    mag_clip_min,
    mag_clip_max,
    lambda_output_distill,
    l2_ema_alpha,
    *,
    use_output_distill=True,
    output_distill_batch_size=1,
    output_distill_ratio=0.10,
    output_distill_every=1,
    output_distill_update_mode="predictor_plus_downstream",
    output_distill_full_backbone_start_step=0,
    output_distill_pair_mode="trunc_normal_centered",
    pair_uniform_anneal_start_step=0,
    pair_uniform_anneal_steps=100000,
    direct_num_pairs=3,
    direct_num_joint_pairs=1,
    direct_num_predictor_only_pairs=2,
    direct_pair_mode="trunc_normal_centered",
    shortcut_loss_mode="direction_magnitude",
    shortcut_activation_huber_delta=1.0,
    debug_gap_log_freq=10000,
    private_use_residual=True,
    private_cosine_mode="bnd",
    private_pair_mode="first",
    predictor_use_timestep=True,
    predictor_normalize_input=True,
    predictor_use_class_input=False,
    learning_rate=1e-4,
    predictor_learning_rate=2e-4,
    timestep_sampling_mode="uniform",
    timestep_logit_mean=0.0,
    timestep_logit_std=1.0,
    pair_center_loc=None,
    pair_center_sigma=0.0,
):
    """Vanilla SiT training step (global timestep; velocity prediction).

    Repo time convention: tau=0 → pure noise, tau=1 → clean data.
      x_tau   = (1 - tau) * x1 + tau * x0
      target  = x0 - x1
      loss    = E[||v_theta(x_tau, tau) - target||^2]
    """
    if direct_num_joint_pairs != 1 or direct_num_predictor_only_pairs not in {0, 1, 2}:
        raise ValueError("Direct shortcut training expects 1 joint pair and 0, 1, or 2 predictor-only pairs.")
    if direct_num_pairs != direct_num_joint_pairs + direct_num_predictor_only_pairs:
        raise ValueError("direct_num_pairs must equal direct_num_joint_pairs + direct_num_predictor_only_pairs.")
    if direct_num_pairs not in {1, 2, 3}:
        raise ValueError("Direct shortcut training supports 1, 2, or 3 static pairs.")
    pair_modes = {
        "trunc_normal",
        "trunc_normal_to_uniform",
        "trunc_normal_centered",
        "trunc_normal_centered_to_uniform",
        "gap2_biased",
    }
    if direct_pair_mode not in pair_modes:
        raise ValueError(f"Unknown direct pair mode: {direct_pair_mode!r}")
    if timestep_sampling_mode not in {"uniform", "logit_normal"}:
        raise ValueError(f"Unknown timestep sampling mode: {timestep_sampling_mode!r}")
    if timestep_logit_std <= 0.0:
        raise ValueError("timestep_logit_std must be positive.")
    if pair_center_sigma < 0.0:
        raise ValueError("pair_center_sigma must be non-negative.")
    if shortcut_loss_mode == "direction_activation_huber":
        shortcut_loss_mode = "direction_activation"
    if shortcut_loss_mode not in {"direction_magnitude", "direction_activation"}:
        raise ValueError(f"Unknown shortcut loss mode: {shortcut_loss_mode!r}")
    if shortcut_activation_huber_delta <= 0.0:
        raise ValueError("shortcut_activation_huber_delta must be positive.")
    if output_distill_pair_mode not in pair_modes:
        raise ValueError(f"Unknown output distill pair mode: {output_distill_pair_mode!r}")
    if output_distill_update_mode not in {
        "predictor_only",
        "predictor_plus_downstream",
        "predictor_plus_all",
        "predictor_only_then_all",
    }:
        raise ValueError(f"Unknown output distill update mode: {output_distill_update_mode!r}")
    if private_cosine_mode not in {"bnd", "nd", "token"}:
        raise ValueError(f"Unknown private cosine mode: {private_cosine_mode!r}")
    if private_pair_mode not in {"first", "random"}:
        raise ValueError(f"Unknown private pair mode: {private_pair_mode!r}")

    x0, y = batch  # x0: [local_B, N, D], y: [local_B]
    local_batch = x0.shape[0]

    (
        rng,
        tau_rng,
        noise_rng,
        drop_rng,
        direct_pair_rng,
        triplet_rng,
        output_subset_rng,
        output_pair_rng,
        output_resume_drop_rng,
        private_pair_rng,
    ) = jax.random.split(rng, 10)
    lambda_private_eff, private_warmup_scale = _scheduled_lambda(
        lambda_private,
        global_step,
        private_start_step,
        private_warmup_iters,
    )
    pair_uniform_mix = pair_sampling_uniform_mix(
        global_step,
        pair_uniform_anneal_start_step,
        pair_uniform_anneal_steps,
    )
    output_distill_full_backbone_active = jnp.logical_or(
        output_distill_update_mode == "predictor_plus_all",
        jnp.logical_and(
            output_distill_update_mode == "predictor_only_then_all",
            global_step >= jnp.asarray(output_distill_full_backbone_start_step, dtype=jnp.int32),
        ),
    )

    q = sample_timestep_indices(
        tau_rng,
        local_batch,
        shortcut_timesteps,
        sampling_mode=timestep_sampling_mode,
        logit_mean=timestep_logit_mean,
        logit_std=timestep_logit_std,
    )
    tau = q.astype(jnp.float32) / jnp.maximum(jnp.float32(shortcut_timesteps - 1), 1.0)
    x1 = jax.random.normal(noise_rng, x0.shape)  # [B, N, D]

    x_tau = (1.0 - tau[:, None, None]) * x1 + tau[:, None, None] * x0
    target = x0 - x1

    def loss_fn(params):
        pred = state.apply_fn(
            {"params": params["backbone"]},
            x_tau,
            timesteps=tau,
            vector=y,
            deterministic=False,
            rngs={"dropout": drop_rng},
            return_hidden_states=True,
        )
        pred, hidden_tuple, dit_time_emb = pred
        loss_gen = jnp.mean((pred - target) ** 2)
        hidden_stack = jnp.stack(hidden_tuple, axis=0)
        hidden_stack_f32 = hidden_stack.astype(jnp.float32)
        hidden_norms = jnp.linalg.norm(hidden_stack_f32, axis=-1, keepdims=True)
        directions = hidden_stack_f32 / (hidden_norms + 1e-6)
        log_magnitudes = jnp.log(hidden_norms + 1e-6)

        def shortcut_direct_mag_loss_for_pair(source_layer, target_layer, source_detach):
            za = hidden_stack_f32[source_layer]
            if source_detach:
                za = jax.lax.stop_gradient(za)
            zb_target = jax.lax.stop_gradient(hidden_stack_f32[target_layer])
            predictor_source, ma_condition = build_predictor_source(
                za,
                normalize_input=predictor_normalize_input,
            )
            ub, _, _ = build_direction_and_norm_map(zb_target)
            t_embed_pair = jax.lax.stop_gradient(dit_time_emb) if source_detach else dit_time_emb
            y_pair, delta_m_pair = state.predictor_apply_fn(
                {"params": params["predictor"]},
                predictor_source,
                source_layer,
                target_layer,
                t_embed_pair,
                ma_condition,
                detach_timestep_embed=source_detach,
                use_timestep_embed=predictor_use_timestep,
                class_labels=y,
            )
            u_pair = l2_normalize_tokens(y_pair)
            cos_pair = jnp.sum(u_pair * ub, axis=-1).mean()
            loss_dir_pair = 1.0 - cos_pair

            if shortcut_loss_mode == "direction_magnitude":
                ma_target = jnp.log(
                    jnp.linalg.norm(jax.lax.stop_gradient(hidden_stack_f32[source_layer]), axis=-1, keepdims=True)
                    + 1e-6
                )
                mb_target = jnp.log(
                    jnp.linalg.norm(jax.lax.stop_gradient(hidden_stack_f32[target_layer]), axis=-1, keepdims=True)
                    + 1e-6
                )
                target_delta_m_raw_pair = jax.lax.stop_gradient(mb_target - ma_target)
                target_delta_m_pair = jax.lax.stop_gradient(
                    jnp.clip(target_delta_m_raw_pair / mag_scale, -1.0, 1.0)
                )
                loss_mag_pair = 0.5 * jnp.mean(jnp.square(delta_m_pair - target_delta_m_pair))
                loss_aux_pair = loss_mag_pair
                delta_m_raw_error_pair = mag_scale * delta_m_pair - target_delta_m_raw_pair
                delta_m_abs_error_pair = jnp.abs(delta_m_raw_error_pair)
                delta_m_mae_pair = jnp.mean(delta_m_abs_error_pair)
                delta_m_rmse_pair = jnp.sqrt(jnp.mean(jnp.square(delta_m_raw_error_pair)))
                delta_m_ratio_error_pair = jnp.mean(jnp.exp(delta_m_abs_error_pair))
                delta_m_clip_rate_pair = jnp.mean((jnp.abs(target_delta_m_raw_pair) > mag_scale).astype(jnp.float32))
            else:
                target_rms_pair = jax.lax.stop_gradient(sample_activation_rms(zb_target))
                pred_rms_pair = sample_activation_rms(y_pair)
                residual_norm_pair = (y_pair - zb_target) / target_rms_pair
                loss_aux_pair = jnp.mean(
                    optax.huber_loss(
                        residual_norm_pair,
                        delta=jnp.asarray(shortcut_activation_huber_delta, dtype=jnp.float32),
                    )
                )
                delta_m_mae_pair = jnp.mean(target_rms_pair)
                delta_m_rmse_pair = jnp.mean(pred_rms_pair)
                delta_m_ratio_error_pair = jnp.float32(0.0)
                delta_m_clip_rate_pair = jnp.sqrt(jnp.mean(jnp.square(residual_norm_pair)))
            pair_loss = lambda_dir * loss_dir_pair + lambda_mag * loss_aux_pair
            y_pair_norm = jnp.linalg.norm(y_pair, axis=-1)
            return (
                pair_loss,
                loss_dir_pair,
                loss_aux_pair,
                cos_pair,
                delta_m_mae_pair,
                delta_m_rmse_pair,
                delta_m_ratio_error_pair,
                delta_m_clip_rate_pair,
                jnp.mean(y_pair_norm),
                jnp.min(y_pair_norm),
                jnp.max(y_pair_norm),
            )

        direct_as, direct_bs = sample_pairs_for_mode(
            directions.shape[0],
            skip_in_loop_max_gap,
            skip_in_loop_gap_loc,
            skip_in_loop_gap_sigma,
            direct_pair_rng,
            direct_num_pairs,
            direct_pair_mode,
            uniform_mix=pair_uniform_mix,
            center_loc=pair_center_loc,
            center_sigma=pair_center_sigma,
        )
        direct_joint_a = direct_as[0]
        direct_joint_b = direct_bs[0]
        joint_values = shortcut_direct_mag_loss_for_pair(
            direct_joint_a,
            direct_joint_b,
            False,
        )
        if direct_num_predictor_only_pairs >= 1:
            direct_ponly_1_a = direct_as[1]
            direct_ponly_1_b = direct_bs[1]
            ponly_1_values = shortcut_direct_mag_loss_for_pair(
                direct_ponly_1_a,
                direct_ponly_1_b,
                True,
            )
        else:
            direct_ponly_1_a = jnp.asarray(-1, dtype=jnp.int32)
            direct_ponly_1_b = jnp.asarray(-1, dtype=jnp.int32)
            ponly_1_values = tuple(jnp.float32(0.0) for _ in range(len(joint_values)))
        if direct_num_pairs == 3:
            direct_ponly_2_a = direct_as[2]
            direct_ponly_2_b = direct_bs[2]
            ponly_2_values = shortcut_direct_mag_loss_for_pair(
                direct_ponly_2_a,
                direct_ponly_2_b,
                True,
            )
        else:
            direct_ponly_2_a = jnp.asarray(-1, dtype=jnp.int32)
            direct_ponly_2_b = jnp.asarray(-1, dtype=jnp.int32)
            ponly_2_values = tuple(jnp.float32(0.0) for _ in range(len(ponly_1_values)))
        loss_direct_joint = joint_values[0]
        loss_direct_ponly_1 = ponly_1_values[0]
        loss_direct_ponly_2 = ponly_2_values[0]
        loss_direct_3pair = (
            loss_direct_joint
            + jnp.asarray(direct_num_predictor_only_pairs >= 1, dtype=jnp.float32) * loss_direct_ponly_1
            + jnp.asarray(direct_num_pairs == 3, dtype=jnp.float32) * loss_direct_ponly_2
        ) / float(direct_num_pairs)
        loss_dir = joint_values[1]
        loss_mag = joint_values[2]
        if direct_num_predictor_only_pairs > 0:
            loss_dir_ponly_mean = (
                ponly_1_values[1]
                + jnp.asarray(direct_num_pairs == 3, dtype=jnp.float32) * ponly_2_values[1]
            ) / float(direct_num_predictor_only_pairs)
            loss_mag_ponly_mean = (
                ponly_1_values[2]
                + jnp.asarray(direct_num_pairs == 3, dtype=jnp.float32) * ponly_2_values[2]
            ) / float(direct_num_predictor_only_pairs)
        else:
            loss_dir_ponly_mean = jnp.float32(0.0)
            loss_mag_ponly_mean = jnp.float32(0.0)
        cos_dir = joint_values[3]
        y_ab_norm_mean = joint_values[8]
        y_ab_norm_min = joint_values[9]
        y_ab_norm_max = joint_values[10]
        delta_m_mae = joint_values[4]
        delta_m_rmse = joint_values[5]
        delta_m_ratio_error = joint_values[6]
        delta_m_clip_rate = joint_values[7]
        direct_gap = direct_joint_b - direct_joint_a

        boot_a, boot_b, boot_c = sample_triplet_uniform(triplet_rng, directions.shape[0] - 1)
        boot_source_u = directions[boot_a] if predictor_normalize_input else hidden_stack_f32[boot_a]
        boot_source_m = log_magnitudes[boot_a]
        boot_source_u = jax.lax.cond(
            bootstrap_detach_source,
            jax.lax.stop_gradient,
            lambda z: z,
            boot_source_u,
        )
        boot_source_m = jax.lax.cond(
            bootstrap_detach_source,
            jax.lax.stop_gradient,
            lambda z: z,
            boot_source_m,
        )
        y_boot_ab, delta_m_boot_ab = state.predictor_apply_fn(
            {"params": params["predictor"]},
            boot_source_u,
            boot_a,
            boot_b,
            dit_time_emb,
            boot_source_m,
            use_timestep_embed=predictor_use_timestep,
            class_labels=y,
        )
        u_boot_ab = l2_normalize_tokens(y_boot_ab)
        if shortcut_loss_mode == "direction_magnitude":
            m_boot_ab = boot_source_m + mag_scale * delta_m_boot_ab
            boot_b_source = u_boot_ab if predictor_normalize_input else y_boot_ab
        else:
            m_boot_ab = jnp.log(jnp.linalg.norm(y_boot_ab.astype(jnp.float32), axis=-1, keepdims=True) + 1e-6)
            boot_b_source = u_boot_ab if predictor_normalize_input else y_boot_ab
        y_abc, delta_m_abc = state.predictor_apply_fn(
            {"params": params["predictor"]},
            boot_b_source,
            boot_b,
            boot_c,
            dit_time_emb,
            m_boot_ab,
            use_timestep_embed=predictor_use_timestep,
            class_labels=y,
        )
        u_abc = l2_normalize_tokens(y_abc)
        y_ac_ema, delta_m_ac_ema = state.predictor_apply_fn(
            {"params": predictor_ema_params},
            boot_source_u,
            boot_a,
            boot_c,
            dit_time_emb,
            boot_source_m,
            use_timestep_embed=predictor_use_timestep,
            class_labels=y,
        )
        u_ac_ema = jax.lax.stop_gradient(l2_normalize_tokens(y_ac_ema))
        cos_boot = jnp.sum(u_abc * u_ac_ema, axis=-1).mean()
        loss_boot = 1.0 - cos_boot
        if shortcut_loss_mode == "direction_magnitude":
            target_delta_m_ac = jax.lax.stop_gradient(delta_m_ac_ema)
            pred_delta_m_ac = delta_m_boot_ab + delta_m_abc
            loss_boot_mag = 0.5 * jnp.mean(jnp.square(pred_delta_m_ac - target_delta_m_ac))
        else:
            y_ac_ema_target = jax.lax.stop_gradient(y_ac_ema)
            boot_target_rms = jax.lax.stop_gradient(sample_activation_rms(y_ac_ema_target))
            boot_pred_rms = sample_activation_rms(y_abc)
            boot_residual_norm = (y_abc - y_ac_ema_target) / boot_target_rms
            loss_boot_mag = jnp.mean(
                optax.huber_loss(
                    boot_residual_norm,
                    delta=jnp.asarray(shortcut_activation_huber_delta, dtype=jnp.float32),
                )
            )
        if shortcut_loss_mode == "direction_magnitude":
            boot_target_rms_metric = jnp.float32(0.0)
            boot_pred_rms_metric = jnp.float32(0.0)
            boot_rel_residual_rms_metric = jnp.float32(0.0)
        else:
            boot_target_rms_metric = jnp.mean(boot_target_rms)
            boot_pred_rms_metric = jnp.mean(boot_pred_rms)
            boot_rel_residual_rms_metric = jnp.sqrt(jnp.mean(jnp.square(boot_residual_norm)))

        def compute_debug_pair_metrics(_):
            def compute_pair_metrics(source_layer, target_layer):
                pair_source, pair_source_m = build_predictor_source(
                    hidden_stack_f32[source_layer],
                    normalize_input=predictor_normalize_input,
                )
                y_pair, delta_m_pair = state.predictor_apply_fn(
                    {"params": params["predictor"]},
                    pair_source,
                    jnp.asarray(source_layer, dtype=jnp.int32),
                    jnp.asarray(target_layer, dtype=jnp.int32),
                    dit_time_emb,
                    pair_source_m,
                    use_timestep_embed=predictor_use_timestep,
                    class_labels=y,
                )
                u_pair = l2_normalize_tokens(y_pair)
                target_u_pair = jax.lax.stop_gradient(directions[target_layer])
                pair_cos = jnp.sum(u_pair * target_u_pair, axis=-1).mean()
                pair_loss_dir = 1.0 - pair_cos
                pair_target_delta_raw = jax.lax.stop_gradient(
                    log_magnitudes[target_layer] - log_magnitudes[source_layer]
                )
                pair_target_delta_norm = jnp.clip(pair_target_delta_raw / mag_scale, -1.0, 1.0)
                pair_loss_mag = 0.5 * jnp.mean(jnp.square(delta_m_pair - pair_target_delta_norm))
                pair_delta_m_mae = jnp.mean(jnp.abs(mag_scale * delta_m_pair - pair_target_delta_raw))
                return pair_loss_dir, pair_loss_mag, pair_cos, pair_delta_m_mae

            values = []
            for pair_a, pair_b in PREDICTOR_DEBUG_PAIRS:
                safe_pair_a = min(pair_a, directions.shape[0] - 1)
                safe_pair_b = min(pair_b, directions.shape[0] - 1)

                def compute_one_pair(_):
                    return compute_pair_metrics(safe_pair_a, safe_pair_b)

                pair_is_valid = pair_b < directions.shape[0]
                values.extend(
                    jax.lax.cond(
                        pair_is_valid,
                        compute_one_pair,
                        lambda _: (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
                        operand=None,
                    )
                )
            for gap in range(1, PREDICTOR_DEBUG_MAX_GAP + 1):
                gap_values = []
                gap_count = jnp.asarray(max(directions.shape[0] - gap, 0), dtype=jnp.float32)
                for source_layer in range(0, 13 - gap):
                    target_layer = source_layer + gap
                    safe_source_layer = min(source_layer, directions.shape[0] - 1)
                    safe_target_layer = min(target_layer, directions.shape[0] - 1)

                    def compute_one_gap_pair(_):
                        return compute_pair_metrics(safe_source_layer, safe_target_layer)

                    pair_values = jax.lax.cond(
                        target_layer < directions.shape[0],
                        compute_one_gap_pair,
                        lambda _: (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
                        operand=None,
                    )
                    gap_values.append(pair_values)
                for metric_idx in range(len(PREDICTOR_DEBUG_METRIC_NAMES)):
                    metric_sum = sum(pair_values[metric_idx] for pair_values in gap_values)
                    values.append(metric_sum / jnp.maximum(gap_count, 1.0))
            return tuple(values)

        debug_metric_count = (
            (len(PREDICTOR_DEBUG_PAIRS) + PREDICTOR_DEBUG_MAX_GAP)
            * len(PREDICTOR_DEBUG_METRIC_NAMES)
        )
        debug_gap_logs_now = jnp.logical_and(
            debug_gap_logs,
            jnp.equal(jnp.mod(global_step, jnp.asarray(debug_gap_log_freq, dtype=jnp.int32)), 0),
        )
        debug_pair_metrics = jax.lax.cond(
            debug_gap_logs_now,
            compute_debug_pair_metrics,
            lambda _: tuple(jnp.float32(0.0) for _ in range(debug_metric_count)),
            operand=None,
        )

        loss_skip_fm = jnp.float32(0.0)
        skip_prob_eff = jnp.float32(0.0)
        skip_do = jnp.asarray(False, dtype=jnp.bool_)
        skip_a_metric = jnp.float32(-1.0)
        skip_b_metric = jnp.float32(-1.0)
        skip_gap_metric = jnp.float32(0.0)

        def compute_output_distill_loss(_):
            subset_idx = jax.random.permutation(output_subset_rng, local_batch)[:output_distill_batch_size]
            x_out = x_tau[subset_idx]
            t_out = tau[subset_idx]
            y_out = y[subset_idx]
            v_teacher = jax.lax.stop_gradient(pred[subset_idx])
            t_emb_teacher = jax.lax.stop_gradient(dit_time_emb[subset_idx])

            output_as, output_bs = sample_pairs_for_mode(
                hidden_stack_f32.shape[0],
                skip_in_loop_max_gap,
                skip_in_loop_gap_loc,
                skip_in_loop_gap_sigma,
                output_pair_rng,
                1,
                output_distill_pair_mode,
                uniform_mix=pair_uniform_mix,
                center_loc=pair_center_loc,
                center_sigma=pair_center_sigma,
            )
            output_a = output_as[0]
            output_b = output_bs[0]
            z_source = hidden_stack_f32[output_a, subset_idx]
            if output_distill_update_mode == "predictor_plus_all":
                pass
            elif output_distill_update_mode == "predictor_only_then_all":
                z_source = jax.lax.cond(
                    output_distill_full_backbone_active,
                    lambda z: z,
                    jax.lax.stop_gradient,
                    z_source,
                )
            else:
                z_source = jax.lax.stop_gradient(z_source)
            predictor_source, m_source = build_predictor_source(
                z_source,
                normalize_input=predictor_normalize_input,
            )
            y_pred, delta_m_pred = state.predictor_apply_fn(
                {"params": params["predictor"]},
                predictor_source,
                output_a,
                output_b,
                t_emb_teacher,
                m_source,
                detach_timestep_embed=True,
                use_timestep_embed=predictor_use_timestep,
                class_labels=y[subset_idx],
            )
            u_pred = l2_normalize_tokens(y_pred)
            if shortcut_loss_mode == "direction_magnitude":
                m_pred = jnp.clip(m_source + mag_scale * delta_m_pred, mag_clip_min, mag_clip_max)
                z_hat_b = jnp.exp(m_pred) * u_pred
            else:
                z_hat_b = y_pred.astype(jnp.float32)

            resume_backbone_params = params["backbone"]
            if output_distill_update_mode == "predictor_only":
                resume_backbone_params = jax.tree_util.tree_map(
                    jax.lax.stop_gradient,
                    resume_backbone_params,
                )
            elif output_distill_update_mode == "predictor_only_then_all":
                resume_backbone_params = jax.tree_util.tree_map(
                    lambda param: jax.lax.cond(
                        output_distill_full_backbone_active,
                        lambda x: x,
                        jax.lax.stop_gradient,
                        param,
                    ),
                    resume_backbone_params,
                )
            elif output_distill_update_mode == "predictor_plus_downstream":
                resume_backbone_params = stop_gradient_except_downstream_backbone_params(
                    resume_backbone_params,
                    output_b,
                )
            elif output_distill_update_mode == "predictor_plus_all":
                pass
            else:
                raise ValueError(f"Unknown output distill update mode: {output_distill_update_mode!r}")
            v_skip = state.apply_fn(
                {"params": resume_backbone_params},
                x_out,
                timesteps=t_out,
                vector=y_out,
                deterministic=False,
                rngs={"dropout": output_resume_drop_rng},
                resume_hidden=z_hat_b,
                resume_start_layer=output_b + 1,
            )
            reduce_axes = tuple(range(1, v_skip.ndim))
            numer = jnp.sum(jnp.square(v_skip - v_teacher), axis=reduce_axes)
            denom = jnp.sum(jnp.square(v_teacher), axis=reduce_axes) + 1e-6
            loss_out = jnp.mean(numer / denom)
            return (
                loss_out,
                output_a.astype(jnp.float32),
                output_b.astype(jnp.float32),
                (output_b - output_a).astype(jnp.float32),
                jnp.asarray(output_distill_batch_size, dtype=jnp.float32),
            )

        output_distill_due = (global_step % jnp.asarray(max(int(output_distill_every), 1), dtype=jnp.int32)) == 0
        output_distill_enabled = jnp.asarray(use_output_distill, dtype=jnp.bool_) & output_distill_due
        loss_output_distill, output_distill_a, output_distill_b, output_distill_gap, output_distill_batch_size_metric = jax.lax.cond(
            output_distill_enabled,
            compute_output_distill_loss,
            lambda _: (
                jnp.float32(0.0),
                jnp.float32(-1.0),
                jnp.float32(-1.0),
                jnp.float32(0.0),
                jnp.asarray(output_distill_batch_size, dtype=jnp.float32),
            ),
            operand=None,
        )
        loss_output_distill_weighted = lambda_output_distill * loss_output_distill

        def compute_private_metrics(_):
            return private_activation_loss(
                hidden_stack[1:],
                max_pairs=private_max_pairs,
                use_residual=private_use_residual,
                cosine_mode=private_cosine_mode,
                pair_mode=private_pair_mode,
                rng=private_pair_rng,
            )

        loss_private, common_norm, private_avg_norm, private_pairwise_cosine = jax.lax.cond(
            private_loss_enabled & (lambda_private_eff > 0.0),
            compute_private_metrics,
            lambda _: (
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.float32(0.0),
            ),
            operand=None,
        )
        loss_total = (
            loss_gen
            + loss_direct_3pair
            + lambda_boot * loss_boot
            + lambda_boot_mag * loss_boot_mag
            + lambda_private_eff * loss_private
            + loss_output_distill_weighted
        )
        v_abs_mean = jnp.mean(jnp.abs(target))
        v_pred_abs_mean = jnp.mean(jnp.abs(pred))
        aux = (
            v_abs_mean,
            v_pred_abs_mean,
            loss_gen,
            loss_dir,
            loss_boot,
            loss_mag,
            loss_boot_mag,
            loss_direct_3pair,
            loss_direct_joint,
            loss_direct_ponly_1,
            loss_direct_ponly_2,
            loss_dir_ponly_mean,
            loss_mag_ponly_mean,
            loss_output_distill,
            loss_output_distill_weighted,
            loss_skip_fm,
            loss_private,
            cos_dir,
            cos_boot,
            common_norm,
            private_avg_norm,
            private_pairwise_cosine,
            lambda_private_eff,
            private_warmup_scale,
            y_ab_norm_mean,
            y_ab_norm_min,
            y_ab_norm_max,
            delta_m_mae,
            delta_m_rmse,
            delta_m_ratio_error,
            delta_m_clip_rate,
            boot_target_rms_metric,
            boot_pred_rms_metric,
            boot_rel_residual_rms_metric,
            direct_gap,
            direct_joint_a.astype(jnp.float32),
            direct_joint_b.astype(jnp.float32),
            direct_ponly_1_a.astype(jnp.float32),
            direct_ponly_1_b.astype(jnp.float32),
            direct_ponly_2_a.astype(jnp.float32),
            direct_ponly_2_b.astype(jnp.float32),
            (direct_ponly_1_b - direct_ponly_1_a).astype(jnp.float32),
            (direct_ponly_2_b - direct_ponly_2_a).astype(jnp.float32),
            skip_do.astype(jnp.float32),
            skip_prob_eff,
            skip_a_metric,
            skip_b_metric,
            skip_gap_metric,
            output_distill_a,
            output_distill_b,
            output_distill_gap,
            output_distill_batch_size_metric,
            pair_uniform_mix,
            output_distill_full_backbone_active.astype(jnp.float32),
            debug_gap_logs_now.astype(jnp.float32),
            debug_pair_metrics,
            hidden_stack,
        )
        return loss_total, aux

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    (
        v_abs,
        v_pred,
        loss_gen,
        loss_dir,
        loss_boot,
        loss_mag,
        loss_boot_mag,
        loss_direct_3pair,
        loss_direct_joint,
        loss_direct_ponly_1,
        loss_direct_ponly_2,
        loss_dir_ponly_mean,
        loss_mag_ponly_mean,
        loss_output_distill,
        loss_output_distill_weighted,
        loss_skip_fm,
        loss_private,
        cos_dir,
        cos_boot,
        common_norm,
        private_avg_norm,
        private_pairwise_cosine,
        lambda_private_eff,
        private_warmup_scale,
        y_ab_norm_mean,
        y_ab_norm_min,
        y_ab_norm_max,
        delta_m_mae,
        delta_m_rmse,
        delta_m_ratio_error,
        delta_m_clip_rate,
        boot_target_rms,
        boot_pred_rms,
        boot_rel_residual_rms,
        direct_gap,
        direct_joint_a,
        direct_joint_b,
        direct_ponly_1_a,
        direct_ponly_1_b,
        direct_ponly_2_a,
        direct_ponly_2_b,
        direct_gap_ponly_1,
        direct_gap_ponly_2,
        skip_do_metric,
        skip_prob_eff,
        skip_a_metric,
        skip_b_metric,
        skip_gap_metric,
        output_distill_a,
        output_distill_b,
        output_distill_gap,
        output_distill_batch_size_metric,
        pair_uniform_mix,
        output_distill_full_backbone_active,
        debug_gap_logs_now,
        debug_pair_metrics,
        hidden_stack,
    ) = aux

    loss = jax.lax.pmean(loss, axis_name="batch")
    v_abs = jax.lax.pmean(v_abs, axis_name="batch")
    v_pred = jax.lax.pmean(v_pred, axis_name="batch")
    loss_gen = jax.lax.pmean(loss_gen, axis_name="batch")
    loss_dir = jax.lax.pmean(loss_dir, axis_name="batch")
    loss_boot = jax.lax.pmean(loss_boot, axis_name="batch")
    loss_mag = jax.lax.pmean(loss_mag, axis_name="batch")
    loss_boot_mag = jax.lax.pmean(loss_boot_mag, axis_name="batch")
    loss_direct_3pair = jax.lax.pmean(loss_direct_3pair, axis_name="batch")
    loss_direct_joint = jax.lax.pmean(loss_direct_joint, axis_name="batch")
    loss_direct_ponly_1 = jax.lax.pmean(loss_direct_ponly_1, axis_name="batch")
    loss_direct_ponly_2 = jax.lax.pmean(loss_direct_ponly_2, axis_name="batch")
    loss_dir_ponly_mean = jax.lax.pmean(loss_dir_ponly_mean, axis_name="batch")
    loss_mag_ponly_mean = jax.lax.pmean(loss_mag_ponly_mean, axis_name="batch")
    loss_output_distill = jax.lax.pmean(loss_output_distill, axis_name="batch")
    loss_output_distill_weighted = jax.lax.pmean(loss_output_distill_weighted, axis_name="batch")
    loss_skip_fm = jax.lax.pmean(loss_skip_fm, axis_name="batch")
    loss_private = jax.lax.pmean(loss_private, axis_name="batch")
    cos_dir = jax.lax.pmean(cos_dir, axis_name="batch")
    cos_boot = jax.lax.pmean(cos_boot, axis_name="batch")
    common_norm = jax.lax.pmean(common_norm, axis_name="batch")
    private_avg_norm = jax.lax.pmean(private_avg_norm, axis_name="batch")
    private_pairwise_cosine = jax.lax.pmean(private_pairwise_cosine, axis_name="batch")
    lambda_private_eff = jax.lax.pmean(lambda_private_eff, axis_name="batch")
    private_warmup_scale = jax.lax.pmean(private_warmup_scale, axis_name="batch")
    y_ab_norm_mean = jax.lax.pmean(y_ab_norm_mean, axis_name="batch")
    y_ab_norm_min = jax.lax.pmin(y_ab_norm_min, axis_name="batch")
    y_ab_norm_max = jax.lax.pmax(y_ab_norm_max, axis_name="batch")
    delta_m_mae = jax.lax.pmean(delta_m_mae, axis_name="batch")
    delta_m_rmse = jax.lax.pmean(delta_m_rmse, axis_name="batch")
    delta_m_ratio_error = jax.lax.pmean(delta_m_ratio_error, axis_name="batch")
    delta_m_clip_rate = jax.lax.pmean(delta_m_clip_rate, axis_name="batch")
    boot_target_rms = jax.lax.pmean(boot_target_rms, axis_name="batch")
    boot_pred_rms = jax.lax.pmean(boot_pred_rms, axis_name="batch")
    boot_rel_residual_rms = jax.lax.pmean(boot_rel_residual_rms, axis_name="batch")
    direct_gap = jax.lax.pmean(direct_gap.astype(jnp.float32), axis_name="batch")
    direct_joint_a = jax.lax.pmean(direct_joint_a.astype(jnp.float32), axis_name="batch")
    direct_joint_b = jax.lax.pmean(direct_joint_b.astype(jnp.float32), axis_name="batch")
    direct_ponly_1_a = jax.lax.pmean(direct_ponly_1_a.astype(jnp.float32), axis_name="batch")
    direct_ponly_1_b = jax.lax.pmean(direct_ponly_1_b.astype(jnp.float32), axis_name="batch")
    direct_ponly_2_a = jax.lax.pmean(direct_ponly_2_a.astype(jnp.float32), axis_name="batch")
    direct_ponly_2_b = jax.lax.pmean(direct_ponly_2_b.astype(jnp.float32), axis_name="batch")
    direct_gap_ponly_1 = jax.lax.pmean(direct_gap_ponly_1.astype(jnp.float32), axis_name="batch")
    direct_gap_ponly_2 = jax.lax.pmean(direct_gap_ponly_2.astype(jnp.float32), axis_name="batch")
    skip_do_metric = jax.lax.pmean(skip_do_metric, axis_name="batch")
    skip_prob_eff = jax.lax.pmean(skip_prob_eff, axis_name="batch")
    skip_a_metric = jax.lax.pmean(skip_a_metric, axis_name="batch")
    skip_b_metric = jax.lax.pmean(skip_b_metric, axis_name="batch")
    skip_gap_metric = jax.lax.pmean(skip_gap_metric, axis_name="batch")
    output_distill_a = jax.lax.pmean(output_distill_a.astype(jnp.float32), axis_name="batch")
    output_distill_b = jax.lax.pmean(output_distill_b.astype(jnp.float32), axis_name="batch")
    output_distill_gap = jax.lax.pmean(output_distill_gap.astype(jnp.float32), axis_name="batch")
    output_distill_batch_size_metric = jax.lax.pmean(
        output_distill_batch_size_metric.astype(jnp.float32),
        axis_name="batch",
    )
    pair_uniform_mix = jax.lax.pmean(pair_uniform_mix, axis_name="batch")
    output_distill_full_backbone_active = jax.lax.pmean(output_distill_full_backbone_active, axis_name="batch")
    debug_gap_logs_now = jax.lax.pmean(debug_gap_logs_now, axis_name="batch")
    debug_pair_metrics = tuple(jax.lax.pmean(value, axis_name="batch") for value in debug_pair_metrics)
    sampled_timestep_q = jax.lax.pmean(jnp.mean(q.astype(jnp.float32)), axis_name="batch")
    sampled_timestep_tau = sampled_timestep_q / jnp.maximum(jnp.float32(shortcut_timesteps - 1), 1.0)
    grads = jax.lax.pmean(grads, axis_name="batch")

    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
    param_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(state.params)))

    l2_ema = update_l2_ema_5bins(l2_ema, jax.lax.stop_gradient(hidden_stack), q, alpha=l2_ema_alpha)
    state = state.apply_gradients(grads=grads)
    ema_params = ema_update(ema_params, state.params["backbone"], ema_decay)
    predictor_ema_params = ema_update(
        predictor_ema_params,
        state.params["predictor"],
        predictor_ema_decay,
    )

    metrics = {
        "train/loss": loss,
        "train/loss_total": loss,
        "train/loss_gen": loss_gen,
        "train/loss_dir": loss_dir,
        "train/loss_boot": loss_boot,
        "train/loss_mag": loss_mag,
        "train/loss_boot_mag": loss_boot_mag,
        "train/loss_boot_aux": loss_boot_mag,
        "train/loss_direct_3pair": loss_direct_3pair,
        "train/loss_direct_npairs": loss_direct_3pair,
        "train/loss_direct_joint": loss_direct_joint,
        "train/loss_direct_ponly_1": loss_direct_ponly_1,
        "train/loss_direct_ponly_2": loss_direct_ponly_2,
        "train/loss_dir_joint": loss_dir,
        "train/loss_mag_joint": loss_mag,
        "train/loss_direct_aux_joint": loss_mag,
        "train/loss_dir_ponly_mean": loss_dir_ponly_mean,
        "train/loss_mag_ponly_mean": loss_mag_ponly_mean,
        "train/loss_direct_aux_ponly_mean": loss_mag_ponly_mean,
        "train/loss_output_distill": loss_output_distill,
        "train/loss_output_distill_weighted": loss_output_distill_weighted,
        "train/loss_skip_fm": loss_skip_fm,
        "train/l_private": loss_private,
        "train/cos_dir": cos_dir,
        "train/cos_boot": cos_boot,
        "train/lambda_private_effective": lambda_private_eff,
        "train/private_warmup_scale": private_warmup_scale,
        "train/common_norm": common_norm,
        "train/private_avg_norm": private_avg_norm,
        "train/private_pairwise_cosine": private_pairwise_cosine,
        "train/private_use_residual": jnp.asarray(1.0 if private_use_residual else 0.0, dtype=jnp.float32),
        "train/private_cosine_mode": jnp.asarray(
            {"bnd": 0.0, "nd": 1.0, "token": 2.0}[private_cosine_mode],
            dtype=jnp.float32,
        ),
        "train/private_pair_mode": jnp.asarray(
            0.0 if private_pair_mode == "first" else 1.0,
            dtype=jnp.float32,
        ),
        "train/predictor_use_timestep": jnp.asarray(1.0 if predictor_use_timestep else 0.0, dtype=jnp.float32),
        "train/predictor_normalize_input": jnp.asarray(1.0 if predictor_normalize_input else 0.0, dtype=jnp.float32),
        "train/predictor_use_class_input": jnp.asarray(1.0 if predictor_use_class_input else 0.0, dtype=jnp.float32),
        "train/y_ab_norm_mean": y_ab_norm_mean,
        "train/y_ab_norm_min": y_ab_norm_min,
        "train/y_ab_norm_max": y_ab_norm_max,
        "train/delta_m_mae": delta_m_mae,
        "train/delta_m_rmse": delta_m_rmse,
        "train/delta_m_ratio_error": delta_m_ratio_error,
        "train/delta_m_clip_rate": delta_m_clip_rate,
        "train/direct_activation_target_rms_joint": (
            delta_m_mae if shortcut_loss_mode == "direction_activation" else jnp.float32(0.0)
        ),
        "train/direct_activation_pred_rms_joint": (
            delta_m_rmse if shortcut_loss_mode == "direction_activation" else jnp.float32(0.0)
        ),
        "train/direct_activation_rel_residual_rms_joint": (
            delta_m_clip_rate if shortcut_loss_mode == "direction_activation" else jnp.float32(0.0)
        ),
        "train/boot_activation_target_rms": boot_target_rms,
        "train/boot_activation_pred_rms": boot_pred_rms,
        "train/boot_activation_rel_residual_rms": boot_rel_residual_rms,
        "train/direct_gap": direct_gap,
        "train/direct_pair_joint_a": direct_joint_a,
        "train/direct_pair_joint_b": direct_joint_b,
        "train/direct_pair_ponly_1_a": direct_ponly_1_a,
        "train/direct_pair_ponly_1_b": direct_ponly_1_b,
        "train/direct_pair_ponly_2_a": direct_ponly_2_a,
        "train/direct_pair_ponly_2_b": direct_ponly_2_b,
        "train/direct_gap_joint": direct_gap,
        "train/direct_gap_ponly_1": direct_gap_ponly_1,
        "train/direct_gap_ponly_2": direct_gap_ponly_2,
        "train/direct_num_pairs": jnp.asarray(direct_num_pairs, dtype=jnp.float32),
        "train/shortcut_loss_mode": jnp.asarray(
            0.0 if shortcut_loss_mode == "direction_magnitude" else 1.0,
            dtype=jnp.float32,
        ),
        "train/direct_loss_mode": jnp.asarray(
            0.0 if shortcut_loss_mode == "direction_magnitude" else 1.0,
            dtype=jnp.float32,
        ),
        "train/shortcut_activation_huber_delta": jnp.asarray(shortcut_activation_huber_delta, dtype=jnp.float32),
        "train/skip_in_loop_do": skip_do_metric,
        "train/skip_in_loop_prob": skip_prob_eff,
        "train/skip_in_loop_source": skip_a_metric,
        "train/skip_in_loop_target": skip_b_metric,
        "train/skip_in_loop_gap": skip_gap_metric,
        "train/output_distill_a": output_distill_a,
        "train/output_distill_b": output_distill_b,
        "train/output_distill_gap": output_distill_gap,
        "train/output_distill_batch_size": output_distill_batch_size_metric,
        "train/output_distill_update_mode": jnp.asarray(
            {
                "predictor_only": 0.0,
                "predictor_plus_downstream": 1.0,
                "predictor_plus_all": 2.0,
                "predictor_only_then_all": 3.0,
            }[output_distill_update_mode],
            dtype=jnp.float32,
        ),
        "train/output_distill_full_backbone_active": output_distill_full_backbone_active,
        "train/output_distill_full_backbone_start_step": jnp.asarray(
            output_distill_full_backbone_start_step,
            dtype=jnp.float32,
        ),
        "train/output_distill_ratio": jnp.asarray(output_distill_ratio, dtype=jnp.float32),
        "train/pair_uniform_mix": pair_uniform_mix,
        "train/pair_center_loc": jnp.asarray(
            -1.0 if pair_center_loc is None else pair_center_loc,
            dtype=jnp.float32,
        ),
        "train/pair_center_sigma": jnp.asarray(pair_center_sigma, dtype=jnp.float32),
        "train/timestep_sampling_mode": jnp.asarray(
            0.0 if timestep_sampling_mode == "uniform" else 1.0,
            dtype=jnp.float32,
        ),
        "train/timestep_logit_mean": jnp.asarray(timestep_logit_mean, dtype=jnp.float32),
        "train/timestep_logit_std": jnp.asarray(timestep_logit_std, dtype=jnp.float32),
        "train/sampled_timestep_q": sampled_timestep_q,
        "train/sampled_timestep_tau": sampled_timestep_tau,
        "train/pair_uniform_anneal_start_step": jnp.asarray(pair_uniform_anneal_start_step, dtype=jnp.float32),
        "train/pair_uniform_anneal_steps": jnp.asarray(pair_uniform_anneal_steps, dtype=jnp.float32),
        "train/lr_backbone": jnp.asarray(learning_rate, dtype=jnp.float32),
        "train/lr_predictor": jnp.asarray(predictor_learning_rate, dtype=jnp.float32),
        "train/debug_gap_logs_active": debug_gap_logs_now,
        "train/debug_gap_log_freq": jnp.asarray(debug_gap_log_freq, dtype=jnp.float32),
        "train/ema_decay": ema_decay,
        "train/predictor_ema_decay": predictor_ema_decay,
        "train/grad_norm": grad_norm,
        "train/param_norm": param_norm,
        "train/v_abs_mean": v_abs,
        "train/v_pred_abs_mean": v_pred,
    }
    for pair_idx, (pair_a, pair_b) in enumerate(PREDICTOR_DEBUG_PAIRS):
        for metric_idx, metric_name in enumerate(PREDICTOR_DEBUG_METRIC_NAMES):
            metrics[f"train/debug_pair_{pair_a}_{pair_b}/{metric_name}"] = debug_pair_metrics[
                pair_idx * len(PREDICTOR_DEBUG_METRIC_NAMES) + metric_idx
            ]
    gap_offset = len(PREDICTOR_DEBUG_PAIRS) * len(PREDICTOR_DEBUG_METRIC_NAMES)
    for gap in range(1, PREDICTOR_DEBUG_MAX_GAP + 1):
        for metric_idx, metric_name in enumerate(PREDICTOR_DEBUG_METRIC_NAMES):
            metrics[f"train/debug_gap_{gap}/{metric_name}"] = debug_pair_metrics[
                gap_offset + (gap - 1) * len(PREDICTOR_DEBUG_METRIC_NAMES) + metric_idx
            ]
    return state, ema_params, predictor_ema_params, l2_ema, metrics, rng


def eval_step(
    state,
    batch,
    rng,
):
    """Vanilla SiT validation step (mirrors train_step; no grads; no EMA teacher)."""
    x0, y = batch
    local_batch = x0.shape[0]

    rng, tau_rng, noise_rng = jax.random.split(rng, 3)

    tau = jax.random.uniform(tau_rng, shape=(local_batch,), minval=0.0, maxval=1.0)
    x1 = jax.random.normal(noise_rng, x0.shape)

    x_tau = (1.0 - tau[:, None, None]) * x1 + tau[:, None, None] * x0
    target = x0 - x1

    pred = state.apply_fn(
        {"params": state.params["backbone"]},
        x_tau,
        timesteps=tau,
        vector=y,
        deterministic=True,
    )
    loss_gen = jnp.mean((pred - target) ** 2)
    v_abs_mean = jnp.mean(jnp.abs(target))
    v_pred_abs_mean = jnp.mean(jnp.abs(pred))

    loss = jax.lax.pmean(loss_gen, axis_name="batch")
    loss_gen = jax.lax.pmean(loss_gen, axis_name="batch")
    v_abs_mean = jax.lax.pmean(v_abs_mean, axis_name="batch")
    v_pred_abs_mean = jax.lax.pmean(v_pred_abs_mean, axis_name="batch")

    metrics = {
        "val/loss": loss,
        "val/loss_total": loss,
        "val/loss_gen": loss_gen,
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


def scalar_to_host_int(value) -> int:
    """Convert a scalar or replicated scalar JAX value to a Python int."""
    try:
        value = jax_utils.unreplicate(value)
    except Exception:
        pass
    value = np.asarray(jax.device_get(value))
    return int(value.reshape(-1)[0])


PRIVATE_TRAIN_METRIC_KEYS = {
    "train/l_private",
    "train/lambda_private_effective",
    "train/private_warmup_scale",
    "train/common_norm",
    "train/private_avg_norm",
    "train/private_pairwise_cosine",
}


def filter_private_metrics(metrics):
    return {key: value for key, value in metrics.items() if key not in PRIVATE_TRAIN_METRIC_KEYS}


class AsyncWandbLogger:
    """Background thread to log metrics without blocking TPU pipeline."""
    def __init__(self, max_queue_size=50, enabled=True, summary_writer=None, history_writers=None):
        self.enabled = enabled
        self.summary_writer = summary_writer
        self.history_writers = tuple(history_writers or ())
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
                if self.summary_writer is not None:
                    section = "val" if any(str(key).startswith("val/") for key in metrics_cpu) else "train"
                    self.summary_writer.update(metrics_cpu, step=step, section=section)
                for history_writer in self.history_writers:
                    history_writer.update(metrics_cpu, step=step)
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


class WandbRunSummaryFile:
    """Maintains one small JSON file with latest and best scalar metrics."""

    MIN_BEST_KEYS = {
        "val/FID",
        "val/sFID",
        "val/FID_skip_3to7",
        "val/sFID_skip_3to7",
        "train/loss",
        "train/loss_total",
        "train/loss_gen",
    }
    MAX_BEST_KEYS = {
        "val/InceptionScore",
        "val/Precision",
        "val/Recall",
        "val/LinearProbeAcc@1",
    }

    def __init__(self, path, enabled=True):
        self.path = path
        self.enabled = enabled
        self.payload = {
            "updated_at": None,
            "latest_step": 0,
            "latest": {},
            "best": {},
        }
        self._lock = threading.Lock()

    def initialize(self, config=None):
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if config is not None:
            self.payload["config"] = self._jsonify(config)
        self._write_locked()
        if getattr(wandb, "run", None) is not None:
            try:
                wandb.save(self.path, policy="live")
            except Exception as exc:
                log_stage(f"WandB summary file registration failed: {exc}")

    def update(self, metrics, step=None, section="train"):
        if not self.enabled:
            return
        scalar_metrics = {}
        for key, value in metrics.items():
            scalar_value = self._to_scalar(value)
            if scalar_value is not None:
                scalar_metrics[key] = scalar_value
        if not scalar_metrics:
            return

        with self._lock:
            if step is not None:
                self.payload["latest_step"] = int(step)
            self.payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.payload["latest"][section] = scalar_metrics
            self._update_best(scalar_metrics, step)
            self._write_locked()

    def _update_best(self, metrics, step):
        for key, value in metrics.items():
            if key in self.MIN_BEST_KEYS:
                self._maybe_update_best(key, value, step, lower_is_better=True)
            elif key in self.MAX_BEST_KEYS:
                self._maybe_update_best(key, value, step, lower_is_better=False)

    def _maybe_update_best(self, key, value, step, lower_is_better):
        current = self.payload["best"].get(key)
        should_update = current is None
        if current is not None:
            old_value = current["value"]
            should_update = value < old_value if lower_is_better else value > old_value
        if should_update:
            self.payload["best"][key] = {
                "value": value,
                "step": int(step) if step is not None else None,
            }

    def _write_locked(self):
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.path)

    @classmethod
    def _to_scalar(cls, value):
        value = cls._jsonify(value)
        if isinstance(value, (int, float, bool)):
            return float(value) if not isinstance(value, bool) else bool(value)
        return None

    @classmethod
    def _jsonify(cls, value):
        if isinstance(value, dict):
            return {str(k): cls._jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._jsonify(v) for v in value]
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            return value.tolist()
        if hasattr(value, "shape"):
            value = jax.device_get(value)
            return cls._jsonify(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


class WandbMetricHistoryFile:
    """Append selected scalar metrics to one JSONL file for time-series downloads."""

    def __init__(self, path, key_prefixes, active_key=None, enabled=True):
        self.path = path
        self.key_prefixes = tuple(key_prefixes)
        self.active_key = active_key
        self.enabled = enabled
        self._lock = threading.Lock()

    def initialize(self):
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            f.write("")
        if getattr(wandb, "run", None) is not None:
            try:
                wandb.save(self.path, policy="live")
            except Exception as exc:
                log_stage(f"WandB history file registration failed for {self.path}: {exc}")

    def update(self, metrics, step=None):
        if not self.enabled:
            return
        if self.active_key is not None and not bool(metrics.get(self.active_key, False)):
            return
        row = {}
        for key, value in metrics.items():
            if key == self.active_key or any(str(key).startswith(prefix) for prefix in self.key_prefixes):
                scalar_value = WandbRunSummaryFile._to_scalar(value)
                if scalar_value is not None:
                    row[key] = scalar_value
        if not row:
            return
        row["step"] = int(step) if step is not None else int(metrics.get("train/step", 0))
        row["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")


def make_sample_latents_fn(config, num_steps=50, cfg_scale=1.0):
    """Build and JIT a sampling function with num_steps and cfg_scale baked in.

    XLA's scan requires a static sequence length, so num_steps cannot be a
    dynamic argument — it is compiled in.  Provide different values for
    different eval modes:
        - Fast TPU monitoring default: num_steps=50, cfg_scale=1.0
        - Paper-like eval: num_steps=250, cfg_scale=1.0
        (paper eval keeps cfg_scale=1.0; higher CFG remains a deviation)
    """
    if cfg_scale > 1.0 and config.get("class_dropout_prob", 0.1) <= 0.0:
        raise ValueError(
            "CFG sampling requires an unconditional label embedding. "
            "Use --cfg-dropout-rate > 0 for training, or keep cfg scale at 1.0."
        )

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
        class_dropout_prob=config["class_dropout_prob"],
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
                [jnp.full_like(class_labels, config["num_classes"]), class_labels],
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


def make_sample_latents_shortcut_fn(
    config,
    num_steps=50,
    cfg_scale=1.0,
    depth_shortcut_skip_source_layer=3,
    depth_shortcut_skip_target_layer=7,
    depth_shortcut_predict_magnitude=False,
    depth_shortcut_mag_scale=3.0,
    depth_shortcut_mag_abs_center=5.5,
    depth_shortcut_mag_abs_scale=1.5,
    depth_shortcut_mag_clip_min=3.0,
    depth_shortcut_mag_clip_max=8.0,
    shortcut_predictor_variant="tiny",
    shortcut_timesteps=50,
    depth_shortcut_normalize_input=True,
    depth_shortcut_skip_timestep_mode="alternate",
    depth_shortcut_output_mode="direction_magnitude",
    depth_shortcut_predictor_overrides=None,
):
    """Build a non-pmapped sampler for preview images with a 3->7 shortcut."""
    if cfg_scale > 1.0 and config.get("class_dropout_prob", 0.1) <= 0.0:
        raise ValueError(
            "CFG sampling requires an unconditional label embedding. "
            "Use --cfg-dropout-rate > 0 for training, or keep cfg scale at 1.0."
        )

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
        class_dropout_prob=config["class_dropout_prob"],
        per_token=False,
    )

    def sample_latents_shortcut(params, predictor_params, l2_ema, class_labels, rng):
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
        x = rearrange(
            noise,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        ).astype(jnp.float32)
        token_h = latent_size // patch_size
        token_w = latent_size // patch_size

        use_cfg = cfg_scale > 1.0
        if use_cfg:
            x = jnp.concatenate([x, x], axis=0)
            class_labels = jnp.concatenate(
                [jnp.full_like(class_labels, config["num_classes"]), class_labels],
                axis=0,
            )

        def model_fn(z_x, t):
            def apply_full(_):
                return model.apply(
                    {"params": params},
                    z_x,
                    timesteps=t,
                    vector=class_labels,
                    deterministic=True,
                )

            def apply_shortcut(_):
                return model.apply(
                    {"params": params},
                    z_x,
                    timesteps=t,
                    vector=class_labels,
                    deterministic=True,
                    depth_shortcut_predictor_params=predictor_params,
                    depth_shortcut_l2_ema=l2_ema,
                    depth_shortcut_variant=shortcut_predictor_variant,
                    depth_shortcut_skip_every_other=True,
                    depth_shortcut_skip_source_layer=int(depth_shortcut_skip_source_layer),
                    depth_shortcut_skip_target_layer=int(depth_shortcut_skip_target_layer),
                    depth_shortcut_predict_magnitude=depth_shortcut_predict_magnitude,
                    depth_shortcut_mag_scale=float(depth_shortcut_mag_scale),
                    depth_shortcut_mag_abs_center=float(depth_shortcut_mag_abs_center),
                    depth_shortcut_mag_abs_scale=float(depth_shortcut_mag_abs_scale),
                    depth_shortcut_mag_clip_min=float(depth_shortcut_mag_clip_min),
                    depth_shortcut_mag_clip_max=float(depth_shortcut_mag_clip_max),
                    depth_shortcut_timesteps=int(shortcut_timesteps),
                    depth_shortcut_normalize_input=bool(depth_shortcut_normalize_input),
                    depth_shortcut_output_mode=depth_shortcut_output_mode,
                    **(depth_shortcut_predictor_overrides or {}),
                )

            if depth_shortcut_skip_timestep_mode == "all":
                return apply_shortcut(None)
            if depth_shortcut_skip_timestep_mode != "alternate":
                raise ValueError(f"Unknown depth shortcut skip timestep mode: {depth_shortcut_skip_timestep_mode!r}")
            t_scalar = t[0] if t.ndim > 0 else t
            step_idx = jnp.rint(t_scalar * float(num_steps - 1) / 0.96).astype(jnp.int32)
            step_idx = jnp.clip(step_idx, 0, int(num_steps) - 1)
            return jax.lax.cond((step_idx % 2) == 0, apply_shortcut, apply_full, operand=None)

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
            reverse=False,
        )

        if use_cfg:
            samples = samples[batch_size:]
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

    return jax.jit(sample_latents_shortcut)


def make_sample_latents_pmap_fn(
    config,
    num_steps=50,
    cfg_scale=1.0,
    depth_shortcut_skip=False,
    depth_shortcut_skip_source_layer=-1,
    depth_shortcut_skip_target_layer=-1,
    depth_shortcut_predict_magnitude=False,
    depth_shortcut_mag_scale=3.0,
    depth_shortcut_mag_abs_center=5.5,
    depth_shortcut_mag_abs_scale=1.5,
    depth_shortcut_mag_clip_min=3.0,
    depth_shortcut_mag_clip_max=8.0,
    shortcut_predictor_variant="tiny",
    shortcut_timesteps=50,
    depth_shortcut_normalize_input=True,
    depth_shortcut_skip_timestep_mode="alternate",
    depth_shortcut_output_mode="direction_magnitude",
    depth_shortcut_predictor_overrides=None,
):
    """Build a sharded (pmap) sampling function for eval.

    Returns a pmapped function:
      sample_latents_pmap(ema_backbone_params_repl, class_labels_sharded, rng_sharded)
        ema_backbone_params_repl: replicated pytree (devices, ...)
        class_labels_sharded: (devices, local_batch) int32
        rng_sharded: (devices, 2) PRNGKey
      -> latents_sharded: (devices, local_batch, 4, 32, 32) float32
    """
    if cfg_scale > 1.0 and config.get("class_dropout_prob", 0.1) <= 0.0:
        raise ValueError(
            "CFG sampling requires an unconditional label embedding. "
            "Use --cfg-dropout-rate > 0 for training, or keep cfg scale at 1.0."
        )

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
        class_dropout_prob=config["class_dropout_prob"],
        per_token=False,
    )

    patch_size = config["patch_size"]
    latent_channels = config["in_channels"]
    latent_size = config["input_size"]
    token_h = latent_size // patch_size
    token_w = latent_size // patch_size

    def _sample_latents_local(ema_params_local, class_labels_local, rng_local):
        batch_size = class_labels_local.shape[0]
        rng_local, noise_rng = jax.random.split(rng_local)
        noise = jax.random.normal(
            noise_rng,
            (batch_size, latent_channels, latent_size, latent_size),
            dtype=jnp.float32,
        )

        from einops import rearrange
        x = rearrange(
            noise,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        ).astype(jnp.float32)

        use_cfg = cfg_scale > 1.0
        if use_cfg:
            x = jnp.concatenate([x, x], axis=0)
            class_labels_local = jnp.concatenate(
                [jnp.full_like(class_labels_local, config["num_classes"]), class_labels_local],
                axis=0,
            )

        def model_fn(z_x, t):
            return model.apply(
                {"params": ema_params_local},
                z_x,
                timesteps=t,
                vector=class_labels_local,
                deterministic=True,
            )

        rng_local, denoise_rng = jax.random.split(rng_local)
        samples = denoise_loop(
            model_fn=model_fn,
            x=x,
            rng=denoise_rng,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            guidance_low=0.0,
            guidance_high=0.7,
            mode="SDE",
            reverse=False,
        )

        if use_cfg:
            samples = samples[batch_size:]

        samples = rearrange(
            samples,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=token_h,
            w=token_w,
            p1=patch_size,
            p2=patch_size,
            c=latent_channels,
        )
        return samples.astype(jnp.float32)

    def _sample_latents_shortcut_local(
        ema_params_local,
        predictor_params_local,
        l2_ema_local,
        class_labels_local,
        rng_local,
    ):
        batch_size = class_labels_local.shape[0]
        rng_local, noise_rng = jax.random.split(rng_local)
        noise = jax.random.normal(
            noise_rng,
            (batch_size, latent_channels, latent_size, latent_size),
            dtype=jnp.float32,
        )

        from einops import rearrange
        x = rearrange(
            noise,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        ).astype(jnp.float32)

        use_cfg = cfg_scale > 1.0
        if use_cfg:
            x = jnp.concatenate([x, x], axis=0)
            class_labels_local = jnp.concatenate(
                [jnp.full_like(class_labels_local, config["num_classes"]), class_labels_local],
                axis=0,
            )

        def model_fn(z_x, t):
            def apply_full(_):
                return model.apply(
                    {"params": ema_params_local},
                    z_x,
                    timesteps=t,
                    vector=class_labels_local,
                    deterministic=True,
                )

            def apply_shortcut(_):
                return model.apply(
                    {"params": ema_params_local},
                    z_x,
                    timesteps=t,
                    vector=class_labels_local,
                    deterministic=True,
                    depth_shortcut_predictor_params=predictor_params_local,
                    depth_shortcut_l2_ema=l2_ema_local,
                    depth_shortcut_variant=shortcut_predictor_variant,
                    depth_shortcut_skip_every_other=True,
                    depth_shortcut_skip_source_layer=int(depth_shortcut_skip_source_layer),
                    depth_shortcut_skip_target_layer=int(depth_shortcut_skip_target_layer),
                    depth_shortcut_predict_magnitude=depth_shortcut_predict_magnitude,
                    depth_shortcut_mag_scale=float(depth_shortcut_mag_scale),
                    depth_shortcut_mag_abs_center=float(depth_shortcut_mag_abs_center),
                    depth_shortcut_mag_abs_scale=float(depth_shortcut_mag_abs_scale),
                    depth_shortcut_mag_clip_min=float(depth_shortcut_mag_clip_min),
                    depth_shortcut_mag_clip_max=float(depth_shortcut_mag_clip_max),
                    depth_shortcut_timesteps=int(shortcut_timesteps),
                    depth_shortcut_normalize_input=bool(depth_shortcut_normalize_input),
                    depth_shortcut_output_mode=depth_shortcut_output_mode,
                    **(depth_shortcut_predictor_overrides or {}),
                )

            if depth_shortcut_skip_timestep_mode == "all":
                return apply_shortcut(None)
            if depth_shortcut_skip_timestep_mode != "alternate":
                raise ValueError(f"Unknown depth shortcut skip timestep mode: {depth_shortcut_skip_timestep_mode!r}")

            # denoise_loop currently uses SDE last_step_size=0.04, so model calls
            # cover t in [0, 0.96]. Alternate by recovered denoise-step index:
            # even steps use the 3->7 shortcut, odd steps run the full DiT.
            t_scalar = t[0] if t.ndim > 0 else t
            step_idx = jnp.rint(t_scalar * float(num_steps - 1) / 0.96).astype(jnp.int32)
            step_idx = jnp.clip(step_idx, 0, int(num_steps) - 1)
            use_shortcut_step = (step_idx % 2) == 0
            return jax.lax.cond(use_shortcut_step, apply_shortcut, apply_full, operand=None)

        rng_local, denoise_rng = jax.random.split(rng_local)
        samples = denoise_loop(
            model_fn=model_fn,
            x=x,
            rng=denoise_rng,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            guidance_low=0.0,
            guidance_high=0.7,
            mode="SDE",
            reverse=False,
        )

        if use_cfg:
            samples = samples[batch_size:]

        samples = rearrange(
            samples,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=token_h,
            w=token_w,
            p1=patch_size,
            p2=patch_size,
            c=latent_channels,
        )
        return samples.astype(jnp.float32)

    if depth_shortcut_skip:
        return jax.pmap(_sample_latents_shortcut_local, axis_name="batch")
    return jax.pmap(_sample_latents_local, axis_name="batch")


def run_preflight_checks(
    state,
    ema_params,
    rng,
    sample_latents_jitted,
    decode_latents,
    inception_fn,
    real_eval_batch,
    preflight_sample_count,
    preflight_fid_samples,
    inception_num_devices,
    inception_local_batch,
    inception_score_enabled=False,
    inception_score_splits=10,
    precision_recall_enabled=False,
    pr_k=3,
    pr_max_samples=5000,
    pr_full_mode=False,
    get_is_worker=None,
    linear_probe_runner=None,
    block_corr_runner=None,
):
    """Smoke-test eval-side decode/metric plumbing before the training loop.

    Uses EMA params for sampling (same as training-time eval).
    decode_latents : callable(latents_nchw) → NHWC float32 [0,1]
    inception_fn   : pmap'd InceptionV3 (from get_inception_network), or None
    real_eval_batch: tuple(real_latents_patchified, real_labels) or None
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
        if real_eval_batch is None:
            raise RuntimeError("Preflight FID requested but no real latents are available.")
        if inception_fn is None:
            raise RuntimeError("Preflight FID requested but InceptionV3 is not initialised.")

        real_latents_patchified, real_labels = real_eval_batch
        real_count = min(preflight_fid_samples, len(real_latents_patchified))
        fake_count = min(preflight_fid_samples, len(fake_latents))
        fid_count = min(real_count, fake_count)
        if fid_count <= 0:
            raise RuntimeError("Preflight FID requested but there are no samples to compare.")

        real_latents_nchw = unpatchify_patchified_latents(real_latents_patchified[:fid_count])
        real_images = decode_latents(real_latents_nchw)   # (N, H, W, 3) [0,1]
        fake_images = decode_latents(fake_latents[:fid_count])
        real_pooled, real_spatial = extract_inception_features_host_images(
            real_images,
            inception_fn,
            num_devices=inception_num_devices,
            local_batch=inception_local_batch,
            mode="pooled+spatial",
        )
        fake_pooled, fake_spatial = extract_inception_features_host_images(
            fake_images,
            inception_fn,
            num_devices=inception_num_devices,
            local_batch=inception_local_batch,
            mode="pooled+spatial",
        )
        fid_val = fid_from_stats(
            np.mean(real_pooled, 0), np.cov(real_pooled, rowvar=False),
            np.mean(fake_pooled, 0), np.cov(fake_pooled, rowvar=False),
        )
        real_spatial_flat = real_spatial.reshape(-1, real_spatial.shape[-1])
        fake_spatial_flat = fake_spatial.reshape(-1, fake_spatial.shape[-1])
        sfid_val = fid_from_stats(
            np.mean(real_spatial_flat, 0), np.cov(real_spatial_flat, rowvar=False),
            np.mean(fake_spatial_flat, 0), np.cov(fake_spatial_flat, rowvar=False),
        )

        summary_parts = [
            f"Preflight FID={fid_val:.2f}",
            f"sFID={sfid_val:.2f}",
            f"(n={fid_count})",
        ]

        if precision_recall_enabled:
            pr_eval_samples = fid_count if pr_full_mode else min(fid_count, int(pr_max_samples))
            pr_mode = "full" if pr_full_mode else ("subset" if pr_eval_samples < fid_count else "full")
            if pr_eval_samples >= (int(pr_k) + 2):
                prec, rec = precision_recall_knn(
                    real_pooled[:pr_eval_samples],
                    fake_pooled[:pr_eval_samples],
                    k=int(pr_k),
                    chunk=512,
                )
                summary_parts.append(f"PR=({prec:.3f},{rec:.3f})[{pr_mode} n={pr_eval_samples}]")
            else:
                summary_parts.append(
                    f"PR=skipped[{pr_mode} n={pr_eval_samples} < k+2]"
                )

        if inception_score_enabled:
            if get_is_worker is None:
                raise RuntimeError("Preflight Inception Score requested but worker factory is unavailable.")
            probs = np.asarray(get_is_worker().infer(fake_images).probs, dtype=np.float64)
            is_mean, is_std, _ = inception_score_from_probs(probs, splits=int(inception_score_splits))
            summary_parts.append(f"IS={is_mean:.2f}")
            if int(inception_score_splits) > 1:
                summary_parts.append(f"IS_std={is_std:.2f}")

        log_stage("  ".join(summary_parts))

    if real_eval_batch is not None and linear_probe_runner is not None:
        real_latents_patchified, real_labels = real_eval_batch
        lp_acc = float(linear_probe_runner(real_latents_patchified, real_labels))
        log_stage(f"Preflight LinearProbeAcc@1 = {lp_acc:.4f}  (clean EMA representation)")

    if real_eval_batch is not None and block_corr_runner is not None:
        real_latents_patchified, real_labels = real_eval_batch
        corr = np.asarray(block_corr_runner(real_latents_patchified, real_labels), dtype=np.float32)
        offdiag = float((np.sum(corr) - np.trace(corr)) / max(corr.size - corr.shape[0], 1))
        log_stage(
            f"Preflight block correlation OK: {corr.shape[0]}x{corr.shape[1]} Pearson heatmap, "
            f"mean_offdiag={offdiag:.4f}"
        )

    return rng


def main():
    parser = argparse.ArgumentParser(description="Train vanilla SiT DiT (JAX)")
    # ── Core training args ────────────────────────────────────────────────────
    parser.add_argument("--batch-size", type=int, default=256, help="Global batch size (divided by device count)")
    parser.add_argument("--model-size", type=str, default="XL", choices=["S", "B", "L", "XL"], help="DiT backbone size: S, B, L, XL")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Backbone AdamW weight decay. Bias, norm, and embedding params are excluded.")
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
    # ── Depth Shortcut Predictor args ────────────────────────────────────────
    parser.add_argument(
        "--shortcut-predictor",
        "--predictor-variant",
        dest="shortcut_predictor",
        type=str,
        default="tiny",
        choices=predictor_variant_names(),
        help="Depth shortcut predictor variant. Includes convnext_*, dilated_*, and attn_hybrid_* families.",
    )
    parser.add_argument(
        "--shortcut-training-mode",
        type=str,
        default="direction-magnitude",
        choices=["direction", "direction-magnitude", "direction-magnitude-skip"],
        help="Depth-shortcut training preset. Stochastic skip-FM is disabled by default and replaced by output distillation.",
    )
    parser.add_argument("--shortcut-lambda-dir", type=float, default=0.5)
    parser.add_argument("--shortcut-lambda-boot", type=float, default=0.25)
    parser.add_argument(
        "--shortcut-bootstrap-detach-source",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Detach bootstrap source U_a and m_a so bootstrap losses update predictor only, not the DiT backbone through the source state.",
    )
    parser.add_argument("--shortcut-lambda-mag", type=float, default=0.375)
    parser.add_argument("--shortcut-lambda-boot-mag", type=float, default=0.1875)
    parser.add_argument(
        "--shortcut-debug-gap-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log fixed-pair predictor debug losses for representative layer gaps.",
    )
    parser.add_argument(
        "--shortcut-debug-gap-log-freq",
        type=int,
        default=10000,
        help="Run fixed-pair predictor debug gap logging every N training steps when --shortcut-debug-gap-logs is enabled.",
    )
    parser.add_argument("--shortcut-lambda-skip-fm", type=float, default=0.0)
    parser.add_argument("--shortcut-skip-in-loop-prob", type=float, default=0.0)
    parser.add_argument(
        "--shortcut-skip-in-loop-gap-mode",
        type=str,
        default="truncated-normal",
        choices=["fixed", "truncated-normal"],
        help="How to sample skip-in-loop gaps. truncated-normal samples discrete gaps on [1, max_gap] with a right tail.",
    )
    parser.add_argument("--shortcut-skip-in-loop-gap", type=int, default=2)
    parser.add_argument("--shortcut-skip-in-loop-max-gap", type=int, default=10)
    parser.add_argument("--shortcut-skip-in-loop-gap-loc", type=float, default=3.0)
    parser.add_argument("--shortcut-skip-in-loop-gap-sigma", type=float, default=2.0)
    parser.add_argument("--shortcut-skip-in-loop-warmup-steps", type=int, default=5000)
    parser.add_argument(
        "--shortcut-skip-in-loop-detach-source",
        dest="shortcut_skip_in_loop_detach_source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy skip-FM option retained for compatibility; skip-FM is disabled by default.",
    )
    parser.add_argument("--shortcut-mag-scale", type=float, default=3.0)
    parser.add_argument("--shortcut-mag-abs-center", type=float, default=5.5)
    parser.add_argument("--shortcut-mag-abs-scale", type=float, default=1.5)
    parser.add_argument("--shortcut-mag-clip-min", type=float, default=3.0)
    parser.add_argument("--shortcut-mag-clip-max", type=float, default=8.0)
    parser.add_argument("--shortcut-timesteps", type=int, default=50)
    parser.add_argument(
        "--timestep-sampling-mode",
        type=str,
        default="uniform",
        choices=["uniform", "logit_normal", "logit-normal"],
        help="Training timestep schedule. logit_normal samples sigmoid(N(mean, std)) then quantizes to shortcut timesteps.",
    )
    parser.add_argument("--timestep-logit-mean", type=float, default=0.0)
    parser.add_argument("--timestep-logit-std", type=float, default=1.0)
    parser.add_argument(
        "--shortcut-predictor-lr",
        "--predictor-learning-rate",
        dest="shortcut_predictor_lr",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--output-distill",
        "--shortcut-output-distill",
        dest="output_distill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-distill-ratio", type=float, default=0.10)
    parser.add_argument(
        "--lambda-output-distill",
        "--shortcut-lambda-output-distill",
        dest="lambda_output_distill",
        type=float,
        default=0.05,
    )
    parser.add_argument("--output-distill-every", type=int, default=1)
    parser.add_argument(
        "--output-distill-update-mode",
        type=str,
        default="predictor_plus_downstream",
        choices=[
            "predictor_only",
            "predictor_plus_downstream",
            "predictor_plus_all",
            "predictor_only_then_all",
        ],
        help=(
            "Gradient routing for output distillation: predictor_only freezes DiT for this branch; "
            "predictor_plus_downstream updates predictor plus blocks after b; "
            "predictor_plus_all also lets gradients flow through source hidden to blocks before a; "
            "predictor_only_then_all starts as predictor_only and switches to predictor_plus_all."
        ),
    )
    parser.add_argument(
        "--output-distill-full-backbone-start-step",
        type=int,
        default=0,
        help=(
            "For --output-distill-update-mode predictor_only_then_all, switch output distillation "
            "from predictor-only updates to predictor plus full-backbone updates at this global step."
        ),
    )
    parser.add_argument(
        "--output-distill-pair-mode",
        type=str,
        default="trunc_normal_centered",
        choices=[
            "trunc_normal",
            "trunc_normal_to_uniform",
            "trunc_normal_centered",
            "trunc_normal_centered_to_uniform",
            "gap2_biased",
        ],
    )
    parser.add_argument(
        "--pair-uniform-anneal-start-step",
        type=int,
        default=0,
        help="Step where trunc_normal_to_uniform pair sampling starts annealing toward uniform gaps.",
    )
    parser.add_argument(
        "--pair-uniform-anneal-steps",
        type=int,
        default=100000,
        help="Number of steps to anneal pair sampling from truncated-normal gaps to uniform gaps.",
    )
    parser.add_argument("--direct-num-pairs", type=int, default=3)
    parser.add_argument("--direct-joint-pairs", type=int, default=1)
    parser.add_argument("--direct-predictor-only-pairs", type=int, default=2)
    parser.add_argument(
        "--direct-pair-mode",
        type=str,
        default="trunc_normal_centered",
        choices=[
            "trunc_normal",
            "trunc_normal_to_uniform",
            "trunc_normal_centered",
            "trunc_normal_centered_to_uniform",
            "gap2_biased",
        ],
    )
    parser.add_argument(
        "--pair-center-loc",
        type=float,
        default=None,
        help="Layer-index midpoint for centered pair sampling. Default is the middle of the model depth.",
    )
    parser.add_argument(
        "--pair-center-sigma",
        type=float,
        default=2.0,
        help="Stddev, in layer-index units, for centered pair sampling modes.",
    )
    parser.add_argument(
        "--shortcut-loss-mode",
        "--direct-loss-mode",
        dest="shortcut_loss_mode",
        type=str,
        default="direction_magnitude",
        choices=["direction_magnitude", "direction_activation", "direction_activation_huber"],
        help=(
            "Shortcut predictor loss/output mode. direction_magnitude keeps the current direction + log-magnitude losses; "
            "direction_activation treats the predictor output as the target activation and uses Huber auxiliary losses. "
            "direction_activation_huber is accepted as a deprecated alias."
        ),
    )
    parser.add_argument(
        "--shortcut-activation-huber-delta",
        "--direct-activation-huber-delta",
        dest="shortcut_activation_huber_delta",
        type=float,
        default=1.0,
        help="Huber delta used when --shortcut-loss-mode=direction_activation.",
    )
    parser.add_argument("--shortcut-predictor-weight-decay", type=float, default=0.1)
    parser.add_argument("--shortcut-predictor-grad-clip", type=float, default=1.0)
    parser.add_argument("--shortcut-predictor-ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--shortcut-predictor-use-timestep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition the shortcut predictor on the DiT timestep embedding. Default true.",
    )
    parser.add_argument(
        "--shortcut-predictor-normalize-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed L2-normalized source hidden directions into the predictor. Disable to feed raw hidden activations.",
    )
    parser.add_argument(
        "--shortcut-predictor-use-class-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Feed image class labels as an additional shortcut predictor input. Default false.",
    )
    parser.add_argument(
        "--shortcut-predictor-class-fusion",
        type=str,
        default="add",
        choices=["add", "concat"],
        help="How class input is fused with the predictor source tokens when class input is enabled.",
    )
    parser.add_argument(
        "--shortcut-predictor-arch",
        type=str,
        default="existing",
        choices=["existing", "deep_dilated_mlp", "hybrid_deep"],
        help="Override the shortcut predictor architecture while keeping --shortcut-predictor as the base variant.",
    )
    parser.add_argument("--shortcut-predictor-hidden-size", type=int, default=None)
    parser.add_argument("--shortcut-predictor-depth", type=int, default=None)
    parser.add_argument("--shortcut-predictor-mlp-ratio", type=float, default=None)
    parser.add_argument(
        "--shortcut-predictor-dilation-cycle",
        type=str,
        default=None,
        help="Comma-separated dilation cycle, e.g. 1,2,4. Repeated to predictor depth.",
    )
    parser.add_argument("--shortcut-predictor-grid-size", type=int, default=None)
    parser.add_argument(
        "--shortcut-predictor-residual-output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether predictor output is source + delta. Default follows the selected architecture.",
    )
    parser.add_argument("--shortcut-predictor-attention-every", type=int, default=None)
    parser.add_argument("--shortcut-predictor-num-heads", type=int, default=None)
    parser.add_argument(
        "--shortcut-predictor-adaln-zero",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use zero-init AdaLN modulation/gates for deep shortcut predictor blocks.",
    )
    parser.add_argument("--shortcut-l2-ema-alpha", type=float, default=0.01)
    parser.add_argument(
        "--private-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable common/private activation loss and its training logs. Default false.",
    )
    parser.add_argument(
        "--lambda-private",
        type=float,
        default=0.0,
        help="Auxiliary common/private activation loss weight. Default 0 disables it.",
    )
    parser.add_argument(
        "--private-max-pairs",
        type=int,
        default=0,
        help="Maximum number of post-block layer pairs for private loss. 0 means all pairs.",
    )
    parser.add_argument(
        "--private-use-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use common-activation residuals for private loss. Disable to use raw activations.",
    )
    parser.add_argument(
        "--private-cosine-mode",
        type=str,
        default="bnd",
        choices=["bnd", "nd", "token"],
        help=(
            "Private cosine reduction: bnd flattens batch/tokens/channels like the original loss; "
            "nd computes per-sample N*D cosine then averages; token computes tokenwise D cosine then averages."
        ),
    )
    parser.add_argument(
        "--private-pair-mode",
        type=str,
        default="first",
        choices=["first", "random"],
        help="Private loss pair selection: first uses deterministic first upper-triangular pairs; random samples pairs each step.",
    )
    parser.add_argument(
        "--private-start-step",
        type=int,
        default=0,
        help="Global step where private activation loss starts contributing.",
    )
    parser.add_argument(
        "--private-warmup-iters",
        type=int,
        default=0,
        help="Linear warmup steps for --lambda-private after --private-start-step.",
    )
    parser.add_argument(
        "--cfg-dropout-rate",
        type=float,
        default=0.1,
        help=(
            "Classifier-free label dropout rate during training. "
            "Set to 0 to train without CFG dropout/unconditional label embedding."
        ),
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
                        help="CFG scale for sample previews. Default 1.0 (paper setting).")
    parser.add_argument(
        "--sample-skip-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also log sample preview images generated with the shortcut 3->7 sampler.",
    )
    # ── FID args (TPU-friendly defaults; not paper-comparable at defaults) ────
    parser.add_argument("--fid-freq", type=int, default=10000,
                        help="Run FID every N steps (0 disables). "
                             "Default cadence is for TPU monitoring, not paper eval.")
    parser.add_argument("--num-fid-samples", type=int, default=4000,
                        help="Number of real/fake samples for FID. "
                             "TPU default: 4000 (monitoring). Paper: 50000.")
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--fid-eval-local-batch", type=int, default=4,
                        help="Per-device eval micro-batch for FID-bundled metrics. "
                             "Global eval batch = num_devices * fid_eval_local_batch. "
                             "Keep this fixed to avoid XLA recompile; prefer multiples of TPU cores.")
    parser.add_argument(
        "--inception-score",
        dest="inception_score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Inception Score (runs torchvision Inception-v3 in an isolated subprocess).",
    )
    parser.add_argument(
        "--inception-score-weights",
        type=str,
        default=None,
        help=(
            "Optional local path to torchvision Inception-v3 weights "
            "(e.g. inception_v3_google-0cc3c7bd.pth). If set, the IS worker "
            "loads this file directly instead of downloading weights."
        ),
    )
    parser.add_argument("--inception-score-splits", type=int, default=10)
    parser.add_argument(
        "--precision-recall",
        dest="precision_recall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Precision/Recall (kNN-manifold) on pooled Inception features.",
    )
    parser.add_argument("--pr-k", type=int, default=3)
    parser.add_argument("--pr-max-samples", type=int, default=5000,
                        help="Monitoring cap for PR. Uses min(num_fid_samples, pr_max_samples).")
    parser.add_argument("--pr-full-mode", action="store_true",
                        help="Allow PR to run on full num_fid_samples (may be O(N^2) heavy).")
    parser.add_argument(
        "--linear-probe",
        dest="linear_probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable linear probe accuracy inference on val representations (same step, not same FID fake pool).",
    )
    parser.add_argument("--probe-save-path", type=str, default=None,
                        help="Path to .npz containing probe weights (W[, b]). Required for --linear-probe.")
    parser.add_argument("--probe-layer", type=int, default=None,
                        help="Backbone layer index to probe. Default: final backbone block depth.")
    parser.add_argument("--probe-eval-batches", type=int, default=4,
                        help="Number of val batches to run probe inference on per eval window.")
    parser.add_argument("--block-corr-freq", type=int, default=0,
                        help="Cadence (steps) for EMA block correlation heatmap diagnostic. 0 disables.")
    parser.add_argument("--block-corr-batches", type=int, default=2,
                        help="Number of val batches for block correlation diagnostic.")
    parser.add_argument("--fid-num-steps", type=int, default=50,
                        help="Denoising steps for FID generation. "
                             "TPU default: 50 (monitoring). Paper: 250.")
    parser.add_argument("--fid-cfg-scale", type=float, default=1.0,
                        help="CFG scale for FID generation. Default 1.0 (paper uses 1.0).")
    parser.add_argument(
        "--fid-skip-eval",
        dest="fid_skip_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute FID/sFID with a single depth shortcut 3->7 using predictor EMA. Direction mode uses L2 EMA magnitudes; direction-magnitude mode uses the magnitude head.",
    )
    parser.add_argument(
        "--fid-skip-timestep-mode",
        type=str,
        default="all",
        choices=["alternate", "all"],
        help=(
            "Validation FID/sFID 3->7 shortcut schedule. alternate uses the shortcut on even denoise "
            "steps and full DiT on odd steps; all uses the shortcut at every denoise step."
        ),
    )
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
    if args.fid_freq > 0 and args.fid_eval_local_batch <= 0:
        raise ValueError("--fid-eval-local-batch must be greater than 0 when FID is enabled")
    if args.inception_score_splits <= 0:
        raise ValueError("--inception-score-splits must be greater than 0")
    if args.pr_k <= 0:
        raise ValueError("--pr-k must be greater than 0")
    if args.pr_max_samples <= 0:
        raise ValueError("--pr-max-samples must be greater than 0")
    if args.probe_eval_batches <= 0:
        raise ValueError("--probe-eval-batches must be greater than 0")
    if args.linear_probe and not args.probe_save_path:
        raise ValueError("--linear-probe requires --probe-save-path")
    if args.block_corr_freq < 0:
        raise ValueError("--block-corr-freq must be >= 0")
    if args.block_corr_batches <= 0:
        raise ValueError("--block-corr-batches must be > 0")
    if args.vae_decode_batch_size <= 0:
        raise ValueError("--vae-decode-batch-size must be greater than 0")
    if not 0.0 <= args.cfg_dropout_rate <= 1.0:
        raise ValueError("--cfg-dropout-rate must be between 0 and 1")
    if args.cfg_dropout_rate == 0.0:
        if args.sample_cfg_scale > 1.0:
            raise ValueError("--sample-cfg-scale > 1 requires --cfg-dropout-rate > 0")
        if args.fid_cfg_scale > 1.0:
            raise ValueError("--fid-cfg-scale > 1 requires --cfg-dropout-rate > 0")
    if args.shortcut_timesteps <= 0:
        raise ValueError("--shortcut-timesteps must be greater than 0")
    args.timestep_sampling_mode = args.timestep_sampling_mode.replace("-", "_")
    if args.timestep_logit_std <= 0.0:
        raise ValueError("--timestep-logit-std must be greater than 0")
    if args.shortcut_mag_scale <= 0:
        raise ValueError("--shortcut-mag-scale must be greater than 0")
    if args.shortcut_mag_abs_scale <= 0:
        raise ValueError("--shortcut-mag-abs-scale must be greater than 0")
    if args.shortcut_mag_clip_min >= args.shortcut_mag_clip_max:
        raise ValueError("--shortcut-mag-clip-min must be smaller than --shortcut-mag-clip-max")
    if not 0.0 <= args.shortcut_skip_in_loop_prob <= 1.0:
        raise ValueError("--shortcut-skip-in-loop-prob must be between 0 and 1")
    if args.shortcut_skip_in_loop_gap <= 0:
        raise ValueError("--shortcut-skip-in-loop-gap must be greater than 0")
    if args.shortcut_skip_in_loop_max_gap <= 0:
        raise ValueError("--shortcut-skip-in-loop-max-gap must be greater than 0")
    if args.shortcut_skip_in_loop_gap_sigma <= 0:
        raise ValueError("--shortcut-skip-in-loop-gap-sigma must be greater than 0")
    if args.shortcut_skip_in_loop_warmup_steps < 0:
        raise ValueError("--shortcut-skip-in-loop-warmup-steps must be >= 0")
    if not 0.0 <= args.shortcut_l2_ema_alpha <= 1.0:
        raise ValueError("--shortcut-l2-ema-alpha must be between 0 and 1")
    if args.shortcut_predictor_lr <= 0.0:
        raise ValueError("--predictor-learning-rate/--shortcut-predictor-lr must be greater than 0")
    if args.shortcut_debug_gap_log_freq <= 0:
        raise ValueError("--shortcut-debug-gap-log-freq must be greater than 0")
    if args.shortcut_predictor_hidden_size is not None and args.shortcut_predictor_hidden_size <= 0:
        raise ValueError("--shortcut-predictor-hidden-size must be greater than 0")
    if args.shortcut_predictor_depth is not None and args.shortcut_predictor_depth <= 0:
        raise ValueError("--shortcut-predictor-depth must be greater than 0")
    if args.shortcut_predictor_mlp_ratio is not None and args.shortcut_predictor_mlp_ratio <= 0:
        raise ValueError("--shortcut-predictor-mlp-ratio must be greater than 0")
    shortcut_predictor_dilation_cycle = parse_int_cycle(args.shortcut_predictor_dilation_cycle)
    if any(dilation <= 0 for dilation in shortcut_predictor_dilation_cycle):
        raise ValueError("--shortcut-predictor-dilation-cycle values must be positive")
    if args.shortcut_predictor_grid_size is not None and args.shortcut_predictor_grid_size <= 0:
        raise ValueError("--shortcut-predictor-grid-size must be greater than 0")
    if args.shortcut_predictor_attention_every is not None and args.shortcut_predictor_attention_every < 0:
        raise ValueError("--shortcut-predictor-attention-every must be non-negative")
    if args.shortcut_predictor_num_heads is not None and args.shortcut_predictor_num_heads <= 0:
        raise ValueError("--shortcut-predictor-num-heads must be greater than 0")
    if not 0.0 <= args.output_distill_ratio <= 1.0:
        raise ValueError("--output-distill-ratio must be between 0 and 1")
    if args.lambda_output_distill < 0.0:
        raise ValueError("--lambda-output-distill must be non-negative")
    if args.output_distill_every <= 0:
        raise ValueError("--output-distill-every must be greater than 0")
    if args.output_distill_full_backbone_start_step < 0:
        raise ValueError("--output-distill-full-backbone-start-step must be non-negative")
    if args.pair_uniform_anneal_start_step < 0:
        raise ValueError("--pair-uniform-anneal-start-step must be non-negative")
    if args.pair_uniform_anneal_steps <= 0:
        raise ValueError("--pair-uniform-anneal-steps must be greater than 0")
    if args.pair_center_sigma < 0.0:
        raise ValueError("--pair-center-sigma must be non-negative")
    if args.direct_joint_pairs != 1 or args.direct_predictor_only_pairs not in {0, 1, 2}:
        raise ValueError("--direct-joint-pairs must be 1 and --direct-predictor-only-pairs must be 0, 1, or 2")
    if args.direct_num_pairs != args.direct_joint_pairs + args.direct_predictor_only_pairs:
        raise ValueError("--direct-num-pairs must equal --direct-joint-pairs + --direct-predictor-only-pairs")
    if args.direct_num_pairs not in {1, 2, 3}:
        raise ValueError("--direct-num-pairs supports static values 1, 2, or 3")
    if args.shortcut_loss_mode == "direction_activation_huber":
        args.shortcut_loss_mode = "direction_activation"
    if args.shortcut_activation_huber_delta <= 0.0:
        raise ValueError("--shortcut-activation-huber-delta must be greater than 0")

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
    output_distill_local_batch_size = max(1, int(round(args.output_distill_ratio * local_batch_size)))
    output_distill_local_batch_size = min(output_distill_local_batch_size, local_batch_size)
    log_stage(f"TPU Cores: {num_devices}. Global Batch: {args.batch_size}, Local Batch: {local_batch_size}")

    # ── Model config ─────────────────────────────────────────────────────────
    config = build_model_config(args.model_size, class_dropout_prob=args.cfg_dropout_rate)
    depth = int(config["depth"])
    shortcut_predictor_overrides = {
        "arch": args.shortcut_predictor_arch,
        "hidden_size": args.shortcut_predictor_hidden_size,
        "depth": args.shortcut_predictor_depth,
        "mlp_ratio": args.shortcut_predictor_mlp_ratio,
        "dilation_cycle": shortcut_predictor_dilation_cycle,
        "grid_size": args.shortcut_predictor_grid_size,
        "residual_output": args.shortcut_predictor_residual_output,
        "attention_every": args.shortcut_predictor_attention_every,
        "num_heads": args.shortcut_predictor_num_heads,
        "adaln_zero": args.shortcut_predictor_adaln_zero,
    }
    depth_shortcut_predictor_overrides = {
        "depth_shortcut_predictor_arch": args.shortcut_predictor_arch,
        "depth_shortcut_predictor_hidden_size": args.shortcut_predictor_hidden_size,
        "depth_shortcut_predictor_depth": args.shortcut_predictor_depth,
        "depth_shortcut_predictor_mlp_ratio": args.shortcut_predictor_mlp_ratio,
        "depth_shortcut_predictor_dilation_cycle": shortcut_predictor_dilation_cycle,
        "depth_shortcut_predictor_grid_size": args.shortcut_predictor_grid_size,
        "depth_shortcut_predictor_residual_output": args.shortcut_predictor_residual_output,
        "depth_shortcut_predictor_attention_every": args.shortcut_predictor_attention_every,
        "depth_shortcut_predictor_num_heads": args.shortcut_predictor_num_heads,
        "depth_shortcut_predictor_adaln_zero": args.shortcut_predictor_adaln_zero,
        "depth_shortcut_predictor_use_class_input": args.shortcut_predictor_use_class_input,
        "depth_shortcut_predictor_class_fusion": args.shortcut_predictor_class_fusion,
    }
    if args.shortcut_skip_in_loop_gap > depth:
        raise ValueError("--shortcut-skip-in-loop-gap must be <= model depth")
    if args.shortcut_skip_in_loop_max_gap > depth:
        raise ValueError("--shortcut-skip-in-loop-max-gap must be <= model depth")
    if args.private_loss and args.lambda_private <= 0.0:
        raise ValueError("--private-loss requires --lambda-private > 0")
    if args.lambda_private < 0.0:
        raise ValueError("--lambda-private must be non-negative")
    if args.private_max_pairs < 0:
        raise ValueError("--private-max-pairs must be non-negative")
    if args.private_cosine_mode not in {"bnd", "nd", "token"}:
        raise ValueError("--private-cosine-mode must be one of: bnd, nd, token")
    if args.private_pair_mode not in {"first", "random"}:
        raise ValueError("--private-pair-mode must be one of: first, random")
    if args.private_start_step < 0:
        raise ValueError("--private-start-step must be non-negative")
    if args.private_warmup_iters < 0:
        raise ValueError("--private-warmup-iters must be non-negative")

    log_stage(
        f"Model=DiT-{args.model_size.upper()} hidden={config['hidden_size']} "
        f"depth={depth} heads={config['num_heads']}"
    )
    log_stage(
        f"Vanilla SiT: ema_decay={args.ema_decay} grad_clip={args.grad_clip} "
        f"weight_decay={args.weight_decay} cfg_dropout_rate={args.cfg_dropout_rate}"
    )
    log_stage(
        f"DepthShortcut: predictor={args.shortcut_predictor} "
        f"mode={args.shortcut_training_mode} "
        f"lambda_dir={args.shortcut_lambda_dir} lambda_boot={args.shortcut_lambda_boot} "
        f"bootstrap_detach_source={args.shortcut_bootstrap_detach_source} "
        f"lambda_mag={args.shortcut_lambda_mag} lambda_boot_mag={args.shortcut_lambda_boot_mag} "
        f"debug_gap_logs={args.shortcut_debug_gap_logs} "
        f"debug_gap_log_freq={args.shortcut_debug_gap_log_freq} "
        f"lambda_skip_fm={args.shortcut_lambda_skip_fm} "
        f"skip_p={args.shortcut_skip_in_loop_prob} "
        f"skip_gap_mode={args.shortcut_skip_in_loop_gap_mode} "
        f"skip_gap={args.shortcut_skip_in_loop_gap} "
        f"skip_max_gap={args.shortcut_skip_in_loop_max_gap} "
        f"skip_gap_loc={args.shortcut_skip_in_loop_gap_loc} "
        f"skip_gap_sigma={args.shortcut_skip_in_loop_gap_sigma} "
        f"skip_warmup={args.shortcut_skip_in_loop_warmup_steps} "
        f"skip_detach_source={args.shortcut_skip_in_loop_detach_source} "
        f"timestep_sampling={args.timestep_sampling_mode} "
        f"timestep_logit=({args.timestep_logit_mean},{args.timestep_logit_std}) "
        f"output_distill={args.output_distill} "
        f"output_distill_ratio={args.output_distill_ratio} "
        f"output_distill_local_batch={output_distill_local_batch_size} "
        f"lambda_output_distill={args.lambda_output_distill} "
        f"output_distill_every={args.output_distill_every} "
        f"output_distill_update_mode={args.output_distill_update_mode} "
        f"output_distill_full_backbone_start_step={args.output_distill_full_backbone_start_step} "
        f"output_distill_pair_mode={args.output_distill_pair_mode} "
        f"pair_uniform_anneal=({args.pair_uniform_anneal_start_step},{args.pair_uniform_anneal_steps}) "
        f"pair_center=({args.pair_center_loc},{args.pair_center_sigma}) "
        f"direct_pairs=({args.direct_joint_pairs} joint,{args.direct_predictor_only_pairs} predictor_only) "
        f"direct_pair_mode={args.direct_pair_mode} "
        f"shortcut_loss_mode={args.shortcut_loss_mode} "
        f"shortcut_activation_huber_delta={args.shortcut_activation_huber_delta} "
        f"mag_scale={args.shortcut_mag_scale} "
        f"mag_abs=({args.shortcut_mag_abs_center},{args.shortcut_mag_abs_scale}) "
        f"mag_clip=({args.shortcut_mag_clip_min},{args.shortcut_mag_clip_max}) "
        f"lr_backbone={args.learning_rate} pred_lr={args.shortcut_predictor_lr} "
        f"pred_wd={args.shortcut_predictor_weight_decay} "
        f"pred_use_t={args.shortcut_predictor_use_timestep} "
        f"pred_norm_input={args.shortcut_predictor_normalize_input} "
        f"pred_use_class_input={args.shortcut_predictor_use_class_input} "
        f"pred_class_fusion={args.shortcut_predictor_class_fusion} "
        f"pred_arch={args.shortcut_predictor_arch} "
        f"pred_hidden={args.shortcut_predictor_hidden_size} "
        f"pred_depth={args.shortcut_predictor_depth} "
        f"pred_mlp_ratio={args.shortcut_predictor_mlp_ratio} "
        f"pred_dilation_cycle={shortcut_predictor_dilation_cycle} "
        f"pred_grid={args.shortcut_predictor_grid_size} "
        f"pred_residual_output={args.shortcut_predictor_residual_output} "
        f"pred_attention_every={args.shortcut_predictor_attention_every} "
        f"pred_heads={args.shortcut_predictor_num_heads} "
        f"pred_adaln_zero={args.shortcut_predictor_adaln_zero}"
    )
    log_stage(
        f"PrivateActivations: enabled={args.private_loss} lambda_private={args.lambda_private} "
        f"max_pairs={args.private_max_pairs} start_step={args.private_start_step} "
        f"warmup_iters={args.private_warmup_iters} use_residual={args.private_use_residual} "
        f"cosine_mode={args.private_cosine_mode} pair_mode={args.private_pair_mode}"
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
    summary_writer = WandbRunSummaryFile(
        os.path.join(args.ckpt_dir, "wandb_run_summary.json"),
        enabled=not args.no_wandb,
    )
    summary_writer.initialize(config=vars(args))
    debug_gap_history_writer = WandbMetricHistoryFile(
        os.path.join(args.ckpt_dir, "wandb_debug_gap_history.jsonl"),
        key_prefixes=("train/debug_pair_", "train/debug_gap_"),
        active_key="train/debug_gap_logs_active",
        enabled=not args.no_wandb,
    )
    debug_gap_history_writer.initialize()
    logger = AsyncWandbLogger(
        enabled=not args.no_wandb,
        summary_writer=summary_writer,
        history_writers=(debug_gap_history_writer,),
    )

    # ── Model, state, EMA ─────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(42)
    state, ema_params, predictor_ema_params, l2_ema = create_train_state(
        rng,
        config,
        args.learning_rate,
        args.grad_clip,
        weight_decay=args.weight_decay,
        predictor_variant=args.shortcut_predictor,
        predictor_lr=args.shortcut_predictor_lr,
        predictor_weight_decay=args.shortcut_predictor_weight_decay,
        predictor_grad_clip=args.shortcut_predictor_grad_clip,
        shortcut_training_mode=args.shortcut_training_mode,
        shortcut_mag_abs_center=args.shortcut_mag_abs_center,
        shortcut_mag_abs_scale=args.shortcut_mag_abs_scale,
        predictor_config_overrides=shortcut_predictor_overrides,
        predictor_use_class_input=args.shortcut_predictor_use_class_input,
        predictor_class_fusion=args.shortcut_predictor_class_fusion,
    )
    predictor_param_count = count_tree_params(state.params["predictor"])
    backbone_param_count = count_tree_params(state.params["backbone"])
    total_param_count = count_tree_params(state.params)
    predictor_bucket = (
        "large"
        if args.shortcut_predictor_arch in {"deep_dilated_mlp", "hybrid_deep"}
        else predictor_size_bucket(args.shortcut_predictor)
    )
    predictor_range = PREDICTOR_PARAM_TARGET_RANGES[predictor_bucket]
    predictor_cfg = predictor_config_from_name(args.shortcut_predictor, config["hidden_size"])
    predictor_cfg = apply_predictor_config_overrides(predictor_cfg, **shortcut_predictor_overrides)
    log_stage(
        f"DepthShortcut params: predictor={predictor_param_count:,} "
        f"backbone={backbone_param_count:,} total={total_param_count:,} "
        f"cfg={predictor_cfg}"
    )
    if not (predictor_range[0] <= predictor_param_count <= predictor_range[1]):
        log_stage(
            f"WARNING: predictor params {predictor_param_count:,} are outside "
            f"{predictor_bucket} target range [{predictor_range[0]:,}, {predictor_range[1]:,}]"
        )
    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(ema_params)
    predictor_ema_params = jax_utils.replicate(predictor_ema_params)
    l2_ema = jax_utils.replicate(l2_ema)
    rng = jax.random.split(rng, num_devices)
    ema_decay_rep = jax_utils.replicate(jnp.float32(args.ema_decay))
    predictor_ema_decay_rep = jax_utils.replicate(jnp.float32(args.shortcut_predictor_ema_decay))
    shortcut_timesteps_rep = jax_utils.replicate(jnp.int32(args.shortcut_timesteps))
    shortcut_lambda_dir_rep = jax_utils.replicate(jnp.float32(args.shortcut_lambda_dir))
    shortcut_lambda_boot_rep = jax_utils.replicate(jnp.float32(args.shortcut_lambda_boot))
    shortcut_bootstrap_detach_source_rep = jax_utils.replicate(jnp.asarray(args.shortcut_bootstrap_detach_source, dtype=jnp.bool_))
    uses_magnitude_losses = True
    effective_lambda_mag = args.shortcut_lambda_mag
    effective_lambda_boot_mag = args.shortcut_lambda_boot_mag
    effective_lambda_skip_fm = 0.0
    effective_skip_prob = 0.0
    shortcut_lambda_mag_rep = jax_utils.replicate(jnp.float32(effective_lambda_mag))
    shortcut_lambda_boot_mag_rep = jax_utils.replicate(jnp.float32(effective_lambda_boot_mag))
    shortcut_lambda_skip_fm_rep = jax_utils.replicate(jnp.float32(effective_lambda_skip_fm))
    shortcut_skip_in_loop_prob_rep = jax_utils.replicate(jnp.float32(effective_skip_prob))
    shortcut_skip_in_loop_gap_mode_rep = jax_utils.replicate(
        jnp.int32(0 if args.shortcut_skip_in_loop_gap_mode == "fixed" else 1)
    )
    shortcut_skip_in_loop_gap_rep = jax_utils.replicate(jnp.int32(args.shortcut_skip_in_loop_gap))
    shortcut_skip_in_loop_max_gap_rep = jax_utils.replicate(jnp.int32(args.shortcut_skip_in_loop_max_gap))
    shortcut_skip_in_loop_gap_loc_rep = jax_utils.replicate(jnp.float32(args.shortcut_skip_in_loop_gap_loc))
    shortcut_skip_in_loop_gap_sigma_rep = jax_utils.replicate(jnp.float32(args.shortcut_skip_in_loop_gap_sigma))
    shortcut_skip_in_loop_warmup_steps_rep = jax_utils.replicate(jnp.int32(args.shortcut_skip_in_loop_warmup_steps))
    shortcut_skip_in_loop_detach_source_rep = jax_utils.replicate(jnp.asarray(args.shortcut_skip_in_loop_detach_source, dtype=jnp.bool_))
    lambda_output_distill_rep = jax_utils.replicate(jnp.float32(args.lambda_output_distill))
    private_loss_enabled_rep = jax_utils.replicate(jnp.asarray(args.private_loss, dtype=jnp.bool_))
    effective_lambda_private = args.lambda_private if args.private_loss else 0.0
    lambda_private_rep = jax_utils.replicate(jnp.float32(effective_lambda_private))
    private_max_pairs_rep = jax_utils.replicate(jnp.int32(args.private_max_pairs))
    private_start_step_rep = jax_utils.replicate(jnp.int32(args.private_start_step))
    private_warmup_iters_rep = jax_utils.replicate(jnp.int32(args.private_warmup_iters))
    shortcut_debug_gap_logs_rep = jax_utils.replicate(jnp.asarray(args.shortcut_debug_gap_logs, dtype=jnp.bool_))
    shortcut_mag_scale_rep = jax_utils.replicate(jnp.float32(args.shortcut_mag_scale))
    shortcut_mag_clip_min_rep = jax_utils.replicate(jnp.float32(args.shortcut_mag_clip_min))
    shortcut_mag_clip_max_rep = jax_utils.replicate(jnp.float32(args.shortcut_mag_clip_max))
    shortcut_l2_ema_alpha_rep = jax_utils.replicate(jnp.float32(args.shortcut_l2_ema_alpha))

    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2

    total_steps = args.epochs * args.steps_per_epoch

    # ── FLOPs estimate (cached once; used for TFLOPs logging) ─────────────────
    # This is a rough throughput metric for monitoring. It intentionally avoids
    # any per-step cost analysis. Accumulated TFLOPs count only training steps.
    def estimate_train_step_flops(cfg, global_batch, n_tokens):
        # Heuristic DiT block cost: attention + MLP ~ O(D^2 * N) per layer.
        D = int(cfg["hidden_size"])
        L = int(cfg["depth"])
        # Scale factors chosen for stable relative monitoring; not paper-accurate.
        flops_per_layer = 12.0 * n_tokens * (D * D)
        fwd = flops_per_layer * L
        bwd = 2.0 * fwd  # backward ~2x forward (rough)
        return float(global_batch) * (fwd + bwd)

    flops_per_train_step = estimate_train_step_flops(config, args.batch_size, n_patches)
    accumulated_train_tflops = 0.0

    # ── Build pmapped training/eval steps ────────────────────────────────────
    pmapped_train_step = jax.pmap(
        functools.partial(
            train_step,
            use_output_distill=args.output_distill,
            output_distill_batch_size=output_distill_local_batch_size,
            output_distill_ratio=args.output_distill_ratio,
            output_distill_every=args.output_distill_every,
            output_distill_update_mode=args.output_distill_update_mode,
            output_distill_full_backbone_start_step=args.output_distill_full_backbone_start_step,
            output_distill_pair_mode=args.output_distill_pair_mode,
            pair_uniform_anneal_start_step=args.pair_uniform_anneal_start_step,
            pair_uniform_anneal_steps=args.pair_uniform_anneal_steps,
            direct_num_pairs=args.direct_num_pairs,
            direct_num_joint_pairs=args.direct_joint_pairs,
            direct_num_predictor_only_pairs=args.direct_predictor_only_pairs,
            direct_pair_mode=args.direct_pair_mode,
            shortcut_loss_mode=args.shortcut_loss_mode,
            shortcut_activation_huber_delta=args.shortcut_activation_huber_delta,
            debug_gap_log_freq=args.shortcut_debug_gap_log_freq,
            private_use_residual=args.private_use_residual,
            private_cosine_mode=args.private_cosine_mode,
            private_pair_mode=args.private_pair_mode,
            predictor_use_timestep=args.shortcut_predictor_use_timestep,
            predictor_normalize_input=args.shortcut_predictor_normalize_input,
            predictor_use_class_input=args.shortcut_predictor_use_class_input,
            learning_rate=args.learning_rate,
            predictor_learning_rate=args.shortcut_predictor_lr,
            timestep_sampling_mode=args.timestep_sampling_mode,
            timestep_logit_mean=args.timestep_logit_mean,
            timestep_logit_std=args.timestep_logit_std,
            pair_center_loc=args.pair_center_loc,
            pair_center_sigma=args.pair_center_sigma,
        ),
        axis_name="batch",
    )
    pmapped_eval_step = jax.pmap(eval_step, axis_name="batch")

    # ── Sample function: num_steps and cfg_scale baked in at JIT time ─────────
    # TPU deviation: default 50 steps for fast monitoring; paper uses 250.
    # Default remains cfg_scale=1.0 to match the paper evaluation setting.
    sample_latents_jitted = make_sample_latents_fn(
        config, num_steps=args.sample_num_steps, cfg_scale=args.sample_cfg_scale
    )
    sample_latents_skip_jitted = None
    if args.sample_skip_eval:
        sample_latents_skip_jitted = make_sample_latents_shortcut_fn(
            config,
            num_steps=args.sample_num_steps,
            cfg_scale=args.sample_cfg_scale,
            depth_shortcut_skip_source_layer=3,
            depth_shortcut_skip_target_layer=7,
            depth_shortcut_predict_magnitude=uses_magnitude_losses,
            depth_shortcut_mag_scale=args.shortcut_mag_scale,
            depth_shortcut_mag_abs_center=args.shortcut_mag_abs_center,
            depth_shortcut_mag_abs_scale=args.shortcut_mag_abs_scale,
            depth_shortcut_mag_clip_min=args.shortcut_mag_clip_min,
            depth_shortcut_mag_clip_max=args.shortcut_mag_clip_max,
            shortcut_predictor_variant=args.shortcut_predictor,
            shortcut_timesteps=args.shortcut_timesteps,
            depth_shortcut_normalize_input=args.shortcut_predictor_normalize_input,
            depth_shortcut_skip_timestep_mode=args.fid_skip_timestep_mode,
            depth_shortcut_output_mode=args.shortcut_loss_mode,
            depth_shortcut_predictor_overrides=depth_shortcut_predictor_overrides,
        )
    # Separate function for FID generation (may differ in num_steps/cfg_scale)
    if args.fid_num_steps != args.sample_num_steps or args.fid_cfg_scale != args.sample_cfg_scale:
        fid_sample_latents_jitted = make_sample_latents_fn(
            config, num_steps=args.fid_num_steps, cfg_scale=args.fid_cfg_scale
        )
    else:
        fid_sample_latents_jitted = sample_latents_jitted

    # Sharded (pmap) sampler for eval hot-path (avoid single-device bottleneck)
    fid_sample_latents_pmapped = make_sample_latents_pmap_fn(
        config, num_steps=args.fid_num_steps, cfg_scale=args.fid_cfg_scale
    )
    fid_sample_latents_skip_pmapped = None
    if args.fid_skip_eval:
        fid_sample_latents_skip_pmapped = make_sample_latents_pmap_fn(
            config,
            num_steps=args.fid_num_steps,
            cfg_scale=args.fid_cfg_scale,
            depth_shortcut_skip=True,
            depth_shortcut_skip_source_layer=3,
            depth_shortcut_skip_target_layer=7,
            depth_shortcut_predict_magnitude=uses_magnitude_losses,
            depth_shortcut_mag_scale=args.shortcut_mag_scale,
            depth_shortcut_mag_abs_center=args.shortcut_mag_abs_center,
            depth_shortcut_mag_abs_scale=args.shortcut_mag_abs_scale,
            depth_shortcut_mag_clip_min=args.shortcut_mag_clip_min,
            depth_shortcut_mag_clip_max=args.shortcut_mag_clip_max,
            shortcut_predictor_variant=args.shortcut_predictor,
            shortcut_timesteps=args.shortcut_timesteps,
            depth_shortcut_normalize_input=args.shortcut_predictor_normalize_input,
            depth_shortcut_skip_timestep_mode=args.fid_skip_timestep_mode,
            depth_shortcut_output_mode=args.shortcut_loss_mode,
            depth_shortcut_predictor_overrides=depth_shortcut_predictor_overrides,
        )

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

    def decode_latents_sharded(latents_nchw_sharded):
        """Sharded decode: (devices, local_batch, 4, 32, 32) → (devices, local_batch, 256, 256, 3).

        Uses TPU Flax backend if available. If using CPU subprocess backend, this
        will host-transfer latents and return images sharded back.
        """
        backend = _ensure_vae_backend()
        if isinstance(backend, VAEDecodeSubprocess):
            # Host bounce fallback: bring latents to host, decode, then reshape back.
            latents = np.asarray(jax.device_get(latents_nchw_sharded), dtype=np.float32).reshape(-1, 4, 32, 32)
            imgs = backend.decode(latents)  # (global, 256,256,3)
            return jnp.array(imgs.reshape(num_devices, -1, 256, 256, 3), dtype=jnp.float32)
        decode_fn, params_repl = backend
        latents_bf16 = latents_nchw_sharded.astype(jnp.bfloat16)
        return decode_fn(latents_bf16, params_repl)

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

    # ── InceptionV3 for FID/sFID: lazy-init, cached per mode ──────────────────
    _inception_fns = {}  # mode -> apply_fn

    def get_inception(mode="pooled"):
        from src.fid_utils import get_inception_network
        mode = str(mode)
        if mode not in _inception_fns:
            log_stage(f"Loading InceptionV3 ({mode})…")
            _inception_fns[mode] = get_inception_network(mode=mode)
            log_stage("InceptionV3 ready.")
        return _inception_fns[mode]

    # ── Torchvision Inception-v3 worker for Inception Score (subprocess) ──────
    _is_worker = [None]

    def get_is_worker():
        if _is_worker[0] is None:
            log_stage("Spawning torchvision Inception-v3 worker for Inception Score…")
            _is_worker[0] = InceptionISSubprocess(weights_path=args.inception_score_weights)
            log_stage("Inception Score worker ready.")
        return _is_worker[0]

    # ── Linear probe weights + pmapped inference (inference-only in fid window) ─
    _probe_weights = [None]  # (W_repl, b_repl)
    _probe_pmapped = {}      # layer -> pmapped_fn

    def get_probe_weights():
        if _probe_weights[0] is None:
            data = np.load(args.probe_save_path)
            if "W" not in data:
                raise ValueError(f"Probe file missing key 'W': {args.probe_save_path!r}")
            W = np.asarray(data["W"], dtype=np.float32)
            b = np.asarray(data["b"], dtype=np.float32) if "b" in data else np.zeros((W.shape[1],), dtype=np.float32)
            W_repl = jax.device_put_replicated(jnp.array(W), jax.local_devices())
            b_repl = jax.device_put_replicated(jnp.array(b), jax.local_devices())
            _probe_weights[0] = (W_repl, b_repl)
        return _probe_weights[0]

    def get_probe_pmapped(layer: int):
        layer = int(layer)
        if layer not in _probe_pmapped:
            def _probe_step(ema_params_local, batch_x_local, batch_y_local, W_local, b_local):
                # Deterministic clean EMA representation for monitoring.
                local_batch = batch_x_local.shape[0]
                clean_t = jnp.ones((local_batch,), dtype=jnp.float32)
                _, feats = state.apply_fn(
                    {"params": ema_params_local},
                    batch_x_local,
                    timesteps=clean_t,
                    vector=batch_y_local,
                    deterministic=True,
                    return_raw_features=layer,
                )
                reps = jnp.mean(feats, axis=1)  # (B, D)
                logits = reps @ W_local + b_local
                pred = jnp.argmax(logits, axis=-1)
                correct = jnp.sum(pred == batch_y_local)
                total = jnp.array(batch_y_local.shape[0], dtype=jnp.int32)
                correct = jax.lax.psum(correct, axis_name="batch")
                total = jax.lax.psum(total, axis_name="batch")
                return correct, total

            _probe_pmapped[layer] = jax.pmap(_probe_step, axis_name="batch")
        return _probe_pmapped[layer]

    # ── EMA block-correlation diagnostic (cadence riêng; async media logging) ─
    _blockcorr_pmapped = [None]

    def get_blockcorr_pmapped():
        if _blockcorr_pmapped[0] is None:
            def _blockcorr_step(ema_params_local, batch_x_local, batch_y_local):
                local_batch = batch_x_local.shape[0]
                clean_t = jnp.ones((local_batch,), dtype=jnp.float32)
                out = state.apply_fn(
                    {"params": ema_params_local},
                    batch_x_local,
                    timesteps=clean_t,
                    vector=batch_y_local,
                    deterministic=True,
                    return_block_summaries=True,
                )
                _, block_summaries = out
                return block_summaries

            _blockcorr_pmapped[0] = jax.pmap(_blockcorr_step, axis_name="batch")
        return _blockcorr_pmapped[0]

    def log_blockcorr_async(corr: np.ndarray, step: int):
        # Avoid blocking the training loop with WandB media encoding/upload.
        def _worker():
            if getattr(wandb, "run", None) is None:
                return
            corr_clip = np.clip(corr, -1.0, 1.0)
            img = ((corr_clip + 1.0) * 0.5 * 255.0).astype(np.uint8)
            img = np.stack([img, img, img], axis=-1)  # HWC
            safe_wandb_log(
                {
                    "diag/block_corr_mean_offdiag": float(
                        (np.sum(corr) - np.trace(corr)) / max(corr.size - corr.shape[0], 1)
                    ),
                    "diag/block_corr_heatmap": wandb.Image(img, caption=f"step {step}"),
                    "train/step": step,
                },
                step=step,
            )

        threading.Thread(target=_worker, daemon=True).start()

    def run_preflight_linear_probe(batch_x_patchified, batch_y):
        probe_layer = int(args.probe_layer if args.probe_layer is not None else depth)
        W_repl, b_repl = get_probe_weights()
        probe_fn = get_probe_pmapped(probe_layer)
        bx = jnp.array(batch_x_patchified).reshape(num_devices, local_batch_size, n_patches, patch_dim)
        by = jnp.array(batch_y).reshape(num_devices, local_batch_size)
        corr, tot = probe_fn(ema_params, bx, by, W_repl, b_repl)
        correct_total = int(jax.device_get(corr[0]))
        count_total = int(jax.device_get(tot[0]))
        if count_total <= 0:
            raise RuntimeError("Preflight linear probe produced zero samples.")
        return float(correct_total / count_total)

    def run_preflight_block_corr(batch_x_patchified, batch_y):
        bc_fn = get_blockcorr_pmapped()
        bx = jnp.array(batch_x_patchified).reshape(num_devices, local_batch_size, n_patches, patch_dim)
        by = jnp.array(batch_y).reshape(num_devices, local_batch_size)
        summaries = bc_fn(ema_params, bx, by)  # (devices, local_batch, depth, hidden)
        summaries_h = np.asarray(jax.device_get(summaries), dtype=np.float32).transpose(2, 0, 1, 3)
        stacked = summaries_h.reshape(summaries_h.shape[0], -1, summaries_h.shape[-1])
        return pearson_corrcoef_rows(stacked.reshape(stacked.shape[0], -1))

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
        inception_local_batch = int(args.fid_eval_local_batch)

        log_stage("[FID probe] running one discarded train step to match training-time memory...")
        probe_x = jnp.array(cached_train_batch[0]).reshape(num_devices, local_batch_size, n_patches, patch_dim)
        probe_y = jnp.array(cached_train_batch[1]).reshape(num_devices, local_batch_size)
        _, _, _, _, probe_metrics, _ = pmapped_train_step(
            state,
            ema_params,
            predictor_ema_params,
            l2_ema,
            (probe_x, probe_y),
            rng,
            jax_utils.replicate(jnp.int32(0)),
            ema_decay_rep,
            predictor_ema_decay_rep,
            shortcut_timesteps_rep,
            shortcut_lambda_dir_rep,
            shortcut_lambda_boot_rep,
            shortcut_bootstrap_detach_source_rep,
            shortcut_lambda_mag_rep,
            shortcut_lambda_boot_mag_rep,
            shortcut_lambda_skip_fm_rep,
            shortcut_skip_in_loop_prob_rep,
            shortcut_skip_in_loop_gap_mode_rep,
            shortcut_skip_in_loop_gap_rep,
            shortcut_skip_in_loop_max_gap_rep,
            shortcut_skip_in_loop_gap_loc_rep,
            shortcut_skip_in_loop_gap_sigma_rep,
            shortcut_skip_in_loop_warmup_steps_rep,
            shortcut_skip_in_loop_detach_source_rep,
            private_loss_enabled_rep,
            lambda_private_rep,
            private_max_pairs_rep,
            private_start_step_rep,
            private_warmup_iters_rep,
            shortcut_debug_gap_logs_rep,
            shortcut_mag_scale_rep,
            shortcut_mag_clip_min_rep,
            shortcut_mag_clip_max_rep,
            lambda_output_distill_rep,
            shortcut_l2_ema_alpha_rep,
        )
        block_pytree(probe_metrics)

        inception_fn = get_inception("pooled+spatial")

        log_stage(
            f"[FID probe] decoding one real validation batch of {args.batch_size} latents "
            f"with VAE micro-batch {args.vae_decode_batch_size}..."
        )
        probe_val_batch, val_data_iter = next_validation_batch(
            val_data_iter, data_pattern=args.val_data_path, batch_size=args.batch_size
        )
        real_latents_nchw = unpatchify_patchified_latents(probe_val_batch[0])
        real_images = decode_latents_batched(real_latents_nchw, args.vae_decode_batch_size)
        extract_inception_features_host_images(
            real_images,
            inception_fn,
            num_devices=num_devices,
            local_batch=inception_local_batch,
            mode="pooled+spatial",
        )

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
        extract_inception_features_host_images(
            fake_images,
            inception_fn,
            num_devices=num_devices,
            local_batch=inception_local_batch,
            mode="pooled+spatial",
        )

        log_stage(
            "[FID probe] success: discarded train step + real/fake pooled+spatial metric batches completed without OOM."
        )
        return val_data_iter, cached_train_batch

    _eval_real_cache = [None]  # dict with pooled/spatial stats and metadata

    def compute_eval_metrics(step, val_data_iter):
        """Hot-path eval for FID-bundled metrics (streaming, sharded, static shapes).

        Computes FID + sFID in a single shared pass over the same real/fake pools.
        (IS/PR/LinearProbe are integrated in later metric modules.)
        """
        from src.fid_utils import fid_from_stats

        local_b = int(args.fid_eval_local_batch)
        global_b = num_devices * local_b
        need = int(args.num_fid_samples)
        current_val_iter = val_data_iter

        inception_fn = get_inception("pooled+spatial")

        pr_enabled = bool(args.precision_recall)
        pr_full = bool(args.pr_full_mode)
        pr_k = int(args.pr_k)
        pr_cap = int(args.pr_max_samples)
        pr_eval_samples = need if pr_full else min(need, pr_cap)
        pr_mode = "full" if pr_full else ("subset" if pr_eval_samples < need else "full")

        def _update_accumulators(acc_pooled, acc_spatial, pooled_feats, spatial_feats, valid_mask):
            pooled = pooled_feats.reshape(num_devices, local_b, 2048)
            spatial = spatial_feats.reshape(num_devices, local_b, spatial_feats.shape[2], spatial_feats.shape[3], 2048)
            bc, bs, bsxx = gaussian_batch_sums_pmap(pooled, valid_mask)
            sc, ss, ssxx = gaussian_spatial_batch_sums_pmap(spatial, valid_mask)
            # Use replica 0 (all replicas identical after psum)
            acc_pooled = gaussian_sums_add(acc_pooled, bc[0], bs[0], bsxx[0])
            acc_spatial = gaussian_sums_add(acc_spatial, sc[0], ss[0], ssxx[0])
            return acc_pooled, acc_spatial

        # ── Real side: build cache once (streaming, no full images kept) ──────
        if _eval_real_cache[0] is None:
            log_stage(f"[EVAL] building real cache: {need} samples (batch={global_b})…")
            acc_real_pooled = init_gaussian_sums(2048)
            acc_real_spatial = init_gaussian_sums(2048)
            pr_real_sampler = ReservoirSampler(pr_eval_samples, seed=0) if pr_enabled else None
            seen = 0
            while seen < need and current_val_iter is not None:
                vbatch, current_val_iter = next_validation_batch(
                    current_val_iter, data_pattern=args.val_data_path, batch_size=args.batch_size
                )
                latents_all = unpatchify_patchified_latents(vbatch[0])  # (B,4,32,32) numpy
                # Stream through this batch in fixed global_b chunks
                for start in range(0, latents_all.shape[0], global_b):
                    if seen >= need:
                        break
                    chunk = latents_all[start:start + global_b]
                    valid = min(chunk.shape[0], need - seen, global_b)
                    if chunk.shape[0] < global_b:
                        pad = global_b - chunk.shape[0]
                        chunk = np.concatenate([chunk, np.zeros((pad, 4, 32, 32), dtype=np.float32)], axis=0)
                    lat_sharded = jnp.array(chunk.reshape(num_devices, local_b, 4, 32, 32), dtype=jnp.float32)
                    imgs_sharded = decode_latents_sharded(lat_sharded)  # (dev,local,256,256,3)
                    pooled, spatial, valid_mask = apply_inception_to_decoded_sharded(
                        imgs_sharded,
                        inception_fn,
                        mode="pooled+spatial",
                        valid_global=valid,
                    )
                    acc_real_pooled, acc_real_spatial = _update_accumulators(
                        acc_real_pooled, acc_real_spatial, pooled, spatial, valid_mask
                    )
                    if pr_real_sampler is not None:
                        pr_real_sampler.add(trim_sharded_batch_to_host(pooled, valid).reshape(valid, -1))
                    seen += valid
            mu_r, cov_r, n_r = finalize_gaussian_sums(acc_real_pooled)
            mu_rs, cov_rs, n_rs = finalize_gaussian_sums(acc_real_spatial)
            pr_real = pr_real_sampler.get() if pr_real_sampler is not None else None
            _eval_real_cache[0] = {
                "pooled": {"mu": mu_r, "cov": cov_r, "count": n_r},
                "spatial": {"mu": mu_rs, "cov": cov_rs, "count": n_rs},
                "pr": {
                    "mode": pr_mode,
                    "eval_samples": int(pr_eval_samples),
                    "k": int(pr_k),
                    "real_feats": pr_real,
                },
                "meta": {"num_samples": min(need, n_r), "feature_dim": 2048, "mode": "pooled+spatial"},
            }
            log_stage(f"[EVAL] real cache ready: pooled n={n_r}, spatial n={n_rs}.")

        real = _eval_real_cache[0]
        mu_real = real["pooled"]["mu"]
        cov_real = real["pooled"]["cov"]
        mu_real_s = real["spatial"]["mu"]
        cov_real_s = real["spatial"]["cov"]

        # ── Fake side: one shared pass for FID + sFID (+ reuse images for IS) ─
        log_stage(f"[EVAL] generating fake pool: {need} samples @ step {step} (batch={global_b})…")
        acc_fake_pooled = init_gaussian_sums(2048)
        acc_fake_spatial = init_gaussian_sums(2048)
        pr_fake_sampler = ReservoirSampler(pr_eval_samples, seed=int(step)) if pr_enabled else None
        # Inception Score streaming accumulators (host-side; reuse same fake images)
        is_enabled = bool(args.inception_score)
        splits = int(args.inception_score_splits)
        if is_enabled:
            is_worker = get_is_worker()
            is_sum_probs = np.zeros((splits, 1000), dtype=np.float64)
            is_sum_p_log_p = np.zeros((splits,), dtype=np.float64)
            is_count = np.zeros((splits,), dtype=np.int64)
        produced = 0
        chunk_idx = 0
        eval_rng = jax.vmap(
            lambda key: jax.random.fold_in(key, jnp.uint32(step & 0xFFFFFFFF))
        )(rng)
        while produced < need:
            valid = min(global_b, need - produced)
            class_rng, sample_rng = make_eval_chunk_rngs(eval_rng, chunk_idx)
            classes = jax.vmap(
                lambda key: jax.random.randint(key, (local_b,), 0, 1000),
                in_axes=0,
            )(class_rng)
            latents_sharded = fid_sample_latents_pmapped(ema_params, classes, sample_rng)
            imgs_sharded = decode_latents_sharded(latents_sharded)
            if is_enabled:
                # Device->host exactly once per batch (uint8) and trim pads by `valid`.
                imgs_u8 = (imgs_sharded * 255.0).clip(0, 255).astype(jnp.uint8)
                imgs_host = np.asarray(jax.device_get(imgs_u8), dtype=np.uint8).reshape(global_b, 256, 256, 3)
                imgs_host = imgs_host[:valid]
                res = is_worker.infer(imgs_host)
                probs = np.asarray(res.probs, dtype=np.float64)  # (valid, 1000)
                # Assign samples to splits by global index within this eval window
                base = produced
                for i in range(valid):
                    split_id = int(((base + i) * splits) // need)
                    p = probs[i]
                    is_sum_probs[split_id] += p
                    # sum_y p log p (avoid log(0))
                    is_sum_p_log_p[split_id] += float(np.sum(p * np.log(np.maximum(p, 1e-12))))
                    is_count[split_id] += 1
            pooled, spatial, valid_mask = apply_inception_to_decoded_sharded(
                imgs_sharded,
                inception_fn,
                mode="pooled+spatial",
                valid_global=valid,
            )
            acc_fake_pooled, acc_fake_spatial = _update_accumulators(
                acc_fake_pooled, acc_fake_spatial, pooled, spatial, valid_mask
            )
            if pr_fake_sampler is not None:
                pr_fake_sampler.add(trim_sharded_batch_to_host(pooled, valid).reshape(valid, -1))
            produced += valid
            chunk_idx += 1

        mu_f, cov_f, _ = finalize_gaussian_sums(acc_fake_pooled)
        mu_fs, cov_fs, _ = finalize_gaussian_sums(acc_fake_spatial)

        fid_val = fid_from_stats(mu_real, cov_real, mu_f, cov_f)
        sfid_val = fid_from_stats(mu_real_s, cov_real_s, mu_fs, cov_fs)

        metrics = {"val/FID": fid_val, "val/sFID": sfid_val, "train/step": step}
        if args.fid_skip_eval and fid_sample_latents_skip_pmapped is not None:
            log_stage(
                f"[EVAL] generating skip fake pool: {need} samples @ step {step} "
                f"(3->7; timestep_mode={args.fid_skip_timestep_mode}; output_mode={args.shortcut_loss_mode})…"
            )
            acc_skip_pooled = init_gaussian_sums(2048)
            acc_skip_spatial = init_gaussian_sums(2048)
            produced_skip = 0
            chunk_idx_skip = 0
            skip_rng = jax.vmap(
                lambda key: jax.random.fold_in(key, jnp.uint32((step + 0x51F) & 0xFFFFFFFF))
            )(rng)
            while produced_skip < need:
                valid = min(global_b, need - produced_skip)
                class_rng, sample_rng = make_eval_chunk_rngs(skip_rng, chunk_idx_skip)
                classes = jax.vmap(
                    lambda key: jax.random.randint(key, (local_b,), 0, 1000),
                    in_axes=0,
                )(class_rng)
                latents_sharded = fid_sample_latents_skip_pmapped(
                    ema_params,
                    predictor_ema_params,
                    l2_ema,
                    classes,
                    sample_rng,
                )
                imgs_sharded = decode_latents_sharded(latents_sharded)
                pooled, spatial, valid_mask = apply_inception_to_decoded_sharded(
                    imgs_sharded,
                    inception_fn,
                    mode="pooled+spatial",
                    valid_global=valid,
                )
                acc_skip_pooled, acc_skip_spatial = _update_accumulators(
                    acc_skip_pooled, acc_skip_spatial, pooled, spatial, valid_mask
                )
                produced_skip += valid
                chunk_idx_skip += 1

            mu_skip, cov_skip, _ = finalize_gaussian_sums(acc_skip_pooled)
            mu_skip_s, cov_skip_s, _ = finalize_gaussian_sums(acc_skip_spatial)
            metrics["val/FID_skip_3to7"] = fid_from_stats(mu_real, cov_real, mu_skip, cov_skip)
            metrics["val/sFID_skip_3to7"] = fid_from_stats(mu_real_s, cov_real_s, mu_skip_s, cov_skip_s)
            metrics["val/FID_skip_timestep_mode"] = 1.0 if args.fid_skip_timestep_mode == "alternate" else 0.0
            metrics["val/FID_skip_output_mode"] = 1.0 if args.shortcut_loss_mode == "direction_activation" else 0.0

        if pr_enabled:
            pr_mode = real.get("pr", {}).get("mode", pr_mode)
            pr_real = real.get("pr", {}).get("real_feats", None)
            pr_fake = pr_fake_sampler.get() if pr_fake_sampler is not None else None
            metrics["val/PR_mode"] = pr_mode
            metrics["val/PR_eval_samples"] = int(pr_eval_samples)
            if pr_real is not None and pr_fake is not None and pr_real.shape[0] >= (pr_k + 2) and pr_fake.shape[0] >= (pr_k + 2):
                prec, rec = precision_recall_knn(pr_real, pr_fake, k=pr_k, chunk=512)
                metrics["val/Precision"] = float(prec)
                metrics["val/Recall"] = float(rec)
        if is_enabled:
            # Compute IS per split: exp(E[p log p] - sum p log p_bar)
            split_scores = []
            for sid in range(splits):
                n = int(is_count[sid])
                if n <= 0:
                    continue
                p_bar = is_sum_probs[sid] / float(n)
                h_pbar = float(np.sum(p_bar * np.log(np.maximum(p_bar, 1e-12))))
                e_plogp = float(is_sum_p_log_p[sid] / float(n))
                split_scores.append(float(np.exp(e_plogp - h_pbar)))
            if split_scores:
                metrics["val/InceptionScore"] = float(np.mean(split_scores))
                if len(split_scores) > 1:
                    metrics["val/InceptionScore_std"] = float(np.std(split_scores, ddof=1))
        if args.linear_probe:
            probe_layer = int(args.probe_layer if args.probe_layer is not None else depth)
            W_repl, b_repl = get_probe_weights()
            probe_fn = get_probe_pmapped(probe_layer)
            correct_total = 0
            count_total = 0
            probe_iter = current_val_iter
            for i in range(int(args.probe_eval_batches)):
                if probe_iter is None:
                    break
                vbatch, probe_iter = next_validation_batch(
                    probe_iter, data_pattern=args.val_data_path, batch_size=args.batch_size
                )
                bx = jnp.array(vbatch[0]).reshape(num_devices, local_batch_size, n_patches, patch_dim)
                by = jnp.array(vbatch[1]).reshape(num_devices, local_batch_size)
                corr, tot = probe_fn(ema_params, bx, by, W_repl, b_repl)
                correct_total += int(jax.device_get(corr[0]))
                count_total += int(jax.device_get(tot[0]))
            if count_total > 0:
                metrics["val/LinearProbeAcc@1"] = float(correct_total / count_total)
            current_val_iter = probe_iter

        summary_parts = [f"[EVAL] step {step}:", f"FID={fid_val:.2f}", f"sFID={sfid_val:.2f}"]
        if "val/FID_skip_3to7" in metrics:
            summary_parts.append(f"FIDskip3to7={metrics['val/FID_skip_3to7']:.2f}")
            summary_parts.append(f"sFIDskip3to7={metrics['val/sFID_skip_3to7']:.2f}")
        if is_enabled and "val/InceptionScore" in metrics:
            summary_parts.append(f"IS={metrics['val/InceptionScore']:.2f}")
        if "val/Precision" in metrics and "val/Recall" in metrics:
            summary_parts.append(
                f"PR=({metrics['val/Precision']:.3f},{metrics['val/Recall']:.3f})[{metrics.get('val/PR_mode', pr_mode)} n={metrics.get('val/PR_eval_samples', pr_eval_samples)}]"
            )
        if "val/LinearProbeAcc@1" in metrics:
            summary_parts.append(f"LP@1={metrics['val/LinearProbeAcc@1']:.4f}")
        summary_parts.append(f"(n={need})")
        log_stage("  ".join(summary_parts))
        safe_wandb_log(metrics, step=step)
        summary_writer.update(metrics, step=step, section="val")
        return current_val_iter

    def compute_block_corr(step, val_data_iter):
        """EMA block correlation diagnostic (cadence riêng)."""
        if args.block_corr_freq <= 0:
            return val_data_iter
        if val_data_iter is None:
            raise RuntimeError("block correlation requires --val-data-path")
        bc_fn = get_blockcorr_pmapped()
        block_batches = []
        for i in range(int(args.block_corr_batches)):
            vbatch, val_data_iter = next_validation_batch(
                val_data_iter, data_pattern=args.val_data_path, batch_size=args.batch_size
            )
            bx = jnp.array(vbatch[0]).reshape(num_devices, local_batch_size, n_patches, patch_dim)
            by = jnp.array(vbatch[1]).reshape(num_devices, local_batch_size)
            summaries = bc_fn(ema_params, bx, by)
            summaries_h = np.asarray(jax.device_get(summaries), dtype=np.float32)
            summaries_h = np.transpose(summaries_h, (1, 0, 2, 3))
            summaries_h = summaries_h.reshape(summaries_h.shape[0], -1, summaries_h.shape[3])
            block_batches.append(summaries_h)
        if not block_batches:
            return val_data_iter
        stacked = np.concatenate(block_batches, axis=1)
        corr = pearson_corrcoef_rows(stacked.reshape(stacked.shape[0], -1))
        log_stage(
            f"[BLOCKCORR] step {step}: computed {corr.shape[0]}x{corr.shape[1]} Pearson heatmap over {stacked.shape[1]} samples."
        )
        log_blockcorr_async(corr.astype(np.float32), step=step)
        return val_data_iter

    # ── Preflight checks ──────────────────────────────────────────────────────
    prefetched_train_batch = None
    if args.preflight_checks:
        inception_fn_for_preflight = get_inception("pooled+spatial") if args.preflight_fid_samples > 0 else None
        preflight_real_batch = None
        if val_iterator is not None:
            preflight_batch, val_iterator = next_validation_batch(
                val_iterator, data_pattern=args.val_data_path, batch_size=args.batch_size,
            )
            preflight_real_batch = preflight_batch
        elif data_iterator is not None:
            prefetched_train_batch = next(data_iterator)
            preflight_real_batch = prefetched_train_batch

        rng = run_preflight_checks(
            state=state,
            ema_params=ema_params,
            rng=rng,
            sample_latents_jitted=sample_latents_jitted,
            decode_latents=decode_latents_batched,
            inception_fn=inception_fn_for_preflight,
            real_eval_batch=preflight_real_batch,
            preflight_sample_count=args.preflight_sample_count,
            preflight_fid_samples=args.preflight_fid_samples,
            inception_num_devices=num_devices,
            inception_local_batch=args.fid_eval_local_batch,
            inception_score_enabled=bool(args.inception_score),
            inception_score_splits=int(args.inception_score_splits),
            precision_recall_enabled=bool(args.precision_recall),
            pr_k=int(args.pr_k),
            pr_max_samples=int(args.pr_max_samples),
            pr_full_mode=bool(args.pr_full_mode),
            get_is_worker=get_is_worker if args.inception_score else None,
            linear_probe_runner=run_preflight_linear_probe if args.linear_probe else None,
            block_corr_runner=run_preflight_block_corr if args.block_corr_freq > 0 else None,
        )

        if args.preflight_fid_memory_probe:
            val_iterator, prefetched_train_batch = run_fid_memory_probe(val_iterator, prefetched_train_batch)

        if args.preflight_only:
            logger.shutdown()
            return

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = scalar_to_host_int(state.step)
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
            state, ema_params, predictor_ema_params, l2_ema, metrics, rng = pmapped_train_step(
                state,
                ema_params,
                predictor_ema_params,
                l2_ema,
                (batch_x, batch_y),
                rng,
                jax_utils.replicate(jnp.int32(global_step)),
                ema_decay_rep,
                predictor_ema_decay_rep,
                shortcut_timesteps_rep,
                shortcut_lambda_dir_rep,
                shortcut_lambda_boot_rep,
                shortcut_bootstrap_detach_source_rep,
                shortcut_lambda_mag_rep,
                shortcut_lambda_boot_mag_rep,
                shortcut_lambda_skip_fm_rep,
                shortcut_skip_in_loop_prob_rep,
                shortcut_skip_in_loop_gap_mode_rep,
                shortcut_skip_in_loop_gap_rep,
                shortcut_skip_in_loop_max_gap_rep,
                shortcut_skip_in_loop_gap_loc_rep,
                shortcut_skip_in_loop_gap_sigma_rep,
                shortcut_skip_in_loop_warmup_steps_rep,
                shortcut_skip_in_loop_detach_source_rep,
                private_loss_enabled_rep,
                lambda_private_rep,
                private_max_pairs_rep,
                private_start_step_rep,
                private_warmup_iters_rep,
                shortcut_debug_gap_logs_rep,
                shortcut_mag_scale_rep,
                shortcut_mag_clip_min_rep,
                shortcut_mag_clip_max_rep,
                lambda_output_distill_rep,
                shortcut_l2_ema_alpha_rep,
            )
            global_step += 1
            accumulated_train_tflops += flops_per_train_step / 1e12

            # Async metric logging
            if args.log_freq > 0 and global_step % args.log_freq == 0:
                cpu_metrics = jax.tree_util.tree_map(lambda m: m[0], metrics)
                t1 = time.time()
                cpu_metrics["perf/train_step_time"] = (t1 - t0) / args.log_freq
                cpu_metrics["perf/train_step_tflops"] = (flops_per_train_step / 1e12) / max(cpu_metrics["perf/train_step_time"], 1e-12)
                cpu_metrics["perf/accumulated_train_tflops"] = accumulated_train_tflops
                cpu_metrics["train/step"] = global_step
                if not args.private_loss:
                    cpu_metrics = filter_private_metrics(cpu_metrics)
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
                    val_metrics, rng = pmapped_eval_step(
                        state, (val_x, val_y), rng
                    )
                    host_val_metrics = replicated_metrics_to_host(val_metrics)
                    for key, value in host_val_metrics.items():
                        metric_sums[key] = metric_sums.get(key, 0.0) + value

                averaged_val_metrics = {k: v / args.eval_batches for k, v in metric_sums.items()}
                averaged_val_metrics["train/step"] = global_step
                logger.log(averaged_val_metrics, step=global_step)

            # Synchronous FID (blocks training; see compute_fid docstring)
            if args.fid_freq > 0 and global_step % args.fid_freq == 0:
                try:
                    val_iterator = compute_eval_metrics(global_step, val_iterator)
                except Exception as exc:
                    log_stage(f"EVAL skipped: {exc}")

            if args.block_corr_freq > 0 and global_step % args.block_corr_freq == 0:
                try:
                    val_iterator = compute_block_corr(global_step, val_iterator)
                except Exception as exc:
                    log_stage(f"BLOCKCORR skipped: {exc}")

            # Sample preview: uses EMA params and configurable num_steps
            if args.sample_freq > 0 and global_step % args.sample_freq == 0:
                print(f"Step {global_step}: Generating sample previews "
                      f"({args.sample_num_steps} steps, cfg={args.sample_cfg_scale})...")
                sample_rng, = jax.random.split(rng[0], 1)
                sample_classes = jax.random.randint(sample_rng, (4,), 0, 1000)
                # Use EMA params for sample generation (paper-faithful eval)
                single_ema_params = jax.tree_util.tree_map(lambda w: w[0], ema_params)
                single_predictor_ema_params = jax.tree_util.tree_map(lambda w: w[0], predictor_ema_params)
                single_l2_ema = jax.tree_util.tree_map(lambda w: w[0], l2_ema)
                latents_dev = sample_latents_jitted(single_ema_params, sample_classes, sample_rng)
                latents_skip_dev = None
                if sample_latents_skip_jitted is not None:
                    latents_skip_dev = sample_latents_skip_jitted(
                        single_ema_params,
                        single_predictor_ema_params,
                        single_l2_ema,
                        sample_classes,
                        sample_rng,
                    )

                def _bg_log(z_dev, z_skip_dev, classes, target_step):
                    z = np.asarray(jax.device_get(z_dev), dtype=np.float32)
                    classes = jax.device_get(classes)
                    images = decode_latents_batched(z, args.vae_decode_batch_size)
                    images = (images * 255).astype(np.uint8)
                    log_payload = {
                        "train/step": target_step,
                        "samples": [wandb.Image(img, caption=f"Class {cls}")
                                    for img, cls in zip(images, classes)],
                    }
                    if z_skip_dev is not None:
                        z_skip = np.asarray(jax.device_get(z_skip_dev), dtype=np.float32)
                        images_skip = decode_latents_batched(z_skip, args.vae_decode_batch_size)
                        images_skip = (images_skip * 255).astype(np.uint8)
                        log_payload["samples_skip_3to7"] = [
                            wandb.Image(img, caption=f"Class {cls} skip 3->7")
                            for img, cls in zip(images_skip, classes)
                        ]
                    safe_wandb_log(log_payload, step=target_step)

                threading.Thread(target=_bg_log,
                                 args=(latents_dev, latents_skip_dev, sample_classes, global_step),
                                 daemon=True).start()

    # ── Checkpoint save (online params + EMA params) ──────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)
    unreplicated_params = jax_utils.unreplicate(state.params)
    unreplicated_ema    = jax_utils.unreplicate(ema_params)
    unreplicated_predictor_ema = jax_utils.unreplicate(predictor_ema_params)
    unreplicated_l2_ema = jax_utils.unreplicate(l2_ema)
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
    checkpoints.save_checkpoint(
        ckpt_dir=os.path.join(args.ckpt_dir, "predictor_ema"),
        target=unreplicated_predictor_ema,
        step=global_step,
    )
    checkpoints.save_checkpoint(
        ckpt_dir=os.path.join(args.ckpt_dir, "l2_ema"),
        target=unreplicated_l2_ema,
        step=global_step,
    )
    if _flax_decode_cache[0] is not None and isinstance(_flax_decode_cache[0], VAEDecodeSubprocess):
        _flax_decode_cache[0].shutdown()
    if _is_worker[0] is not None:
        _is_worker[0].shutdown()
    logger.shutdown()


if __name__ == "__main__":
    main()
