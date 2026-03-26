#!/usr/bin/env python3
"""
Precompute VAE Latents for ImageNet and Save as ArrayRecords.
Optimized for Kaggle TPU v5e-8 (JAX/Flax).

Usage:
    python prepare_data_tpu.py \
        --split train \
        --data-dir /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC \
        --output-dir ./imagenet_latents \
        --batch-size 128 \
        --num-shards 1024 \
        --group-size 1

    python prepare_data_tpu.py \
        --split train val \
        --data-dir /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC \
        --output-dir ./imagenet_latents \
        --batch-size 128 \
        --num-shards 1024 \
        --group-size 1
"""

import os
import argparse
import pickle
import gc
import csv
import zipfile
import concurrent.futures
from tqdm import tqdm
from PIL import Image
import warnings

warnings.filterwarnings("ignore", message=".*Flax classes are deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import jax
import jax.numpy as jnp
try:
    from diffusers.models import FlaxAutoencoderKL
except Exception:
    FlaxAutoencoderKL = None

try:
    from array_record.python.array_record_module import ArrayRecordWriter
except Exception:
    ArrayRecordWriter = None

# Load HuggingFace Token from Kaggle Secrets if available 
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
except Exception:
    pass


SUPPORTED_SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")


class FastImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        try:
            self.classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
        except FileNotFoundError:
            self.classes = []
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset_fast(root)
        
    def _make_dataset_fast(self, root):
        samples = []
        def process_dir(cls_name):
            d_path = os.path.join(root, cls_name)
            class_idx = self.class_to_idx[cls_name]
            try:
                filenames = os.listdir(d_path)
                return [(os.path.join(d_path, fname), class_idx) for fname in filenames
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
            except Exception:
                return []
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(process_dir, self.classes))
            
        for r in results:
            samples.extend(r)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


class FlatImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


def resolve_split_dir(data_dir, split):
    split_dir_candidate = os.path.join(data_dir, split)
    if os.path.isdir(split_dir_candidate):
        return split_dir_candidate
    if os.path.basename(os.path.normpath(data_dir)).lower() == split.lower() and os.path.isdir(data_dir):
        # Accept both .../CLS-LOC and .../CLS-LOC/train as --data-dir.
        print(f"[prepare_data_tpu] Detected split directory passed directly: {data_dir}")
        return data_dir
    return split_dir_candidate


def list_image_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name)) and name.endswith(IMAGE_EXTENSIONS)
    )


def find_metadata_file(start_path, filename):
    current = os.path.abspath(start_path)
    while True:
        candidate = os.path.join(current, filename)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def build_class_to_idx(data_dir):
    train_dir = resolve_split_dir(data_dir, "train")
    classes = sorted([d.name for d in os.scandir(train_dir) if d.is_dir()]) if os.path.isdir(train_dir) else []
    if not classes:
        raise RuntimeError(
            "Could not build class index from the train split. "
            f"Expected class directories under {train_dir}."
        )
    return {cls_name: i for i, cls_name in enumerate(classes)}


def load_flat_split_samples(split_dir, split, data_dir):
    image_paths = list_image_files(split_dir)
    if not image_paths:
        raise RuntimeError(f"No image files found in flat split directory: {split_dir}")

    image_map = {os.path.splitext(os.path.basename(path))[0]: path for path in image_paths}
    if split == "test":
        return [(path, -1) for _, path in sorted(image_map.items())]

    metadata_file = find_metadata_file(split_dir, f"LOC_{split}_solution.csv")
    if metadata_file is None:
        metadata_file = find_metadata_file(data_dir, f"LOC_{split}_solution.csv")
    if metadata_file is None:
        raise RuntimeError(
            f"Could not find LOC_{split}_solution.csv for flat {split} split. "
            "This file is required to recover class labels."
        )

    class_to_idx = build_class_to_idx(data_dir)
    labels_by_image = {}
    with open(metadata_file, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image_id = row["ImageId"]
            prediction = row["PredictionString"].strip()
            if not prediction:
                continue
            synset = prediction.split()[0]
            if synset not in class_to_idx:
                raise RuntimeError(f"Unknown synset '{synset}' found in {metadata_file}")
            labels_by_image[image_id] = class_to_idx[synset]

    missing_labels = sorted(image_id for image_id in image_map if image_id not in labels_by_image)
    if missing_labels:
        raise RuntimeError(
            f"Missing labels for {len(missing_labels)} image(s) in {split_dir}. "
            f"Example image id: {missing_labels[0]}"
        )

    return [(image_map[image_id], labels_by_image[image_id]) for image_id in sorted(image_map)]

def get_dataloader(data_dir, split, batch_size, num_workers=4):
    split_dir = resolve_split_dir(data_dir, split)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    print(f"Scanning directory {split_dir} (Fast parallel scan for Kaggle)...")
    class_dirs = [d.name for d in os.scandir(split_dir) if d.is_dir()] if os.path.isdir(split_dir) else []
    if class_dirs:
        dataset = FastImageFolder(split_dir, transform=transform)
    else:
        dataset = FlatImageDataset(load_flat_split_samples(split_dir, split, data_dir), transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(
            "No images found for encoding. "
            f"Resolved directory: {split_dir}. "
            "If you passed a split folder already, keep --split matching that folder (e.g. --split train)."
        )
    
    # Batch size needs to be perfectly divisible by drop_last for JAX splitting
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic order; Grain IndexSampler handles shuffle at training time
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True # Keep perfectly shaped batches for JAX
    )
    return dataloader, len(dataset)


_VAE_PARAMS_FILENAME = "vae_params_bf16.msgpack"


def save_vae_params(vae_params, cache_zip_path):
    """Convert Flax VAE params sang msgpack rồi zip lại để dễ tải."""
    import flax.serialization

    os.makedirs(os.path.dirname(os.path.abspath(cache_zip_path)), exist_ok=True)
    tmp_msgpack = cache_zip_path.replace(".zip", "")
    params_bytes = flax.serialization.to_bytes(vae_params)
    with open(tmp_msgpack, "wb") as f:
        f.write(params_bytes)
    with zipfile.ZipFile(cache_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(tmp_msgpack, arcname=_VAE_PARAMS_FILENAME)
    os.remove(tmp_msgpack)
    size_mb = os.path.getsize(cache_zip_path) / (1024 * 1024)
    print(f"[VAE Cache] Saved Flax params → {cache_zip_path} ({size_mb:.1f} MB)")


def load_vae_params_from_zip(vae_model, cache_zip_path):
    """Load Flax VAE params từ file zip đã cache, chỉ tải config model (không cần PyTorch)."""
    import flax.serialization

    print(f"[VAE Cache] Loading cached Flax params from {cache_zip_path} ...")
    with zipfile.ZipFile(cache_zip_path, "r") as zf:
        with zf.open(_VAE_PARAMS_FILENAME) as f:
            vae_params = flax.serialization.from_bytes(None, f.read())

    # Chỉ download config.json để build model architecture, không cần download PyTorch weights
    vae = FlaxAutoencoderKL.from_config(FlaxAutoencoderKL.load_config(vae_model))
    vae_params = jax.tree_util.tree_map(jnp.array, vae_params)
    size_mb = os.path.getsize(cache_zip_path) / (1024 * 1024)
    print(f"[VAE Cache] Loaded ({size_mb:.1f} MB, skipped PyTorch conversion)")
    return vae, vae_params


def load_vae(vae_model, vae_cache=None):
    """
    Load VAE Flax params. Ưu tiên load từ cache zip nếu có.
    Nếu không có cache, convert từ PyTorch rồi tự động lưu zip.
    """
    if vae_cache and os.path.exists(vae_cache):
        return load_vae_params_from_zip(vae_model, vae_cache)

    print(f"[VAE] Loading Flax VAE from {vae_model!r} (converting from PyTorch)...")
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(vae_model, from_pt=True)
    vae_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), vae_params)

    if vae_cache:
        cache_dir = os.path.dirname(os.path.abspath(vae_cache))
        print(f"[VAE Cache] Saving converted params to {vae_cache} ...")
        save_vae_params(vae_params, vae_cache)
        # Lưu config.json cùng thư mục để train.py load local, tránh HF download
        # (HF download kích hoạt lazy import C extension → SIGSEGV trong main JAX process)
        vae.save_config(cache_dir)
        print(f"[VAE Cache] Saved config.json → {os.path.join(cache_dir, 'config.json')}")

    return vae, vae_params


def validate_dependencies():
    missing_deps = []
    if FlaxAutoencoderKL is None:
        missing_deps.append("diffusers[flax]/jax/flax")
    if ArrayRecordWriter is None:
        missing_deps.append("array-record")
    if missing_deps:
        raise ImportError(
            "Missing dependencies for TPU encoding: "
            + ", ".join(missing_deps)
            + ". Install them in the Kaggle environment before running online encoding."
        )


def resolve_splits(split_args):
    resolved = []
    for item in split_args:
        for token in item.split(","):
            split = token.strip().lower()
            if not split:
                continue
            if split == "all":
                resolved.extend(SUPPORTED_SPLITS)
                continue
            if split not in SUPPORTED_SPLITS:
                raise ValueError(
                    f"Unsupported split '{split}'. "
                    f"Expected one of {', '.join(SUPPORTED_SPLITS)} or 'all'."
                )
            resolved.append(split)

    deduped = []
    seen = set()
    for split in resolved:
        if split in seen:
            continue
        seen.add(split)
        deduped.append(split)

    if not deduped:
        raise ValueError("No valid splits were provided.")
    return deduped


def format_arrayrecord_options(group_size):
    if group_size <= 0:
        raise ValueError("--group-size must be greater than 0")
    return f"group_size:{group_size}"


def run_encoding(
    split,
    data_dir,
    output_dir,
    batch_size=128,
    num_shards=256,
    vae_model="stabilityai/sd-vae-ft-ema",
    group_size=1,
    vae_cache=None,
):
    validate_dependencies()
    os.makedirs(output_dir, exist_ok=True)
    writer_options = format_arrayrecord_options(group_size)
    print(
        f"[prepare_data_tpu] data-dir={data_dir} split={split} output-dir={output_dir} "
        f"group_size={group_size}"
    )

    # Verify JAX devices
    num_devices = jax.device_count()
    print(f"JAX detects {num_devices} devices.")
    assert batch_size % num_devices == 0, f"Batch size must be divisible by {num_devices}"
    batch_per_device = batch_size // num_devices

    # 1. Load VAE (dùng cache zip nếu có, ngược lại convert từ PyTorch rồi tự lưu cache)
    vae, vae_params = load_vae(vae_model, vae_cache=vae_cache)
    
    SCALE_FACTOR = 0.18215 

    # 2. PMAP Encoding Function
    @jax.pmap
    def encode_fn(images, params):
        # Flax models from Diffusers (with from_pt=True) expect NCHW input format,
        # otherwise they mistake the Height dimension for the Channel dimension.
        
        # Apply VAE
        latent_dist = vae.apply({"params": params}, images, method=vae.encode).latent_dist
        
        # Using MEAN instead of random sampling to make it deterministic 
        # (similar to stable diffusion training latents cache)
        latents_nhwc = latent_dist.mean * SCALE_FACTOR
        
        # Diffusers Flax VAE outputs latents in NHWC. Let's transpose back to NCHW for matching the standard.
        latents_nchw = jnp.transpose(latents_nhwc, (0, 3, 1, 2))
        return latents_nchw

    # Replicate PMAP Params across devices
    from flax.jax_utils import replicate
    vae_params_repl = replicate(vae_params)
    
    # 3. Setup DataLoader
    dataloader, num_samples = get_dataloader(data_dir, split, batch_size)
    print(f"Found {num_samples} images in {split} split.")
    
    samples_per_shard = (num_samples + num_shards - 1) // num_shards
    
    current_shard = 0
    samples_in_current_shard = 0
    def get_writer(shard_idx):
        path = os.path.join(output_dir, f"{split}-{shard_idx:05d}-of-{num_shards:05d}.ar")
        return ArrayRecordWriter(path, options=writer_options)

    writer = get_writer(current_shard)
    
    for images, labels, paths in tqdm(dataloader, desc=f"Encoding {split}"):

        # Reshape to (num_devices, batch_per_device, C, H, W)
        images_np = images.numpy()
        images_jax = jnp.array(images_np.reshape((num_devices, batch_per_device, 3, 256, 256)), dtype=jnp.bfloat16)

        # PMAP Encode (Executes simultaneously on all 8 TPUs)
        latents = encode_fn(images_jax, vae_params_repl)

        # Flatten back CPU numpy (Batch, 4, 32, 32)
        latents_np = jax.device_get(latents).reshape((-1, 4, 32, 32)).astype("float32")
        labels_np = labels.numpy()

        for latent, label, path in zip(latents_np, labels_np, paths):
            payload = {
                "latent": latent,
                "label": int(label),
                "image_path": path,
            }
            serialized = pickle.dumps(payload)
            writer.write(serialized)
            
            samples_in_current_shard += 1
            if samples_in_current_shard >= samples_per_shard:
                writer.close()
                current_shard += 1
                if current_shard < num_shards:
                    writer = get_writer(current_shard)
                    samples_in_current_shard = 0
                    
    writer.close()
    print("TPU Data preparation complete.")
    del vae, vae_params, vae_params_repl, dataloader
    gc.collect()


def run_multi_split_encoding(
    splits,
    data_dir,
    output_dir,
    batch_size=128,
    num_shards=256,
    vae_model="stabilityai/sd-vae-ft-ema",
    group_size=1,
    vae_cache=None,
):
    for split in resolve_splits(splits):
        run_encoding(
            split=split,
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            num_shards=num_shards,
            vae_model=vae_model,
            group_size=group_size,
            vae_cache=vae_cache,
        )


def main():
    parser = argparse.ArgumentParser(description="Encode ImageNet using JAX/TPU v5e-8.")
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train"],
        help="One or more splits to encode. Examples: --split train, --split train val, --split all",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Base directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save .ar files")
    parser.add_argument("--batch-size", type=int, default=128, help="Global batch size (mutiple of 8)")
    parser.add_argument("--num-shards", type=int, default=1024, help="Number of .ar shards")
    parser.add_argument("--group-size", type=int, default=1, help="ArrayRecord group_size to write. Use 1 for Grain training.")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-ema", help="HF VAE")
    parser.add_argument(
        "--vae-cache",
        type=str,
        default=None,
        help=(
            "Path to cache file for converted Flax VAE params (e.g. ./vae_params_bf16.zip). "
            "Nếu file chưa tồn tại: convert từ PyTorch rồi tự động lưu zip. "
            "Nếu file đã có: load thẳng, bỏ qua bước convert PyTorch."
        ),
    )

    args = parser.parse_args()
    splits = resolve_splits(args.split)
    print(f"[prepare_data_tpu] Encoding splits: {', '.join(splits)}")
    run_multi_split_encoding(
        splits=splits,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_shards=args.num_shards,
        group_size=args.group_size,
        vae_model=args.vae_model,
        vae_cache=args.vae_cache,
    )

if __name__ == "__main__":
    main()
