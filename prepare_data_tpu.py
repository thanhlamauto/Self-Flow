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
        --num-shards 1024

    python prepare_data_tpu.py \
        --split train val \
        --data-dir /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC \
        --output-dir ./imagenet_latents \
        --batch-size 128 \
        --num-shards 1024
"""

import os
import argparse
import pickle
import gc
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
        return sample, target

def get_dataloader(data_dir, split, batch_size, num_workers=4):
    split_dir_candidate = os.path.join(data_dir, split)
    if os.path.isdir(split_dir_candidate):
        split_dir = split_dir_candidate
    elif os.path.basename(os.path.normpath(data_dir)).lower() == split.lower() and os.path.isdir(data_dir):
        # Accept both .../CLS-LOC and .../CLS-LOC/train as --data-dir.
        split_dir = data_dir
        print(f"[prepare_data_tpu] Detected split directory passed directly: {split_dir}")
    else:
        split_dir = split_dir_candidate

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    print(f"Scanning directory {split_dir} (Fast parallel scan for Kaggle)...")
    dataset = FastImageFolder(split_dir, transform=transform)
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
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True # Keep perfectly shaped batches for JAX
    )
    return dataloader, len(dataset)


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


def run_encoding(split, data_dir, output_dir, batch_size=128, num_shards=1024, vae_model="stabilityai/sd-vae-ft-ema"):
    validate_dependencies()
    os.makedirs(output_dir, exist_ok=True)
    print(f"[prepare_data_tpu] data-dir={data_dir} split={split} output-dir={output_dir}")
    
    # Verify JAX devices
    num_devices = jax.device_count()
    print(f"JAX detects {num_devices} devices.")
    assert batch_size % num_devices == 0, f"Batch size must be divisible by {num_devices}"
    batch_per_device = batch_size // num_devices
    
    # 1. Load VAE natively in Flax (from standard PyTorch weights)
    print(f"Loading Flax VAE: {vae_model}")
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(vae_model, from_pt=True)
    # Cast to bf16 for TPU optimization
    vae_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), vae_params)
    
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
        return ArrayRecordWriter(path, options="")

    writer = get_writer(current_shard)
    
    for images, labels in tqdm(dataloader, desc=f"Encoding {split}"):
        
        # Reshape to (num_devices, batch_per_device, C, H, W)
        images_np = images.numpy()
        images_jax = jnp.array(images_np.reshape((num_devices, batch_per_device, 3, 256, 256)), dtype=jnp.bfloat16)
        
        # PMAP Encode (Executes simultaneously on all 8 TPUs)
        latents = encode_fn(images_jax, vae_params_repl)
        
        # Flatten back CPU numpy (Batch, 4, 32, 32)
        latents_np = jax.device_get(latents).reshape((-1, 4, 32, 32)).astype("float32")
        labels_np = labels.numpy()
        
        for latent, label in zip(latents_np, labels_np):
            payload = {
                "latent": latent,
                "label": int(label)
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
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-ema", help="HF VAE")
    
    args = parser.parse_args()
    splits = resolve_splits(args.split)
    print(f"[prepare_data_tpu] Encoding splits: {', '.join(splits)}")
    for split in splits:
        run_encoding(
            split=split,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_shards=args.num_shards,
            vae_model=args.vae_model,
        )

if __name__ == "__main__":
    main()
