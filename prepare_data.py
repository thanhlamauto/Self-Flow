#!/usr/bin/env python3
"""
Precompute VAE Latents for ImageNet and Save as ArrayRecords.
Designed to run on Kaggle GPU P100 environment.

Usage:
    python prepare_data.py \
        --split train \
        --data-dir /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC \
        --output-dir /kaggle/working/imagenet_latents \
        --batch-size 128 \
        --num-shards 1024
"""

import os
import argparse
import pickle
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from diffusers.models import AutoencoderKL
import array_record

# Load HuggingFace Token from Kaggle Secrets if available 
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
except Exception:
    pass # Fallback to system environment variable if not on Kaggle

import concurrent.futures
from PIL import Image
from torch.utils.data import Dataset

class FastImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        # Get all subdirectories (classes)
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
            # Use os.listdir to avoid slow stat calls on Kaggle's FUSE filesystem
            try:
                filenames = os.listdir(d_path)
                return [(os.path.join(d_path, fname), class_idx) for fname in filenames
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
            except Exception:
                return []
                
        # Use ThreadPoolExecutor to hide network latency when scanning 1,000 directories
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
    """Create PyTorch DataLoader for ImageNet folder structure."""
    split_dir = os.path.join(data_dir, split)
    
    # Standard Latent Diffusion preprocessing:
    # Resize shortest edge to 256, center crop 256x256
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    print(f"Scanning directory {split_dir} (Fast parallel scan for Kaggle)...")
    dataset = FastImageFolder(split_dir, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle randomly before sharding
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader, len(dataset)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Encode ImageNet to VAE Latents safely for JAX ArrayRecord TPU pipeline.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", type=str, required=True, help="Base directory containing train/val/test folders")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save .ar files")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for VAE encoding")
    parser.add_argument("--num-shards", type=int, default=1024, help="Number of .ar shards (use more for train, less for val)")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-ema", help="HuggingFace Hub VAE path")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load VAE
    print(f"Loading VAE: {args.vae_model}")
    vae = AutoencoderKL.from_pretrained(args.vae_model)
    # Convert VAE weights to FP16 to massively speed up P100 encoding and save memory
    vae = vae.to(device, dtype=torch.float16)
    vae.eval()
    
    SCALE_FACTOR = 0.18215 # SD-VAE scale factor
    
    # 2. Setup DataLoader
    dataloader, num_samples = get_dataloader(args.data_dir, args.split, args.batch_size)
    print(f"Found {num_samples} images in {args.split} split.")
    
    samples_per_shard = (num_samples + args.num_shards - 1) // args.num_shards
    print(f"Targeting ~{samples_per_shard} samples per shard across {args.num_shards} shards.")
    
    # 3. Encoding Loop
    current_shard = 0
    samples_in_current_shard = 0
    
    def get_writer(shard_idx):
        path = os.path.join(args.output_dir, f"{args.split}-{shard_idx:05d}-of-{args.num_shards:05d}.ar")
        return array_record.ArrayRecordWriter(path, options="")

    writer = get_writer(current_shard)
    
    for images, labels in tqdm(dataloader, desc=f"Encoding {args.split}"):
        
        # Move images to GPU and FP16 to match VAE
        images = images.to(device, dtype=torch.float16)
        
        # Encode with VAE
        # using sample() gives the latents, applying scale right now.
        latent_dist = vae.encode(images).latent_dist
        latents = latent_dist.sample() * SCALE_FACTOR
        
        # Move latents back to CPU as Float32 numpy arrays for serialization
        latents_np = latents.cpu().to(torch.float32).numpy()
        labels_np = labels.cpu().numpy()
        
        # Iterate over batch and write each single record iteratively
        for latent, label in zip(latents_np, labels_np):
            
            # Serialize payload natively (Pickle is simple and fast for Grain reading later)
            payload = {
                "latent": latent, # Shape (4, 32, 32)
                "label": label    # Int
            }
            serialized = pickle.dumps(payload)
            writer.write(serialized)
            
            samples_in_current_shard += 1
            
            # Shard Rotation checking
            if samples_in_current_shard >= samples_per_shard:
                writer.close()
                current_shard += 1
                if current_shard < args.num_shards:
                    writer = get_writer(current_shard)
                    samples_in_current_shard = 0
                    
    # File cleanup in case last shard wasn't perfectly full
    writer.close()
    
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
