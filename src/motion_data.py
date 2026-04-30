"""NumPy/JAX dataloading utilities for MDM-style motion arrays."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class MotionDataset:
    motions: np.ndarray
    lengths: np.ndarray
    actions: Optional[np.ndarray] = None
    text_embeds: Optional[np.ndarray] = None
    captions: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        return int(self.motions.shape[0])


def lengths_to_mask(lengths: np.ndarray, max_frames: int) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int32)
    return np.arange(int(max_frames))[None, :] < lengths[:, None]


def _to_mdm_shape(motions: np.ndarray, *, njoints: Optional[int], nfeats: Optional[int]) -> np.ndarray:
    arr = np.asarray(motions, dtype=np.float32)
    if arr.ndim == 4:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected motions rank 3 or 4, got shape {arr.shape}")
    if njoints is None or nfeats is None:
        raise ValueError("Flat motion arrays require njoints and nfeats")
    batch, frames, features = arr.shape
    expected = int(njoints) * int(nfeats)
    if features != expected:
        raise ValueError(f"Expected feature dim {expected}, got {features}")
    arr = arr.reshape(batch, frames, int(njoints), int(nfeats))
    return np.transpose(arr, (0, 2, 3, 1))


def load_motion_npz(path: str, *, njoints: Optional[int] = None, nfeats: Optional[int] = None) -> MotionDataset:
    """Load an ``.npz`` dataset.

    Required key:
      - ``motions`` or ``motion``: either [N,J,F,T] or [N,T,J*F]

    Optional keys:
      - ``lengths``: [N], defaults to full sequence length
      - ``actions`` or ``action``: [N]
      - ``text_embeds`` or ``text_embed``: [N,D]
    """
    with np.load(os.path.expanduser(path), allow_pickle=True) as data:
        motion_key = "motions" if "motions" in data else "motion"
        if motion_key not in data:
            raise KeyError(f"{path} must contain a 'motions' or 'motion' array")
        motions = _to_mdm_shape(data[motion_key], njoints=njoints, nfeats=nfeats)
        lengths = np.asarray(data["lengths"], dtype=np.int32) if "lengths" in data else None
        if lengths is None:
            lengths = np.full((motions.shape[0],), motions.shape[-1], dtype=np.int32)
        action_key = "actions" if "actions" in data else "action"
        actions = np.asarray(data[action_key], dtype=np.int32) if action_key in data else None
        text_key = "text_embeds" if "text_embeds" in data else "text_embed"
        text_embeds = np.asarray(data[text_key], dtype=np.float32) if text_key in data else None
    return MotionDataset(motions=motions, lengths=lengths, actions=actions, text_embeds=text_embeds)


def load_motion_npy_dir(
    path: str,
    *,
    njoints: int,
    nfeats: int,
    max_frames: Optional[int] = None,
) -> MotionDataset:
    """Load a directory of per-sample ``.npy`` files and pad to a fixed length.

    Each file may be [T,J*F], [J,F,T], or [T,J,F].
    """
    files = sorted(glob.glob(os.path.join(os.path.expanduser(path), "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found under {path}")

    samples = []
    lengths = []
    for file in files:
        arr = np.asarray(np.load(file), dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.reshape(arr.shape[0], int(njoints), int(nfeats))
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 3 and arr.shape[0] != njoints:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim != 3:
            raise ValueError(f"Unsupported motion shape {arr.shape} in {file}")
        samples.append(arr)
        lengths.append(arr.shape[-1])

    target_frames = int(max_frames or max(lengths))
    motions = np.zeros((len(samples), int(njoints), int(nfeats), target_frames), dtype=np.float32)
    clipped_lengths = np.minimum(np.asarray(lengths, dtype=np.int32), target_frames)
    for idx, arr in enumerate(samples):
        n = int(clipped_lengths[idx])
        motions[idx, :, :, :n] = arr[:, :, :n]
    return MotionDataset(motions=motions, lengths=clipped_lengths)


def load_humanml_dataset(
    root: str = "motion-diffusion-model/dataset/HumanML3D",
    *,
    split: str = "train",
    max_frames: int = 196,
    normalize: bool = True,
) -> MotionDataset:
    """Load native HumanML3D files into MDM's ``[N,263,1,196]`` convention.

    Expected layout under ``root``:
      - ``new_joint_vecs/*.npy`` motion feature files with shape [T,263]
      - ``texts/*.txt`` captions in HumanML format
      - ``train.txt`` / ``val.txt`` / ``test.txt`` split files
      - ``Mean.npy`` and ``Std.npy`` for training normalization
    """
    root = os.path.expanduser(root)
    split_file = os.path.join(root, f"{split}.txt")
    motion_dir = os.path.join(root, "new_joint_vecs")
    text_dir = os.path.join(root, "texts")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(
            f"HumanML split file not found: {split_file}. "
            "Download/copy HumanML3D to motion-diffusion-model/dataset/HumanML3D."
        )
    if not os.path.isdir(motion_dir):
        raise FileNotFoundError(f"HumanML motion dir not found: {motion_dir}")

    mean = std = None
    if normalize:
        mean_path = os.path.join(root, "Mean.npy")
        std_path = os.path.join(root, "Std.npy")
        if not os.path.isfile(mean_path) or not os.path.isfile(std_path):
            raise FileNotFoundError(
                f"HumanML normalization files not found under {root}: Mean.npy and Std.npy are required"
            )
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)

    with open(split_file) as f:
        names = [line.strip() for line in f if line.strip()]

    motions = []
    lengths = []
    captions = []
    for name in names:
        motion_path = os.path.join(motion_dir, name + ".npy")
        if not os.path.isfile(motion_path):
            continue
        motion = np.asarray(np.load(motion_path), dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] != 263:
            raise ValueError(f"Expected HumanML motion [T,263], got {motion.shape} in {motion_path}")
        if motion.shape[0] < 40 or motion.shape[0] >= 200:
            continue
        length = min(int(motion.shape[0]), int(max_frames))
        motion = motion[:length]
        if normalize:
            motion = (motion - mean) / std
        padded = np.zeros((int(max_frames), 263), dtype=np.float32)
        padded[:length] = motion
        motions.append(padded.T[:, None, :])
        lengths.append(length)

        caption = ""
        text_path = os.path.join(text_dir, name + ".txt")
        if os.path.isfile(text_path):
            with open(text_path) as tf:
                first = tf.readline().strip()
            caption = first.split("#")[0] if first else ""
        captions.append(caption)

    if not motions:
        raise ValueError(f"No usable HumanML motions found in {root} split={split}")
    return MotionDataset(
        motions=np.stack(motions, axis=0).astype(np.float32),
        lengths=np.asarray(lengths, dtype=np.int32),
        captions=np.asarray(captions, dtype=object),
    )


class MotionDataLoader:
    """Small deterministic host dataloader yielding JAX arrays."""

    def __init__(
        self,
        dataset: MotionDataset,
        *,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def __iter__(self) -> Iterator[dict[str, jnp.ndarray]]:
        rng = np.random.default_rng(self.seed)
        order = np.arange(self.dataset.size)
        if self.shuffle:
            rng.shuffle(order)
        for start in range(0, self.dataset.size, self.batch_size):
            idx = order[start : start + self.batch_size]
            if idx.shape[0] < self.batch_size and self.drop_last:
                continue
            yield self._make_batch(idx)

    def _make_batch(self, idx: np.ndarray) -> dict[str, jnp.ndarray]:
        motions = self.dataset.motions[idx]
        lengths = self.dataset.lengths[idx]
        batch = {
            "motion": jnp.asarray(motions, dtype=jnp.float32),
            "lengths": jnp.asarray(lengths, dtype=jnp.int32),
            "mask": jnp.asarray(lengths_to_mask(lengths, motions.shape[-1])),
        }
        if self.dataset.actions is not None:
            batch["action"] = jnp.asarray(self.dataset.actions[idx], dtype=jnp.int32)
        if self.dataset.text_embeds is not None:
            batch["text_embed"] = jnp.asarray(self.dataset.text_embeds[idx], dtype=jnp.float32)
        return batch


def get_motion_dataloader(
    path: str,
    *,
    batch_size: int,
    njoints: Optional[int] = None,
    nfeats: Optional[int] = None,
    max_frames: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 0,
    split: str = "train",
) -> MotionDataLoader:
    expanded = os.path.expanduser(path)
    if path == "humanml" or (
        os.path.isdir(expanded)
        and os.path.isdir(os.path.join(expanded, "new_joint_vecs"))
        and os.path.isfile(os.path.join(expanded, f"{split}.txt"))
    ):
        root = "motion-diffusion-model/dataset/HumanML3D" if path == "humanml" else expanded
        dataset = load_humanml_dataset(root, split=split, max_frames=max_frames or 196)
    elif os.path.isdir(expanded):
        if njoints is None or nfeats is None:
            raise ValueError("Directory datasets require njoints and nfeats")
        dataset = load_motion_npy_dir(expanded, njoints=njoints, nfeats=nfeats, max_frames=max_frames)
    else:
        dataset = load_motion_npz(expanded, njoints=njoints, nfeats=nfeats)
    return MotionDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
