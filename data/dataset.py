#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader utilities for climbing board data.

Provides:
- ClimbDataset: PyTorch Dataset wrapper for processed .pt files
- get_dataloader: Creates DataLoader with weighted sampling
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset


class ClimbDataset(Dataset):
    """
    PyTorch Dataset for processed climbing board data.

    Each sample contains:
        - sequence: token IDs (L_max,)
        - seq_length: number of active holds
        - difficulty: numeric difficulty
        - angle: wall angle in degrees
        - board_size: board size index
        - is_classic: benchmark/classic status
        - quality: average star rating
    """

    def __init__(self, data_path: str | Path):
        """Load processed dataset from .pt file."""
        self.data = torch.load(data_path, weights_only=False)
        self.vocab_meta = self.data["vocab_meta"]

    def __len__(self) -> int:
        return len(self.data["sequences"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "sequence": self.data["sequences"][idx],
            "seq_length": self.data["seq_lengths"][idx],
            "difficulty": self.data["difficulty"][idx],
            "angle": self.data["angle"][idx],
            "board_size": self.data["board_size"][idx],
            "is_classic": self.data["is_classic"][idx],
            "quality": self.data["quality"][idx],
        }

    @property
    def sample_weights(self) -> torch.Tensor:
        """Get sample weights for weighted sampling."""
        return self.data["sample_weight"]

    @property
    def board(self) -> str:
        return self.data["board"]

    @property
    def L_max(self) -> int:
        return self.vocab_meta["L_max"]

    @property
    def V_total(self) -> int:
        return self.vocab_meta["V_total"]

    @property
    def MASK_TOKEN(self) -> int:
        return self.vocab_meta["MASK_TOKEN"]

    @property
    def PAD_TOKEN(self) -> int:
        return self.vocab_meta["PAD_TOKEN"]

    @property
    def board_sizes(self) -> list:
        return self.vocab_meta.get("board_sizes", [])

    @property
    def n_sizes(self) -> int:
        return len(self.board_sizes)

    def get_size_logit_mask(self, size_idx: int) -> torch.Tensor:
        """Get logit mask for a specific board size."""
        return self.vocab_meta["size_logit_masks"].get(size_idx, None)


def get_dataloader(
    dataset: ClimbDataset,
    batch_size: int,
    weighted: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader with optional weighted sampling.

    Args:
        dataset: ClimbDataset instance
        batch_size: Batch size
        weighted: Use weighted random sampling based on sample_weight
        num_workers: Number of worker processes (0 for MPS compatibility)
        shuffle: Shuffle data (ignored if weighted=True)

    Returns:
        DataLoader instance
    """
    if weighted:
        sampler = WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,  # MPS uses unified memory
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Custom collate function (default works fine, included for extensibility)."""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }


def train_val_split(
    dataset: ClimbDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple["TrainValDataset", "TrainValDataset"]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Full ClimbDataset
        val_fraction: Fraction of data for validation (default 10%)
        seed: Random seed for reproducibility

    Returns:
        (train_dataset, val_dataset) tuple
    """
    n = len(dataset)
    n_val = int(n * val_fraction)
    n_train = n - n_val

    # Deterministic split
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = TrainValDataset(dataset, train_indices)
    val_dataset = TrainValDataset(dataset, val_indices)

    return train_dataset, val_dataset


class TrainValDataset(Dataset):
    """
    Subset wrapper that preserves ClimbDataset properties.
    """

    def __init__(self, parent: ClimbDataset, indices: List[int]):
        self.parent = parent
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.parent[self.indices[idx]]

    @property
    def vocab_meta(self):
        return self.parent.vocab_meta

    @property
    def sample_weights(self) -> torch.Tensor:
        """Get sample weights for this subset."""
        return self.parent.sample_weights[self.indices]

    @property
    def board(self) -> str:
        return self.parent.board

    @property
    def L_max(self) -> int:
        return self.parent.L_max

    @property
    def V_total(self) -> int:
        return self.parent.V_total

    @property
    def MASK_TOKEN(self) -> int:
        return self.parent.MASK_TOKEN

    @property
    def PAD_TOKEN(self) -> int:
        return self.parent.PAD_TOKEN

    @property
    def board_sizes(self) -> list:
        return self.parent.board_sizes

    @property
    def n_sizes(self) -> int:
        return self.parent.n_sizes

    def get_size_logit_mask(self, size_idx: int) -> torch.Tensor:
        return self.parent.get_size_logit_mask(size_idx)
