#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader utilities for climbing board data.

Provides:
- ClimbDataset: PyTorch Dataset wrapper for processed .pt files
- get_dataloader: Creates DataLoader with weighted sampling
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class ClimbDataset(Dataset):
    """
    PyTorch Dataset for processed climbing board data.

    Each sample contains:
        - sequence: token IDs (L_max,)
        - seq_length: number of active holds
        - difficulty: numeric difficulty
        - angle: wall angle in degrees
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
