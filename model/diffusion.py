#!/usr/bin/env python3
"""
Discrete masked diffusion process.

Implements:
- Forward process: mask tokens with probability gamma(t)
- Noise schedules (cosine, linear)
- Loss computation on masked positions only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """Cosine noise schedule: gamma(t) = 1 - cos(pi*t/2)^2"""
    return 1 - torch.cos(torch.pi * t / 2) ** 2


def linear_schedule(t: torch.Tensor) -> torch.Tensor:
    """Linear noise schedule: gamma(t) = t"""
    return t


class MaskedDiffusion(nn.Module):
    """
    Masked discrete diffusion process for token sequences.

    Forward process: independently replace each active token with MASK
    with probability gamma(t).
    """

    def __init__(
        self,
        MASK_TOKEN: int,
        PAD_TOKEN: int,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.MASK_TOKEN = MASK_TOKEN
        self.PAD_TOKEN = PAD_TOKEN

        if schedule == "cosine":
            self.gamma = cosine_schedule
        elif schedule == "linear":
            self.gamma = linear_schedule
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply forward masking process.

        Args:
            x_0: Clean token sequence (B, L)
            t: Noise level (B,) in [0, 1]

        Returns:
            x_t: Noised sequence with some tokens replaced by MASK
        """
        B, L = x_0.shape
        device = x_0.device

        # Compute masking probability for each sample
        gamma_t = self.gamma(t)  # (B,)

        # Generate mask: True = replace with MASK token
        mask_prob = gamma_t.unsqueeze(1).expand(-1, L)  # (B, L)
        should_mask = torch.rand(B, L, device=device) < mask_prob

        # Don't mask PAD tokens
        is_pad = x_0 == self.PAD_TOKEN
        should_mask = should_mask & ~is_pad

        # Apply masking
        x_t = x_0.clone()
        x_t[should_mask] = self.MASK_TOKEN

        return x_t, should_mask

    def compute_loss(
        self,
        logits: torch.Tensor,
        x_0: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss only at masked positions.

        Args:
            logits: Model output (B, L, V-1)
            x_0: Ground truth tokens (B, L)
            mask: Boolean mask (B, L), True = was masked

        Returns:
            loss: Scalar loss value
        """
        B, L, V_minus_1 = logits.shape

        # Flatten for loss computation
        logits_flat = logits.view(-1, V_minus_1)  # (B*L, V-1)
        targets_flat = x_0.view(-1)  # (B*L,)
        mask_flat = mask.view(-1)  # (B*L,)

        # Only compute loss on masked positions
        if mask_flat.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        masked_logits = logits_flat[mask_flat]  # (n_masked, V-1)
        masked_targets = targets_flat[mask_flat]  # (n_masked,)

        loss = F.cross_entropy(masked_logits, masked_targets)
        return loss


class DiffusionTrainer:
    """Training utilities for masked diffusion."""

    def __init__(
        self,
        model: nn.Module,
        diffusion: MaskedDiffusion,
        p_uncond: float = 0.1,
    ):
        self.model = model
        self.diffusion = diffusion
        self.p_uncond = p_uncond

    def training_step(
        self,
        batch: dict,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Single training step.

        Args:
            batch: Dictionary with sequence and conditioning labels
            device: Target device

        Returns:
            loss: Scalar loss
        """
        # Move batch to device
        x_0 = batch["sequence"].to(device)
        difficulty = batch["difficulty"].to(device)
        angle = batch["angle"].to(device)
        board_size = batch["board_size"].to(device)
        is_classic = batch["is_classic"].to(device)
        quality = batch["quality"].to(device)

        B = x_0.shape[0]

        # Sample random timestep
        t = torch.rand(B, device=device)

        # Forward process: mask tokens
        x_t, mask = self.diffusion.forward_process(x_0, t)

        # CFG: randomly drop conditioning with probability p_uncond
        mask_diff = torch.rand(B, device=device) < self.p_uncond
        mask_angle = torch.rand(B, device=device) < self.p_uncond
        mask_size = torch.rand(B, device=device) < self.p_uncond
        mask_classic = torch.rand(B, device=device) < self.p_uncond
        mask_quality = torch.rand(B, device=device) < self.p_uncond

        # Sometimes drop all labels together
        mask_all = torch.rand(B, device=device) < self.p_uncond
        mask_diff = mask_diff | mask_all
        mask_angle = mask_angle | mask_all
        mask_size = mask_size | mask_all
        mask_classic = mask_classic | mask_all
        mask_quality = mask_quality | mask_all

        # Forward pass
        logits = self.model(
            x_t, t, difficulty, angle, board_size, is_classic, quality,
            mask_diff, mask_angle, mask_size, mask_classic, mask_quality,
        )

        # Compute loss
        loss = self.diffusion.compute_loss(logits, x_0, mask)

        return loss
