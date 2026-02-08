#!/usr/bin/env python3
"""
Classifier-free guidance utilities.

Provides functions for applying CFG during sampling.
"""

import torch
import torch.nn.functional as F


def apply_cfg(
    logits_cond: torch.Tensor,
    logits_uncond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    Apply classifier-free guidance.

    Args:
        logits_cond: Conditional logits (B, L, V)
        logits_uncond: Unconditional logits (B, L, V)
        guidance_scale: CFG strength (s >= 0, typical 1-5)

    Returns:
        Guided logits
    """
    return (1 + guidance_scale) * logits_cond - guidance_scale * logits_uncond


def get_conditional_logits(
    model,
    x_t: torch.Tensor,
    t: torch.Tensor,
    difficulty: torch.Tensor = None,
    angle: torch.Tensor = None,
    is_classic: torch.Tensor = None,
    quality: torch.Tensor = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Get CFG-adjusted logits for sampling.

    Args:
        model: Diffusion transformer model
        x_t: Current noisy sequence (B, L)
        t: Current timestep (B,)
        difficulty, angle, is_classic, quality: Optional conditioning
        guidance_scale: CFG strength

    Returns:
        CFG-adjusted logits (B, L, V-1)
    """
    B, L = x_t.shape
    device = x_t.device

    # Default conditioning values if not provided
    if difficulty is None:
        difficulty = torch.zeros(B, device=device)
    if angle is None:
        angle = torch.zeros(B, dtype=torch.long, device=device)
    if is_classic is None:
        is_classic = torch.zeros(B, dtype=torch.bool, device=device)
    if quality is None:
        quality = torch.zeros(B, device=device)

    # Conditional forward pass (no masking)
    logits_cond = model(
        x_t, t, difficulty, angle, is_classic, quality,
        mask_diff=None, mask_angle=None, mask_classic=None, mask_quality=None,
    )

    if guidance_scale == 0.0:
        return logits_cond

    # Unconditional forward pass (all labels masked)
    all_true = torch.ones(B, dtype=torch.bool, device=device)
    logits_uncond = model(
        x_t, t, difficulty, angle, is_classic, quality,
        mask_diff=all_true, mask_angle=all_true,
        mask_classic=all_true, mask_quality=all_true,
    )

    # Apply CFG
    return apply_cfg(logits_cond, logits_uncond, guidance_scale)
