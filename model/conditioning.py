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
    Apply classifier-free guidance (standard formula).

    Args:
        logits_cond: Conditional logits (B, L, V)
        logits_uncond: Unconditional logits (B, L, V)
        guidance_scale: CFG strength (s >= 1 typical, s=1 means no guidance)
            s=1: just conditional
            s>1: amplify difference from unconditional

    Returns:
        Guided logits
    """
    # Standard CFG: uncond + s * (cond - uncond) = (1-s)*uncond + s*cond
    # When s=1, returns cond. When s>1, extrapolates beyond cond.
    return logits_uncond + guidance_scale * (logits_cond - logits_uncond)


def get_conditional_logits(
    model,
    x_t: torch.Tensor,
    t: torch.Tensor,
    difficulty: torch.Tensor = None,
    angle: torch.Tensor = None,
    board_size: torch.Tensor = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Get CFG-adjusted logits for sampling.

    Args:
        model: Diffusion transformer model
        x_t: Current noisy sequence (B, L)
        t: Current timestep (B,)
        difficulty, angle, board_size: Optional conditioning
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
    if board_size is None:
        board_size = torch.zeros(B, dtype=torch.long, device=device)

    # Conditional forward pass (no masking)
    logits_cond = model(
        x_t, t, difficulty, angle, board_size,
        mask_diff=None, mask_angle=None, mask_size=None,
    )

    if guidance_scale == 0.0:
        return logits_cond

    # Unconditional forward pass (all labels masked)
    all_true = torch.ones(B, dtype=torch.bool, device=device)
    logits_uncond = model(
        x_t, t, difficulty, angle, board_size,
        mask_diff=all_true, mask_angle=all_true, mask_size=all_true,
    )

    # Apply CFG
    return apply_cfg(logits_cond, logits_uncond, guidance_scale)
