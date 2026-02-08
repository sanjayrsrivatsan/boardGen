#!/usr/bin/env python3
"""
Sampling (reverse process) for discrete masked diffusion.

Implements confidence-based iterative unmasking with size logit masking.
"""

from typing import Optional, List, Dict

import torch
import torch.nn.functional as F

from model.conditioning import get_conditional_logits
from model.diffusion import cosine_schedule


class DiffusionSampler:
    """
    Iterative unmasking sampler for discrete masked diffusion.

    Uses confidence-based selection to unmask tokens progressively.
    Applies size logit masks to constrain generation to valid placements.
    """

    def __init__(
        self,
        model,
        MASK_TOKEN: int,
        PAD_TOKEN: int,
        L_max: int,
        size_logit_masks: Dict[int, torch.Tensor] = None,
        schedule: str = "cosine",
    ):
        self.model = model
        self.MASK_TOKEN = MASK_TOKEN
        self.PAD_TOKEN = PAD_TOKEN
        self.L_max = L_max
        self.size_logit_masks = size_logit_masks or {}

        if schedule == "cosine":
            self.gamma = cosine_schedule
        else:
            self.gamma = lambda t: t

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_holds: int,
        device: torch.device,
        n_steps: int = 32,
        difficulty: Optional[torch.Tensor] = None,
        angle: Optional[torch.Tensor] = None,
        board_size: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        size_idx: int = None,
        fixed_holds: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Generate climb sequences via iterative unmasking.

        Args:
            n_samples: Number of climbs to generate
            n_holds: Target number of holds per climb
            device: Compute device
            n_steps: Number of diffusion steps
            difficulty, angle, board_size: Conditioning
            guidance_scale: CFG strength
            temperature: Sampling temperature
            size_idx: Board size index for logit masking (required for multi-size boards)
            fixed_holds: Optional list of token IDs to keep fixed

        Returns:
            Generated sequences (n_samples, L_max)
        """
        self.model.eval()

        # Initialize: k MASK tokens + (L_max - k) PAD tokens
        x = torch.full((n_samples, self.L_max), self.PAD_TOKEN, dtype=torch.long, device=device)
        x[:, :n_holds] = self.MASK_TOKEN

        # Handle fixed holds
        if fixed_holds is not None:
            for i, tok in enumerate(fixed_holds):
                if i < n_holds:
                    x[:, i] = tok

        # Track which positions are fixed (never remask)
        fixed_mask = x != self.MASK_TOKEN

        # Get size logit mask
        size_mask = None
        if size_idx is not None and size_idx in self.size_logit_masks:
            size_mask = self.size_logit_masks[size_idx].to(device)

        # Prepare conditioning tensors
        if difficulty is not None and not isinstance(difficulty, torch.Tensor):
            difficulty = torch.full((n_samples,), difficulty, device=device)
        if angle is not None and not isinstance(angle, torch.Tensor):
            angle = torch.full((n_samples,), angle, dtype=torch.long, device=device)
        if board_size is not None and not isinstance(board_size, torch.Tensor):
            board_size = torch.full((n_samples,), board_size, dtype=torch.long, device=device)
        elif board_size is None and size_idx is not None:
            board_size = torch.full((n_samples,), size_idx, dtype=torch.long, device=device)

        # Diffusion steps from t=1 to t=0
        for step in range(n_steps):
            t = 1.0 - step / n_steps
            t_next = 1.0 - (step + 1) / n_steps

            t_tensor = torch.full((n_samples,), t, device=device)

            # Get model predictions with CFG
            logits = get_conditional_logits(
                self.model, x, t_tensor,
                difficulty, angle, board_size,
                guidance_scale,
            )

            # Apply temperature
            logits = logits / temperature

            # Apply size logit mask (hard constraint)
            if size_mask is not None:
                logits[:, :, ~size_mask] = float("-inf")

            # Compute probabilities
            probs = F.softmax(logits, dim=-1)

            # Find MASK positions
            is_mask = x == self.MASK_TOKEN

            # Compute confidence (max prob) at each position
            max_probs, max_tokens = probs.max(dim=-1)

            # Number of tokens to unmask this step
            gamma_t = self.gamma(torch.tensor(t))
            gamma_next = self.gamma(torch.tensor(t_next))
            n_masked_now = (is_mask.float().sum(dim=1)).int()
            frac_to_unmask = (gamma_t - gamma_next) / gamma_t if gamma_t > 0 else 1.0
            n_to_unmask = (n_masked_now.float() * frac_to_unmask).ceil().int()

            # For each sample, unmask highest-confidence MASK positions
            for i in range(n_samples):
                mask_positions = is_mask[i].nonzero(as_tuple=True)[0]
                if len(mask_positions) == 0:
                    continue

                # Get confidences at mask positions
                confidences = max_probs[i, mask_positions]

                # Select top-k positions to unmask
                k = min(n_to_unmask[i].item(), len(mask_positions))
                if k == 0:
                    continue

                _, top_indices = confidences.topk(k)
                positions_to_unmask = mask_positions[top_indices]

                # Sample tokens for selected positions
                for pos in positions_to_unmask:
                    token = torch.multinomial(probs[i, pos], 1).item()
                    x[i, pos] = token

        return x

    @torch.no_grad()
    def sample_with_length_dist(
        self,
        n_samples: int,
        length_dist: torch.Tensor,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample with varying sequence lengths from a distribution.

        Args:
            n_samples: Number of samples
            length_dist: (L_max,) probability distribution over lengths
            device: Compute device
            **kwargs: Other arguments passed to sample()

        Returns:
            Generated sequences (n_samples, L_max)
        """
        # Sample lengths
        lengths = torch.multinomial(length_dist, n_samples, replacement=True)

        # Generate each length separately (could be batched more efficiently)
        results = []
        for length in lengths:
            x = self.sample(1, length.item(), device, **kwargs)
            results.append(x)

        return torch.cat(results, dim=0)
