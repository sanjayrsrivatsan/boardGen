#!/usr/bin/env python3
"""
Embedding layers for the discrete masked diffusion model.

Provides:
- TokenEmbedding: Joint token + coordinate + position embeddings
- TimestepEmbedding: Sinusoidal timestep encoding
- LabelEmbedding: Conditioning label embeddings with null tokens for CFG
"""

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous timesteps."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps of shape (B,) in [0, 1]

        Returns:
            Embeddings of shape (B, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepEmbedding(nn.Module):
    """Timestep embedding with sinusoidal encoding + MLP."""

    def __init__(self, d_model: int, d_time: int = 128):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(d_time)
        self.mlp = nn.Sequential(
            nn.Linear(d_time, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps of shape (B,)

        Returns:
            Embeddings of shape (B, d_model)
        """
        return self.mlp(self.sinusoidal(t))


class TokenEmbedding(nn.Module):
    """
    Combined token + coordinate + sequence position embedding.

    Each token receives:
    - Learned embedding for the (placement, role) token ID
    - Projected spatial (x, y) coordinates from the board
    - Learned sequence position embedding
    """

    def __init__(
        self,
        V_total: int,
        d_model: int,
        L_max: int,
        placement_coords: torch.Tensor,
        MASK_TOKEN: int,
        PAD_TOKEN: int,
    ):
        super().__init__()
        self.V_total = V_total
        self.d_model = d_model
        self.L_max = L_max
        self.MASK_TOKEN = MASK_TOKEN
        self.PAD_TOKEN = PAD_TOKEN

        # Token embeddings for all vocab including MASK and PAD
        self.token_embed = nn.Embedding(V_total, d_model)

        # Coordinate projection
        self.coord_proj = nn.Linear(2, d_model)

        # Store placement coordinates, adding coords for MASK and PAD
        # MASK gets learned coords, PAD gets zeros
        V_pos = placement_coords.shape[0]
        full_coords = torch.zeros(V_total, 2)
        # Repeat coords for each role
        V_role = (V_total - 2) // V_pos
        for i in range(V_pos):
            for j in range(V_role):
                full_coords[i * V_role + j] = placement_coords[i]
        # MASK and PAD coordinates (will be projected, MASK uses learned)
        self.register_buffer("placement_coords", full_coords)
        self.mask_coord = nn.Parameter(torch.zeros(2))

        # Sequence position embeddings
        self.pos_embed = nn.Embedding(L_max, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Token IDs of shape (B, L)

        Returns:
            Embeddings of shape (B, L, d_model)
        """
        B, L = token_ids.shape

        # Token embedding
        tok_emb = self.token_embed(token_ids)  # (B, L, d_model)

        # Coordinate lookup and projection
        coords = self.placement_coords[token_ids]  # (B, L, 2)
        # Replace MASK coordinates with learned embedding
        mask_positions = token_ids == self.MASK_TOKEN
        coords = coords.clone()
        coords[mask_positions] = self.mask_coord

        coord_emb = self.coord_proj(coords)  # (B, L, d_model)

        # Sequence position embedding
        positions = torch.arange(L, device=token_ids.device)
        pos_emb = self.pos_embed(positions).unsqueeze(0)  # (1, L, d_model)

        return tok_emb + coord_emb + pos_emb


class LabelEmbedding(nn.Module):
    """
    Conditioning label embeddings with null tokens for classifier-free guidance.

    Supports:
    - difficulty: continuous
    - angle: discrete (optional, can be disabled for fixed-angle boards)
    - board_size: discrete (optional, can be disabled for single-size boards)
    - is_classic: binary
    - quality: continuous
    """

    def __init__(
        self,
        d_model: int,
        d_cond: int = 64,
        angle_conditioning: bool = True,
        size_conditioning: bool = True,
        max_angle: int = 70,
        n_sizes: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cond = d_cond
        self.angle_conditioning = angle_conditioning
        self.size_conditioning = size_conditioning

        # Individual label embedders
        self.diff_embed = nn.Sequential(
            nn.Linear(1, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        if angle_conditioning:
            self.angle_embed = nn.Embedding(max_angle + 1, d_cond)
        else:
            self.angle_embed = None

        if size_conditioning:
            self.size_embed = nn.Embedding(n_sizes, d_cond)
        else:
            self.size_embed = None

        self.classic_embed = nn.Embedding(2, d_cond)

        self.quality_embed = nn.Sequential(
            nn.Linear(1, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        # Null embeddings for CFG
        self.null_diff = nn.Parameter(torch.zeros(d_cond))
        self.null_angle = nn.Parameter(torch.zeros(d_cond))
        self.null_size = nn.Parameter(torch.zeros(d_cond))
        self.null_classic = nn.Parameter(torch.zeros(d_cond))
        self.null_quality = nn.Parameter(torch.zeros(d_cond))

        # Combiner - count active conditioning signals
        n_conds = 3  # diff, classic, quality always present
        if angle_conditioning:
            n_conds += 1
        if size_conditioning:
            n_conds += 1

        self.combiner = nn.Sequential(
            nn.Linear(n_conds * d_cond, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        difficulty: torch.Tensor,
        angle: torch.Tensor,
        board_size: torch.Tensor,
        is_classic: torch.Tensor,
        quality: torch.Tensor,
        mask_diff: torch.Tensor = None,
        mask_angle: torch.Tensor = None,
        mask_size: torch.Tensor = None,
        mask_classic: torch.Tensor = None,
        mask_quality: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            difficulty: (B,) continuous difficulty
            angle: (B,) discrete angle
            board_size: (B,) discrete size index
            is_classic: (B,) binary
            quality: (B,) continuous quality
            mask_*: (B,) bool tensors, True = use null embedding

        Returns:
            Combined embedding of shape (B, d_model)
        """
        B = difficulty.shape[0]
        device = difficulty.device

        # Difficulty
        diff_emb = self.diff_embed(difficulty.float().unsqueeze(-1))
        if mask_diff is not None:
            diff_emb = torch.where(
                mask_diff.unsqueeze(-1),
                self.null_diff.expand(B, -1),
                diff_emb,
            )

        # Angle
        if self.angle_conditioning:
            angle_emb = self.angle_embed(angle.long().clamp(0, 70))
            if mask_angle is not None:
                angle_emb = torch.where(
                    mask_angle.unsqueeze(-1),
                    self.null_angle.expand(B, -1),
                    angle_emb,
                )
        else:
            angle_emb = None

        # Board size
        if self.size_conditioning:
            size_emb = self.size_embed(board_size.long())
            if mask_size is not None:
                size_emb = torch.where(
                    mask_size.unsqueeze(-1),
                    self.null_size.expand(B, -1),
                    size_emb,
                )
        else:
            size_emb = None

        # Classic
        classic_emb = self.classic_embed(is_classic.long())
        if mask_classic is not None:
            classic_emb = torch.where(
                mask_classic.unsqueeze(-1),
                self.null_classic.expand(B, -1),
                classic_emb,
            )

        # Quality
        quality_emb = self.quality_embed(quality.float().unsqueeze(-1))
        if mask_quality is not None:
            quality_emb = torch.where(
                mask_quality.unsqueeze(-1),
                self.null_quality.expand(B, -1),
                quality_emb,
            )

        # Combine all active embeddings
        parts = [diff_emb]
        if angle_emb is not None:
            parts.append(angle_emb)
        if size_emb is not None:
            parts.append(size_emb)
        parts.extend([classic_emb, quality_emb])

        combined = torch.cat(parts, dim=-1)
        return self.combiner(combined)

    def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get fully null conditioning for unconditional generation."""
        parts = [self.null_diff]
        if self.angle_conditioning:
            parts.append(self.null_angle)
        if self.size_conditioning:
            parts.append(self.null_size)
        parts.extend([self.null_classic, self.null_quality])

        combined = torch.cat(parts)
        return self.combiner(combined.unsqueeze(0).expand(batch_size, -1))
