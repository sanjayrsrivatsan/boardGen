#!/usr/bin/env python3
"""
Transformer encoder architecture for discrete masked diffusion.

Implements a bidirectional transformer with:
- Adaptive Layer Normalization (AdaLN) for timestep/label conditioning
- Attention masking for PAD tokens
- Output projection to vocabulary logits
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):
    """Adaptive Layer Normalization - modulates LayerNorm with conditioning."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Conditioning produces scale and shift
        self.cond_proj = nn.Linear(d_model, 2 * d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, L, d_model)
            cond: Conditioning of shape (B, d_model)

        Returns:
            Modulated output of shape (B, L, d_model)
        """
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        # Expand for sequence dimension
        scale = scale.unsqueeze(1)  # (B, 1, d_model)
        shift = shift.unsqueeze(1)  # (B, 1, d_model)
        return self.norm(x) * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """Single transformer block with AdaLN conditioning."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Self-attention
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN for both sublayers
        self.adaln1 = AdaLN(d_model)
        self.adaln2 = AdaLN(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, L, d_model)
            cond: Conditioning of shape (B, d_model)
            attn_mask: Attention mask of shape (B, L, L), True = masked

        Returns:
            Output of shape (B, L, d_model)
        """
        # Self-attention with residual
        normed = self.adaln1(x, cond)

        # Expand attention mask for multihead attention
        if attn_mask is not None:
            # Convert (B, L, L) bool mask to (B*n_heads, L, L) float mask
            B, L, _ = attn_mask.shape
            # True means "do not attend", need to convert to float with -inf
            attn_mask_float = attn_mask.float().masked_fill(attn_mask, float("-inf"))
            attn_mask_float = attn_mask_float.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            attn_mask_float = attn_mask_float.reshape(B * self.n_heads, L, L)
        else:
            attn_mask_float = None

        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask_float)
        x = x + self.dropout(attn_out)

        # Feedforward with residual
        normed = self.adaln2(x, cond)
        x = x + self.ff(normed)

        return x


class DiffusionTransformer(nn.Module):
    """
    Transformer encoder for discrete masked diffusion.

    Takes token sequences and conditioning, outputs logits over vocabulary.
    """

    def __init__(
        self,
        V_total: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        L_max: int,
        placement_coords: torch.Tensor,
        MASK_TOKEN: int,
        PAD_TOKEN: int,
        angle_conditioning: bool = True,
        size_conditioning: bool = True,
        n_sizes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.V_total = V_total
        self.d_model = d_model
        self.L_max = L_max
        self.MASK_TOKEN = MASK_TOKEN
        self.PAD_TOKEN = PAD_TOKEN

        # Import here to avoid circular imports
        from model.embeddings import TokenEmbedding, TimestepEmbedding, LabelEmbedding

        # Embeddings
        self.token_embedding = TokenEmbedding(
            V_total, d_model, L_max, placement_coords, MASK_TOKEN, PAD_TOKEN
        )
        self.timestep_embedding = TimestepEmbedding(d_model)
        self.label_embedding = LabelEmbedding(
            d_model,
            angle_conditioning=angle_conditioning,
            size_conditioning=size_conditioning,
            n_sizes=n_sizes,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection (predict over all non-PAD tokens)
        self.output_proj = nn.Linear(d_model, V_total - 1)

    def forward(
        self,
        token_ids: torch.Tensor,
        t: torch.Tensor,
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
        Forward pass.

        Args:
            token_ids: (B, L) token sequences
            t: (B,) timesteps in [0, 1]
            difficulty, angle, board_size, is_classic, quality: conditioning labels
            mask_*: CFG masking for each label

        Returns:
            logits: (B, L, V_total - 1) logits over non-PAD vocabulary
        """
        B, L = token_ids.shape

        # Token embeddings
        x = self.token_embedding(token_ids)  # (B, L, d_model)

        # Timestep conditioning
        t_emb = self.timestep_embedding(t)  # (B, d_model)

        # Label conditioning
        y_emb = self.label_embedding(
            difficulty, angle, board_size, is_classic, quality,
            mask_diff, mask_angle, mask_size, mask_classic, mask_quality,
        )  # (B, d_model)

        # Combined conditioning
        cond = t_emb + y_emb  # (B, d_model)

        # Attention mask: don't attend to/from PAD positions
        pad_mask = token_ids == self.PAD_TOKEN  # (B, L)
        # Create (B, L, L) mask where True means "do not attend"
        attn_mask = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, cond, attn_mask)

        # Final norm and output projection
        x = self.final_norm(x)
        logits = self.output_proj(x)  # (B, L, V_total - 1)

        return logits

    @classmethod
    def from_config(cls, config: dict, vocab_meta: dict) -> "DiffusionTransformer":
        """Create model from config and vocabulary metadata."""
        model_size = config.get("model_size", "small")

        if model_size == "small":
            d_model, n_heads, n_layers, d_ff = 128, 4, 4, 512
        else:  # "full"
            d_model, n_heads, n_layers, d_ff = 256, 8, 6, 1024

        n_sizes = len(vocab_meta.get("board_sizes", ["default"]))

        return cls(
            V_total=vocab_meta["V_total"],
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            L_max=vocab_meta["L_max"],
            placement_coords=vocab_meta["placement_coords"],
            MASK_TOKEN=vocab_meta["MASK_TOKEN"],
            PAD_TOKEN=vocab_meta["PAD_TOKEN"],
            angle_conditioning=config.get("angle_conditioning", True),
            size_conditioning=config.get("size_conditioning", True),
            n_sizes=n_sizes,
        )
