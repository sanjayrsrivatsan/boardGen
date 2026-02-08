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
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, L, d_model)
            cond: Conditioning of shape (B, d_model)
            key_padding_mask: (B, L) bool mask, True = PAD position to ignore

        Returns:
            Output of shape (B, L, d_model)
        """
        # Self-attention with residual
        normed = self.adaln1(x, cond)

        # Use key_padding_mask to ignore PAD positions in attention
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
        )
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
        mask_diff: torch.Tensor = None,
        mask_angle: torch.Tensor = None,
        mask_size: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            token_ids: (B, L) token sequences
            t: (B,) timesteps in [0, 1]
            difficulty, angle, board_size: conditioning labels
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
            difficulty, angle, board_size,
            mask_diff, mask_angle, mask_size,
        )  # (B, d_model)

        # Combined conditioning
        cond = t_emb + y_emb  # (B, d_model)

        # Key padding mask: True for PAD positions to ignore
        key_padding_mask = token_ids == self.PAD_TOKEN  # (B, L)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, cond, key_padding_mask)

        # Final norm and output projection
        x = self.final_norm(x)
        logits = self.output_proj(x)  # (B, L, V_total - 1)

        return logits

    # Model size configurations for scaling experiments
    MODEL_SIZES = {
        "tiny":   {"d_model": 64,  "n_heads": 2,  "n_layers": 2,  "d_ff": 256},
        "small":  {"d_model": 128, "n_heads": 4,  "n_layers": 4,  "d_ff": 512},
        "medium": {"d_model": 192, "n_heads": 6,  "n_layers": 6,  "d_ff": 768},
        "large":  {"d_model": 256, "n_heads": 8,  "n_layers": 8,  "d_ff": 1024},
        "xlarge": {"d_model": 384, "n_heads": 12, "n_layers": 12, "d_ff": 1536},
    }

    @classmethod
    def from_config(cls, config: dict, vocab_meta: dict) -> "DiffusionTransformer":
        """Create model from config and vocabulary metadata."""
        model_size = config.get("model_size", "small")

        if model_size in cls.MODEL_SIZES:
            size_config = cls.MODEL_SIZES[model_size]
            d_model = size_config["d_model"]
            n_heads = size_config["n_heads"]
            n_layers = size_config["n_layers"]
            d_ff = size_config["d_ff"]
        elif model_size == "full":  # Legacy support
            d_model, n_heads, n_layers, d_ff = 256, 8, 6, 1024
        else:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(cls.MODEL_SIZES.keys())}")

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
