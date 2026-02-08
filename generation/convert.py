#!/usr/bin/env python3
"""
Token sequence <-> frames string conversion utilities.

Handles conversion between:
- Token IDs (model representation)
- Aurora frames strings (board app format)
- Human-readable hold lists
"""

from typing import List, Dict, Tuple

import torch


def tokens_to_frames(token_ids: torch.Tensor, vocab_meta: dict) -> str:
    """
    Convert token ID sequence to Aurora frames string.

    Args:
        token_ids: (L,) tensor of token IDs
        vocab_meta: Vocabulary metadata

    Returns:
        Frames string like "p1083r15p1117r15..."
    """
    V_role = vocab_meta["V_role"]
    MASK = vocab_meta["MASK_TOKEN"]
    PAD = vocab_meta["PAD_TOKEN"]
    placement_idx_to_id = vocab_meta["placement_index_to_id"]
    role_idx_to_id = vocab_meta["role_index_to_id"]

    parts = []
    for tok in token_ids.tolist():
        if tok == MASK or tok == PAD:
            continue

        placement_idx = tok // V_role
        role_idx = tok % V_role

        placement_id = placement_idx_to_id[placement_idx]
        role_id = role_idx_to_id[role_idx]

        parts.append(f"p{placement_id}r{role_id}")

    return "".join(parts)


def frames_to_tokens(frames: str, vocab_meta: dict) -> torch.Tensor:
    """
    Convert Aurora frames string to token sequence.

    Args:
        frames: Frames string like "p1083r15p1117r15..."
        vocab_meta: Vocabulary metadata

    Returns:
        (L,) tensor of token IDs (not padded)
    """
    import re

    V_role = vocab_meta["V_role"]
    id_to_placement_idx = vocab_meta["id_to_placement_index"]
    id_to_role_idx = vocab_meta["id_to_role_index"]

    pattern = r"p(\d+)r(\d+)"
    matches = re.findall(pattern, frames)

    tokens = []
    for p_str, r_str in matches:
        p_id = int(p_str)
        r_id = int(r_str)

        if p_id not in id_to_placement_idx or r_id not in id_to_role_idx:
            continue

        p_idx = id_to_placement_idx[p_id]
        r_idx = id_to_role_idx[r_id]

        token_id = p_idx * V_role + r_idx
        tokens.append(token_id)

    return torch.tensor(tokens, dtype=torch.long)


def tokens_to_hold_list(token_ids: torch.Tensor, vocab_meta: dict) -> List[Dict]:
    """
    Convert token sequence to human-readable hold list.

    Args:
        token_ids: (L,) tensor of token IDs
        vocab_meta: Vocabulary metadata

    Returns:
        List of hold dictionaries with placement_id, role, coordinates
    """
    V_role = vocab_meta["V_role"]
    MASK = vocab_meta["MASK_TOKEN"]
    PAD = vocab_meta["PAD_TOKEN"]
    placement_idx_to_id = vocab_meta["placement_index_to_id"]
    role_idx_to_id = vocab_meta["role_index_to_id"]
    role_names = vocab_meta["role_names"]
    coords = vocab_meta["placement_coords"]

    holds = []
    for tok in token_ids.tolist():
        if tok == MASK or tok == PAD:
            continue

        placement_idx = tok // V_role
        role_idx = tok % V_role

        placement_id = placement_idx_to_id[placement_idx]
        role_id = role_idx_to_id[role_idx]
        role_name = role_names[role_idx] if role_idx < len(role_names) else f"role_{role_id}"

        x, y = coords[placement_idx].tolist()

        holds.append({
            "placement_id": placement_id,
            "role_id": role_id,
            "role": role_name,
            "x": x,
            "y": y,
        })

    return holds


def hold_list_to_tokens(holds: List[Dict], vocab_meta: dict) -> torch.Tensor:
    """
    Convert hold list to token sequence.

    Args:
        holds: List of hold dicts with placement_id and role_id
        vocab_meta: Vocabulary metadata

    Returns:
        (L,) tensor of token IDs
    """
    V_role = vocab_meta["V_role"]
    id_to_placement_idx = vocab_meta["id_to_placement_index"]
    id_to_role_idx = vocab_meta["id_to_role_index"]

    tokens = []
    for hold in holds:
        p_id = hold["placement_id"]
        r_id = hold["role_id"]

        if p_id not in id_to_placement_idx or r_id not in id_to_role_idx:
            continue

        p_idx = id_to_placement_idx[p_id]
        r_idx = id_to_role_idx[r_id]

        token_id = p_idx * V_role + r_idx
        tokens.append(token_id)

    return torch.tensor(tokens, dtype=torch.long)
