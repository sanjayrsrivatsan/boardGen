#!/usr/bin/env python3
"""
Post-processing and validity checks for generated climbs.

Applies hard constraints:
- No duplicate placements
- At least 1 start hold
- At least 1 finish hold
- Minimum/maximum hold count
"""

from typing import Tuple, Set

import torch


def apply_validity_checks(
    sequence: torch.Tensor,
    vocab_meta: dict,
    min_holds: int = 4,
    max_holds: int = 20,
) -> Tuple[torch.Tensor, bool]:
    """
    Apply validity checks and fix violations where possible.

    Args:
        sequence: (L,) tensor of token IDs
        vocab_meta: Vocabulary metadata
        min_holds: Minimum required holds
        max_holds: Maximum allowed holds

    Returns:
        (fixed_sequence, is_valid) tuple
    """
    V_role = vocab_meta["V_role"]
    PAD = vocab_meta["PAD_TOKEN"]
    MASK = vocab_meta["MASK_TOKEN"]
    coords = vocab_meta["placement_coords"]

    # Extract active holds
    active_mask = (sequence != PAD) & (sequence != MASK)
    active_tokens = sequence[active_mask].tolist()

    if len(active_tokens) < min_holds:
        return sequence, False

    if len(active_tokens) > max_holds:
        return sequence, False

    # Check for duplicate placements
    placements = [tok // V_role for tok in active_tokens]
    if len(placements) != len(set(placements)):
        # Remove duplicates, keeping first occurrence
        seen: Set[int] = set()
        deduped = []
        for tok in active_tokens:
            p = tok // V_role
            if p not in seen:
                seen.add(p)
                deduped.append(tok)
        active_tokens = deduped

        if len(active_tokens) < min_holds:
            return sequence, False

    # Infer role indices from role_names
    role_names = vocab_meta["role_names"]
    start_role_idx = None
    finish_role_idx = None

    for i, name in enumerate(role_names):
        name_lower = name.lower()
        if "start" in name_lower:
            start_role_idx = i
        elif "finish" in name_lower or "end" in name_lower or "top" in name_lower:
            finish_role_idx = i

    # Check for start/finish holds
    roles = [tok % V_role for tok in active_tokens]
    has_start = start_role_idx is not None and start_role_idx in roles
    has_finish = finish_role_idx is not None and finish_role_idx in roles

    # Fix missing start: assign to lowest hold
    if not has_start and start_role_idx is not None and len(active_tokens) > 0:
        # Find lowest hold (min y)
        min_y = float("inf")
        min_idx = 0
        for i, tok in enumerate(active_tokens):
            p_idx = tok // V_role
            y = coords[p_idx, 1].item()
            if y < min_y:
                min_y = y
                min_idx = i

        # Change role to start
        old_tok = active_tokens[min_idx]
        p_idx = old_tok // V_role
        active_tokens[min_idx] = p_idx * V_role + start_role_idx
        has_start = True

    # Fix missing finish: assign to highest hold
    if not has_finish and finish_role_idx is not None and len(active_tokens) > 0:
        # Find highest hold (max y)
        max_y = float("-inf")
        max_idx = 0
        for i, tok in enumerate(active_tokens):
            p_idx = tok // V_role
            y = coords[p_idx, 1].item()
            if y > max_y:
                max_y = y
                max_idx = i

        # Change role to finish
        old_tok = active_tokens[max_idx]
        p_idx = old_tok // V_role
        active_tokens[max_idx] = p_idx * V_role + finish_role_idx
        has_finish = True

    # Rebuild sequence
    L_max = vocab_meta["L_max"]
    new_seq = torch.full((L_max,), PAD, dtype=torch.long)
    for i, tok in enumerate(active_tokens):
        if i < L_max:
            new_seq[i] = tok

    is_valid = len(active_tokens) >= min_holds and has_start and has_finish

    return new_seq, is_valid


def check_climb_validity(
    sequence: torch.Tensor,
    vocab_meta: dict,
    min_holds: int = 4,
    max_holds: int = 20,
) -> dict:
    """
    Check validity without modifying the sequence.

    Returns dictionary with detailed validity information.
    """
    V_role = vocab_meta["V_role"]
    PAD = vocab_meta["PAD_TOKEN"]
    MASK = vocab_meta["MASK_TOKEN"]

    active_mask = (sequence != PAD) & (sequence != MASK)
    active_tokens = sequence[active_mask].tolist()
    n_holds = len(active_tokens)

    placements = [tok // V_role for tok in active_tokens]
    has_duplicates = len(placements) != len(set(placements))

    role_names = vocab_meta["role_names"]
    roles = [tok % V_role for tok in active_tokens]

    has_start = any("start" in role_names[r].lower() for r in roles if r < len(role_names))
    has_finish = any(
        any(x in role_names[r].lower() for x in ["finish", "end", "top"])
        for r in roles if r < len(role_names)
    )

    return {
        "n_holds": n_holds,
        "min_holds_ok": n_holds >= min_holds,
        "max_holds_ok": n_holds <= max_holds,
        "no_duplicates": not has_duplicates,
        "has_start": has_start,
        "has_finish": has_finish,
        "is_valid": (
            n_holds >= min_holds
            and n_holds <= max_holds
            and not has_duplicates
            and has_start
            and has_finish
        ),
    }
