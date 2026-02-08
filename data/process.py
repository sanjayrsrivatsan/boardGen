#!/usr/bin/env python3
"""
Stage 1.2-1.8: Data Processing Pipeline

Converts SQLite database to token sequences for diffusion model training.
Handles vocabulary construction, coordinate normalization, canonical ordering,
filtering, and sample weighting.

Usage:
    python data/process.py --config configs/tension.yaml
    python data/process.py --config configs/kilter.yaml
    python data/process.py --config configs/moonboard.yaml
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> dict:
    """Load board configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def connect_db(db_path: str) -> sqlite3.Connection:
    """Connect to SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def discover_layouts_and_sets(conn: sqlite3.Connection) -> None:
    """Print available layouts and hold sets for configuration."""
    print("\n=== Available Layouts ===")
    cursor = conn.execute("""
        SELECT l.id, l.name, l.product_id, l.is_mirrored
        FROM layouts l
        ORDER BY l.id
    """)
    for row in cursor:
        print(f"  Layout {row['id']}: {row['name']} (mirrored={row['is_mirrored']})")

    print("\n=== Available Product Size/Layout/Set Combinations ===")
    cursor = conn.execute("""
        SELECT psls.id, psls.product_size_id, psls.layout_id, psls.set_id,
               l.name as layout_name
        FROM product_sizes_layouts_sets psls
        JOIN layouts l ON psls.layout_id = l.id
        ORDER BY psls.layout_id, psls.set_id
    """)
    for row in cursor:
        print(f"  ID {row['id']}: layout={row['layout_id']} ({row['layout_name']}), set={row['set_id']}")


def get_valid_placements(
    conn: sqlite3.Connection,
    layout_id: int,
    set_ids: List[int],
) -> Dict[int, Tuple[float, float]]:
    """
    Get valid placements for the target layout and hold sets.
    Returns dict: placement_id -> (x, y) coordinates.
    """
    set_ids_str = ",".join(str(s) for s in set_ids)
    cursor = conn.execute(f"""
        SELECT p.id as placement_id, h.x, h.y
        FROM placements p
        JOIN holes h ON p.hole_id = h.id
        WHERE p.set_id IN ({set_ids_str})
        ORDER BY p.id
    """)
    return {row["placement_id"]: (row["x"], row["y"]) for row in cursor}


def get_board_extents(conn: sqlite3.Connection, layout_id: int) -> Tuple[float, float, float, float]:
    """Get board coordinate extents for normalization."""
    cursor = conn.execute("""
        SELECT ps.edge_left, ps.edge_right, ps.edge_bottom, ps.edge_top
        FROM product_sizes ps
        JOIN product_sizes_layouts_sets psls ON ps.id = psls.product_size_id
        WHERE psls.layout_id = ?
        LIMIT 1
    """, (layout_id,))
    row = cursor.fetchone()
    if row:
        return row["edge_left"], row["edge_right"], row["edge_bottom"], row["edge_top"]
    # Fallback: compute from holes
    cursor = conn.execute("SELECT MIN(x), MAX(x), MIN(y), MAX(y) FROM holes")
    row = cursor.fetchone()
    return row[0], row[1], row[2], row[3]


def get_roles(conn: sqlite3.Connection) -> Dict[int, str]:
    """Get role ID -> name mapping."""
    cursor = conn.execute("SELECT id, full_name FROM placement_roles ORDER BY id")
    return {row["id"]: row["full_name"] for row in cursor}


def parse_frames(frames: str) -> List[Tuple[int, int]]:
    """
    Parse Aurora frames string into list of (placement_id, role_id) pairs.
    Format: p{placement_id}r{role_id}p{placement_id}r{role_id}...
    """
    pattern = r"p(\d+)r(\d+)"
    matches = re.findall(pattern, frames)
    return [(int(p), int(r)) for p, r in matches]


def get_climbs(
    conn: sqlite3.Connection,
    layout_id: int,
    valid_placement_ids: set,
    L_max: int,
) -> List[Dict[str, Any]]:
    """
    Fetch valid climbs with their stats and conditioning labels.
    Applies filtering criteria from spec section 1.5.6.
    """
    cursor = conn.execute("""
        SELECT
            c.uuid,
            c.frames,
            c.layout_id,
            cs.angle,
            cs.display_difficulty,
            cs.difficulty_average,
            cs.quality_average,
            cs.benchmark_difficulty,
            cs.ascent_count
        FROM climbs c
        JOIN climb_stats cs ON c.uuid = cs.climb_uuid
        WHERE c.is_listed = 1
          AND c.is_draft = 0
          AND c.layout_id = ?
          AND cs.ascent_count >= 1
        ORDER BY c.uuid, cs.angle
    """, (layout_id,))

    climbs = []
    for row in cursor:
        holds = parse_frames(row["frames"])

        # Filter: all placements must be valid
        if not all(p in valid_placement_ids for p, r in holds):
            continue

        # Filter: respect L_max
        if len(holds) > L_max:
            continue

        # Filter: minimum holds
        if len(holds) < 4:
            continue

        climbs.append({
            "uuid": row["uuid"],
            "holds": holds,
            "angle": row["angle"],
            "difficulty": row["display_difficulty"] or row["difficulty_average"] or 0,
            "quality": row["quality_average"] or 0,
            "is_classic": row["benchmark_difficulty"] is not None,
            "ascent_count": row["ascent_count"] or 0,
        })

    return climbs


def build_vocabulary(
    placements: Dict[int, Tuple[float, float]],
    roles: Dict[int, str],
    board_extents: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """
    Build token vocabulary and coordinate mappings.
    Token ID = placement_index * V_role + role_index
    """
    # Create index mappings
    placement_ids = sorted(placements.keys())
    role_ids = sorted(roles.keys())

    placement_id_to_index = {pid: idx for idx, pid in enumerate(placement_ids)}
    placement_index_to_id = {idx: pid for idx, pid in enumerate(placement_ids)}
    role_id_to_index = {rid: idx for idx, rid in enumerate(role_ids)}
    role_index_to_id = {idx: rid for idx, rid in enumerate(role_ids)}

    V_pos = len(placement_ids)
    V_role = len(role_ids)
    V_total = V_pos * V_role + 2  # +2 for MASK and PAD

    MASK_TOKEN = V_pos * V_role
    PAD_TOKEN = V_pos * V_role + 1

    # Normalize coordinates
    x_min, x_max, y_min, y_max = board_extents
    coords = np.zeros((V_pos, 2), dtype=np.float32)
    for pid, (x, y) in placements.items():
        idx = placement_id_to_index[pid]
        coords[idx, 0] = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        coords[idx, 1] = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.5

    return {
        "placement_index_to_id": placement_index_to_id,
        "id_to_placement_index": placement_id_to_index,
        "role_index_to_id": role_index_to_id,
        "id_to_role_index": role_id_to_index,
        "role_names": [roles[role_index_to_id[i]] for i in range(V_role)],
        "placement_coords": torch.from_numpy(coords),
        "V_pos": V_pos,
        "V_role": V_role,
        "V_total": V_total,
        "MASK_TOKEN": MASK_TOKEN,
        "PAD_TOKEN": PAD_TOKEN,
    }


def tokenize_climb(
    holds: List[Tuple[int, int]],
    vocab_meta: Dict,
    L_max: int,
) -> Tuple[torch.Tensor, int]:
    """
    Convert holds to token sequence with canonical ordering.
    Returns (token_ids, seq_length).
    """
    V_role = vocab_meta["V_role"]
    placement_to_idx = vocab_meta["id_to_placement_index"]
    role_to_idx = vocab_meta["id_to_role_index"]
    coords = vocab_meta["placement_coords"]
    PAD = vocab_meta["PAD_TOKEN"]

    # Convert to tokens with coordinates for sorting
    tokens_with_coords = []
    for placement_id, role_id in holds:
        p_idx = placement_to_idx[placement_id]
        r_idx = role_to_idx[role_id]
        token_id = p_idx * V_role + r_idx
        y, x = coords[p_idx, 1].item(), coords[p_idx, 0].item()
        tokens_with_coords.append((token_id, y, x))

    # Canonical ordering: sort by y (bottom to top), then x (left to right)
    tokens_with_coords.sort(key=lambda t: (t[1], t[2]))

    # Build padded sequence
    seq = torch.full((L_max,), PAD, dtype=torch.long)
    for i, (tok, _, _) in enumerate(tokens_with_coords):
        seq[i] = tok

    return seq, len(holds)


def compute_sample_weights(
    climbs: List[Dict],
    alpha: float,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """
    Compute sample weights: w = alpha * is_classic + beta * log(1 + ascent) + gamma * quality
    """
    weights = []
    for c in climbs:
        w = (
            alpha * float(c["is_classic"]) +
            beta * np.log1p(c["ascent_count"]) +
            gamma * c["quality"]
        )
        weights.append(max(w, 0.1))  # minimum weight
    return torch.tensor(weights, dtype=torch.float32)


def process_board(config: dict) -> dict:
    """Main processing pipeline for a single board."""
    board = config["board"]
    db_path = config["db_path"]
    layout_id = config["layout_id"]
    set_ids = config["set_ids"]
    L_max = config["L_max"]
    weighting = config.get("weighting", {"alpha": 2.0, "beta": 1.0, "gamma": 0.5})

    print(f"\nProcessing {board} board...")
    print(f"  Database: {db_path}")
    print(f"  Layout ID: {layout_id}, Set IDs: {set_ids}")

    conn = connect_db(db_path)

    # Discover available configurations (for reference)
    discover_layouts_and_sets(conn)

    # Get placements and roles
    placements = get_valid_placements(conn, layout_id, set_ids)
    print(f"\n  Found {len(placements)} valid placements")

    roles = get_roles(conn)
    print(f"  Found {len(roles)} roles: {list(roles.values())}")

    board_extents = get_board_extents(conn, layout_id)
    print(f"  Board extents: {board_extents}")

    # Build vocabulary
    vocab_meta = build_vocabulary(placements, roles, board_extents)
    vocab_meta["board"] = board
    vocab_meta["L_max"] = L_max
    print(f"  Vocabulary: V_pos={vocab_meta['V_pos']}, V_role={vocab_meta['V_role']}, V_total={vocab_meta['V_total']}")

    # Get climbs
    valid_placement_ids = set(placements.keys())
    climbs = get_climbs(conn, layout_id, valid_placement_ids, L_max)
    print(f"  Found {len(climbs)} valid climbs")

    if len(climbs) == 0:
        print("  WARNING: No valid climbs found! Check layout_id and set_ids.")
        conn.close()
        return None

    # Tokenize climbs
    sequences = []
    seq_lengths = []
    for climb in climbs:
        seq, length = tokenize_climb(climb["holds"], vocab_meta, L_max)
        sequences.append(seq)
        seq_lengths.append(length)

    # Extract conditioning labels
    difficulties = torch.tensor([c["difficulty"] for c in climbs], dtype=torch.float32)
    angles = torch.tensor([c["angle"] for c in climbs], dtype=torch.int64)
    is_classic = torch.tensor([c["is_classic"] for c in climbs], dtype=torch.bool)
    quality = torch.tensor([c["quality"] for c in climbs], dtype=torch.float32)
    ascent_count = torch.tensor([c["ascent_count"] for c in climbs], dtype=torch.int64)

    # Compute sample weights
    sample_weight = compute_sample_weights(
        climbs,
        weighting["alpha"],
        weighting["beta"],
        weighting["gamma"],
    )

    conn.close()

    # Print statistics
    print(f"\n  Dataset Statistics:")
    print(f"    Difficulty range: {difficulties.min():.1f} - {difficulties.max():.1f}")
    print(f"    Unique angles: {torch.unique(angles).tolist()}")
    print(f"    Classics: {is_classic.sum().item()} ({100*is_classic.float().mean():.1f}%)")
    print(f"    Hold count range: {min(seq_lengths)} - {max(seq_lengths)}")

    return {
        "board": board,
        "layout_id": layout_id,
        "set_ids": set_ids,
        "sequences": torch.stack(sequences),
        "seq_lengths": torch.tensor(seq_lengths, dtype=torch.int64),
        "difficulty": difficulties,
        "angle": angles,
        "is_classic": is_classic,
        "quality": quality,
        "ascent_count": ascent_count,
        "sample_weight": sample_weight,
        "climb_uuids": [c["uuid"] for c in climbs],
        "vocab_meta": vocab_meta,
    }


def main():
    parser = argparse.ArgumentParser(description="Process climbing board database")
    parser.add_argument("--config", required=True, help="Path to board config YAML")
    parser.add_argument("--output-dir", type=Path, default=Path("processed"))
    args = parser.parse_args()

    config = load_config(args.config)

    # Check database exists
    if not Path(config["db_path"]).exists():
        print(f"Error: Database not found at {config['db_path']}")
        print("Run 'python data/download.py' first.")
        return

    dataset = process_board(config)

    if dataset is None:
        return

    # Save processed dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{config['board']}.pt"
    torch.save(dataset, output_path)
    print(f"\nSaved processed dataset to {output_path}")
    print(f"  Total samples: {len(dataset['sequences'])}")


if __name__ == "__main__":
    main()
