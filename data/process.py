#!/usr/bin/env python3
"""
Stage 1.2-1.8: Data Processing Pipeline

Converts SQLite database to token sequences for diffusion model training.
Handles vocabulary construction, coordinate normalization, canonical ordering,
filtering, board size detection, and sample weighting.

Usage:
    python data/process.py --config configs/tension.yaml
    python data/process.py --config configs/kilter.yaml
    python data/process.py --config configs/moonboard.yaml
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

import numpy as np
import torch
import yaml


# Standard roles for climbing boards (order matters for role indices)
# Kilter has duplicate role entries with different IDs - we collapse them by name
STANDARD_ROLES = ["Start", "Middle", "Finish", "Foot Only"]

# Mapping for color roles that sometimes appear in databases
# Colors map to standard roles based on Kilter convention
COLOR_ROLE_MAP = {
    "Green": "Start",
    "Blue": "Middle",
    "Cyan": "Middle",
    "Pink": "Finish",
    "Magenta": "Finish",
    "Yellow": "Foot Only",
    "Red": "Middle",  # Less common, treat as middle
}


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
        print(f"  ID {row['id']}: layout={row['layout_id']} ({row['layout_name']}), set={row['set_id']}, size={row['product_size_id']}")


def parse_size_name(name: str, width: float, height: float) -> str:
    """
    Parse database size name into normalized format like '12x12'.
    Tries to extract dimensions from name like '12 high x 12 wide'.
    """
    import re
    # Try to match patterns like "12 high x 12 wide" or "10 high x 8 wide"
    match = re.search(r'(\d+)\s*high\s*x\s*(\d+)\s*wide', name, re.IGNORECASE)
    if match:
        h, w = match.groups()
        return f"{h}x{w}"
    # Try simpler pattern like "12x12"
    match = re.search(r'(\d+)\s*x\s*(\d+)', name)
    if match:
        return f"{match.group(1)}x{match.group(2)}"
    # Fall back to computing from edges (assuming 12 units per dimension for TB2)
    # TB2 is 12 rows high, 12 columns wide at max
    h = int(round(height / 12))
    w = int(round(width / 12))
    return f"{h}x{w}"


def get_board_sizes(conn: sqlite3.Connection, layout_id: int) -> Dict[str, Dict]:
    """
    Get all available board sizes for a layout.
    Returns dict: size_name -> {id, edge_left, edge_right, edge_bottom, edge_top}
    """
    cursor = conn.execute("""
        SELECT DISTINCT ps.id, ps.edge_left, ps.edge_right, ps.edge_bottom, ps.edge_top,
               ps.name, ps.description
        FROM product_sizes ps
        JOIN product_sizes_layouts_sets psls ON ps.id = psls.product_size_id
        WHERE psls.layout_id = ?
        ORDER BY (ps.edge_right - ps.edge_left) * (ps.edge_top - ps.edge_bottom)
    """, (layout_id,))

    sizes = {}
    for row in cursor:
        width = row["edge_right"] - row["edge_left"]
        height = row["edge_top"] - row["edge_bottom"]
        # Parse the name into normalized format
        raw_name = row["name"] or row["description"] or ""
        name = parse_size_name(raw_name, width, height)
        sizes[name] = {
            "id": row["id"],
            "edge_left": row["edge_left"],
            "edge_right": row["edge_right"],
            "edge_bottom": row["edge_bottom"],
            "edge_top": row["edge_top"],
        }
    return sizes


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


def get_board_extents(conn: sqlite3.Connection, layout_id: int, size_name: str = None) -> Tuple[float, float, float, float]:
    """Get board coordinate extents for normalization (uses largest size by default)."""
    if size_name:
        # Get specific size
        cursor = conn.execute("""
            SELECT ps.edge_left, ps.edge_right, ps.edge_bottom, ps.edge_top
            FROM product_sizes ps
            JOIN product_sizes_layouts_sets psls ON ps.id = psls.product_size_id
            WHERE psls.layout_id = ? AND (ps.name = ? OR ps.description LIKE ?)
            LIMIT 1
        """, (layout_id, size_name, f"%{size_name}%"))
    else:
        # Get largest size
        cursor = conn.execute("""
            SELECT ps.edge_left, ps.edge_right, ps.edge_bottom, ps.edge_top
            FROM product_sizes ps
            JOIN product_sizes_layouts_sets psls ON ps.id = psls.product_size_id
            WHERE psls.layout_id = ?
            ORDER BY (ps.edge_right - ps.edge_left) * (ps.edge_top - ps.edge_bottom) DESC
            LIMIT 1
        """, (layout_id,))

    row = cursor.fetchone()
    if row:
        return row["edge_left"], row["edge_right"], row["edge_bottom"], row["edge_top"]
    # Fallback: compute from holes
    cursor = conn.execute("SELECT MIN(x), MAX(x), MIN(y), MAX(y) FROM holes")
    row = cursor.fetchone()
    return row[0], row[1], row[2], row[3]


def get_roles(conn: sqlite3.Connection, collapse_roles: bool = False) -> Tuple[Dict[int, str], Dict[int, int]]:
    """
    Get role ID -> name mapping.

    If collapse_roles=True, collapses duplicate role names and color roles
    to the 4 standard functional roles (Start, Middle, Finish, Foot Only).

    Returns:
        roles: dict mapping role_id -> role_name (collapsed if requested)
        role_id_remap: dict mapping original_role_id -> collapsed_role_id (or identity if not collapsing)
    """
    cursor = conn.execute("SELECT id, full_name FROM placement_roles ORDER BY id")
    raw_roles = {row["id"]: row["full_name"] for row in cursor}

    if not collapse_roles:
        # No collapsing - return identity mapping
        role_id_remap = {rid: rid for rid in raw_roles}
        return raw_roles, role_id_remap

    # Collapse to standard roles by name
    # Normalize role names: standard roles stay as-is, colors get mapped
    def normalize_role(name: str) -> str:
        if name in STANDARD_ROLES:
            return name
        if name in COLOR_ROLE_MAP:
            return COLOR_ROLE_MAP[name]
        # Unknown role - try to match partial name
        for std in STANDARD_ROLES:
            if std.lower() in name.lower():
                return std
        return name  # Keep as-is if no match

    # Determine which standard roles are actually used
    used_roles = set()
    for name in raw_roles.values():
        used_roles.add(normalize_role(name))

    # Build collapsed role set (maintain standard order for known roles)
    collapsed_roles = []
    for std_role in STANDARD_ROLES:
        if std_role in used_roles:
            collapsed_roles.append(std_role)
            used_roles.discard(std_role)
    # Add any remaining non-standard roles
    for role in sorted(used_roles):
        collapsed_roles.append(role)

    # Create new role_id -> name mapping with sequential IDs
    roles = {i: name for i, name in enumerate(collapsed_roles)}

    # Create mapping from original role_id -> collapsed role_id
    role_id_remap = {}
    collapsed_name_to_id = {name: i for i, name in enumerate(collapsed_roles)}
    for orig_id, orig_name in raw_roles.items():
        new_name = normalize_role(orig_name)
        role_id_remap[orig_id] = collapsed_name_to_id[new_name]

    return roles, role_id_remap


def parse_frames(frames: str, role_id_remap: Dict[int, int] = None) -> List[Tuple[int, int]]:
    """
    Parse Aurora frames string into list of (placement_id, role_id) pairs.
    Format: p{placement_id}r{role_id}p{placement_id}r{role_id}...

    If role_id_remap is provided, remaps role IDs (used for collapsing color roles).
    """
    pattern = r"p(\d+)r(\d+)"
    matches = re.findall(pattern, frames)
    if role_id_remap:
        return [(int(p), role_id_remap.get(int(r), int(r))) for p, r in matches]
    return [(int(p), int(r)) for p, r in matches]


def get_compatible_sizes(
    climb_placements: Set[int],
    placement_coords: Dict[int, Tuple[float, float]],
    board_sizes: Dict[str, Dict],
) -> List[str]:
    """
    Return list of board sizes this climb is compatible with.
    A climb is compatible if all its placements fall within the size's boundaries.
    """
    compatible = []
    for size_name, size_info in board_sizes.items():
        fits = True
        for p_id in climb_placements:
            if p_id not in placement_coords:
                fits = False
                break
            x, y = placement_coords[p_id]
            if not (size_info["edge_left"] <= x <= size_info["edge_right"] and
                    size_info["edge_bottom"] <= y <= size_info["edge_top"]):
                fits = False
                break
        if fits:
            compatible.append(size_name)
    return compatible


def get_minimum_compatible_size(
    compatible_sizes: List[str],
    size_order: List[str],
) -> str:
    """Get the smallest compatible size (first in the ordered list)."""
    for size in size_order:
        if size in compatible_sizes:
            return size
    return size_order[-1] if size_order else ""


def get_climbs(
    conn: sqlite3.Connection,
    layout_id: int,
    valid_placement_ids: set,
    placement_coords: Dict[int, Tuple[float, float]],
    board_sizes: Dict[str, Dict],
    size_order: List[str],
    L_max: int,
    role_id_remap: Dict[int, int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch valid climbs with their stats and conditioning labels.
    Applies filtering criteria from spec section 1.5.6.
    Tags each climb with minimum compatible board size.
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
            cs.ascensionist_count
        FROM climbs c
        JOIN climb_stats cs ON c.uuid = cs.climb_uuid
        WHERE c.is_listed = 1
          AND c.is_draft = 0
          AND c.layout_id = ?
          AND cs.ascensionist_count >= 1
        ORDER BY c.uuid, cs.angle
    """, (layout_id,))

    climbs = []
    for row in cursor:
        holds = parse_frames(row["frames"], role_id_remap)

        # Filter: all placements must be valid
        placement_set = set(p for p, r in holds)
        if not placement_set.issubset(valid_placement_ids):
            continue

        # Filter: respect L_max
        if len(holds) > L_max:
            continue

        # Filter: minimum holds
        if len(holds) < 4:
            continue

        # Determine compatible sizes
        compatible = get_compatible_sizes(placement_set, placement_coords, board_sizes)
        if not compatible:
            continue

        min_size = get_minimum_compatible_size(compatible, size_order)
        min_size_idx = size_order.index(min_size) if min_size in size_order else 0

        climbs.append({
            "uuid": row["uuid"],
            "holds": holds,
            "angle": row["angle"],
            "difficulty": row["display_difficulty"] or row["difficulty_average"] or 0,
            "quality": row["quality_average"] or 0,
            "is_classic": row["benchmark_difficulty"] is not None,
            "ascent_count": row["ascensionist_count"] or 0,
            "board_size": min_size,
            "board_size_idx": min_size_idx,
            "compatible_sizes": compatible,
        })

    return climbs


def build_vocabulary(
    placements: Dict[int, Tuple[float, float]],
    roles: Dict[int, str],
    board_extents: Tuple[float, float, float, float],
    board_sizes: Dict[str, Dict],
    size_order: List[str],
) -> Dict[str, Any]:
    """
    Build token vocabulary, coordinate mappings, and size logit masks.
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

    # Normalize coordinates relative to largest board
    x_min, x_max, y_min, y_max = board_extents
    coords = np.zeros((V_pos, 2), dtype=np.float32)
    raw_coords = {}  # Store unnormalized for size mask computation
    for pid, (x, y) in placements.items():
        idx = placement_id_to_index[pid]
        coords[idx, 0] = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        coords[idx, 1] = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        raw_coords[idx] = (x, y)

    # Build size logit masks
    # size_logit_masks[size_idx] = BoolTensor of shape (V_total - 1,)
    # True for tokens whose placement falls within that size's boundaries
    size_logit_masks = {}
    for size_idx, size_name in enumerate(size_order):
        if size_name not in board_sizes:
            # Default to all valid if size not found
            mask = torch.ones(V_total - 1, dtype=torch.bool)
        else:
            size_info = board_sizes[size_name]
            mask = torch.zeros(V_total - 1, dtype=torch.bool)

            for p_idx in range(V_pos):
                x, y = raw_coords[p_idx]
                valid = (size_info["edge_left"] <= x <= size_info["edge_right"] and
                        size_info["edge_bottom"] <= y <= size_info["edge_top"])
                if valid:
                    # Mark all roles for this placement as valid
                    for r_idx in range(V_role):
                        token_id = p_idx * V_role + r_idx
                        mask[token_id] = True

            # MASK token is always valid (it's at index V_pos * V_role which is V_total - 2)
            mask[MASK_TOKEN] = True

        size_logit_masks[size_idx] = mask

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
        "board_sizes": size_order,
        "size_logit_masks": size_logit_masks,
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
    available_sizes = config.get("available_sizes", [])
    default_size = config.get("default_size", available_sizes[-1] if available_sizes else "")

    print(f"\nProcessing {board} board...")
    print(f"  Database: {db_path}")
    print(f"  Layout ID: {layout_id}, Set IDs: {set_ids}")
    print(f"  Available sizes: {available_sizes}")
    print(f"  Default (largest) size: {default_size}")

    conn = connect_db(db_path)

    # Discover available configurations (for reference)
    discover_layouts_and_sets(conn)

    # Get board sizes from database
    db_sizes = get_board_sizes(conn, layout_id)
    print(f"\n  Found {len(db_sizes)} board sizes in database: {list(db_sizes.keys())}")

    # Use config sizes if available, otherwise use discovered sizes
    if available_sizes:
        # Map config size names to database size info
        board_sizes = {}
        for size_name in available_sizes:
            if size_name in db_sizes:
                board_sizes[size_name] = db_sizes[size_name]
            else:
                # Try to find a matching size
                for db_name, db_info in db_sizes.items():
                    if size_name in db_name or db_name in size_name:
                        board_sizes[size_name] = db_info
                        break
        if not board_sizes:
            board_sizes = db_sizes
    else:
        board_sizes = db_sizes
        available_sizes = list(db_sizes.keys())

    print(f"  Using sizes: {list(board_sizes.keys())}")

    # Get placements and roles
    placements = get_valid_placements(conn, layout_id, set_ids)
    print(f"\n  Found {len(placements)} valid placements")

    # Collapse color-based roles for Kilter board
    collapse_roles = board.lower() == "kilter"
    roles, role_id_remap = get_roles(conn, collapse_roles=collapse_roles)
    if collapse_roles:
        print(f"  Collapsed roles to {len(roles)}: {list(roles.values())}")
    else:
        print(f"  Found {len(roles)} roles: {list(roles.values())}")

    # Get extents from largest size
    board_extents = get_board_extents(conn, layout_id, default_size)
    print(f"  Board extents: {board_extents}")

    # Build vocabulary with size masks
    vocab_meta = build_vocabulary(placements, roles, board_extents, board_sizes, available_sizes)
    vocab_meta["board"] = board
    vocab_meta["L_max"] = L_max
    print(f"  Vocabulary: V_pos={vocab_meta['V_pos']}, V_role={vocab_meta['V_role']}, V_total={vocab_meta['V_total']}")
    print(f"  Size logit masks computed for {len(vocab_meta['size_logit_masks'])} sizes")

    # Get climbs with size tagging
    valid_placement_ids = set(placements.keys())
    climbs = get_climbs(conn, layout_id, valid_placement_ids, placements, board_sizes, available_sizes, L_max, role_id_remap)
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
    board_size_indices = torch.tensor([c["board_size_idx"] for c in climbs], dtype=torch.int64)
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
    print(f"    Size distribution:")
    for i, size_name in enumerate(available_sizes):
        count = (board_size_indices == i).sum().item()
        print(f"      {size_name}: {count} climbs ({100*count/len(climbs):.1f}%)")
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
        "board_size": board_size_indices,
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
