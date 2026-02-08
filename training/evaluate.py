#!/usr/bin/env python3
"""
Evaluation metrics for generated climbs.

Computes:
- Validity rate
- Hold count distribution similarity
- Spatial density comparison
- Diversity (pairwise Jaccard distance)
- Novelty (fraction not matching training set)

Usage:
    python training/evaluate.py --board tension --n_samples 1000
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Set

import torch
import numpy as np

from data.dataset import ClimbDataset


def compute_validity_rate(
    sequences: torch.Tensor,
    vocab_meta: dict,
) -> float:
    """
    Compute fraction of samples passing validity checks.

    Checks:
    - At least 4 holds
    - At most 20 holds
    - No duplicate placements
    """
    PAD = vocab_meta["PAD_TOKEN"]
    V_role = vocab_meta["V_role"]

    valid_count = 0
    for seq in sequences:
        holds = seq[seq != PAD].tolist()
        n_holds = len(holds)

        # Check hold count
        if n_holds < 4 or n_holds > 20:
            continue

        # Check for duplicate placements
        placements = [h // V_role for h in holds]
        if len(placements) != len(set(placements)):
            continue

        valid_count += 1

    return valid_count / len(sequences)


def extract_hold_counts(sequences: torch.Tensor, PAD_TOKEN: int) -> List[int]:
    """Extract number of holds per sequence."""
    return [(seq != PAD_TOKEN).sum().item() for seq in sequences]


def compute_hold_count_similarity(
    generated_counts: List[int],
    training_counts: List[int],
) -> float:
    """Compute histogram intersection between hold count distributions."""
    gen_hist = Counter(generated_counts)
    train_hist = Counter(training_counts)

    all_keys = set(gen_hist.keys()) | set(train_hist.keys())

    # Normalize to probabilities
    gen_total = sum(gen_hist.values())
    train_total = sum(train_hist.values())

    intersection = 0.0
    for k in all_keys:
        p_gen = gen_hist.get(k, 0) / gen_total
        p_train = train_hist.get(k, 0) / train_total
        intersection += min(p_gen, p_train)

    return intersection


def compute_jaccard_distance(seq1: Set[int], seq2: Set[int]) -> float:
    """Compute Jaccard distance between two hold sets."""
    if len(seq1) == 0 and len(seq2) == 0:
        return 0.0
    intersection = len(seq1 & seq2)
    union = len(seq1 | seq2)
    return 1.0 - (intersection / union) if union > 0 else 0.0


def compute_diversity(sequences: torch.Tensor, PAD_TOKEN: int) -> float:
    """Compute average pairwise Jaccard distance."""
    hold_sets = []
    for seq in sequences:
        holds = set(seq[seq != PAD_TOKEN].tolist())
        hold_sets.append(holds)

    n = len(hold_sets)
    if n < 2:
        return 0.0

    # Sample pairs for efficiency
    n_pairs = min(1000, n * (n - 1) // 2)
    distances = []

    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        d = compute_jaccard_distance(hold_sets[i], hold_sets[j])
        distances.append(d)

    return np.mean(distances)


def compute_novelty(
    generated: torch.Tensor,
    training: torch.Tensor,
    PAD_TOKEN: int,
) -> float:
    """Compute fraction of generated climbs not in training set."""
    # Convert to frozensets for comparison
    training_sets = set()
    for seq in training:
        holds = frozenset(seq[seq != PAD_TOKEN].tolist())
        training_sets.add(holds)

    novel_count = 0
    for seq in generated:
        holds = frozenset(seq[seq != PAD_TOKEN].tolist())
        if holds not in training_sets:
            novel_count += 1

    return novel_count / len(generated)


def evaluate(board: str, n_samples: int = 1000):
    """Run evaluation metrics."""
    from model.transformer import DiffusionTransformer
    from model.sampler import DiffusionSampler

    # Load model
    checkpoint_path = Path("checkpoints") / board / "best.pt"
    if not checkpoint_path.exists():
        print(f"Error: No trained model found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint["config"]
    vocab_meta = checkpoint["vocab_meta"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = DiffusionTransformer.from_config(config, vocab_meta)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load training data
    data_path = Path("processed") / f"{board}.pt"
    dataset = ClimbDataset(data_path)

    # Generate samples
    print(f"Generating {n_samples} samples...")
    sampler = DiffusionSampler(
        model,
        vocab_meta["MASK_TOKEN"],
        vocab_meta["PAD_TOKEN"],
        vocab_meta["L_max"],
    )

    # Sample lengths from training distribution
    train_lengths = dataset.data["seq_lengths"]
    length_counts = torch.bincount(train_lengths, minlength=vocab_meta["L_max"] + 1)
    length_dist = length_counts.float() / length_counts.sum()

    generated = sampler.sample_with_length_dist(
        n_samples,
        length_dist,
        device,
        n_steps=32,
        guidance_scale=2.0,
    )

    # Compute metrics
    print("\n=== Evaluation Results ===")

    validity = compute_validity_rate(generated, vocab_meta)
    print(f"Validity rate: {validity:.1%}")

    gen_counts = extract_hold_counts(generated, vocab_meta["PAD_TOKEN"])
    train_counts = extract_hold_counts(dataset.data["sequences"], vocab_meta["PAD_TOKEN"])
    count_sim = compute_hold_count_similarity(gen_counts, train_counts)
    print(f"Hold count distribution similarity: {count_sim:.3f}")

    diversity = compute_diversity(generated, vocab_meta["PAD_TOKEN"])
    print(f"Diversity (avg Jaccard distance): {diversity:.3f}")

    novelty = compute_novelty(generated, dataset.data["sequences"], vocab_meta["PAD_TOKEN"])
    print(f"Novelty (fraction new): {novelty:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated climbs")
    parser.add_argument("--board", required=True, choices=["tension", "kilter", "moonboard"])
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()

    evaluate(args.board, args.n_samples)


if __name__ == "__main__":
    main()
