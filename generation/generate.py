#!/usr/bin/env python3
"""
CLI for generating climbing problems.

Usage:
    python generation/generate.py --board tension --difficulty 14 --angle 40 --n 10
    python generation/generate.py --board kilter --difficulty 16 --angle 45 --n 10
    python generation/generate.py --board moonboard --difficulty 18 --n 10 --classic
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch

from generation.convert import tokens_to_frames, tokens_to_hold_list
from generation.postprocess import apply_validity_checks


def load_model(board: str):
    """Load trained model for a board."""
    from model.transformer import DiffusionTransformer

    checkpoint_path = Path("checkpoints") / board / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No trained model at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint["config"]
    vocab_meta = checkpoint["vocab_meta"]

    model = DiffusionTransformer.from_config(config, vocab_meta)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, config, vocab_meta


def generate(
    board: str,
    n_samples: int = 10,
    difficulty: Optional[float] = None,
    angle: Optional[int] = None,
    is_classic: bool = False,
    quality: Optional[float] = None,
    guidance_scale: float = 3.0,
    temperature: float = 0.8,
    n_holds: Optional[int] = None,
    n_steps: int = 32,
) -> List[Dict]:
    """
    Generate climbing problems.

    Args:
        board: Board name (tension, kilter, moonboard)
        n_samples: Number of climbs to generate
        difficulty: Target numeric difficulty (None = unconditional)
        angle: Wall angle in degrees (ignored for moonboard)
        is_classic: Target classic/benchmark quality
        quality: Target star rating
        guidance_scale: CFG strength
        temperature: Sampling temperature
        n_holds: Fixed number of holds (None = sample from distribution)
        n_steps: Diffusion steps

    Returns:
        List of generated climb dictionaries
    """
    from model.sampler import DiffusionSampler
    from data.dataset import ClimbDataset

    # Load model
    model, config, vocab_meta = load_model(board)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create sampler
    sampler = DiffusionSampler(
        model,
        vocab_meta["MASK_TOKEN"],
        vocab_meta["PAD_TOKEN"],
        vocab_meta["L_max"],
    )

    # Determine sequence length
    if n_holds is None:
        # Sample from training distribution
        data_path = Path("processed") / f"{board}.pt"
        if data_path.exists():
            dataset = ClimbDataset(data_path)
            lengths = dataset.data["seq_lengths"]
            n_holds = lengths[torch.randint(len(lengths), (1,))].item()
        else:
            n_holds = 8  # default

    # Handle angle for moonboard
    if not config.get("angle_conditioning", True):
        angle = config.get("default_angle", 40)

    # Prepare conditioning
    diff_tensor = torch.full((n_samples,), difficulty, device=device) if difficulty else None
    angle_tensor = torch.full((n_samples,), angle or 40, dtype=torch.long, device=device)
    classic_tensor = torch.full((n_samples,), is_classic, dtype=torch.bool, device=device)
    quality_tensor = torch.full((n_samples,), quality or 0, device=device) if quality else None

    # Generate
    print(f"Generating {n_samples} climbs for {board}...")
    sequences = sampler.sample(
        n_samples=n_samples,
        n_holds=n_holds,
        device=device,
        n_steps=n_steps,
        difficulty=diff_tensor,
        angle=angle_tensor,
        is_classic=classic_tensor,
        quality=quality_tensor,
        guidance_scale=guidance_scale,
        temperature=temperature,
    )

    # Convert to output format
    results = []
    for i, seq in enumerate(sequences):
        # Apply validity checks
        seq, valid = apply_validity_checks(seq, vocab_meta)

        frames = tokens_to_frames(seq, vocab_meta)
        holds = tokens_to_hold_list(seq, vocab_meta)

        results.append({
            "id": i,
            "tokens": seq.tolist(),
            "frames_string": frames,
            "hold_list": holds,
            "n_holds": len(holds),
            "valid": valid,
            "metadata": {
                "board": board,
                "difficulty": difficulty,
                "angle": angle,
                "is_classic": is_classic,
                "quality": quality,
                "guidance_scale": guidance_scale,
                "temperature": temperature,
            },
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate climbing problems")
    parser.add_argument("--board", required=True, choices=["tension", "kilter", "moonboard"])
    parser.add_argument("--n", type=int, default=10, help="Number of climbs to generate")
    parser.add_argument("--difficulty", type=float, help="Target numeric difficulty")
    parser.add_argument("--angle", type=int, help="Wall angle in degrees")
    parser.add_argument("--classic", action="store_true", help="Target classic quality")
    parser.add_argument("--quality", type=float, help="Target star rating")
    parser.add_argument("--guidance", type=float, default=3.0, help="CFG strength")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--n_holds", type=int, help="Fixed number of holds")
    parser.add_argument("--n_steps", type=int, default=32, help="Diffusion steps")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    climbs = generate(
        board=args.board,
        n_samples=args.n,
        difficulty=args.difficulty,
        angle=args.angle,
        is_classic=args.classic,
        quality=args.quality,
        guidance_scale=args.guidance,
        temperature=args.temperature,
        n_holds=args.n_holds,
        n_steps=args.n_steps,
    )

    # Print results
    print(f"\nGenerated {len(climbs)} climbs:")
    for climb in climbs:
        status = "VALID" if climb["valid"] else "INVALID"
        print(f"  [{status}] {climb['n_holds']} holds: {climb['frames_string'][:50]}...")

    valid_count = sum(1 for c in climbs if c["valid"])
    print(f"\nValidity rate: {valid_count}/{len(climbs)} ({100*valid_count/len(climbs):.1f}%)")

    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(climbs, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
