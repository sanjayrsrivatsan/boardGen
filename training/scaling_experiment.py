#!/usr/bin/env python3
"""
Scaling Law Experiment

Trains models of varying sizes on the same data to discover scaling laws.
Tracks validation loss vs model parameters/compute.

Usage:
    python training/scaling_experiment.py --config configs/kilter.yaml --sizes tiny,small,medium
    python training/scaling_experiment.py --config configs/kilter.yaml --sizes all
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import yaml
import matplotlib.pyplot as plt


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_scaling_experiment(
    config_path: str,
    sizes: list,
    epochs: int = 100,
    output_dir: Path = Path("scaling_results"),
):
    """Run scaling experiment across multiple model sizes."""

    config = load_config(config_path)
    board = config["board"]

    output_dir = output_dir / board
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"scaling_results_{timestamp}.csv"

    print(f"\n{'='*60}")
    print(f"Scaling Law Experiment: {board}")
    print(f"Sizes: {sizes}")
    print(f"Epochs per size: {epochs}")
    print(f"Results: {results_file}")
    print(f"{'='*60}\n")

    # Import here to get model sizes
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.transformer import DiffusionTransformer
    from data.dataset import ClimbDataset, train_val_split, get_dataloader
    from model.diffusion import MaskedDiffusion, DiffusionTrainer

    # Load dataset once
    data_path = Path("processed") / f"{board}.pt"
    full_dataset = ClimbDataset(data_path)
    train_dataset, val_dataset = train_val_split(full_dataset, val_fraction=0.1)

    print(f"Dataset: {len(full_dataset)} samples (train: {len(train_dataset)}, val: {len(val_dataset)})")

    # Get device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    # Results storage
    all_results = []

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Training {size.upper()} model")
        print(f"{'='*60}")

        # Update config with size
        config["model_size"] = size

        # Create model
        model = DiffusionTransformer.from_config(config, train_dataset.vocab_meta)
        n_params = count_parameters(model)
        model = model.to(device)

        print(f"Parameters: {n_params:,}")

        # Training setup
        batch_size = config.get("training", {}).get("batch_size", 64)
        lr = config.get("training", {}).get("learning_rate", 3e-4)
        p_uncond = config.get("training", {}).get("p_uncond", 0.15)

        train_loader = get_dataloader(train_dataset, batch_size, weighted=True)
        val_loader = get_dataloader(val_dataset, batch_size, weighted=False, shuffle=False)

        diffusion = MaskedDiffusion(
            MASK_TOKEN=train_dataset.MASK_TOKEN,
            PAD_TOKEN=train_dataset.PAD_TOKEN,
        )
        trainer = DiffusionTrainer(model, diffusion, p_uncond)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Training loop
        best_val_loss = float("inf")
        size_results = []
        total_steps = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()
                loss = trainer.training_step(batch, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1
                total_steps += 1

            avg_train_loss = train_loss / n_batches

            # Validate
            model.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    loss = trainer.training_step(batch, device)
                    val_loss += loss.item()
                    n_val_batches += 1

            avg_val_loss = val_loss / n_val_batches

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}")

            size_results.append({
                "size": size,
                "params": n_params,
                "epoch": epoch + 1,
                "steps": total_steps,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
            })

        all_results.extend(size_results)

        # Save checkpoint
        checkpoint_path = output_dir / f"model_{size}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "vocab_meta": train_dataset.vocab_meta,
            "params": n_params,
            "best_val_loss": best_val_loss,
            "epochs": epochs,
        }, checkpoint_path)
        print(f"  Saved: {checkpoint_path}")

        # Free memory
        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save all results to CSV
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to: {results_file}")

    # Generate scaling plot
    plot_scaling_results(all_results, output_dir, timestamp)

    return all_results


def plot_scaling_results(results: list, output_dir: Path, timestamp: str):
    """Generate scaling law plots."""
    import pandas as pd

    df = pd.DataFrame(results)

    # Get final results for each size
    final_results = df.groupby("size").last().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Val loss vs Parameters (log-log)
    ax1 = axes[0]
    sizes_order = ["tiny", "small", "medium", "large", "xlarge"]
    final_sorted = final_results.set_index("size").loc[[s for s in sizes_order if s in final_results["size"].values]].reset_index()

    ax1.loglog(final_sorted["params"], final_sorted["best_val_loss"], "bo-", markersize=10, linewidth=2)
    for _, row in final_sorted.iterrows():
        ax1.annotate(row["size"], (row["params"], row["best_val_loss"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax1.set_xlabel("Parameters")
    ax1.set_ylabel("Best Validation Loss")
    ax1.set_title("Scaling Law: Loss vs Parameters")
    ax1.grid(True, alpha=0.3, which="both")

    # Plot 2: Training curves for each size
    ax2 = axes[1]
    colors = {"tiny": "C0", "small": "C1", "medium": "C2", "large": "C3", "xlarge": "C4"}
    for size in df["size"].unique():
        size_df = df[df["size"] == size]
        ax2.plot(size_df["epoch"], size_df["val_loss"],
                label=f"{size} ({size_df['params'].iloc[0]/1e6:.1f}M)",
                color=colors.get(size, "gray"))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Training Curves by Model Size")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"scaling_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Run scaling law experiment")
    parser.add_argument("--config", required=True, help="Path to board config YAML")
    parser.add_argument("--sizes", default="tiny,small,medium",
                       help="Comma-separated list of sizes or 'all'")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Epochs per model size")
    parser.add_argument("--output-dir", type=Path, default=Path("scaling_results"),
                       help="Output directory for results")
    args = parser.parse_args()

    if args.sizes == "all":
        sizes = ["tiny", "small", "medium", "large", "xlarge"]
    else:
        sizes = [s.strip() for s in args.sizes.split(",")]

    run_scaling_experiment(
        config_path=args.config,
        sizes=sizes,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
