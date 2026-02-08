#!/usr/bin/env python3
"""
Training script for discrete masked diffusion model.

Usage:
    python training/train.py --config configs/tension.yaml
    python training/train.py --config configs/kilter.yaml
    python training/train.py --config configs/moonboard.yaml
"""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from data.dataset import ClimbDataset, get_dataloader
from model.transformer import DiffusionTransformer
from model.diffusion import MaskedDiffusion, DiffusionTrainer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Get best available device (MPS for Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def train(config: dict):
    """Main training loop."""
    import sys
    # Unbuffered output for real-time logging
    sys.stdout.reconfigure(line_buffering=True)

    board = config["board"]
    training_cfg = config.get("training", {})

    batch_size = training_cfg.get("batch_size", 64)
    lr = training_cfg.get("learning_rate", 3e-4)
    warmup_steps = training_cfg.get("warmup_steps", 1000)
    epochs = training_cfg.get("epochs", 300)
    ema_decay = training_cfg.get("ema_decay", 0.9999)
    p_uncond = training_cfg.get("p_uncond", 0.1)

    device = get_device()
    print(f"\nTraining {board} model on {device}")

    # Load dataset
    data_path = Path("processed") / f"{board}.pt"
    if not data_path.exists():
        print(f"Error: Processed data not found at {data_path}")
        print("Run 'python data/process.py --config <config>' first.")
        return

    dataset = ClimbDataset(data_path)
    dataloader = get_dataloader(dataset, batch_size, weighted=True)
    print(f"Dataset: {len(dataset)} samples")

    # Create model
    model = DiffusionTransformer.from_config(config, dataset.vocab_meta)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Create diffusion and trainer
    diffusion = MaskedDiffusion(
        MASK_TOKEN=dataset.MASK_TOKEN,
        PAD_TOKEN=dataset.PAD_TOKEN,
    )
    trainer = DiffusionTrainer(model, diffusion, p_uncond)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (epochs * len(dataloader) - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA
    ema = EMA(model, ema_decay)

    # Training
    checkpoint_dir = Path("checkpoints") / board
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Loss logging
    log_file = checkpoint_dir / "training_log.csv"
    with open(log_file, "w") as f:
        f.write("epoch,loss,lr\n")

    best_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            loss = trainer.training_step(batch, device)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            ema.update()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / n_batches
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # Log to CSV
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{current_lr:.2e}\n")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            ema.apply_shadow()
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "vocab_meta": dataset.vocab_meta,
                "loss": avg_loss,
            }
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch+1}.pt")
            ema.restore()

        if avg_loss < best_loss:
            best_loss = avg_loss
            ema.apply_shadow()
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": config,
                "vocab_meta": dataset.vocab_meta,
                "loss": avg_loss,
            }
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            ema.restore()
            print(f"  New best model saved (loss: {best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--config", required=True, help="Path to board config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
