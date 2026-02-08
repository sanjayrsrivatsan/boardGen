# BoardGen

A discrete masked diffusion model for generating boulder problems on LED climbing boards (Tension Board, Kilter Board, MoonBoard).

## Overview

BoardGen uses a transformer-based discrete diffusion model to generate novel climbing routes conditioned on:
- **Difficulty** (V-grade)
- **Wall angle** (0-70°)
- **Board size** (e.g., 12x12, 10x8)

The model learns from thousands of user-created climbs and generates new problems that respect the physical constraints of each board.

## Features

- **Multi-board support**: Tension Board 2, Kilter Board, MoonBoard
- **Conditional generation**: Control difficulty, angle, and board size
- **Classifier-free guidance**: Adjustable guidance scale for generation quality
- **Web interface**: Visualize generated climbs on actual board images
- **Train/validation split**: Monitor for overfitting during training

## Architecture

- **Discrete masked diffusion** (MDLM) for token sequences
- **Token vocabulary**: `token_id = placement_index * V_role + role_index`
- **4 hold roles**: Start, Middle, Finish, Foot Only
- **AdaLN conditioning**: Adaptive layer normalization for difficulty/angle/size
- **EMA weights**: Exponential moving average for stable generation

## Installation

```bash
git clone https://github.com/sanjayrsrivatsan/boardGen.git
cd boardGen
pip install -r requirements.txt
```

## Quick Start

### 1. Download board data

```bash
python data/download.py --board tension
python data/download.py --board kilter
```

### 2. Process data

```bash
python data/process.py --config configs/tension.yaml
python data/process.py --config configs/kilter.yaml
```

### 3. Train model

```bash
python training/train.py --config configs/tension.yaml
```

Training logs are saved to `checkpoints/<board>/training_log.csv` with train/val loss.

### 4. Generate climbs

```bash
python generation/generate.py --board tension --difficulty 16 --angle 40 --n_samples 5
```

### 5. Launch web interface

```bash
python interface/server.py
```

Open http://localhost:8000/app/ to visualize generated climbs.

## Project Structure

```
boardGen/
├── configs/           # Board configuration files (YAML)
├── data/
│   ├── download.py    # Download board databases
│   ├── process.py     # Process data into training format
│   └── dataset.py     # PyTorch dataset and dataloader
├── model/
│   ├── transformer.py # Diffusion transformer architecture
│   ├── diffusion.py   # Masked diffusion process
│   ├── embeddings.py  # Position and label embeddings
│   ├── conditioning.py# Classifier-free guidance
│   └── sampler.py     # Generation sampling
├── training/
│   └── train.py       # Training loop with validation
├── generation/
│   └── generate.py    # CLI generation script
├── interface/
│   ├── server.py      # FastAPI backend
│   └── frontend/      # React web interface
├── checkpoints/       # Saved model weights
└── processed/         # Processed training data
```

## Configuration

Each board has a YAML config file:

```yaml
board: tension
db_path: data/tension.sqlite
layout_id: 10
set_ids: [12, 13]
angle_conditioning: true
angle_range: [0, 65]
size_conditioning: true
available_sizes: ["10x8", "12x8", "10x12", "12x12"]
L_max: 24
model_size: small

training:
  batch_size: 64
  learning_rate: 3.0e-4
  epochs: 1000
  p_uncond: 0.15  # CFG dropout probability
```

## Model Sizes

| Board | Placements | Roles | V_total | Parameters |
|-------|------------|-------|---------|------------|
| Tension | 996 | 4 | 3,986 | 2.17M |
| Kilter | 488 | 4 | 1,954 | 1.65M |

## Training

Training uses:
- **AdamW optimizer** with cosine LR decay
- **Warmup** for first 1000 steps
- **EMA** (decay=0.9999) for stable weights
- **Gradient clipping** (max norm=1.0)
- **10% validation split** for overfitting detection

Monitor training with the CSV log:
```csv
epoch,train_loss,val_loss,lr
1,6.57,7.71,1.77e-04
2,5.10,7.61,3.00e-04
...
```

## Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `difficulty` | Target difficulty (10-32 internal scale) | 14 |
| `angle` | Wall angle in degrees | 40 |
| `board_size` | Board dimensions (e.g., "12x12") | largest |
| `guidance_scale` | CFG strength (0=unconditional, higher=stronger) | 3.0 |
| `temperature` | Sampling temperature | 0.8 |
| `n_samples` | Number of climbs to generate | 10 |

## License

MIT

## Acknowledgments

- [boardlib](https://github.com/lemeryferti662/boardlib) for Aurora board data access
- Tension Climbing, Kilter, and Moon Climbing for the LED board systems
