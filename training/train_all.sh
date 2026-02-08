#!/bin/bash
# Train all board models sequentially

set -e

echo "=== Training Tension Board model ==="
python training/train.py --config configs/tension.yaml

echo ""
echo "=== Training Kilter Board model ==="
python training/train.py --config configs/kilter.yaml

echo ""
echo "=== Training MoonBoard model ==="
python training/train.py --config configs/moonboard.yaml

echo ""
echo "=== All models trained ==="
