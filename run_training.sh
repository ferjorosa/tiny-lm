#!/usr/bin/env bash

set -euo pipefail

# uv run python scripts/training/train.py \
#   --model-config configs/models/gpt2-8k.yaml \
#   --training-config configs/training/smoke.yaml \
#   --data-config configs/data/tinystories-8k.yaml

uv run python scripts/training/train.py \
  --model-config configs/models/gpt2-8k-2l.yaml \
  --training-config configs/training/gpt2-8k.yaml \
  --data-config configs/data/tinystories-8k.yaml
