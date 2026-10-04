#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 40 --mu 0.1 --restart 100000000 \
    --starts 0.5 --strat 1 --out runs/052/predflip_starts_399.json > runs/052/predflip_starts_399.out 2>&1
echo done
