#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 20 --updates 1000 --mu 0.1 --restart 100000000 \
    --starts 0.5 --strat 1 --colour 1 --colours red,green,blue --tint 0 --noise 0 --collect-steps 150000 \
    --out runs/052/s2f_old_399.json > runs/052/s2f_old_399.out 2>&1
echo done
