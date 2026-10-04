#!/usr/bin/env bash
# step 2e's rho = 1 arm, run after the check arm (out of GPU memory with three at once)
cd "$(dirname "$0")/../.."
until grep -q "'checkpoint': 40," runs/052/s2e_check_399.out; do sleep 30; done
bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 40 --mu 0.1 --restart 100000000 --starts 0.5 --strat 1 --colour 1 \
    --rel 1 --out runs/052/s2e_rel1_399.json > runs/052/s2e_rel1_399.out 2>&1
echo done
