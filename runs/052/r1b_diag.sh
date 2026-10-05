#!/usr/bin/env bash
# card 052 step R1b, the short diagnostic: arm A with alignment and uniformity as published, 10 checkpoints
cd "$(dirname "$0")/../.."
bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 10 --updates 2000 --mu 0.1 --restart 100000000 \
   --starts 0.5 --strat 1 --colour 1 --anchor align_uniform --codebook fixed --drift 0.05 --collect-steps 100000 \
   --interaction effect --out runs/052/r1b_diag_A_399.json > runs/052/r1b_diag_A_399.out 2>&1
echo done
