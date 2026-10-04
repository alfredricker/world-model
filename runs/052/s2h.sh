#!/usr/bin/env bash
# card 052 step 2h: arms A and B of step 2g on the main setting (nine colours, tint and noise)
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 40 --updates 2000 --mu 0.1 --restart 100000000 \
   --starts 0.5 --strat 1 --colour 1 --anchor uniformity --codebook fixed"
$P --interaction effect --out runs/052/s2h_A_399.json > runs/052/s2h_A_399.out 2>&1 &
$P --interaction diff   --out runs/052/s2h_B_399.json > runs/052/s2h_B_399.out 2>&1 &
wait
echo done
