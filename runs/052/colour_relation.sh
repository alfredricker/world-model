#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 40 --mu 0.1 --restart 100000000 --starts 0.5 --strat 1 --colour 1"
$P          --out runs/052/s2e_check_399.json  > runs/052/s2e_check_399.out 2>&1 &
$P --rel 0.1 --out runs/052/s2e_rel01_399.json > runs/052/s2e_rel01_399.out 2>&1 &
$P --rel 1   --out runs/052/s2e_rel1_399.json  > runs/052/s2e_rel1_399.out 2>&1 &
wait
echo done
