#!/usr/bin/env bash
# card 052 step T: transitions read through sparse conditions; no gates; the step-2h baseline; all without tint
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 40 --updates 2000 --mu 0.1 --starts 0.5 --strat 1 --colour 1 --tint 0"
$P --anchor transition --interaction transition --ema 0.99 --restart 250 --gates 1 --out runs/052/sT_T_399.json  > runs/052/sT_T_399.out 2>&1 &
$P --anchor transition --interaction transition --ema 0.99 --restart 250 --gates 0 --out runs/052/sT_T0_399.json > runs/052/sT_T0_399.out 2>&1 &
$P --anchor uniformity --codebook fixed --interaction effect --restart 100000000 --out runs/052/sT_base_399.json > runs/052/sT_base_399.out 2>&1 &
wait
echo done
