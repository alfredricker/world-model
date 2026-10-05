#!/usr/bin/env bash
# card 053 step 1 (the R1 checks): step T's harness with masks, per-action measured penalty, no restarts; T and T0
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 40 --updates 2000 --mu 0.1 --starts 0.5 --strat 1 --colour 1 --tint 0 --anchor transition --interaction transition --ema 0.99 --restart 100000000"
$P --gates 1 --sp-w auto --out runs/053/s1_T_399.json  > runs/053/s1_T_399.out 2>&1 &
$P --gates 0 --out runs/053/s1_T0_399.json > runs/053/s1_T0_399.out 2>&1 &
wait
echo done
