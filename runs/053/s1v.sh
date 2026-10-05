#!/usr/bin/env bash
# card 053 step 1 revised (fix 1): the visibility margin; V (gates) and V0 (no gates)
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 12 --updates 2000 --mu 0.1 --starts 0.5 --strat 1 --colour 1 --tint 0 --collect-steps 100000 --anchor transition --interaction transition --ema 0.99 --restart 100000000 --vis-w 1"
$P --gates 1 --sp-w auto --out runs/053/s1v_V_399.json  > runs/053/s1v_V_399.out 2>&1 &
$P --gates 0 --out runs/053/s1v_V0_399.json > runs/053/s1v_V0_399.out 2>&1 &
wait
echo done
