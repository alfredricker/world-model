#!/usr/bin/env bash
# card 052 step 2g: collapse prevention x interaction, on the old setting (step 2f is the pixels x effect cell)
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 20 --updates 1000 --mu 0.1 --restart 100000000 \
   --starts 0.5 --strat 1 --colour 1 --colours red,green,blue --tint 0 --noise 0 --collect-steps 150000"
$P --anchor uniformity --codebook fixed --interaction effect --out runs/052/s2g_A_399.json > runs/052/s2g_A_399.out 2>&1 &
$P --anchor uniformity --codebook fixed --interaction diff   --out runs/052/s2g_B_399.json > runs/052/s2g_B_399.out 2>&1 &
$P --anchor pixels --interaction diff                        --out runs/052/s2g_C_399.json > runs/052/s2g_C_399.out 2>&1 &
wait
echo done
