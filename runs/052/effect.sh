#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card052/effect.py --seed 399 --checkpoints 40"
$P --mu 0.1 --restart 100000000       --out runs/052/effect_plain_mu01_399.json > runs/052/effect_plain_mu01_399.out 2>&1 &
$P --mu 1   --restart 100000000       --out runs/052/effect_plain_mu1_399.json  > runs/052/effect_plain_mu1_399.out 2>&1 &
$P --mu 0.1 --ema 0.99 --restart 2000 --out runs/052/effect_ema_mu01_399.json   > runs/052/effect_ema_mu01_399.out 2>&1 &
$P --mu 1   --ema 0.99 --restart 2000 --out runs/052/effect_ema_mu1_399.json    > runs/052/effect_ema_mu1_399.out 2>&1 &
wait
echo done
