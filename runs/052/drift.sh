#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card052/drift.py --seed 399 --checkpoints 40"
$P --ema 0.99 --restart 2000 --out runs/052/drift_ema_399.json > runs/052/drift_ema_399.out 2>&1 &
$P --restart 100000000      --out runs/052/drift_plain_399.json > runs/052/drift_plain_399.out 2>&1 &
wait
echo done
