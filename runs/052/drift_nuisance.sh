#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card052/drift.py --seed 399 --checkpoints 20 --ema 0.99 --restart 2000"
$P --tint 0 --out runs/052/drift_notint_399.json > runs/052/drift_notint_399.out 2>&1 &
$P --noise 0 --out runs/052/drift_nonoise_399.json > runs/052/drift_nonoise_399.out 2>&1 &
$P --tint 0 --noise 0 --out runs/052/drift_clean_399.json > runs/052/drift_clean_399.out 2>&1 &
wait
echo done
