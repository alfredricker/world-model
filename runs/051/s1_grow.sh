#!/usr/bin/env bash
# Card 051 step 1, criterion 2: time per step as memory grows (seed 399).
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card051/step1.py grow --seed 399 --n 10 --limit 1200"
$P --recall index --keys 0,5000,20000,50000 --out runs/051/s1_grow_index_399.json > runs/051/s1_grow_index_399.out 2>&1 &
$P --recall own   --keys 0,5000,20000       --out runs/051/s1_grow_own_399.json   > runs/051/s1_grow_own_399.out 2>&1 &
wait
echo done
