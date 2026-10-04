#!/usr/bin/env bash
# Card 051 step 1, criterion 1: same decisions as card 050 (seed 399).
cd "$(dirname "$0")/../.."
P="bin/prun python"
$P tools/card051/step1.py same --recall own   --seed 399 --n 30 --out runs/051/s1_same_own_399.json   > runs/051/s1_same_own_399.out 2>&1 &
$P tools/card051/step1.py same --recall index --seed 399 --n 30 --out runs/051/s1_same_index_399.json > runs/051/s1_same_index_399.out 2>&1 &
$P tools/card050/own.py   --dev --arm A --seeds 399-399 --layouts 30 --out runs/051/s1_familiar_own_399.json   > runs/051/s1_familiar_own_399.out 2>&1 &
$P tools/card051/index.py --dev --arm A --seeds 399-399 --layouts 30 --out runs/051/s1_familiar_index_399.json > runs/051/s1_familiar_index_399.out 2>&1 &
wait
echo done
