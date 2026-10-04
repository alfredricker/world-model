#!/usr/bin/env bash
# Card 051 step 4a on seed 399: two doors, one door, clutter, familiar worlds.
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card051/threats.py"
$P --chain two_doors --seed 399 --n 30  --out runs/051/s4a_two_399.json     > runs/051/s4a_two_399.out 2>&1 &
$P --chain one_door  --seed 399 --n 100 --out runs/051/s4a_one_399.json     > runs/051/s4a_one_399.out 2>&1 &
$P --clutter         --seed 399 --n 100 --out runs/051/s4a_clutter_399.json > runs/051/s4a_clutter_399.out 2>&1 &
$P --dev --arm A --seeds 399-399 --layouts 30 --out runs/051/s4a_familiar_399.json > runs/051/s4a_familiar_399.out 2>&1 &
wait
echo done
