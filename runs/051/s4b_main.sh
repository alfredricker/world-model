#!/usr/bin/env bash
# Card 051 step 4b on seeds 400-404.
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card051/commit.py"
for s in 400 401 402 403 404; do
  (
    $P --chain two_doors --seed $s --n 30  --out runs/051/s4b_two_$s.json     > runs/051/s4b_two_$s.out 2>&1
    $P --chain one_door  --seed $s --n 100 --out runs/051/s4b_one_$s.json     > runs/051/s4b_one_$s.out 2>&1
    $P --clutter         --seed $s --n 100 --out runs/051/s4b_clutter_$s.json > runs/051/s4b_clutter_$s.out 2>&1
    $P --dev --arm A --seeds $s-$s --layouts 30 --out runs/051/s4b_familiar_$s.json > runs/051/s4b_familiar_$s.out 2>&1
  ) &
done
wait
echo done
