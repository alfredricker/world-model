#!/usr/bin/env bash
# Card 051 step 4a on seeds 400-404 (one lane per seed), and step 1's checks again after fitting ways once.
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card051/threats.py"
for s in 400 401 402 403 404; do
  (
    $P --chain two_doors --seed $s --n 30  --out runs/051/s4a_two_$s.json     > runs/051/s4a_two_$s.out 2>&1
    $P --chain one_door  --seed $s --n 100 --out runs/051/s4a_one_$s.json     > runs/051/s4a_one_$s.out 2>&1
    $P --clutter         --seed $s --n 100 --out runs/051/s4a_clutter_$s.json > runs/051/s4a_clutter_$s.out 2>&1
    $P --dev --arm A --seeds $s-$s --layouts 30 --out runs/051/s4a_familiar_$s.json > runs/051/s4a_familiar_$s.out 2>&1
  ) &
done
Q="bin/prun python tools/card051/step1.py"
$Q same --recall index --seed 399 --n 30 --out runs/051/s1b_same_index_399.json > runs/051/s1b_same_index_399.out 2>&1 &
bin/prun python tools/card051/index.py --dev --arm A --seeds 399-399 --layouts 30 --out runs/051/s1b_familiar_index_399.json > runs/051/s1b_familiar_index_399.out 2>&1 &
$Q grow --recall index --seed 399 --keys 0,5000,20000,50000 --n 10 --limit 1200 --out runs/051/s1b_grow_index_399.json > runs/051/s1b_grow_index_399.out 2>&1 &
wait
echo done
