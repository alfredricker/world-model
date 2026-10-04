#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
Q="bin/prun python tools/card051/step1.py"
$Q same --recall index --seed 399 --n 30 --out runs/051/s1c_same_index_399.json > runs/051/s1c_same_index_399.out 2>&1 &
$Q grow --recall index --seed 399 --keys 0,5000,20000,50000 --n 10 --limit 1200 --out runs/051/s1c_grow_index_399.json > runs/051/s1c_grow_index_399.out 2>&1 &
wait
echo done
