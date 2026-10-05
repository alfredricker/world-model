#!/usr/bin/env bash
# card 064: version 14 on the chained rooms with two doors and the cluttered key world
cd "$(dirname "$0")/../.."
A="--arm B --occlude --frontier --keep-look"
R="bin/prun python tools/card062/believed.py tools/card057/partial.py runs/054/b_m0.5_399.pt $A"
$R --out runs/064/two_399.json @@ --chain two_doors --seed 399 --n 30 > runs/064/two_399.out 2>&1 &
$R --out runs/064/clutter_399.json @@ --clutter --seed 399 --n 100 > runs/064/clutter_399.out 2>&1 &
wait
echo done
