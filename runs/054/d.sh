#!/usr/bin/env bash
# card 054 step D: card 051's chained rooms (one door) and cluttered world on one frozen encoder with identity codes.
# Usage: runs/054/d.sh NAME WEIGHTS.pt SEED
cd "$(dirname "$0")/../.."
N=$1; W=$2; S=$3
P="bin/prun python tools/card053/planner_check.py $W --gates 0 --codes noise @@"
$P --chain one_door --seed $S --n 100 --out runs/054/d_one_$N.json > runs/054/d_one_$N.out 2>&1 &
$P --clutter --seed $S --n 100 --out runs/054/d_clutter_$N.json > runs/054/d_clutter_$N.out 2>&1 &
wait
echo done
