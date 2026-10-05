#!/usr/bin/env bash
# card 056: goals from example frames on card 054's four encoders (identity up to noise as in its steps B-D)
cd "$(dirname "$0")/../.."
for s in 399 400 401 402; do
  bin/prun python tools/card056/goals.py runs/054/b_m0.5_$s.pt --out runs/056/main_$s.json --layouts 30 > runs/056/main_$s.out 2>&1 &
done
wait
echo done
