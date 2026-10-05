#!/usr/bin/env bash
# card 056, declared revision: one goal frame per episode (a draw of five frames comes from five episodes)
cd "$(dirname "$0")/../.."
for s in 399 400 401 402; do
  bin/prun python tools/card056/goals.py runs/054/b_m0.5_$s.pt --out runs/056/rev_$s.json --layouts 30 --one-per-episode > runs/056/rev_$s.out 2>&1 &
done
wait
echo done
