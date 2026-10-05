#!/usr/bin/env bash
# card 063: card 054's encoder recipe (m = 0.5) with prioritized replay of the transition tries; then card 054's evaluation
cd "$(dirname "$0")/../.."
for S in 399 400 401 402; do
  (
    bin/prun python tools/card063/priority.py tools/card052/predflip.py --seed $S --checkpoints 12 --updates 2000 --mu 0.1 --starts 0.5 --strat 1 --colour 1 --tint 0 --collect-steps 100000 --anchor transition --interaction transition --ema 0.99 --restart 100000000 --vis-w 1 --gates 0 --pair-m 0.5 --out runs/063/b_$S.json > runs/063/b_$S.out 2>&1
    bin/prun python tools/card053/recall_probe.py runs/063/b_$S.pt --interaction transition --gates 0 --codes noise --out runs/063/recall_$S.json > runs/063/recall_$S.out 2>&1
    bin/prun python tools/card053/planner_check.py runs/063/b_$S.pt --gates 0 --codes noise --out runs/063/planner_$S.json > runs/063/planner_$S.out 2>&1
  ) &
done
wait
echo done
