#!/usr/bin/env bash
# card 054 step B: V0's recipe with the variance floor replaced by the margin between visibly different tiles (m = 0.5, 1.0)
cd "$(dirname "$0")/../.."
P="bin/prun python tools/card052/predflip.py --seed ${SEED:-399} --checkpoints 12 --updates 2000 --mu 0.1 --starts 0.5 --strat 1 --colour 1 --tint 0 --collect-steps 100000 --anchor transition --interaction transition --ema 0.99 --restart 100000000 --vis-w 1 --gates 0"
for m in ${MS:-0.5 1.0}; do
  $P --pair-m $m --out runs/054/b_m${m}_${SEED:-399}.json > runs/054/b_m${m}_${SEED:-399}.out 2>&1 &
done
wait
echo done
