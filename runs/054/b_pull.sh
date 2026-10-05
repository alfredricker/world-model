#!/usr/bin/env bash
# card 054 step B, third arm: margin 1.0 and noisy copies pulled together (weight 1)
cd "$(dirname "$0")/../.."
bin/prun python tools/card052/predflip.py --seed ${SEED:-399} --checkpoints 12 --updates 2000 --mu 0.1 --starts 0.5 --strat 1 --colour 1 --tint 0 --collect-steps 100000 --anchor transition --interaction transition --ema 0.99 --restart 100000000 --vis-w 1 --gates 0 --pair-m 1.0 --pull-w 1 --out runs/054/b_pull_${SEED:-399}.json > runs/054/b_pull_${SEED:-399}.out 2>&1
echo done
