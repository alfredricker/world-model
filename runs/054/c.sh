#!/usr/bin/env bash
# card 054 step C: m = 0.5 on seeds 400-402, then the evaluation of each
cd "$(dirname "$0")/../.."
for s in 400 401 402; do
  ( SEED=$s MS=0.5 runs/054/b.sh > /dev/null 2>&1 && runs/054/eval.sh m05_$s runs/054/b_m0.5_$s.pt > /dev/null 2>&1 ) &
done
wait
echo done
