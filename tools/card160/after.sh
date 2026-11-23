#!/usr/bin/env bash
# usage: after.sh <experience>   after the test, labelled as such: the same test with the planner imagining an effect's trace in
# memory too (FIX=1); plus, for experience 1, the upper bound (rules) and random actions on the test tasks
cd "$(dirname "$0")/../.."
e=$1
[ -f runs/160/effects_${e}_sparse_test.json ] || tools/card160/run.sh 590 python tools/card160/evaluate.py effects $e sparse test > runs/160/effects_${e}_sparse.log 2>&1
for k in sparse mlp; do
  while [ ! -f runs/160/model_${e}_$k.pkl ]; do sleep 10; done
  FIX=1 tools/card160/run.sh 590 env FIX=1 python tools/card160/evaluate.py run $e $k test > runs/160/fix_${e}_$k.log 2>&1
done
if [ "$e" = 1 ]; then for k in rules random; do tools/card160/run.sh 590 python tools/card160/evaluate.py run 1 $k test > runs/160/test_1_$k.log 2>&1; done; fi
tail -q -n 5 runs/160/fix_${e}_sparse.log runs/160/fix_${e}_mlp.log | cut -c1-130
