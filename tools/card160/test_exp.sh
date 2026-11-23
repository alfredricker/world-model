#!/usr/bin/env bash
# usage: test_exp.sh <experience>   the frozen test for one experience: fit both models, run both agents on the five test tasks, score predicted events
cd "$(dirname "$0")/../.."
e=$1
for k in sparse mlp; do
  [ -f runs/160/model_${e}_$k.pkl ] || tools/card160/run.sh 590 python tools/card160/evaluate.py fit $e $k > runs/160/fit_${e}_$k.log 2>&1
done
for k in sparse mlp; do
  tools/card160/run.sh 590 python tools/card160/evaluate.py run $e $k test > runs/160/test_${e}_$k.log 2>&1
  tools/card160/run.sh 590 python tools/card160/evaluate.py effects $e $k test > runs/160/effects_${e}_$k.log 2>&1
done
tail -n 5 runs/160/test_${e}_sparse.log runs/160/test_${e}_mlp.log runs/160/effects_${e}_sparse.log runs/160/effects_${e}_mlp.log | cut -c1-330
