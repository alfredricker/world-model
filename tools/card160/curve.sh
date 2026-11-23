#!/usr/bin/env bash
# usage: curve.sh <name> <steps per world>   a smaller experience (GOAL P19): play, fit the energy model, the frozen test
cd "$(dirname "$0")/../.."
tools/card160/run.sh 500 python tools/card160/grid.py play $1 $2 > runs/160/play_$1.log 2>&1
tools/card160/run.sh 590 python tools/card160/evaluate.py fit $1 sparse > runs/160/fit_$1_sparse.log 2>&1
tools/card160/run.sh 590 python tools/card160/evaluate.py run $1 sparse test > runs/160/test_$1_sparse.log 2>&1
grep "experienced at least" runs/160/play_$1.log; grep "statements right" runs/160/fit_$1_sparse.log; cut -c1-120 runs/160/test_$1_sparse.log
