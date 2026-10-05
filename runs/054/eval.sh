#!/usr/bin/env bash
# card 054: evaluate one frozen encoder with identity-up-to-noise codes: the gap, recall on generator tries, the planner
# in the familiar worlds. Usage: runs/054/eval.sh NAME WEIGHTS.pt
cd "$(dirname "$0")/../.."
N=$1; W=$2
bin/prun python tools/card054/gap.py $W 2>&1 | grep -v -i warn > runs/054/gap_$N.txt &
bin/prun python tools/card053/recall_probe.py $W --interaction transition --gates 0 --codes noise --out runs/054/recall_$N.json > runs/054/recall_$N.out 2>&1 &
bin/prun python tools/card053/planner_check.py $W --gates 0 --codes noise --out runs/054/planner_$N.json > runs/054/planner_$N.out 2>&1 &
wait
echo done
