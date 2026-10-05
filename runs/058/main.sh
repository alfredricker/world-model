#!/usr/bin/env bash
# card 058: walking compares a route with clearing the way; card 056's goals and forks (one frame per episode),
# and version 10's episode-end tests (familiar worlds, chained rooms with one door, cluttered) on seed 399's encoder
cd "$(dirname "$0")/../.."
W=runs/054/b_m0.5_399.pt
R="bin/prun python tools/card058/shorter.py"
$R tools/card056/goals.py $W --out runs/058/goals_399.json --layouts 30 --one-per-episode > runs/058/goals_399.out 2>&1 &
$R tools/card053/planner_check.py $W --gates 0 --codes noise --out runs/058/planner_399.json > runs/058/planner_399.out 2>&1 &
$R tools/card053/planner_check.py $W --gates 0 --codes noise @@ --chain one_door --seed 399 --n 100 --out runs/058/d_one_399.json > runs/058/d_one_399.out 2>&1 &
$R tools/card053/planner_check.py $W --gates 0 --codes noise @@ --clutter --seed 399 --n 100 --out runs/058/d_clutter_399.json > runs/058/d_clutter_399.out 2>&1 &
wait
echo done
