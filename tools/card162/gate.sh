#!/usr/bin/env bash
# usage: tools/card162/gate.sh <experience>    card 162's feasibility gate for one experience (arm A and card 161's agent), on layouts
# outside every test; every run capped at 590 s
cd "$(dirname "$0")/../.."; e=$1; export CPUQ=400 OMP_NUM_THREADS=2
r() { tools/card162/run.sh 590 python tools/card162/ev.py "$@" 2>&1 | grep -v -i warn; }
[ -f runs/162/net_A_$e.pkl ] || r fit A $e
for a in base A; do r doors $a $e; r moves $a $e; done
for a in base A; do r run $a $e gate KeyCorridorS3R3; done
r run A $e gate UnlockPickup BlockedUnlockPickup
r run A $e gate DoorKey-8x8
echo "gate $e done"
