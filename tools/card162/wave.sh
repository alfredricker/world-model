#!/usr/bin/env bash
# usage: tools/card162/wave.sh <experience>    the frozen test of card 162 for one experience; every run capped at 590 s
cd "$(dirname "$0")/../.."; e=$1; export CPUQ=400 OMP_NUM_THREADS=2
r() { tools/card162/run.sh 590 python tools/card162/ev.py "$@" 2>&1 | grep -v -i warn; }
for a in base A; do r run $a $e fresh DoorKey-8x8 DoorKey-16x16; r run $a $e fresh UnlockPickup BlockedUnlockPickup; r run $a $e fresh KeyCorridorS3R3; done
echo "wave $e done"
