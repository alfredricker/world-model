#!/usr/bin/env bash
# usage: tools/card161/wave.sh <experience> <1|2>    the frozen test of card 161 for one experience, in two parts; every run capped at 590 s
cd "$(dirname "$0")/../.."; e=$1; export CPUQ=400 OMP_NUM_THREADS=2
r() { tools/card161/run.sh 590 python tools/card161/ev.py "$@" 2>&1 | grep -v -i warn; }
runs() { for s in "${@:2}"; do r run $1 $e $s DoorKey-8x8 DoorKey-16x16; r run $1 $e $s UnlockPickup BlockedUnlockPickup; r run $1 $e $s KeyCorridorS3R3; done; }
if [ "$2" = 1 ]; then
  [ -f runs/160/model_${e}_shared.pkl ] || r fit $e shared
  runs base fresh; runs A fresh old
else
  runs B fresh old; r hands mlp $e; r hands shared $e
fi
echo "wave $e $2 done"
