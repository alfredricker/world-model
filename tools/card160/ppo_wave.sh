#!/usr/bin/env bash
# usage: ppo_wave.sh <seed> <view...>   the baseline on the five test tasks for one seed, capped runs side by side, each under
# ten minutes; tested at the energy model's number of steps (700k) and at twice that.  (Seed 0 ran both views at once: ten
# runs slowed each other and only the first test was reached.)
cd "$(dirname "$0")/../.."
s=$1; shift
for v in "$@"; do for t in DoorKey-8x8 DoorKey-16x16 UnlockPickup BlockedUnlockPickup KeyCorridorS3R3; do
  CPUQ=200 OMP_NUM_THREADS=2 tools/card160/run.sh 590 python tools/card160/ppo.py $t $v $s 700000 1400000 > runs/160/ppo_${t}_${v}_$s.log 2>&1 &
done; done
wait; for v in "$@"; do tail -q -n 1 runs/160/ppo_*_${v}_$s.log; done
