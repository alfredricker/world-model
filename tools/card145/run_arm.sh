#!/usr/bin/env bash
# Card 145: one capped training run that survives a GPU fault. usage: run_arm.sh <name> <minutes> <train.py arguments...>
# Each try is capped (memory, CPU, time); after a crash it resumes from the last 5-minute checkpoint, at most 4 tries.
cd "$(dirname "$0")/../.."
name=$1; minutes=$2; shift 2
V=$HOME/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13
for try in 1 2 3 4; do
  OMP_NUM_THREADS=4 timeout $((minutes * 60 + 300)) systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=800% \
    nice -n 10 bin/prun env PATH=$V/bin:$PATH python tools/card145/train.py --mode state --minutes $minutes --out $name --resume 1 "$@" >> runs/145/$name.log 2>&1
  grep -q '^done' runs/145/$name.log && break
  echo "try $try stopped; resuming" >> runs/145/$name.log
done
tail -n 1 runs/145/$name.log
