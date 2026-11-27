#!/usr/bin/env bash
# usage: tools/card167/run.sh <seconds> <command...>   a capped run (CHARTER rule 5) in the project's .venv; cards 160 to 167 on the path
set -euo pipefail
T=$1; shift
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONPATH="$PWD/tools/card167:$PWD/tools/card166:$PWD/tools/card164:$PWD/tools/card162:$PWD/tools/card161:$PWD/tools/card160"
exec timeout "$T" systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=${CPUQ:-800}% \
  nice -n 10 env PATH="$PWD/.venv/bin:$PATH" "$@"
