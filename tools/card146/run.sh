#!/usr/bin/env bash
# usage: tools/card146/run.sh <seconds> <command...>   capped run (CHARTER rule 5) in card 146's environment:
# data/ext/venv146 is a venv that adds accelerate, ema_pytorch, timm, einops, tensorboard and fastcluster on
# top of the older project environment (a .pth file points at it). data/ext/akorn is a clone of
# github.com/autonomousvision/akorn at eabbe27. Both are git-ignored; the card says how to rebuild them.
# With AKORN=1 the command runs inside the clone (its code reads ./data and writes ./runs, both links).
set -euo pipefail
T=$1; shift
cd "$(dirname "$0")/../.."
VENV146="$PWD/data/ext/venv146"
[ "${AKORN:-0}" = 1 ] && cd data/ext/akorn
find_lib() {  # newest 64-bit store directory holding the wanted library (as bin/prun does)
    local dir
    for dir in $(ls -dt /nix/store/$1 2>/dev/null); do
        if [[ -e "$dir/$2" ]] && [[ $(od -An -j4 -N1 -tu1 "$dir/$2") -eq 2 ]]; then echo "$dir"; return; fi
    done
}
gcc_lib=$(find_lib "*-gcc-*-lib/lib" libstdc++.so.6)
zlib_lib=$(find_lib "*-zlib-*/lib" libz.so.1)
export LD_LIBRARY_PATH="/run/opengl-driver/lib:${gcc_lib}:${zlib_lib}"
export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib OMP_NUM_THREADS=4
exec timeout "$T" systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=800% \
  nice -n 10 env PATH="$VENV146/bin:$PATH" "$@"
