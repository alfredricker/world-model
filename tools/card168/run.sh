#!/usr/bin/env bash
# usage: tools/card168/run.sh <seconds> <command...>   capped OGBench run (CHARTER rule 5) in poetry venv with NixOS GL/libz
set -euo pipefail
T=$1; shift
cd "$(dirname "$0")/../.."
V="$HOME/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13"
GLVND=$(cat runs/143/glvnd.txt 2>/dev/null || echo "")
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_DIRS=/run/opengl-driver/share/glvnd/egl_vendor.d
if [[ -n "$GLVND" ]]; then
  export LD_LIBRARY_PATH="$GLVND/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$PWD/tools/card168:$PWD/tools/card141:$PWD/tools/card132${PYTHONPATH:+:$PYTHONPATH}"
exec timeout "$T" systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=${CPUQ:-800}% \
  nice -n 10 bin/prun env PATH="$V/bin:$PATH" "$@"
