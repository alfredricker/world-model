#!/usr/bin/env bash
# Card 107: build the two environments and the Craftium world under runs/107/envs (git-ignored).
#   craftium_venv  Python 3.12 with Craftium's wheel (v0.0.1, GitHub release; bundles Luanti and VoxeLibre)
#   depth_venv     an overlay on the project's poetry env (its torch, read-only) plus Depth Anything 3's few extras
#   da3_src        Depth Anything 3 at a pinned commit; da3_stubs: empty modules it imports but depth never uses
#   wm_env         Craftium's open world (voxel-libre2) with this card's probe mod (tools/card107/wm_probe)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
E=$ROOT/runs/107/envs
cd "$E"
WHEEL=craftium-0.0.1-cp312-cp312-manylinux_2_28_x86_64.whl
[ -f $WHEEL ] || curl -sL -o $WHEEL https://github.com/mikelma/craftium/releases/download/v0.0.1/$WHEEL
[ -d craftium_venv ] || { uv venv -q --python 3.12 craftium_venv; VIRTUAL_ENV=$E/craftium_venv uv pip install -q ./$WHEEL numpy scipy; }
PV=$(cd "$ROOT" && poetry env info -p)
if [ ! -d depth_venv ]; then
    uv venv -q --python "$PV/bin/python" depth_venv
    echo "$PV/lib/python3.13/site-packages" > depth_venv/lib/python3.13/site-packages/_project_env.pth
    VIRTUAL_ENV=$E/depth_venv uv pip install -q huggingface-hub==2.2.0
    VIRTUAL_ENV=$E/depth_venv uv pip install -q --no-deps addict==2.4.0 einops==0.8.2 evo==1.38.0 safetensors==0.8.0 \
        opencv-python-headless==5.0.0.93 scipy==1.18.1 trimesh==5.1.1 moviepy==1.0.3 decorator==5.3.1 proglog==0.1.12 \
        imageio-ffmpeg==0.6.0 plyfile==1.1.5 pillow-heif==1.8.1 tqdm==4.70.1
fi
if [ ! -d da3_src ]; then
    git clone -q https://github.com/ByteDance-Seed/Depth-Anything-3.git da3_src
    git -C da3_src checkout -q 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
fi
mkdir -p da3_stubs/pycolmap && echo "# stub: imported by Depth Anything 3's export code, unused for depth" > da3_stubs/pycolmap/__init__.py
if [ ! -d wm_env ]; then
    cp -r "$E/craftium_venv/lib/python3.12/site-packages/craftium/craftium-envs/voxel-libre2" wm_env
    echo "load_mod_wm_probe = mods/wm_probe" >> wm_env/worlds/world/world.mt
fi
rm -rf wm_env/mods/wm_probe && cp -r "$ROOT/tools/card107/wm_probe" wm_env/mods/wm_probe
echo "card 107 environments ready in $E"
