"""Card 107: collect a Craftium clip — frames, the agent's pose, and the probe mod's log (scoring only).

  tools/card107/cprun tools/card107/probe.py <steps> <stem> [--mode pasture|roam] [--cx X --cz Z] [--n 8]
        [--mob mobs_mc:cow] [--seed S] [--mouse 0.05] [--fwd 0.2] [--w 640] [--h 360] [--voxels]
      → runs/107/clips/<stem>.npz (frames, poses: x, y, z, yaw, pitch in degrees, dtime; acts), <stem>.log,
        and with --voxels <stem>_vox.npz and <stem>_nodes.txt (scoring only)

The world is runs/107/envs/wm_env (tools/card107/setup.sh). In lockstep (sync_mode), one game frame per step.
Policy, each held with p = 0.9:
  pasture: nop 0.6 - fwd, turn left 0.2, turn right 0.2, forward fwd (card 107's test clip: mostly standing)
  roam:    forward 0.5, turn left 0.15, turn right 0.15, jump 0.1, nop 0.1 (walking, for view synthesis)
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from craftium import CraftiumEnv
from craftium.wrappers import DiscreteActionWrapper

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "107"
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
N, STEM = int(sys.argv[1]), sys.argv[2]
MODE = arg("--mode", "pasture")
MOB, NMOB = arg("--mob", "mobs_mc:cow"), int(arg("--n", 8))
MM, W, H = float(arg("--mouse", 0.05)), int(arg("--w", 640)), int(arg("--h", 360))
FWD, SEED = float(arg("--fwd", 0.2)), int(arg("--seed", 1021))
CX, CZ = int(arg("--cx", 400)), int(arg("--cz", 400))
VOX = "--voxels" in sys.argv
OUT = RUNS / "clips"
OUT.mkdir(parents=True, exist_ok=True)
ACTIONS = ["forward", "backward", "left", "right", "jump", "sneak", "dig", "place", "slot_1", "slot_2", "slot_3",
           "slot_4", "slot_5", "mouse x+", "mouse x-", "mouse y+", "mouse y-"]
NOP, XP, XM = 0, 1 + ACTIONS.index("mouse x+"), 1 + ACTIONS.index("mouse x-")
FW, JUMP = 1 + ACTIONS.index("forward"), 1 + ACTIONS.index("jump")
if MODE == "roam":
    CHOICES, PROBS = [FW, XP, XM, JUMP, NOP], [0.5, 0.15, 0.15, 0.1, 0.1]
else:
    CHOICES, PROBS = [NOP, XP, XM, FW], [0.6 - FWD, 0.2, 0.2, FWD]

conf = dict(max_block_generate_distance=3, mcl_logging_mobs_spawn=True, hud_scaling=0.5, fov=90, console_alpha=0,
            smooth_lighting=False, performance_tradeoffs=True, enable_particles=False,
            wm_n=NMOB, wm_mob=MOB, wm_seed=SEED, wm_mode=MODE, wm_cx=CX, wm_cz=CZ)
t0 = time.monotonic()
base = CraftiumEnv(env_dir=str(RUNS / "envs" / "wm_env"), obs_width=W, obs_height=H, init_frames=200,
                   game_id="VoxeLibre", _minetest_conf=conf, sync_mode=True, seed=SEED,
                   run_dir_prefix=str(RUNS / "luanti_runs"), max_timesteps=None,
                   enable_voxel_obs=VOX, _voxel_obs_available=VOX)
base.mt.proc_env = {**os.environ, "SDL_VIDEODRIVER": "offscreen"}          # NixOS: keep the library paths
env = DiscreteActionWrapper(base, actions=ACTIONS, mouse_mov=MM)
obs, info = env.reset()
t_reset = time.monotonic() - t0
rng = np.random.default_rng(SEED)
pose = lambda i: [*np.asarray(i["player_pos"], float).tolist(), float(i["player_yaw"]), float(i["player_pitch"]),
                  float(i["mt_dtime"])]
frames, poses, acts = [obs], [pose(info)], []
vox = [np.asarray(info["voxel_obs"])[..., 0].astype(np.uint16)] if VOX else []
a = NOP
t1 = time.monotonic()
for t in range(N):
    if rng.random() > 0.9:
        a = int(rng.choice(CHOICES, p=PROBS))
    obs, r, term, trunc, info = env.step(a)
    frames.append(obs)
    poses.append(pose(info))
    acts.append(a)
    if VOX:
        vox.append(np.asarray(info["voxel_obs"])[..., 0].astype(np.uint16))
    if term or trunc:
        print("episode ended at", t, flush=True)
        break
dt = time.monotonic() - t1
world = Path(base.mt.run_dir) / "worlds" / "world"
shutil.copy(world / "wm_probe.log", OUT / f"{STEM}.log")
if VOX:
    shutil.copy(world / "wm_nodes.txt", OUT / f"{STEM}_nodes.txt")
    np.savez_compressed(OUT / f"{STEM}_vox.npz", vox=np.stack(vox))
env.close()
np.savez_compressed(OUT / f"{STEM}.npz", frames=np.stack(frames), poses=np.array(poses), acts=np.array(acts))
P = np.array(poses)
rep = {"stem": STEM, "mode": MODE, "place": [CX, CZ] if MODE == "roam" else [120, 92], "steps": len(acts),
       "seed": SEED, "reset seconds": round(t_reset, 1), "steps per second": round(len(acts) / dt, 1),
       "walked (blocks)": round(float(np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1).sum()), 1),
       "start": P[0, :3].round(1).tolist()}
print(json.dumps(rep), flush=True)
json.dump(rep, open(OUT / f"{STEM}.json", "w"), indent=1)
