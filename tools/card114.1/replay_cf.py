"""Card 114.1: factual and counterfactual replays of card 114's held-out pulls in OGBench's scene simulator.

For each held-out pull (start frame t) the simulator is set to the stored joint positions, velocities and button
states at t - 10, with that slide's lock flipped (counterfactual) or not (factual), and the 30 recorded actions
t - 10 .. t + 19 are replayed. Kept: the model's history inputs at t - 10, t - 5, t (rendered frames for the visual
data, the 40-number observation for the state data) and the slide's position at t, t + 5, ..., t + 20 (the truth).

  MUJOCO_GL=egl ... bin/prun python tools/card114.1/replay_cf.py --data visual|state [--limit N]
  → runs/114.1/replay_<data>.npz
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
OUT = ROOT / 'runs/114.1'
EP, SKIP, K = 1001, 5, 4
SLIDES = {'drawer': (23, 0), 'window': (24, 1)}        # qpos index, button index
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def main():
    which = arg('--data', 'visual')
    limit = int(arg('--limit', 0))
    name = 'visual-scene-play-v0' if which == 'visual' else 'scene-play-v0'
    ev = np.load(ROOT / 'runs/114' / ('events.npz' if which == 'visual' else 'sevents.npz'))
    srcs = {'val': np.load(DATA / f'{name}-val.npz'), 'tail': np.load(DATA / f'{name}.npz')}
    off = {'val': 0, 'tail': 900 * EP}
    env = gym.make(name.replace('-play', ''), terminate_at_goal=False, mode='data_collection')
    env.reset(seed=0)
    u = env.unwrapped
    rec = {k: [] for k in ['split', 'slide', 't', 'dir', 'locked', 'cf', 'hist', 'pos', 'pos_rec', 'act', 'hist_rec']}
    t0 = time.time()
    for split, src in srcs.items():
        qpos, qvel, btn, acts = src['qpos'], src['qvel'], src['button_states'], src['actions']
        obs = (np.load(ROOT / 'runs/114' / f"obs_{name}{'-val' if split == 'val' else ''}.npy", mmap_mode='r')
               if which == 'visual' else None)
        for slide, (qi, bi) in SLIDES.items():
            ts, ds, lk = ev[f'{split}/{slide}/t'], ev[f'{split}/{slide}/dir'], ev[f'{split}/{slide}/locked']
            if limit:
                ts, ds, lk = ts[:limit], ds[:limit], lk[:limit]
            for t, d, l in zip(ts, ds, lk):
                g = t + off[split]                              # row in the source file
                s0 = g - 2 * SKIP
                for cf in (False, True):
                    b = btn[s0].copy()
                    if cf:
                        b[bi] = 1 - b[bi]
                    u.set_state(qpos[s0].copy(), qvel[s0].copy(), b)
                    hist = [u.compute_observation()]
                    pos = []
                    for j in range(3 * SKIP + 3 * SKIP):       # t-10 .. t+19
                        ob, *_ = u.step(acts[s0 + j])
                        n = j + 1                                # frames after s0
                        if n in (SKIP, 2 * SKIP):
                            hist.append(ob)
                        if n >= 2 * SKIP and (n - 2 * SKIP) % SKIP == 0:
                            pos.append(u._data.qpos[qi])
                    rec['split'].append(split); rec['slide'].append(slide); rec['t'].append(t)
                    rec['dir'].append(d); rec['locked'].append(l); rec['cf'].append(cf)
                    rec['hist'].append(np.stack(hist)); rec['pos'].append(np.array(pos, np.float32))
                    rec['pos_rec'].append(qpos[g + SKIP * np.arange(K + 1), qi].astype(np.float32))
                    rec['act'].append(acts[s0:s0 + 6 * SKIP].reshape(6, 25).astype(np.float32))
                    if which == 'visual':
                        rec['hist_rec'].append(np.asarray(obs[g + SKIP * np.arange(-2, 1)]))
            print(f'{split} {slide}: {len(ts)} pulls, {time.time() - t0:.0f}s', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f'replay_{which}.npz', **{k: np.array(v) for k, v in rec.items()})
    print(f'{len(rec["t"])} replays in {time.time() - t0:.0f}s → {OUT / f"replay_{which}.npz"}')


if __name__ == '__main__':
    main()
