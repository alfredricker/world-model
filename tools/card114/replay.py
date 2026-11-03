"""Card 114, gate step 1 (data check): the evaluator's state of every frame of visual-scene-play-v0.

Replays the stored qpos through OGBench's scene model (no physics step, kinematics only) to recover what the
evaluator knows: the gripper's position and opening, both handles' positions, the drawer's and window's slide
positions, the cube's position and both button states. Used for scoring only; the model never sees it.

  bin/prun python tools/card114/replay.py --split val|train [--data state] → runs/114/state_<split>.npz
  (--data state: OGBench's state dataset scene-play-v0, for the upper bound → runs/114/sstate_<split>.npz)
"""
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import ogbench  # noqa: F401  (registers the envs)

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def replay(qpos):
    env = gym.make('scene-v0', terminate_at_goal=False)
    env.reset(seed=0)
    u = env.unwrapped
    m, d = u._model, u._data
    pinch = u._pinch_site_id
    drawer, window = u._drawer_site_id, u._window_site_id
    grip = u._gripper_opening_joint_id
    n = len(qpos)
    out = {k: np.zeros((n, 3), np.float32) for k in ['effector', 'drawer_handle', 'window_handle']}
    out['gripper'] = np.zeros(n, np.float32)
    for t in range(n):
        d.qpos[:] = qpos[t]
        mujoco.mj_kinematics(m, d)
        out['effector'][t] = d.site_xpos[pinch]
        out['drawer_handle'][t] = d.site_xpos[drawer]
        out['window_handle'][t] = d.site_xpos[window]
        out['gripper'][t] = np.clip(qpos[t, grip] / 0.8, 0, 1)
    env.close()
    return out


def main():
    split = arg('--split', 'val')
    state_data = arg('--data', 'visual') == 'state'
    name = ('scene-play-v0' if state_data else 'visual-scene-play-v0') + ('-val' if split == 'val' else '')
    z = np.load(DATA / f'{name}.npz')
    qpos, buttons, actions, terminals = z['qpos'], z['button_states'], z['actions'], z['terminals']
    st = replay(qpos)
    st.update(
        drawer=qpos[:, 23].astype(np.float32), window=qpos[:, 24].astype(np.float32),
        cube=qpos[:, 14:17].astype(np.float32), buttons=buttons.astype(np.int8),
        actions=actions.astype(np.float32), terminals=terminals,
    )
    out = ROOT / 'runs/114'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{'sstate' if state_data else 'state'}_{split}.npz"
    np.savez(path, **st)
    print('frames', len(qpos), 'saved', path)
    print('drawer range', st['drawer'].min(), st['drawer'].max(), 'window range', st['window'].min(), st['window'].max())


if __name__ == '__main__':
    main()
