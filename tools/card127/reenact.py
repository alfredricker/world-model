"""Card 127, feasibility gate: can a remembered touch be done again in the live environment?

Memory: the touches of training episodes 400-499 (card 125.2's set), with the recorded proprioception and actions.
The skill "redo touch i":
1. reach: lift to a safe height (the 90th percentile of the effector's height in memory), move over the touch's
   approach point (the recorded effector pose PRE frames before contact), descend to it, with the recorded gripper
   opening. Closed loop on proprioception; the map from an action to the effector's displacement is a least-squares
   fit on the memory's own steps.
2. follow: track the recorded effector pose frame by frame from PRE frames before contact to POST frames after
   release, the same closed loop.
Nothing is trained and the scene is not looked at. Tested in OGBench's scene-v0 (state observations, used only for
proprioception; the truth is read from the simulator for scoring). For each kind of touch, N trials: the
environment is reset to a benchmark task's start, a remembered touch of that kind is drawn (chosen with the truth
in this gate: its slide started where the environment's is, and it had the effect), and redone.

  MUJOCO_GL=egl bin/prun python tools/card127/reenact.py [--n 20] → runs/127/reenact.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
EP, PRE, POST = 1001, 10, 10
POSE = slice(12, 18)                       # effector xyz (scaled), cos yaw, sin yaw, gripper opening (scaled)
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


class Memory:
    def __init__(self, name='m400', first=400):
        z = np.load(DATA / 'scene-play-v0.npz')
        sl = slice(first * EP, (first + 100) * EP)
        self.obs, self.act, self.qpos, self.btn = z['observations'][sl], z['actions'][sl], z['qpos'][sl], z['button_states'][sl]
        c = np.load(ROOT / f'runs/122/computed_{name}.npz')
        ok = (c['st'] % EP >= PRE + 1) & (c['en'] % EP <= EP - POST - 2)
        self.st, self.en = c['st'][ok], c['en'][ok]
        # action → displacement of the proprioceptive pose, from the memory's own steps
        same = np.arange(len(self.obs) - 1) % EP != EP - 1
        d = (self.obs[1:, 12:15] - self.obs[:-1, 12:15])[same]
        self.gain = np.linalg.lstsq(self.act[:-1, :3][same], d, rcond=None)[0]             # [3, 3]
        yaw = np.arctan2(self.obs[:, 16], self.obs[:, 15])
        dy = np.angle(np.exp(1j * (yaw[1:] - yaw[:-1])))[same]
        self.yaw_gain = float(np.linalg.lstsq(self.act[:-1, 3:4][same], dy, rcond=None)[0][0])
        dg = (self.obs[1:, 17] - self.obs[:-1, 17])[same]
        self.grip_gain = float(np.linalg.lstsq(self.act[:-1, 4:5][same], dg, rcond=None)[0][0])
        self.safe_z = float(np.quantile(self.obs[:, 14], 0.9))
        a, b = self.st - 2, self.en + 2
        lo = self.st - 5
        self.effect = dict(press0=self.btn[b, 0] != self.btn[lo, 0], press1=self.btn[b, 1] != self.btn[lo, 1],
                           drawer=self.qpos[b, 23] - self.qpos[a, 23], window=self.qpos[b, 24] - self.qpos[a, 24],
                           cube=np.linalg.norm(self.qpos[b, 14:17] - self.qpos[a, 14:17], axis=1))
        self.start = dict(drawer=self.qpos[a, 23], window=self.qpos[a, 24])

    def pose(self, t):
        o = self.obs[t]
        return np.r_[o[12:15], np.arctan2(o[16], o[15]), o[17]]


def pose_of(ob):
    return np.r_[ob[12:15], np.arctan2(ob[16], ob[15]), ob[17]]


def action_to(mem, cur, tgt, move=(1, 1, 1)):
    a = np.zeros(5, np.float32)
    want = (tgt[:3] - cur[:3]) * np.array(move)
    a[:3] = np.linalg.solve(mem.gain.T, want)
    a[3] = np.angle(np.exp(1j * (tgt[3] - cur[3]))) / mem.yaw_gain
    a[4] = (tgt[4] - cur[4]) / mem.grip_gain
    return np.clip(a, -1, 1)


def redo(env, mem, ob, i, budget=200):
    """Redo remembered touch i from the current state. Returns the last observation and the steps used."""
    t0, t1 = mem.st[i] - PRE, mem.en[i] + POST
    tgt, steps = mem.pose(t0), 0
    up = tgt.copy(); up[2] = mem.safe_z
    stages = [(np.r_[pose_of(ob)[:2], mem.safe_z, pose_of(ob)[3:]], (0, 0, 1)), (up, (1, 1, 1)), (tgt, (1, 1, 1))]
    for goal, move in stages:
        for _ in range(60):
            cur = pose_of(ob)
            if np.abs((goal[:3] - cur[:3]) * np.array(move)).max() < 0.05 and steps:
                break
            g = goal.copy()
            if move == (0, 0, 1):
                g[3:] = cur[3:]
            ob, *_ = env.step(action_to(mem, cur, g, move))
            steps += 1
            if steps >= budget:
                return ob, steps
    for t in range(t0 + 1, t1 + 1):
        ob, *_ = env.step(action_to(mem, pose_of(ob), mem.pose(t)))
        steps += 1
    return ob, steps


def main():
    n = int(arg('--n', 20))
    rng = np.random.default_rng(0)
    mem = Memory()
    env = gym.make('scene-v0', terminate_at_goal=False)
    u = env.unwrapped
    truth = lambda: dict(btn=u._cur_button_states.copy(), drawer=float(u._data.joint('drawer_slide').qpos[0]),
                         window=float(u._data.joint('window_slide').qpos[0]))
    E = mem.effect
    # kind → (task whose start is used, remembered touches of that kind, test of success)
    near = lambda s, v: np.abs(mem.start[s] - v) < 0.02
    only = (E['cube'] < 0.01)
    kinds = {
        'press button 1 (drawer lock)': (1, E['press0'] & only, lambda a, b: a['btn'][0] != b['btn'][0]),
        'press button 2 (window lock)': (1, E['press1'] & only, lambda a, b: a['btn'][1] != b['btn'][1]),
        'open the drawer': (1, (E['drawer'] < -0.1) & near('drawer', 0.0) & only, lambda a, b: b['drawer'] < -0.12),
        'open the window': (1, (E['window'] > 0.1) & near('window', 0.0) & only, lambda a, b: b['window'] > 0.16),
        'press button 1, locked start': (2, E['press0'] & only, lambda a, b: a['btn'][0] != b['btn'][0]),
        'press button 2, locked start': (2, E['press1'] & only, lambda a, b: a['btn'][1] != b['btn'][1]),
    }
    res = dict(gain=mem.gain.round(3).tolist(), safe_z=round(mem.safe_z, 2), kinds={})
    for name, (task, cand, good) in kinds.items():
        idx = np.where(cand)[0]
        wins, steps = [], []
        for _ in range(n):
            ob, _ = env.reset(options=dict(task_id=task))
            before = truth()
            ob, s = redo(env, mem, ob, int(rng.choice(idx)))
            wins.append(bool(good(before, truth()))); steps.append(s)
        res['kinds'][name] = dict(remembered=int(len(idx)), trials=n, success=round(float(np.mean(wins)), 2),
                                  mean_steps=round(float(np.mean(steps)), 1))
        print(name, res['kinds'][name], flush=True)
    # a chain, with the truth choosing the touches: from task 2's start (both locked, both open) close the drawer
    wins = []
    for _ in range(n):
        ob, _ = env.reset(options=dict(task_id=2))
        ob, s1 = redo(env, mem, ob, int(rng.choice(np.where(E['press0'] & only)[0])))
        cur = truth()['drawer']
        cand = np.where((E['drawer'] > 0.1) & (np.abs(mem.start['drawer'] - cur) < 0.02) & only)[0]
        if len(cand):
            ob, s2 = redo(env, mem, ob, int(rng.choice(cand)))
        wins.append(bool(truth()['drawer'] > -0.04))
    res['chain_unlock_then_close_drawer'] = dict(trials=n, success=round(float(np.mean(wins)), 2))
    print(json.dumps(res, indent=1))
    json.dump(res, open(ROOT / 'runs/127/reenact.json', 'w'), indent=1)
    os._exit(0)


if __name__ == '__main__':
    main()
