"""Card 128, feasibility gate: can a remembered cube move be shifted to where the cube is now and where it should go?

Memory: card 127's (episodes 400-499), the touches across which the cube moved more than 5 cm, both ends on the
table. A remembered move has the cube's position before (A) and after (B); the gripper is centred on the cube at
contact (within 1 mm in memory), so A and B are also where the gripper closed and opened.
Live: scene-v0, the cube at C, to go to G. The move recalled is the one with the smallest |C-A| + |G-B|. Its
recorded gripper path is followed with an offset added: C-A up to contact, G-B from release on, blended linearly
in between.
Arms (the truth supplies C, G and the cube's yaw here; card 128's main run must read them from the frame):
- replay: no offset (the trivial baseline);
- shift: the offset;
- shift+yaw: the offset, and the gripper's yaw turned by the difference between the live and remembered cube's yaw
  (to the nearest quarter turn);
- shift, C wrong by 1 cm / 2 cm in a random direction: how exact the reading of the frame must be.
Problems: the benchmark's task 3 (cube 20 cm along the table), and random starts and goals on the table.
Success: the cube ends within 4 cm of G (the benchmark's rule).

  MUJOCO_GL=egl bin/prun python tools/card128/gate.py [--n 30] → runs/128/gate.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card127'))
from reenact import POST, PRE, ROOT, Memory, action_to, arg, pose_of  # noqa: E402

CEN = np.array([0.425, 0, 0])
yaw_of = lambda q: 2 * np.arctan2(q[..., 3], q[..., 0])
quarter = lambda a: (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def follow(env, mem, ob, i, off0, off1, dyaw=0.0):
    st, en = mem.st[i], mem.en[i]

    def tgt(t):
        p = mem.pose(t).copy()
        w = np.clip((t - st) / max(en - st, 1), 0, 1)
        p[:2] += ((1 - w) * off0 + w * off1) * 10
        p[3] += dyaw
        return p
    first = tgt(st - PRE)
    up = first.copy(); up[2] = mem.safe_z
    cur = pose_of(ob)
    for goal, move in [(np.r_[cur[:2], mem.safe_z, cur[3:]], (0, 0, 1)), (up, (1, 1, 1)), (first, (1, 1, 1))]:
        for _ in range(60):
            cur = pose_of(ob)
            if np.abs((goal[:3] - cur[:3]) * np.array(move)).max() < 0.05:
                break
            g = goal.copy()
            if move == (0, 0, 1):
                g[3:] = cur[3:]
            ob, *_ = env.step(action_to(mem, cur, g, move))
    for t in range(st - PRE + 1, en + POST + 1):
        ob, *_ = env.step(action_to(mem, pose_of(ob), tgt(t)))
    return ob


def main():
    n = int(arg('--n', 30))
    rng = np.random.default_rng(0)
    mem = Memory()
    a, b = mem.st - 2, mem.en + 2
    A, B = mem.qpos[a, 14:17], mem.qpos[b, 14:17]
    ok = (mem.effect['cube'] > 0.05) & (A[:, 2] < 0.03) & (B[:, 2] < 0.03)
    idx = np.where(ok)[0]
    yawA = yaw_of(mem.qpos[a, 17:21])
    env = gym.make('scene-v0', terminate_at_goal=False)
    u = env.unwrapped
    cube = lambda: u._data.joint('object_joint_0').qpos.copy()
    lo, hi = A[idx, :2].min(0) + 0.03, A[idx, :2].max(0) - 0.03
    lo[1] = max(lo[1], -0.2)                                             # keep off the drawer's side
    arms = {'replay': (0, False, 0.0), 'shift': (1, False, 0.0), 'shift+yaw': (1, True, 0.0),
            'shift, C wrong by 1 cm': (1, False, 0.01), 'shift, C wrong by 2 cm': (1, False, 0.02)}
    res = dict(remembered_moves=int(len(idx)), trials=n, problems={})
    for prob in ('task 3', 'random'):
        out = {}
        for name, (use, yaw, noise) in arms.items():
            wins, dist, near = [], [], []
            for _ in range(n):
                ob, _ = env.reset(options=dict(task_id=3))
                if prob == 'random':
                    while True:
                        c, g = rng.uniform(lo, hi), rng.uniform(lo, hi)
                        if np.linalg.norm(c - g) > 0.08:
                            break
                    q = cube(); q[:2] = c
                    u._data.joint('object_joint_0').qpos[:] = q
                    ob, *_ = env.step(np.zeros(5, np.float32))
                    G = np.r_[g, 0.02]
                else:
                    G = u.cur_task_info['goal']['block_xyzs'][0]
                q = cube()
                C = q[:3].copy()
                i = idx[np.argmin(np.linalg.norm(A[idx, :2] - C[:2], axis=1) + np.linalg.norm(B[idx, :2] - G[:2], axis=1))]
                near.append(np.linalg.norm(A[i, :2] - C[:2]) + np.linalg.norm(B[i, :2] - G[:2]))
                th = rng.uniform(0, 2 * np.pi)
                Cn = C[:2] + noise * np.array([np.cos(th), np.sin(th)])
                follow(env, mem, ob, i, use * (Cn - A[i, :2]), use * (G[:2] - B[i, :2]),
                       quarter(yaw_of(q[3:7]) - yawA[i]) if yaw else 0.0)
                d = float(np.linalg.norm(cube()[:3] - G))
                wins.append(d <= 0.04); dist.append(d)
            out[name] = dict(success=round(float(np.mean(wins)), 2), median_cm_from_goal=round(float(np.median(dist)) * 100, 1),
                             mean_cm_recalled_move_off=round(float(np.mean(near)) * 100, 1))
            print(prob, name, out[name], flush=True)
        res['problems'][prob] = out
    json.dump(res, open(ROOT / 'runs/128/gate.json', 'w'), indent=1)
    os._exit(0)


if __name__ == '__main__':
    main()
