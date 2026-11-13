"""Card 128: card 127's planner with the cube as one more variable, whose value is a position.

Added to plan.py's round, before its own choice:
- the read-out (readout.py) gives the cube's position in the gripper's coordinates from the frame now (C) and from
  the goal frame (G);
- remembered cube moves: m400's touches at places with no variable and no cause role, across which a change was
  seen, whose release is at the height of the grasp (within 1 cm); A = the gripper's position at contact, B = at
  release, both from proprioception. A move is real if |B - A| is above Otsu's threshold on its log;
- the cube differs if |C - G| is above that same threshold;
- then: recall the real move with the smallest |C - A| + |G - B| (one of the best 3, at random) and follow its
  recorded path with the offset C - A up to contact, G - B from release on, blended in between.
Otherwise the round is plan.py's.
Problems: --task 3 (the benchmark's), or --cube N: N self-made problems, the cube's start and goal drawn at random
inside the range of remembered grasps, at least 8 cm apart, locks and slides as in task 1's start and unchanged.
--noshift 1 replays the recalled move with no offset (baseline).

  MUJOCO_GL=egl ... bin/prun python tools/card128/plan_cube.py --cube 30 [--tag x] → runs/128/plan_<tag>.json
"""
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'card127'))
from readout import CEN, OUT, ReadOut  # noqa: E402
from plan import PRE, POST, R, ROOT, Agent, Done, Runner, arg  # noqa: E402
from reenact import action_to  # noqa: E402
from gate import otsu  # noqa: E402  (card 125's)


class CubeAgent(Agent):
    def __init__(self, rng):
        super().__init__(rng)
        k = np.load(ROOT / 'runs/127/know.npz')
        used = set(self.effs) | set(self.conds) | {v['cause'] for v in self.conds.values()}
        from reenact import EP
        inside = (self.st % EP >= PRE + 1) & (self.en % EP <= EP - POST - 2)
        idx = np.where(~np.isin(self.place, list(used)) & k['changed'] & inside)[0]
        A, B = self.mem.obs[self.st[idx] + 2, 12:15], self.mem.obs[self.en[idx], 12:15]
        idx, A, B = (x[np.abs(B[:, 2] - A[:, 2]) < 0.1] for x in (idx, A, B))
        d = np.linalg.norm(B[:, :2] - A[:, :2], axis=1)
        self.cut = float(np.exp(otsu(np.log(d + 1e-3))))
        real = d > self.cut
        self.moves, self.A, self.B = idx[real], A[real, :2], B[real, :2]
        self.ro = ReadOut().cuda()
        self.ro.load_state_dict(torch.load(OUT / 'readout.pt'))
        self.ro.eval()

    def where(self, F):
        with torch.no_grad():
            return self.ro(torch.from_numpy(F)[None].cuda())[0][0, :2].cpu().numpy()

    def recall_move(self, C, G):
        s = np.linalg.norm(self.A - C, axis=1) + np.linalg.norm(self.B - G, axis=1)
        j = self.rng.choice(np.argsort(s)[:3])
        return int(self.moves[j]), C - self.A[j], G - self.B[j]


def follow(run, mem, st, en, off0, off1):
    def tgt(t):
        p = mem.pose(t).copy()
        w = np.clip((t - st) / max(en - st, 1), 0, 1)
        p[:2] += (1 - w) * off0 + w * off1
        return p
    first = tgt(st - PRE)
    up = first.copy(); up[2] = mem.safe_z
    run.lift(mem)
    run.goto(mem, up)
    run.goto(mem, first)
    for t in range(st - PRE + 1, en + POST + 1):
        run.step(action_to(mem, run.pose(), tgt(t)))


def cube_task(u, rng, lo, hi):
    t = copy.deepcopy(u.task_infos[0])
    t['goal'] = copy.deepcopy(t['init'])
    while True:
        c, g = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if np.linalg.norm(c - g) > 0.08:
            break
    t['init']['block_xyzs'] = np.array([[*c, 0.02]])
    t['goal']['block_xyzs'] = np.array([[*g, 0.02]])
    return t


def main():
    ncube, task, noshift = int(arg('--cube', 0)), int(arg('--task', 0)), bool(int(arg('--noshift', 0)))
    n = ncube or int(arg('--n', 50))
    tag = arg('--tag', (f'cube{ncube}' if ncube else f'task{task}') + ('_noshift' if noshift else ''))
    rng = np.random.default_rng(int(arg('--seed', 0)))
    ag = CubeAgent(rng)
    env = gym.make('visual-scene-v0', terminate_at_goal=False, width=R, height=R)
    env.reset(seed=int(arg('--seed', 0)))
    run = Runner(env)
    lo, hi = ag.A.min(0) / 10 + CEN[:2] + 0.02, ag.A.max(0) / 10 + CEN[:2] - 0.02
    print('remembered moves', len(ag.moves), 'threshold cm', round(ag.cut * 10, 1), 'range', lo.round(2), hi.round(2), flush=True)
    cube = lambda: run.u._data.joint('object_joint_0').qpos[:3].copy()
    eps = []
    for e in range(n):
        run.reset(task_info=cube_task(run.u, rng, lo, hi)) if ncube else run.reset(task_id=task)
        gF = ag.read(run.goal)
        G = ag.where(gF)
        gtrue = np.array(run.u.cur_task_info['goal']['block_xyzs'][0])
        ep = dict(goal_read_error_cm=round(float(np.linalg.norm(G / 10 + CEN[:2] - gtrue[:2]) * 100), 2), rounds=[])
        try:
            while True:
                run.rest(ag.mem, ag.rest)
                F = ag.read(run.frame)
                C = ag.where(F)
                rec = dict(at=run.steps, truth=run.truth(), cube_cm_from_goal=round(float(np.linalg.norm(cube() - gtrue) * 100), 1),
                           read_error_cm=round(float(np.linalg.norm(C / 10 + CEN[:2] - cube()[:2]) * 100), 2))
                ep['rounds'].append(rec)
                if np.linalg.norm(C - G) > ag.cut:
                    i, o0, o1 = ag.recall_move(C, G)
                    rec['act'] = ('move', i)
                    follow(run, ag.mem, ag.st[i], ag.en[i], o0 * (not noshift), o1 * (not noshift))
                    continue
                act, _ = ag.decide(F, gF)
                rec['act'] = act
                if act is None:
                    break
                run.redo(ag.mem, ag.st[act[2]] - PRE, ag.en[act[2]] + POST)
        except Done:
            pass
        ep.update(success=run.success, steps=run.steps, end=run.truth(), cube_end_cm_from_goal=round(float(np.linalg.norm(cube() - gtrue) * 100), 1))
        eps.append(ep)
        print(e, 'success' if run.success else 'FAIL', run.steps, 'steps', [r['act'][0] if r.get('act') else None for r in ep['rounds']],
              'cube', ep['cube_end_cm_from_goal'], 'cm from goal; read errors', [r['read_error_cm'] for r in ep['rounds']][:4], flush=True)
    res = dict(tag=tag, episodes=n, success=round(float(np.mean([x['success'] for x in eps])), 3),
               mean_steps_when_success=round(float(np.mean([x['steps'] for x in eps if x['success']] or [0])), 1),
               cube_within_4cm=round(float(np.mean([x['cube_end_cm_from_goal'] <= 4 for x in eps])), 3),
               median_read_error_cm=round(float(np.median([r['read_error_cm'] for x in eps for r in x['rounds']])), 2), log=eps)
    json.dump(res, open(OUT / f'plan_{tag}.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    sys.stdout.write(json.dumps({k: v for k, v in res.items() if k != 'log'}) + '\n'); sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
