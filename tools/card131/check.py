"""Card 131, data check: does the cube read-out show when the cube is out of view?

og4's read-out (runs/129/readout_q.pt) answers "where would the gripper close on this group" as a softmax over
patches; it has no output for "nowhere in view". Live scenes, arm at og4's rest pose, three kinds:
  table   the cube at random on the table (inside the range of remembered grasps), drawer open or closed at random
  hidden  the cube inside the closed drawer (as in the goals of tasks 4 and 5)
  indrawer the cube inside the open drawer
For the cube's group: the largest patch score before the softmax, the largest softmax weight, the entropy of the
weights, and the error of the answer against the true cube. A threshold with no truth: the 5th percentile of each
measure over memory's held-out frames that preceded a cube touch (runs/129/feats.npy, episodes 80-99).

  MUJOCO_GL=egl ... bin/prun python tools/card131/check.py [--n 40] → runs/131/check.json
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.argv += ['--places', '1']
spec = importlib.util.spec_from_file_location('og3plan', HERE.parent / 'card129.1' / 'plan.py')
og3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(og3)
ROOT, arg, CEN = og3.ROOT, og3.arg, og3.CEN
GRP = 3


def measures(ro, F):
    with torch.no_grad():
        f = torch.from_numpy(np.asarray(F, dtype=np.float32))[None].cuda()
        s = ro.score(ro.trunk(f))[0, :, GRP]
        w = torch.softmax(s, 0)
        pos = ro(f)[0, GRP].cpu().numpy()
    return dict(score=float(s.max()), weight=float(w.max()), entropy=float(-(w * torch.log(w + 1e-12)).sum())), pos


def main():
    n = int(arg('--n', 40))
    rng = np.random.default_rng(0)
    ag = og3.UAgent(rng)
    env = og3.gym.make('visual-scene-v0', terminate_at_goal=False, width=og3.R, height=og3.R)
    env.reset(seed=0)
    run = og3.Runner(env)
    v = ag.aims[GRP]
    lo, hi = v['A'].min(0)[:2] / 10 + CEN[:2] + 0.02, v['A'].max(0)[:2] / 10 + CEN[:2] - 0.02
    X, M = np.load(ROOT / 'runs/129/feats.npy', mmap_mode='r'), np.load(ROOT / 'runs/129/meta.npy')
    G = np.load(ROOT / 'runs/129/groups.npy')
    rows = np.where((G[M[:, 9].astype(int)] == GRP) & (M[:, 10] >= 80))[0][:200]
    mem = [measures(ag.ro, X[i])[0] for i in rows]
    cut = {k: float(np.quantile([m[k] for m in mem], 0.95 if k == 'entropy' else 0.05)) for k in mem[0]}
    print('memory frames', len(rows), 'thresholds', cut, flush=True)
    out = dict(thresholds=cut, kinds={})
    cube = lambda: run.u._data.joint('object_joint_0').qpos[:3].copy()
    for kind in ('table', 'hidden', 'indrawer'):
        rec = []
        for e in range(n):
            t = og3.dev_task(run.u, rng)
            if kind == 'table':
                t['init']['block_xyzs'] = np.array([[*rng.uniform(lo, hi), 0.02]])
            else:
                t['init']['drawer_pos'] = 0.0 if kind == 'hidden' else -0.16
                t['init']['block_xyzs'] = np.array([[0.33, -0.36 - t['init']['drawer_pos'], 0.07]])
            t['goal'] = {k: (x.copy() if hasattr(x, 'copy') else x) for k, x in t['init'].items()}
            t['goal']['window_pos'] = 0.2 - t['init']['window_pos']           # a goal that differs, so no instant success
            run.reset(task_info=t)
            try:
                run.rest(ag.mem, ag.rest)
            except og3.Done:
                pass
            m, pos = measures(ag.ro, ag.read(run.frame))
            c = cube()
            m.update(err_cm=float(np.linalg.norm(pos / 10 + CEN - c) * 100), cube=c.round(3).tolist(), drawer=round(run.truth()['drawer'], 3),
                     read=(pos / 10 + CEN).round(3).tolist())
            rec.append(m)
        out['kinds'][kind] = rec
        print(kind, {k: [round(float(q), 3) for q in np.quantile([r[k] for r in rec], [0.1, 0.5, 0.9])] for k in ('score', 'weight', 'entropy', 'err_cm')},
              'cube e.g.', rec[0]['cube'], 'read', rec[0]['read'], flush=True)
    vis, hid = out['kinds']['table'], out['kinds']['hidden']
    for k in cut:
        sgn = -1 if k == 'entropy' else 1
        a, b = np.array([sgn * r[k] for r in vis]), np.array([sgn * r[k] for r in hid])
        auc = float((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean())
        out[f'{k}_auc'] = auc
        print(k, 'AUC table over hidden', round(auc, 3), 'called visible at memory threshold: table', round(float((a >= sgn * cut[k]).mean()), 2),
              'hidden', round(float((b >= sgn * cut[k]).mean()), 2),
              'indrawer', round(float((np.array([sgn * r[k] for r in out['kinds']['indrawer']]) >= sgn * cut[k]).mean()), 2), flush=True)
    json.dump(out, open(ROOT / 'runs/131/check.json', 'w'), indent=1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
