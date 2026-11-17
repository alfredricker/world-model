"""Card 143, criterion 3: put-away problems, live.

A problem: the cube at random on the table, the drawer open (-0.16) at the start and in the goal, locks and window
at random and the same in start and goal; the goal frame shows the cube inside the drawer, at a position the cube
had in recorded play with the drawer open (m400; used only to make the problem). Scored by the simulator: the cube
ends inside the drawer (card 141's rule); the benchmark's own success flag is given beside it.

The agent is og4 (tools/card130/plan.py, frozen counts). Options:
  --readout runs/143/fit_rel4_0   the aim read-out of this card (its .pt, and its .npz for the "differs" cut by og4's rule)
  --ends 1                        a remembered move of an aimed group that the hand never found again in the episode
                                  ends where the hand let go (og4 keeps only moves whose end is the group's next contact
                                  point, so it remembers no move that ends in the drawer)

  MUJOCO_GL=egl ... python tools/card143/put.py --n 50 [--readout ... --ends 1] --tag x → runs/143/put_<tag>.json
"""
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('og4plan', HERE.parent / 'card130' / 'plan.py')
og4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(og4)
og3, arg, ROOT = og4.og3, og4.arg, og4.og3.ROOT
OUT = ROOT / 'runs/143'
CEN, EP = og3.CEN, og3.EP


def inside(c):
    return bool((abs(c[0] - 0.33) < 0.12) and (c[2] > 0.04) and (c[1] < -0.1))


def main():
    n, seed, tag = int(arg('--n', 50)), int(arg('--seed', 0)), arg('--tag', 'og4')
    rng = np.random.default_rng(seed)
    ag = og4.OAgent(rng)
    ro = arg('--readout', '')
    if ro:
        ag.ro.load_state_dict(torch.load(ROOT / f'{ro}.pt', weights_only=False)['state'])
        d = np.load(ROOT / f'{ro}.npz')
        for g in ag.aims:
            ag.aims[g]['cut'] = 2 * float(np.quantile(d['own_err_cm'][d['own_group'] == g], 0.9)) / 10
    if int(arg('--ends', 0)):
        k = np.load(ROOT / 'runs/127/know.npz')
        G = np.load(ROOT / 'runs/129/groups.npy')
        ok = (ag.st % EP >= og3.PRE + 1) & (ag.en % EP <= EP - og3.POST - 2)
        for g, v in ag.aims.items():
            own = np.where((G[ag.place] == g) & k['changed'])[0]
            last = np.array([a for j, a in enumerate(own) if ok[a] and (j + 1 == len(own) or ag.st[own[j + 1]] // EP != ag.st[a] // EP)])
            A, B = ag.mem.obs[ag.st[last] + 2, 12:15], ag.mem.obs[ag.en[last], 12:15]
            m = np.linalg.norm(B - A, axis=1) > v['cut']
            print('group', g, 'moves', len(v['idx']), 'plus', int(m.sum()), 'that end where the hand let go', flush=True)
            v['idx'], v['A'], v['B'] = np.r_[v['idx'], last[m]], np.r_[v['A'], A[m]], np.r_[v['B'], B[m]]
    env = og3.gym.make('visual-scene-v0', terminate_at_goal=False, width=og3.R, height=og3.R)
    env.reset(seed=seed)
    run = og3.Runner(env)
    lo = np.min([v['A'].min(0) for v in ag.aims.values()], 0)[:2] / 10 + CEN[:2] + 0.02
    hi = np.max([v['A'].max(0) for v in ag.aims.values()], 0)[:2] / 10 + CEN[:2] - 0.02
    ev = np.load(ROOT / 'runs/132/ev_m400.npz')
    spots = ev['cube'][np.array([inside(c) for c in ev['cube']]) & (ev['drawer'] < -0.15)]
    print('aims', {g: (len(v['idx']), round(v['cut'] * 10, 2)) for g, v in ag.aims.items()}, 'drawer spots', len(spots), flush=True)
    cube = lambda: run.u._data.joint('object_joint_0').qpos[:3].copy()

    def problem():
        t = copy.deepcopy(run.u.task_infos[0])
        b, w = rng.integers(0, 2, 2), float(rng.choice([0.0, 0.2]))
        for side in ('init', 'goal'):
            t[side]['button_states'], t[side]['drawer_pos'], t[side]['window_pos'] = b.copy(), -0.16, w
        lo2 = np.maximum(lo, [-1, -0.05])                                   # start clear of the open drawer
        t['init']['block_xyzs'] = np.array([[*rng.uniform(lo2, hi), 0.02]])
        t['goal']['block_xyzs'] = spots[rng.integers(len(spots))][None].astype(np.float64)
        return t
    eps = []
    for e in range(n):
        run.reset(task_info=problem())
        gF = ag.read(run.goal)
        gtrue = np.array(run.u.cur_task_info['goal']['block_xyzs'][0])
        g_read = {g: ag.aim(gF)[g] / 10 + CEN for g in ag.aims}
        ep = dict(goal_read_error_cm={g: round(float(np.linalg.norm((p - gtrue)[:2]) * 100), 2) for g, p in g_read.items()}, touches=[])
        try:
            while True:
                run.rest(ag.mem, ag.rest)
                F = ag.read(run.frame)
                act, log = ag.decide(F, gF)
                ep['touches'].append(dict(at=run.steps, act=act[:3] if act else None, cube_cm_from_goal=round(float(np.linalg.norm(cube() - gtrue) * 100), 1),
                                          read_error_cm=round(float(np.linalg.norm((ag.aim(F)[min(ag.aims)] / 10 + CEN - cube())[:2]) * 100), 2),
                                          cube_inside=inside(cube()), differs=log['differs']))
                if act is None or len(ep['touches']) > 12:
                    break
                og3.follow(run, ag.mem, ag.st[act[2]], ag.en[act[2]], act[3], act[4])
        except og3.Done:
            pass
        ep.update(success=run.success, inside=inside(cube()), steps=run.steps, end=run.truth(), cube_end_cm=round(float(np.linalg.norm(cube() - gtrue) * 100), 1),
                  gave_up=bool(ep['touches'] and ep['touches'][-1]['act'] is None))
        eps.append(ep)
        print(e, 'IN' if ep['inside'] else 'out', 'success' if run.success else '', run.steps, 'steps', [a['act'][:2] if a['act'] else None for a in ep['touches']],
              'cube', ep['cube_end_cm'], 'cm; goal read error', ep['goal_read_error_cm'], flush=True)
    res = dict(tag=tag, episodes=n, cube_inside=round(float(np.mean([x['inside'] for x in eps])), 3), success=round(float(np.mean([x['success'] for x in eps])), 3),
               cube_within_4cm=round(float(np.mean([x['cube_end_cm'] <= 4 for x in eps])), 3),
               stopped_by_itself_with_cube_inside=round(float(np.mean([x['inside'] and x['gave_up'] for x in eps])), 3),
               goal_read_error_cm_median=round(float(np.median([list(x['goal_read_error_cm'].values())[0] for x in eps])), 2),
               touches_mean=round(float(np.mean([len(x['touches']) for x in eps])), 1), log=eps)
    json.dump(res, open(OUT / f'put_{tag}.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    sys.stdout.write(json.dumps({k: v for k, v in res.items() if k != 'log'}) + '\n'); sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
