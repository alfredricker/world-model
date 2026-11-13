"""Card 129.1: one planner, one skill; every group of places has two readings, each used where memory shows it informative.

Knowledge, all read with no truth:
- groups of places (card 129): places joined by remembered moves both ways;
- per group (card 127's know.py run on groups): an appearance variable if its touches change its 10 patches at
  least 5 times the general level; a condition, its states and cause group (cards 125-125.2);
- per group (aim.py): the aim (the group-asked read-out of where the gripper would close) is informative if its
  error on held-out touches is under half that of the touch's place's mean contact point.
A group has something to plan for if it has either reading.
One round: rest, read the frame; for each such group
- it differs from the goal if its appearance differs (card 127's rule: (now, goal) nearer a remembered change than
  a remembered non-change) or its aim differs (now and goal aims farther apart than twice the read-out's 90th-percentile
  error on held-out touches; Otsu on the lengths of remembered moves gave 11 cm and missed 8-10 cm differences);
- ready = differs, and its condition, if it has one, reads the enabling state.
Choose: a ready group at random; else a press at the cause group of a differing group's condition; else a press
that restores a condition to its state in the goal frame; else stop. (Conditions count only for groups that have
something to plan for.)
Recall for a ready group: by aim where informative (the remembered real move with the smallest |aim now - contact|
+ |aim in goal - end|; a move's end is where the hand next found the target, the contact point of the group's next
touch in the episode; the gripper's own release point left the cube 3 cm off), else by appearance (card 127). One of the best 3 at random.
One skill for every touch: the recorded gripper path with an offset: (aim now - contact) up to contact,
(aim in goal - end) from release on, blended between; both zero where the aim is not informative.
Problems: --task k; --dev N (locks and slides at random); --cube N (cube start and goal at random on the table);
--mix N (all at once).

  MUJOCO_GL=egl ... bin/prun python tools/card129.1/plan.py --task 3 [--tag x] → runs/129.1/plan_<tag>.json
"""
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
sys.path.insert(0, str(HERE.parent / 'card129'))
sys.path.insert(0, str(HERE.parent / 'card127'))
from readout_q import ReadOutQ  # noqa: E402
from plan import POST, PRE, R, ROOT, Agent, Done, Runner, arg, dev_task  # noqa: E402
from reenact import EP, action_to  # noqa: E402
from gate import otsu  # noqa: E402  (card 125's)

OUT = ROOT / 'runs/129.1'
CEN = np.array([0.425, 0, 0])


class UAgent(Agent):
    def __init__(self, rng):
        # --places 1 (diagnostic after the main run): appearance and conditions per place (card 127's knowledge);
        # groups only say which touches share an aim
        per_place = bool(int(arg('--places', 0)))
        super().__init__(rng, 'runs/127/know.npz' if per_place else 'runs/129.1/know.npz')
        k = np.load(ROOT / ('runs/127/know.npz' if per_place else 'runs/129.1/know.npz'))
        G = np.load(ROOT / 'runs/129/groups.npy')
        member = lambda g: (G[self.place] == g) if per_place else (self.place == g)
        ck = torch.load(ROOT / 'runs/129/readout_q.pt', weights_only=False)
        self.ro = ReadOutQ().cuda()
        self.ro.load_state_dict(ck['state'])
        self.ro.eval()
        inside = (self.st % EP >= PRE + 1) & (self.en % EP <= EP - POST - 2)
        self.aims = {}
        for g, v in json.load(open(OUT / 'aim.json')).items():
            if v['informative']:
                # a move's end: where the hand next found the target (the group's next touch in the episode)
                own = np.where(member(int(g)) & k['changed'])[0]
                pair = [(a, b) for a, b in zip(own[:-1], own[1:]) if self.st[a] // EP == self.st[b] // EP and inside[a]]
                idx, nxt = np.array([a for a, _ in pair]), np.array([b for _, b in pair])
                A, B = self.mem.obs[self.st[idx] + 2, 12:15], self.mem.obs[self.st[nxt] + 2, 12:15]
                d = np.linalg.norm(B - A, axis=1)
                cut = 2 * v['aim_p90_cm'] / 10                              # two readings, each within its 90th percentile
                self.aims[int(g)] = dict(cut=cut, idx=idx[d > cut], A=A[d > cut], B=B[d > cut])
        self.groups = sorted(set(self.effs) | set(self.aims))
        self.conds = {g: v for g, v in self.conds.items() if g in self.groups}
        # rest pose: card 127's rule over every group in use, the aimed ones included (without them it sat low over
        # the table and knocked the cube about 5 cm on the way to other touches)
        used = sorted(set(self.groups) | {v['cause'] for v in self.conds.values()})
        centres = np.stack([self.mem.obs[self.st[member(c) if c in self.aims else self.place == c] + 2, 12:15].mean(0) for c in used])
        cand = np.where(inside)[0]
        P = self.mem.obs[self.st[cand] - 2, 12:15]
        m = cand[np.argmax(np.linalg.norm(P[:, None] - centres[None], axis=-1).min(1))]
        print('rest pose before', self.rest.round(2), end=' ')
        self.rest = self.mem.pose(self.st[m] - 2)
        self.rest[4] = np.quantile(self.mem.obs[:, 17], 0.9)
        print('now', self.rest.round(2), flush=True)

    def aim(self, F):
        with torch.no_grad():
            return self.ro(torch.from_numpy(F)[None].cuda())[0].cpu().numpy()          # [groups, 3]

    def decide(self, now, goal, nocond=False):
        an, ag = self.aim(now), self.aim(goal)
        log = dict(differs={}, state={c: (self.state(now, c), self.state(goal, c), self.conds[c]['enabled']) for c in self.conds})
        ready, enable = [], []
        for g in self.groups:
            d = max(self.differs(now, goal, g) if g in self.effs else 0.0,
                    float(np.linalg.norm(an[g] - ag[g]) / self.aims[g]['cut']) if g in self.aims else 0.0)
            log['differs'][g] = round(d, 2)
            if d > 1:
                blocked = g in self.conds and not nocond and log['state'][g][0] != self.conds[g]['enabled']
                (enable if blocked else ready).append(g)
        zero = np.zeros(3)
        if ready:
            g = int(self.rng.choice(ready))
            if g in self.aims:
                v = self.aims[g]
                j = self.rng.choice(np.argsort(np.linalg.norm(v['A'] - an[g], axis=1) + np.linalg.norm(v['B'] - ag[g], axis=1))[:3])
                return ('move', g, int(v['idx'][j]), an[g] - v['A'][j], ag[g] - v['B'][j]), log
            return ('move', g, self.recall(now, goal, g)[0], zero, zero), log
        restore = [c for c in self.conds if log['state'][c][0] != log['state'][c][1]]
        for kind, cs in (('enable', enable), ('restore', restore)):
            for c in cs:
                v = self.conds[c]
                cand = v['press'][v['pfrom'] == log['state'][c][0]]
                if len(cand):
                    return (kind, v['cause'], int(self.rng.choice(cand)), zero, zero), log
        return None, log


def follow(run, mem, st, en, off0, off1):
    def tgt(t):
        p = mem.pose(t).copy()
        w = np.clip((t - st) / max(en - st, 1), 0, 1)
        p[:3] += (1 - w) * off0 + w * off1
        return p
    first = tgt(st - PRE)
    up = first.copy(); up[2] = mem.safe_z
    run.lift(mem)
    run.goto(mem, up)
    run.goto(mem, first)
    for t in range(st - PRE + 1, en + POST + 1):
        run.step(action_to(mem, run.pose(), tgt(t)))


def main():
    task, dev, ncube, mix = (int(arg(k, 0)) for k in ('--task', '--dev', '--cube', '--mix'))
    n = dev or ncube or mix or int(arg('--n', 50))
    tag = arg('--tag', f'task{task}' if task else f'dev{dev}' if dev else f'cube{ncube}' if ncube else f'mix{mix}')
    rng = np.random.default_rng(int(arg('--seed', 0)))
    ag = UAgent(rng)
    env = gym.make('visual-scene-v0', terminate_at_goal=False, width=R, height=R)
    env.reset(seed=int(arg('--seed', 0)))
    run = Runner(env)
    lo = np.min([v['A'].min(0) for v in ag.aims.values()], 0)[:2] / 10 + CEN[:2] + 0.02
    hi = np.max([v['A'].max(0) for v in ag.aims.values()], 0)[:2] / 10 + CEN[:2] - 0.02
    print('groups with appearance', list(ag.effs), 'with aim', {g: (len(v['idx']), round(v['cut'] * 10, 1)) for g, v in ag.aims.items()},
          'with condition', {g: v['cause'] for g, v in ag.conds.items()}, 'table range', lo.round(2), hi.round(2), flush=True)
    cube = lambda: run.u._data.joint('object_joint_0').qpos[:3].copy()

    def problem():
        t = dev_task(run.u, rng)
        if ncube:
            t['goal'] = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in t['init'].items()}
        if ncube or mix:
            while True:
                c, g = rng.uniform(lo, hi), rng.uniform(lo, hi)
                if np.linalg.norm(c - g) > 0.08:
                    break
            t['init']['block_xyzs'], t['goal']['block_xyzs'] = np.array([[*c, 0.02]]), np.array([[*g, 0.02]])
        return t
    eps = []
    for e in range(n):
        run.reset(task_id=task) if task else run.reset(task_info=problem())
        gF = ag.read(run.goal)
        t = run.u.cur_task_info
        gtrue = np.array(t['goal']['block_xyzs'][0])
        ep = dict(start=run.truth(), goal=dict(locks=[int(x) for x in t['goal']['button_states']], drawer=t['goal']['drawer_pos'],
                                               window=t['goal']['window_pos']), touches=[])
        try:
            while True:
                run.rest(ag.mem, ag.rest)
                act, log = ag.decide(ag.read(run.frame), gF)
                ep['touches'].append(dict(at=run.steps, act=act[:3] if act else None, truth=run.truth(),
                                          cube_cm_from_goal=round(float(np.linalg.norm(cube() - gtrue) * 100), 1),
                                          differs=log['differs'], state={c: list(v) for c, v in log['state'].items()}))
                if act is None:
                    break
                follow(run, ag.mem, ag.st[act[2]], ag.en[act[2]], act[3], act[4])
        except Done:
            pass
        ep.update(success=run.success, steps=run.steps, end=run.truth(), cube_end_cm=round(float(np.linalg.norm(cube() - gtrue) * 100), 1),
                  gave_up=bool(ep['touches'] and ep['touches'][-1]['act'] is None))
        eps.append(ep)
        print(e, 'success' if run.success else 'FAIL', run.steps, 'steps', [a['act'][:2] if a['act'] else None for a in ep['touches']],
              'cube', ep['cube_end_cm'], 'cm', ep['end'], '→', ep['goal'], flush=True)
    res = dict(tag=tag, episodes=n, success=round(float(np.mean([x['success'] for x in eps])), 3),
               mean_steps_when_success=round(float(np.mean([x['steps'] for x in eps if x['success']] or [0])), 1),
               gave_up=round(float(np.mean([x['gave_up'] and not x['success'] for x in eps])), 3),
               cube_within_4cm=round(float(np.mean([x['cube_end_cm'] <= 4 for x in eps])), 3), log=eps)
    json.dump(res, open(OUT / f'plan_{tag}.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    sys.stdout.write(json.dumps({k: v for k, v in res.items() if k != 'log'}) + '\n'); sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
