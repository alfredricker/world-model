"""Card 127: plans from remembered touches, in OGBench's live visual scene (256-px frames).

The agent sees the frame, the goal frame and its proprioception. It knows what know.py read from memory:
places, each slide place's variable (effect patches), each condition's patches, two states, enabling state and
cause place, and the presses (single touches at a cause place across which the condition's state changed).

One round:
1. rest: lift, move to the rest pose (the remembered gripper position farthest from every place in use), hold still, read the frame (blocks 2 and 12 per patch).
2. compare with the goal frame:
   - a pair of readings (before, after) is matched to a remembered touch by the squared distance over the
     variable's patches, before to before plus after to after. Remembered pairs are split into changed and
     unchanged by Otsu on the log distance between their own before and after. A place's variable differs if
     (now, goal) is closer to its nearest remembered changed pair than to its nearest unchanged pair;
   - a condition differs if its state now is not its state in the goal frame.
3. choose one touch:
   - a variable differs and its condition reads the enabling state (or it has none): recall, among that place's
     touches that changed the variable and across which a change was seen (card 125's outcome; without it a
     brief contact before an unsensed pull is recalled, and redoing it does nothing), the one whose before matches now and whose after matches the goal over
     the variable's patches (one of the best 3, at random, so that a failed touch is not repeated for ever);
   - else a variable differs and its condition reads the other state: a press of its cause place whose
     before-state is the state now;
   - else a condition differs from the goal: the same kind of press. (So a condition is restored only when no
     step that needs it is pending.)
   - nothing differs: stop.
4. redo the touch (reenact.py), go to 1.
--nocond removes the condition check in 3 (a differing variable is pulled at once).
Problems: --task 1|2 (the benchmark's), or --dev N: N self-made problems, each lock and each slide's start and
goal drawn at random (slides open or closed), the cube unmoved, start and goal not equal.
The simulator's truth is used only to score and to log.

  MUJOCO_GL=egl ... bin/prun python tools/card127/plan.py --dev 20 [--nocond 1] [--tag x] → runs/127/plan_<tag>.json
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
import ogbench  # noqa: E402,F401
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'card125'))
sys.path.insert(0, str(HERE.parent / 'card124'))
sys.path.insert(0, str(HERE.parent / 'card115'))
from reenact import EP, POST, PRE, ROOT, Memory, action_to  # noqa: E402
from gate import otsu  # noqa: E402
from train import features  # noqa: E402
from encode import load_encoder  # noqa: E402

R, LIMIT, HOLD = 256, 750, 3
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


class Done(Exception):
    pass


class Agent:
    def __init__(self, rng):
        self.rng = rng
        self.mem = Memory()
        k = np.load(ROOT / 'runs/127/know.npz')
        self.st, self.en, self.place, self.steady = k['st'], k['en'], k['place'], k['steady']
        before = np.load(ROOT / 'runs/125/before_m400.npy', mmap_mode='r')
        n = len(self.st)
        inside = (self.st % EP >= PRE + 1) & (self.en % EP <= EP - POST - 2)
        self.conds, self.effs = {}, {}
        for c in range(int(self.place.max()) + 1):
            if f'cond{c}' in k:
                self.conds[c] = dict(top=k[f'cond{c}'], centre=k[f'centre{c}'], states=k[f'states{c}'],
                                     enabled=int(k[f'enabled{c}']), cause=int(k[f'cause{c}']),
                                     press=k[f'press{c}'], pfrom=k[f'pressfrom{c}'])
                ok = inside[self.conds[c]['press']]
                self.conds[c]['press'], self.conds[c]['pfrom'] = self.conds[c]['press'][ok], self.conds[c]['pfrom'][ok]
            if f'eff{c}' in k:
                eff = k[f'eff{c}']
                idx = np.where(k['has_next'])[0]
                B = np.stack([np.asarray(before[i][eff], np.float32) for i in idx])
                A = np.stack([np.asarray(before[i + 1][eff], np.float32) for i in idx])
                w = self.steady[idx][:, eff] & self.steady[idx + 1][:, eff]
                keep = w.sum(1) >= 3
                idx, B, A, w = idx[keep], B[keep], A[keep], w[keep].astype(np.float32)
                d = (((A - B) ** 2).sum(-1) * w).sum(1) / w.sum(1)
                here = self.place[idx] == c
                ch = d > float(np.exp(otsu(np.log(d[here]))))
                self.effs[c] = dict(eff=eff, idx=idx, B=B, A=A, w=w, changed=ch, redo=ch & here & inside[idx] & k['changed'][idx])
        # rest pose: the remembered gripper position farthest from every place in use (a variable's or a cause place)
        centres = np.stack([self.mem.obs[self.st[self.place == c] + 2, 12:15].mean(0) for c in range(int(self.place.max()) + 1)])
        used = sorted(set(self.effs) | set(self.conds) | {v['cause'] for v in self.conds.values()})
        cand = np.where(inside)[0]
        P = self.mem.obs[self.st[cand] - 2, 12:15]
        m = cand[np.argmax(np.linalg.norm(P[:, None] - centres[used][None], axis=-1).min(1))]
        self.rest = self.mem.pose(self.st[m] - 2)
        self.rest[4] = np.quantile(self.mem.obs[:, 17], 0.9)
        self.enc = load_encoder('cuda')

    def read(self, frame):
        with torch.no_grad():
            return features(self.enc, torch.from_numpy(np.ascontiguousarray(frame))[None].to('cuda'))[0].cpu().numpy()

    def state(self, F, c):
        v = self.conds[c]
        r = (F[v['top']] - v['centre']).mean(0)
        return int(np.argmin(((v['states'] - r) ** 2).sum(1)))

    def match(self, now, goal, c):
        v = self.effs[c]
        return ((((v['B'] - now[v['eff']]) ** 2).sum(-1) + ((v['A'] - goal[v['eff']]) ** 2).sum(-1)) * v['w']).sum(1) / v['w'].sum(1)

    def differs(self, now, goal, c):
        """Above 1: the pair (now, goal) is closer to a remembered change of the variable than to a remembered non-change."""
        s, ch = self.match(now, goal, c), self.effs[c]['changed']
        return float(s[~ch].min() / s[ch].min())

    def recall(self, now, goal, c):
        v = self.effs[c]
        s = np.where(v['redo'], self.match(now, goal, c), np.inf)
        j = self.rng.choice(np.argsort(s)[:3])
        return int(v['idx'][j]), float(s[j])

    def decide(self, now, goal, nocond=False):
        """→ (kind, place, touch) or None, and a log of the readings."""
        log = dict(differs={c: round(self.differs(now, goal, c), 2) for c in self.effs},
                   state={c: (self.state(now, c), self.state(goal, c), self.conds[c]['enabled']) for c in self.conds})
        pulls, enable = [], []
        for c in self.effs:
            if log['differs'][c] > 1:
                blocked = c in self.conds and not nocond and log['state'][c][0] != self.conds[c]['enabled']
                (enable if blocked else pulls).append(c)
        restore = [c for c in self.conds if log['state'][c][0] != log['state'][c][1]]
        if pulls:
            c, (i, s) = min(((c, self.recall(now, goal, c)) for c in pulls), key=lambda x: x[1][1])
            return ('pull', c, i), log
        for kind, cs in (('enable', enable), ('restore', restore)):
            for c in cs:
                v = self.conds[c]
                cand = v['press'][v['pfrom'] == log['state'][c][0]]
                if len(cand):
                    return (kind, v['cause'], int(self.rng.choice(cand))), log
        return None, log


class Runner:
    """Steps the environment, counts steps and stops the episode at success or at the limit."""

    def __init__(self, env):
        self.env, self.u = env, env.unwrapped

    def reset(self, **options):
        self.frame, info = self.env.reset(options=options)
        self.goal, self.steps, self.success = info['goal'], 0, False
        return info

    def pose(self):
        o = self.u.compute_ob_info()
        return np.r_[(o['proprio/effector_pos'] - np.array([0.425, 0, 0])) * 10, o['proprio/effector_yaw'][0],
                     o['proprio/gripper_opening'][0] * 3]

    def step(self, a):
        self.frame, _, _, _, info = self.env.step(a)
        self.steps += 1
        if info['success']:
            self.success = True
            raise Done
        if self.steps >= LIMIT:
            raise Done

    def goto(self, mem, goal, move=(1, 1, 1), tol=0.05, most=60):
        for _ in range(most):
            cur = self.pose()
            if np.abs((goal[:3] - cur[:3]) * np.array(move)).max() < tol:
                break
            g = goal.copy()
            if move == (0, 0, 1):
                g[3:] = cur[3:]
            self.step(action_to(mem, cur, g, move))

    def lift(self, mem):
        cur = self.pose()
        self.goto(mem, np.r_[cur[:2], mem.safe_z, cur[3:]], (0, 0, 1))

    def redo(self, mem, t0, t1):
        tgt = mem.pose(t0)
        up = tgt.copy(); up[2] = mem.safe_z
        self.lift(mem)
        self.goto(mem, up)
        self.goto(mem, tgt)
        for t in range(t0 + 1, t1 + 1):
            self.step(action_to(mem, self.pose(), mem.pose(t)))

    def rest(self, mem, pose):
        self.lift(mem)
        up = pose.copy(); up[2] = max(mem.safe_z, pose[2])
        self.goto(mem, up)
        self.goto(mem, pose)
        for _ in range(HOLD):
            self.step(np.zeros(5, np.float32))

    def truth(self):
        return dict(locks=[int(x) for x in self.u._cur_button_states],
                    drawer=round(float(self.u._data.joint('drawer_slide').qpos[0]), 3),
                    window=round(float(self.u._data.joint('window_slide').qpos[0]), 3))


def dev_task(u, rng):
    while True:
        t = copy.deepcopy(u.task_infos[0])
        for side in ('init', 'goal'):
            t[side]['button_states'] = rng.integers(0, 2, 2)
            t[side]['drawer_pos'] = float(rng.choice([0.0, -0.16]))
            t[side]['window_pos'] = float(rng.choice([0.0, 0.2]))
        t['goal']['block_xyzs'] = t['init']['block_xyzs'].copy()
        i, g = t['init'], t['goal']
        if (i['button_states'] != g['button_states']).any() or i['drawer_pos'] != g['drawer_pos'] or i['window_pos'] != g['window_pos']:
            return t


def main():
    dev, task, nocond = int(arg('--dev', 0)), int(arg('--task', 0)), bool(int(arg('--nocond', 0)))
    n = dev or int(arg('--n', 50))
    tag = arg('--tag', f'dev{dev}' if dev else f'task{task}' + ('_nocond' if nocond else ''))
    rng = np.random.default_rng(int(arg('--seed', 0)))
    ag = Agent(rng)
    env = gym.make('visual-scene-v0', terminate_at_goal=False, width=R, height=R)
    env.reset(seed=int(arg('--seed', 0)))
    run = Runner(env)
    print('rest pose', ag.rest.round(2), 'places with a variable', list(ag.effs), 'with a condition', list(ag.conds),
          'remembered pulls', {c: int(v['redo'].sum()) for c, v in ag.effs.items()}, flush=True)
    eps = []
    for e in range(n):
        info = run.reset(task_info=dev_task(run.u, rng)) if dev else run.reset(task_id=task)
        goal = ag.read(run.goal)
        t = run.u.cur_task_info
        ep = dict(start=run.truth(), goal=dict(locks=[int(x) for x in t['goal']['button_states']],
                                               drawer=t['goal']['drawer_pos'], window=t['goal']['window_pos']), touches=[])
        try:
            while True:
                run.rest(ag.mem, ag.rest)
                act, log = ag.decide(ag.read(run.frame), goal, nocond)
                rec = dict(at=run.steps, act=act, truth=run.truth(), differs=log['differs'],
                           state={c: list(v) for c, v in log['state'].items()})
                ep['touches'].append(rec)
                if act is None:
                    break
                i = act[2]
                run.redo(ag.mem, ag.st[i] - PRE, ag.en[i] + POST)
        except Done:
            pass
        ep.update(success=run.success, steps=run.steps, end=run.truth(), gave_up=bool(ep['touches'] and ep['touches'][-1]['act'] is None))
        eps.append(ep)
        print(e, 'success' if run.success else 'FAIL', run.steps, 'steps', len(ep['touches']), 'rounds',
              [a['act'][:2] if a['act'] else None for a in ep['touches']], ep['start'], '→', ep['goal'], flush=True)
    res = dict(tag=tag, episodes=n, nocond=nocond, success=round(float(np.mean([x['success'] for x in eps])), 3),
               mean_steps_when_success=round(float(np.mean([x['steps'] for x in eps if x['success']] or [0])), 1),
               gave_up=round(float(np.mean([x['gave_up'] and not x['success'] for x in eps])), 3), log=eps)
    print(json.dumps({k: v for k, v in res.items() if k != 'log'}))
    json.dump(res, open(ROOT / f'runs/127/plan_{tag}.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os._exit(0)


if __name__ == '__main__':
    main()
