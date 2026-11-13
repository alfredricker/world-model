"""Card 130: the agent's own tries come first (version 27's rule, carried to og3).

og3 (tools/card129.1/plan.py --places 1) recalls a remembered touch by how well it matches now and the goal, one of
the best 3 at random; a touch that has just failed can be recalled again. Here the agent keeps its own tries:
- after each touch it reads the scene again. A move worked if its variable no longer differs from the goal; a
  press worked if the condition's state changed;
- this episode: a touch that failed is not recalled again while others remain (version 27: one try decides);
- long term: each remembered touch has its own count of tries and successes. Among the 10 best matches (all
  candidates for a press) the touch with the highest P = (successes + BETA * p0) / (tries + BETA) is redone, the
  better match on a tie; p0 is the agent's own success rate so far for that variable and kind of touch, 0.5 with
  no tries.
--practice 1: the long-term counts are updated and saved (runs/130/own.json); otherwise they are loaded and
frozen, and only the episode's own failures are added. --own 0: og3 unchanged (baseline).
Other arguments as tools/card129.1/plan.py (--task, --dev, --cube, --mix, --seed, --tag); results in runs/130/.

  MUJOCO_GL=egl ... bin/prun python tools/card130/plan.py --mix 50 --practice 1 --seed 31
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.argv += ['--places', '1']
spec = importlib.util.spec_from_file_location('og3plan', HERE.parent / 'card129.1' / 'plan.py')
og3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(og3)

OUT = og3.ROOT / 'runs/130'
BETA, TOP = 1.0, 10
arg = og3.arg


class OAgent(og3.UAgent):
    def __init__(self, rng):
        super().__init__(rng)
        self.practice = bool(int(arg('--practice', 0)))
        f = OUT / 'own.json'
        self.own = {int(k): v for k, v in json.load(open(f)).items()} if f.exists() else {}   # touch → [successes, tries, kind key]
        self._goal, self.failed, self.pending = None, set(), None

    def p(self, i, key):
        tot = [v for v in self.own.values() if v[2] == key]
        p0 = (sum(v[0] for v in tot) + 0.5) / (sum(v[1] for v in tot) + 1)
        s, n, _ = self.own.get(i, (0, 0, key))
        return (s + BETA * p0) / (n + BETA)

    def pick(self, cands, key):
        left = [i for i in cands if i not in self.failed] or list(cands)
        return max(left, key=lambda i: (self.p(i, key), -left.index(i)))

    def decide(self, now, goal, nocond=False):
        if goal is not self._goal:                                         # a new episode
            self._goal, self.failed, self.pending = goal, set(), None
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
        if self.pending:                                                   # what did the last touch do?
            i, key, g, was = self.pending
            ok = bool(log['differs'][g] <= 1) if was is None else bool(log['state'][g][0] != was)
            if not ok:
                self.failed.add(i)
            if self.practice:
                s, n, _ = self.own.get(i, (0, 0, key))
                self.own[i] = [s + ok, n + 1, key]
                json.dump(self.own, open(OUT / 'own.json', 'w'))
            log['last_worked'] = ok
            self.pending = None
        zero = np.zeros(3)
        if ready:
            g = int(self.rng.choice(ready))
            key = f'move{g}'
            if g in self.aims:
                v = self.aims[g]
                order = np.argsort(np.linalg.norm(v['A'] - an[g], axis=1) + np.linalg.norm(v['B'] - ag[g], axis=1))[:TOP]
                at = {int(v['idx'][j]): j for j in order}
                i = self.pick(list(at), key)
                self.pending = (i, key, g, None)
                return ('move', g, i, an[g] - v['A'][at[i]], ag[g] - v['B'][at[i]]), log
            v = self.effs[g]
            s = np.where(v['redo'], self.match(now, goal, g), np.inf)
            order = [j for j in np.argsort(s)[:TOP] if np.isfinite(s[j])]
            i = self.pick([int(v['idx'][j]) for j in order], key)
            self.pending = (i, key, g, None)
            return ('move', g, i, zero, zero), log
        restore = [c for c in self.conds if log['state'][c][0] != log['state'][c][1]]
        for kind, cs in (('enable', enable), ('restore', restore)):
            for c in cs:
                v = self.conds[c]
                cand = v['press'][v['pfrom'] == log['state'][c][0]]
                if len(cand):
                    i = self.pick([int(x) for x in self.rng.permutation(cand)], f'press{c}')
                    self.pending = (i, f'press{c}', c, log['state'][c][0])
                    return (kind, v['cause'], i, zero, zero), log
        return None, log


if __name__ == '__main__':
    base = OAgent if int(arg('--own', 1)) else og3.UAgent

    def make(rng):                                                         # og3 reads its knowledge first, then writes here
        ag = base(rng)
        og3.OUT = OUT
        return ag
    og3.UAgent = make
    og3.main()
