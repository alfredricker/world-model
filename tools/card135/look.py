"""Card 135, active form: memory decides when to look again.

Before its own touch of object A the agent may look at the objects whose records recall uses for "A moves" (chosen
by contrast over the store, card 134.1); a look shows the true value and costs one look. Policies:
  never / always
  age     look when the store says a value of this age is too often wrong: each time the agent saw an object again
          it stored (object, age of the memory, was it wrong); the estimate for an age is the share wrong among the
          25 stored re-sightings of that object nearest in age; an object never seen counts as the oldest age.
          Looks when the estimate exceeds a threshold (swept).
  random  the same number of looks as each "age" threshold, at random
  oracle  looks exactly when the remembered value is wrong or missing (not available to the agent)
Recall: card 134.1's, record with the time item; the store holds no looks.

  bin/prun python tools/card135/look.py gate → runs/135/look_gate.json  (store only, fifths of episodes held out)
  bin/prun python tools/card135/look.py      → runs/135/look.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift as D
from drift import B, C, E, F

THETAS = (0.12, 0.2, 0.25, 0.3)                                                          # 0.2 is the declared threshold


def staleness(rs):
    """→ f(object, age or None) = share of the 25 nearest stored re-sightings (in log age) that were wrong."""
    tab = {int(o): rs[rs[:, 0] == o][:, 1:] for o in np.unique(rs[:, 0])}

    def f(o, age):
        if o not in tab:
            return 1.0
        t = tab[o]
        a = t[:, 0].max() if age is None else np.log1p(age)
        return float(t[np.argsort(np.abs(t[:, 0] - a))[:25], 1].mean())
    return f


def evaluate(mem_args, test_args, cond, policy, fold=None):
    """the store is the agent's own past under the same looking policy; fold: store only, that fifth held out."""
    Rm, _, Ym, tm, mm = D.events(*mem_args, look=policy, cond=cond)
    keep = np.where(~mm['mixed'] & ((mm['ep'] % D.FOLDS != fold) if fold is not None else True))[0]
    sub = {k: v[keep] for k, v in Rm.items()}
    sel = F.select(sub, tm[keep], Ym[keep], mm['ep'][keep])
    if fold is None:
        Rt, _, _, tt, mt = D.events(*test_args, look=policy, cond=cond)
        rows = np.arange(len(tt))
    else:
        Rt, tt, mt, rows = Rm, tm, mm, np.where(mm['ep'] % D.FOLDS == fold)[0]
    pull, cube, prob = F.answer(sel, sub, tm[keep], Ym[keep], {k: v[rows] for k, v in Rt.items()}, tt[rows], mt['slide'][rows])
    m = {k: v[rows] for k, v in mt.items() if k != 'resight'}
    q = m['pull']
    g0 = q & (m['group'] == 0)
    return dict(looks_per_pull=round(float(m['looked'][q].mean()), 3), pulls=round(100 * B.bal(pull[q], m['moved'][q]), 1),
                lock_touched=round(100 * B.bal(pull[g0], m['moved'][g0]), 1), logloss=round(E.logloss(m['moved'][q], prob[q]), 3),
                by_group_looks=[round(float(m['looked'][q & (m['group'] == g)].mean()), 3) for g in range(3)],
                by_group=[round(100 * B.bal(pull[q & (m['group'] == g)], m['moved'][q & (m['group'] == g)]), 1) for g in range(3)])


def policies(stale, rng_seed=0):
    out = {'never': lambda e, i, o, age, w: False, 'always': lambda e, i, o, age, w: True, 'oracle': lambda e, i, o, age, w: w}
    for th in THETAS:
        out[f'age_{th}'] = lambda e, i, o, age, w, th=th: stale(o, age) > th
    return out


def main():
    gate = len(sys.argv) > 1 and sys.argv[1] == 'gate'
    mem, test = C.load(C.MEM), C.load(C.TEST)
    tgm, chm, _ = C.targets(mem)
    tgt, cht, _ = C.targets(test)
    lk, _ = B.lock_of(mem, tgm, chm)
    out = {}
    for p in D.PS[1:]:
        Rm, _, Ym, tm, mm = D.events(mem, tgm, chm, lk, p, 0)
        folds = range(D.FOLDS) if gate else (None,)
        res = {}
        mem_args, test_args = (mem, tgm, chm, lk, p, 0), (test, tgt, cht, lk, p, 1)
        for f in folds:
            keep = np.where(~mm['mixed'] & ((mm['ep'] % D.FOLDS != f) if gate else True))[0]
            sel = F.select({k: v[keep] for k, v in Rm.items()}, tm[keep], Ym[keep], mm['ep'][keep])
            cond = {k: [F.NAMES.index(x) for x in sel.get((k, k), []) if x != 'contact'] for k in (1, 2)}
            stale = staleness(mm['resight'])
            for name, fn in policies(stale).items():
                res.setdefault(name, []).append(evaluate(mem_args, test_args, cond, fn, f))
                if name.startswith('age_'):
                    rate = res[name][-1]['looks_per_pull']
                    rr = []
                    for s_ in range(1 if gate else 3):
                        rr.append(evaluate(mem_args, test_args, cond, lambda e, i, o, age, w, s_=s_: np.random.default_rng([s_, e, i, o]).random() < rate, f))
                    res.setdefault('random_' + name[4:], []).append({k: (np.mean([x[k] for x in rr], 0).round(3).tolist()) for k in rr[0]})
            res.setdefault('conditions', []).append({F.NAMES[k]: [F.NAMES[o] for o in v] for k, v in cond.items()})
        o = {k: ({kk: np.mean([x[kk] for x in v], 0).round(3).tolist() for kk in v[0]} if k != 'conditions' else v) for k, v in res.items()}
        tab = {int(ob): mm['resight'][mm['resight'][:, 0] == ob] for ob in (3, 4)}
        o['stored_resightings'] = {F.NAMES[ob]: [int(len(t)), round(float(t[:, 2].mean()), 3)] for ob, t in tab.items()}
        o['staleness_by_age'] = {str(a): round(staleness(mm['resight'])(3, a), 3) for a in (1, 2, 4, 8, 16, None)}
        out[str(p)] = o
        print('P', p, o['conditions'], o['stored_resightings'], o['staleness_by_age'], flush=True)
        for k, v in o.items():
            if isinstance(v, dict) and 'pulls' in v:
                print(f'  {k:12s} looks {v["looks_per_pull"]:.3f}  pulls {v["pulls"]:.1f}  lock touched {v["lock_touched"]:.1f}  logloss {v["logloss"]:.3f}  groups {v["by_group"]} looks {v["by_group_looks"]}', flush=True)
    json.dump(out, open(C.ROOT / f'runs/135/look{"_gate" if gate else ""}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
