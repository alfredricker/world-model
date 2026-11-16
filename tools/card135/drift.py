"""Card 135: state that changes out of sight.

Test bed of cards 132 to 134.1 (true visible state; the agent sees only what it touches or changes; a lock is seen
only when the lock is touched). New: a share P of the touches of every episode, drawn at random, are made by
"another hand": they happen (the recorded physics is unchanged) but the agent neither sees them nor stores them, and
its records are not updated. So a remembered value can be wrong, more often the longer ago it was seen. Only the
agent's own touches are stored and scored. Recall is card 134.1's (per object, records chosen by contrast).
Arms (the object records):
  plain   card 134.1's record: seen flag, last seen value, touched flag, last effect, "that touch changed it"
  time    the same + the time since the object was last seen (log of 1 + touches elapsed, the other hand's included)
  truth   the true locks and whether the cube is truly in the drawer in place of those three records (upper bound)

  bin/prun python tools/card135/drift.py gate  → runs/135/gate.json   (plain and truth on test; time on the store, episodes held out)
  bin/prun python tools/card135/drift.py       → runs/135/drift.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card134.1'))
import factored as F
from factored import B, C, E

U, PS, NAMES, FOLDS = 0.02, (0.0, 0.25, 0.5), F.NAMES, F.FOLDS
DIM = (3, 1, 1, 1, 1)
BINS = ((1, 2), (3, 5), (6, 10), (11, 99))


def events(eps, tg, ch, lk, p, seed, look=None, cond=None):
    """look(e, i, o, age, wrong) → bool: before its own touch of k the agent may look at each object o in cond[k]; a look shows the
    true value. Without looks, m['resight'] lists (object, log(1 + age), was the remembered value wrong) each time an object is seen again."""
    TH = (0.01, 0.01, 0.01, 0.5, 0.5); resight = []
    R, Tr, Y, tgt = {n: [] for n in ('contact',) + NAMES}, {n: [] for n in ('contact',) + NAMES}, [], []
    m = {k: [] for k in ('pull', 'moved', 'reopen', 'cube_in', 'ep', 'mixed', 'slide', 'group', 'stale', 'el', 'looked')}
    for e, ((s, h), t, c) in enumerate(zip(eps, tg, ch)):
        ins = C.inside(s)
        hid = np.random.default_rng([seed, e]).random(len(t)) < p
        seen, eff, when = {n: None for n in NAMES}, {n: None for n in NAMES}, {n: None for n in NAMES}
        touched_lock, last = [False, False], {1: None, 2: None}
        for i in range(len(t)):
            if hid[i]:
                continue
            k = t[i]
            y = np.r_[(s[i + 1, :3] - s[i, :3]) / U, (s[i + 1, 3:5] - s[i, 3:5]) / U, 3 * (s[i + 1, 5:7] - s[i, 5:7])]
            val = lambda o, j: s[j, :3] if o == 0 else s[j, 2 + o]
            wrong = lambda o: seen[NAMES[o]] is None or bool(np.linalg.norm(np.atleast_1d(val(o, i) - seen[NAMES[o]])) > TH[o])
            j0 = lk[k] if k in (1, 2) else 0
            group = 0 if touched_lock[j0] else 1 if k in (1, 2) and last[k] is not None else 2
            stale0 = k in (1, 2) and seen[f'l{j0}'] is not None and seen[f'l{j0}'] != s[i, 5 + j0]
            el0 = i - when[f'l{j0}'] if k in (1, 2) and when[f'l{j0}'] is not None else -1
            looks = 0
            if look is not None:
                for o in cond.get(k, ()):
                    if look(e, i, o, None if when[NAMES[o]] is None else i - when[NAMES[o]], wrong(o)):
                        seen[NAMES[o]] = val(o, i); when[NAMES[o]] = i; looks += 1
            for o, (name, n) in enumerate(zip(NAMES, DIM)):
                unit = 1.0 if name[0] == 'l' else U
                a = np.zeros(n) if seen[name] is None else np.atleast_1d(seen[name]) / unit
                b = np.zeros(n) if eff[name] is None else eff[name]
                R[name].append(np.r_[float(seen[name] is not None), a, float(eff[name] is not None), b, float(np.linalg.norm(b) > 0.5),
                                     0.0 if when[name] is None else np.log1p(i - when[name])])
                Tr[name].append(R[name][-1] if o in (1, 2) else np.r_[float(ins[i])] if o == 0 else np.r_[s[i, 2 + o]])
            R['contact'].append(h[i] / U); Tr['contact'].append(h[i] / U); Y.append(y); tgt.append(k)
            mixed = k in (1, 2) and bool(c[i, 3] or c[i, 4])
            j = lk[k] if k in (1, 2) else 0
            ln = f'l{j}'
            m['group'].append(group); m['stale'].append(stale0); m['el'].append(el0); m['looked'].append(looks)
            m['mixed'].append(mixed); m['pull'].append(k in (1, 2) and not mixed); m['moved'].append(bool(c[i, 1] or c[i, 2]))
            m['reopen'].append(abs(s[i, 3]) < 0.03 and abs(s[i + 1, 3]) > 0.10); m['cube_in'].append(bool(ins[i]))
            m['ep'].append(e); m['slide'].append(k if k in (1, 2) else 1)
            if k in (1, 2):
                last[k] = bool(c[i, k])
            if k >= 0:
                eff[NAMES[k]] = y[:3] if k == 0 else y[2 + k:3 + k]
            for o, name in enumerate(NAMES):
                if k == o or c[i, o]:
                    if seen[name] is not None:
                        resight.append((o, np.log1p(i - when[name]), float(wrong(o))))
                    seen[name] = s[i + 1, :3] if o == 0 else s[i + 1, 2 + o]; when[name] = i + 1
                    if o >= 3:
                        touched_lock[o - 3] = True
    f = lambda d: {k: np.array(v, dtype=np.float32) for k, v in d.items()}
    m = {k: np.array(v) for k, v in m.items()}; m['resight'] = np.array(resight).reshape(-1, 3)
    return f(R), f(Tr), np.array(Y, dtype=np.float32), np.array(tgt), m


def arm(R, Tr, name):
    if name == 'truth':
        return Tr
    return R if name == 'time' else {k: (v if k == 'contact' else v[:, :-1]) for k, v in R.items()}


def score(pull, cube, prob, m):
    out = E.score(pull, cube, prob, m, None)
    q = m['pull'] & (m['group'] == 0)
    out['lock_touched_stale_share'] = round(float(m['stale'][q].mean()), 3)
    out['by_elapsed'] = {}
    for lo, hi in BINS:
        b = q & (m['el'] >= lo) & (m['el'] <= hi)
        if b.sum() > 20:
            out['by_elapsed'][f'{lo}-{hi}'] = dict(n=int(b.sum()), stale=round(float(m['stale'][b].mean()), 3), moved=round(float(m['moved'][b].mean()), 3),
                                                   mean_prob=round(float(prob[b].mean()), 3), logloss=round(E.logloss(m['moved'][b], prob[b]), 3),
                                                   bal=round(100 * B.bal(pull[b], m['moved'][b]), 1))
    return out


def run(r, tg, Y, ep, keep, rq, tq, slide):
    sub = {k: v[keep] for k, v in r.items()}
    log = {}
    sel = F.select(sub, tg[keep], Y[keep], ep[keep], log)
    return F.answer(sel, sub, tg[keep], Y[keep], rq, tq, slide), {k: v['added'] for k, v in log.items() if k in ('drawer->drawer', 'window->window', 'drawer->cube')}


def main():
    gate = len(sys.argv) > 1 and sys.argv[1] == 'gate'
    mem, test = C.load(C.MEM), C.load(C.TEST)
    tgm, chm, _ = C.targets(mem)
    tgt, cht, _ = C.targets(test)
    lk, _ = B.lock_of(mem, tgm, chm)
    out = {}
    for p in PS:
        Rm, Tm, Ym, tm, mm = events(mem, tgm, chm, lk, p, 0)
        Rt, Tt, Yt, tt, mt = events(test, tgt, cht, lk, p, 1)
        keep = np.where(~mm['mixed'])[0]
        o = dict(n={g: [int((mt['pull'] & (mt['group'] == i)).sum()), int((mt['pull'] & (mt['group'] == i) & ~mt['moved']).sum())] for i, g in enumerate(E.GROUPS)},
                 reopen=int(mt['reopen'].sum()), stored_events=int(len(keep)))
        q = mt['pull'] & (mt['group'] == 0)
        base = float(mm['moved'][mm['pull'] & (mm['group'] == 0)].mean())
        o['base_rate_lock_touched'] = dict(moves=round(base, 3), logloss=round(E.logloss(mt['moved'][q], np.full(q.sum(), base)), 3))
        for a in (('plain', 'truth') if gate else ('plain', 'time', 'truth')):
            ans, sel = run(arm(Rm, Tm, a), tm, Ym, mm['ep'], keep, arm(Rt, Tt, a), tt, mt['slide'])
            o[a] = dict(score(*ans, mt), chosen=sel)
        if gate:                                                                         # the new arm on the store only
            pull, cube, prob = np.zeros(len(tm), bool), np.zeros(len(tm), bool), np.zeros(len(tm))
            for f in range(FOLDS):
                tr, te = np.where(~mm['mixed'] & (mm['ep'] % FOLDS != f))[0], np.where(mm['ep'] % FOLDS == f)[0]
                (a_, b_, c_), _ = run(Rm, tm, Ym, mm['ep'], tr, {k: v[te] for k, v in Rm.items()}, tm[te], mm['slide'][te])
                pull[te], cube[te], prob[te] = a_, b_, c_
            o['time_store_held_out'] = score(pull, cube, prob, mm)
        out[str(p)] = o
        print('P', p, json.dumps({k: v for k, v in o.items()}, indent=None), flush=True)
    json.dump(out, open(C.ROOT / f'runs/135/{"gate" if gate else "drift"}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
