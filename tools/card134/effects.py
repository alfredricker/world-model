"""Card 134: unseen state from effects.

Test bed of cards 132 to 133.1 (true visible state, narrow view), with a stricter view: a lock is seen only when
the lock itself is touched. Keys are scaled to unit spread per item over the store, equal weight (card 133.1's
prior); nothing is trained. Recall: 5 nearest stored events, median effect; the probability that a pull moves is
the share of the 25 nearest stored events in which that slide moved.
Arms (contents of the key, after contact point and touched object):
  seen      last seen value of each object (card 133.1's contents)
  effects   the same + for each object the effect of the last touch on it, with a "touched before" flag
  truth     the true locks and whether the cube is truly in the drawer
  last_pull the outcome of the last pull of this slide (a reference, middle group only)
Test pulls are split by what the agent knows of the slide's lock: lock touched earlier / slide pulled earlier
only / nothing.

  bin/prun python tools/card134/effects.py → runs/134/effects.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card132'))
import bed as B
import check as C

ROOT, U, K, KP = C.ROOT, 0.02, 5, 25
GROUPS = ('lock_touched', 'pull_only', 'nothing')
OBJ = (('cube', 3), ('drawer', 1), ('window', 1), ('l0', 1), ('l1', 1))                 # targets 0..4


def events(eps, tg, ch, lk):
    Xs, Xe, T, Y = [], [], [], []
    m = dict(pull=[], moved=[], reopen=[], cube_in=[], seen_cube=[], ep=[], mixed=[], slide=[], group=[], last=[])
    for e, ((s, h), t, c) in enumerate(zip(eps, tg, ch)):
        ins = C.inside(s)
        seen, eff = {n: None for n, _ in OBJ}, {n: None for n, _ in OBJ}
        touched_lock, last = [False, False], {1: None, 2: None}
        for i in range(len(t)):
            k = t[i]
            y = np.r_[(s[i + 1, :3] - s[i, :3]) / U, (s[i + 1, 3:5] - s[i, 3:5]) / U, 3 * (s[i + 1, 5:7] - s[i, 5:7])]
            head = np.r_[h[i] / U, 1.0 * (np.arange(-1, 5) == k)]
            f, g = [], []
            for name, n in OBJ:
                unit = 1.0 if name[0] == 'l' else U
                f += [float(seen[name] is not None), *(np.zeros(n) if seen[name] is None else np.atleast_1d(seen[name]) / unit)]
                g += [float(eff[name] is not None), *(np.zeros(n) if eff[name] is None else eff[name])]
            Xs.append(np.r_[head, f]); Xe.append(np.r_[head, f, g]); T.append(np.r_[head, s[i, 5:7], float(ins[i])]); Y.append(y)
            mixed = k in (1, 2) and bool(c[i, 3] or c[i, 4])
            j = lk[k] if k in (1, 2) else 0
            m['group'].append(0 if touched_lock[j] else 1 if k in (1, 2) and last[k] is not None else 2)
            m['last'].append(bool(last[k]) if k in (1, 2) and last[k] is not None else True)
            m['mixed'].append(mixed); m['pull'].append(k in (1, 2) and not mixed); m['moved'].append(bool(c[i, 1] or c[i, 2]))
            m['reopen'].append(abs(s[i, 3]) < 0.03 and abs(s[i + 1, 3]) > 0.10); m['cube_in'].append(bool(ins[i]))
            m['ep'].append(e); m['seen_cube'].append(seen['cube'] is not None); m['slide'].append(k if k in (1, 2) else 1)
            if k in (1, 2):
                last[k] = bool(c[i, k])
            if k >= 0:                                                                   # the effect of this touch on the object it touched
                name = OBJ[k][0]
                eff[name] = y[:3] if k == 0 else y[2 + k:3 + k]
            if k == 0 or c[i, 0]:
                seen['cube'] = s[i + 1, :3]
            if k == 1 or c[i, 1]:
                seen['drawer'] = s[i + 1, 3]
            if k == 2 or c[i, 2]:
                seen['window'] = s[i + 1, 4]
            for q in (0, 1):
                if k == 3 + q or c[i, 3 + q]:
                    seen[f'l{q}'] = s[i + 1, 5 + q]; touched_lock[q] = True
    f32 = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    return dict(seen=f32(Xs), effects=f32(Xe), truth=f32(T)), f32(Y), {k: np.array(v) for k, v in m.items()}


def answer(Es, Ys, Eq, slide):
    d = torch.cdist(Eq, Es)
    R = Ys[d.topk(K, largest=False).indices].median(1).values.numpy()
    near = Ys[d.topk(KP, largest=False).indices].numpy()                                 # [Q,25,7]
    r = np.arange(len(Eq))
    return np.abs(R[r, 2 + slide]) > 0.5, np.linalg.norm(R[:, :3], axis=1) > 2.5, (np.abs(near[r, :, 2 + slide]) > 0.5).mean(1)


def logloss(y, p):
    p = np.clip(p, 0.02, 0.98)                                                           # 25 neighbours cannot say less than 1 in 25
    return float(-(y * np.log(p) + (~y) * np.log(1 - p)).mean())


def calibration(y, p):
    b = np.minimum((p * 10).astype(int), 9)
    return float(sum((b == i).mean() * abs(p[b == i].mean() - y[b == i].mean()) for i in range(10) if (b == i).any()))


def score(pred, cube, prob, m, base):
    pc = lambda v: None if v is None else round(100 * v, 1)
    out = {}
    for gi, g in enumerate(GROUPS):
        q = m['pull'] & (m['group'] == gi)
        out[g] = pc(B.bal(pred[q], m['moved'][q]))
        if prob is not None:
            out[g + '_logloss'] = round(logloss(m['moved'][q], prob[q]), 3)
            out[g + '_mean_prob'] = round(float(prob[q].mean()), 3)
    out['pull'] = pc(B.bal(pred[m['pull']], m['moved'][m['pull']]))
    if cube is not None:
        out['reopen'] = pc(B.bal(cube[m['reopen']], m['cube_in'][m['reopen']]))
    if prob is not None:
        out['calibration_gap'] = pc(calibration(m['moved'][m['pull']], prob[m['pull']]))
    return out


def main():
    mem, test = C.load(C.MEM), C.load(C.TEST)
    tgm, chm, _ = C.targets(mem)
    tgt, cht, _ = C.targets(test)
    lk, _ = B.lock_of(mem, tgm, chm)
    Xm, Ym, mm = events(mem, tgm, chm, lk)
    Xt, Yt, mt = events(test, tgt, cht, lk)
    out = dict(n={g: [int((mt['pull'] & (mt['group'] == i)).sum()), int((mt['pull'] & (mt['group'] == i) & ~mt['moved']).sum())] for i, g in enumerate(GROUPS)},
               reopen=int(mt['reopen'].sum()), stores={})
    q = mt['pull'] & (mt['group'] == 2)
    base = float((mm['moved'][mm['pull'] & (mm['group'] == 2)]).mean())
    out['base_rate_nothing'] = dict(moves=round(base, 3), logloss=round(logloss(mt['moved'][q], np.full(q.sum(), base)), 3))
    out['last_pull'] = score(mt['last'], None, None, mt, base)['pull_only']
    for n, draws in ((400, 1), (100, 3)):
        rows = {a: [] for a in Xm}
        for s in range(draws):
            keep = ~mm['mixed'] & np.isin(mm['ep'], np.random.default_rng(s).choice(400, n, replace=False))
            for a in Xm:
                A, Q = Xm[a][keep], Xt[a]
                if a == 'truth':
                    w = torch.tensor(np.r_[np.ones(3), 10 * np.ones(6), 3, 3, 3], dtype=torch.float32); A, Q = A * w, Q * w
                else:
                    mu, sd = A.mean(0), A.std(0) + 1e-6; A, Q = (A - mu) / sd, (Q - mu) / sd
                rows[a].append(score(*answer(A, Ym[keep], Q, mt['slide']), mt, base))
        out['stores'][n] = {a: {k: round(float(np.mean([r[k] for r in rs])), 3 if 'logloss' in k or 'prob' in k else 1) for k in rs[0]} for a, rs in rows.items()}
        print(n, json.dumps(out['stores'][n], indent=1), flush=True)
    print({k: v for k, v in out.items() if k != 'stores'})
    json.dump(out, open(ROOT / 'runs/134/effects.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
