"""Card 125, feasibility gate: does contrast over remembered touches find what decides a touch's outcome?

Memory: the touches of episodes 900-999 (tail). Test: the touches of the validation episodes.
The agent's view of a touch, no labels:
- place: the gripper's position at contact (proprioception, observations 12:15); k-means over the memory's touches;
- outcome: card 122's lasting-change score for the touch (block 2, mean of the top 4 patches), split into
  "something changed" / "nothing changed" by Otsu's threshold on its log over the memory;
- the scene before: blocks 2 and 12 per patch (memory.py).
Contrast, within one place: per patch, the squared difference of the before-features' class means over their
pooled variance, summed over the 768 numbers (a Fisher score). The 10 highest patches are the candidate condition.
Read-outs:
- where the candidate patches lie (truth masks: button, slide), against the same contrast with outcomes shuffled;
- prediction: nearest class mean on the candidate patches, fitted on the memory, tested on validation touches of
  the same place, against the truth "did the slide move"; baselines: all patches (no selection), the gripper's
  position alone; upper bound: the true lock.

  bin/prun python tools/card125/gate.py [--k 10] → runs/125/gate.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
EP, TOP = 1001, 10
OBJECTS = ['arm', 'cube', 'button0', 'button1', 'drawer', 'window']
SLIDE = {'drawer': (23, 0, 4, 2), 'window': (24, 1, 5, 3)}      # qpos index, button, slide's mask, button's mask
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def otsu(x):
    xs = np.sort(x)
    best, cut = -1, xs[0]
    for c in np.quantile(xs, np.linspace(0.02, 0.98, 97)):
        a, b = x[x <= c], x[x > c]
        if len(a) and len(b):
            v = len(a) * len(b) * (a.mean() - b.mean()) ** 2
            if v > best:
                best, cut = v, c
    return cut


def load(name, first=900):
    z = np.load(DATA / ('scene-play-v0-val.npz' if name == 'val' else 'scene-play-v0.npz'))
    cut = 0 if name == 'val' else first * EP
    sl = slice(cut, cut + 100 * EP)
    c = np.load(ROOT / f'runs/122/computed_{name}.npz')
    st, en = c['st'], c['en']
    qpos, btn, obs = z['qpos'][sl], z['button_states'][sl], z['observations'][sl]
    d = dict(st=st, en=en, pos=obs[st + 2, 12:15], tnum=c['tnum'], tden=c['tden'],
             masks=c['masks'][st - 2].astype(bool), btn=btn[st],
             before=np.load(ROOT / f'runs/125/before_{name}.npy', mmap_mode='r'))
    for s, (qi, bi, _, _) in SLIDE.items():
        d[f'{s}_moved'] = np.abs(qpos[en, qi] - qpos[st, qi]) > 0.01
        d[f'{s}_locked'] = btn[st, bi] == 0
    d['flipb'] = btn[np.minimum(en + 2, len(btn) - 1)] != btn[st]
    d['flip'] = d['flipb'].any(1)
    return d


def fisher(X, y):
    """X [n, P, F] float32, y bool → per-patch score [P]."""
    a, b = X[y], X[~y]
    num = (a.mean(0) - b.mean(0)) ** 2
    den = (a.var(0) * len(a) + b.var(0) * len(b)) / len(X) + 1e-4
    return (num / den).sum(-1)


def ncm(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    a, b = ((Xtr[ytr] - mu) / sd).mean(0), ((Xtr[~ytr] - mu) / sd).mean(0)
    z = (Xte - mu) / sd
    return ((z - a) ** 2).sum(1) < ((z - b) ** 2).sum(1)


def bal(pred, true):
    if true.all() or (~true).all():
        return None
    return round(float((pred[true].mean() + (~pred[~true]).mean()) / 2), 3)


def main():
    k = int(arg('--k', 10))
    rng = np.random.default_rng(0)
    M, V = load('tail'), load('val')
    eps = np.median(M['tden'])
    score = lambda d: np.log(np.sort(d['tnum'] / (d['tden'] + eps), 1)[:, -4:].mean(1))
    sm, sv = score(M), score(V)
    cut = otsu(sm)
    ym, yv = sm > cut, sv > cut
    km = KMeans(k, n_init=5, random_state=0).fit(M['pos'])
    pm, pv = km.labels_, km.predict(V['pos'])
    res = dict(touches=dict(memory=len(sm), val=len(sv)), changed_share=dict(memory=round(float(ym.mean()), 3),
               val=round(float(yv.mean()), 3)), places=[])
    for c in range(k):
        im, iv = np.where(pm == c)[0], np.where(pv == c)[0]
        e = dict(place=c, memory=len(im), val=len(iv), changed=round(float(ym[im].mean()), 3),
                 truth={s: dict(moved=round(float(M[f'{s}_moved'][im].mean()), 3),
                                locked=round(float(M[f'{s}_locked'][im].mean()), 3)) for s in SLIDE},
                 flip=round(float(M['flip'][im].mean()), 3))
        n1, n0 = int(ym[im].sum()), int((~ym[im]).sum())
        if min(n1, n0) >= 15 and len(iv) >= 20:
            X = np.asarray(M['before'][im], np.float32)
            y = ym[im]
            f = fisher(X, y)
            top = np.argsort(f)[-TOP:]
            fs = [fisher(X, rng.permutation(y)) for _ in range(5)]
            mk = M['masks'][im].mean(0) > 0.5                              # [6, P] usual place of each object
            on = lambda idx: {o: round(float(mk[oi][idx].mean()), 2) for oi, o in enumerate(OBJECTS)}
            e['contrast'] = dict(top_score=round(float(f[top].mean()), 2),
                                 shuffled_top_score=round(float(np.mean([np.sort(x)[-TOP:].mean() for x in fs])), 2),
                                 top_patches_on=on(top), shuffled_on=on(np.argsort(fs[0])[-TOP:]),
                                 patch_share={o: round(float(mk[oi].mean()), 3) for oi, o in enumerate(OBJECTS)})
            Xv = np.asarray(V['before'][iv], np.float32)
            sel = lambda A, idx: A[:, idx].reshape(len(A), -1)
            pred = dict(candidate=ncm(sel(X, top), y, sel(Xv, top)),
                        all_patches=ncm(X.mean(1), y, Xv.mean(1)),
                        position=ncm(M['pos'][im], y, V['pos'][iv]))
            e['predict_outcome_val'] = {n: bal(p, yv[iv]) for n, p in pred.items()}
            for s in SLIDE:
                t = V[f'{s}_moved'][iv]
                if t.mean() > 0.1:
                    e[f'predict_{s}_moved_val'] = {n: bal(p, t) for n, p in pred.items()}
                    e[f'predict_{s}_moved_val']['true_lock'] = bal(~V[f'{s}_locked'][iv], t)
                    e[f'predict_{s}_moved_val']['outcome_label_itself'] = bal(yv[iv], t)
        res['places'].append(e)
    print(json.dumps(res, indent=1))
    json.dump(res, open(ROOT / 'runs/125/gate.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
