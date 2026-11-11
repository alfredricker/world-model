"""Card 122: strict scoring of cards 119, 120 and 121's signals, the same way on any patch grid.

  --dir runs/122 --grid 32   the 256-pixel run (compute.py)
  --dir runs/120 --grid 14   card 120's 64-pixel run, for a like-for-like comparison (no extent samples there)

Strict measures (no slack). For a transition inside a touch, the object's patches are those it covers just before
or just after the touch; for a lock flip, 'button' is the button itself and 'handle' the drawer's or window's body.
- touch: rank of the object's best patch in card 120's signal → top patch exact (rank 0), found (top 1% of patches);
- the same with the arm's patches removed (upper bound) and with the signal shuffled over patches (chance);
- fixed: card 119's signal at its best frame inside the transition (object's patches at that frame and 5 before);
- card 119's thresholded transitions F1, a boundary right if its top patch lies on the object (no slack);
- extent (grid 32 only): card 121's gate, grown from a true seed, blocks 2 and 12.

  bin/prun python tools/card122/score.py --dir runs/122 --grid 32 → <dir>/score122.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card115'))
from masks import OBJECTS  # noqa: E402
from score import frame_score, peaks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EP, SKIP, TOL, CLEAR = 1001, 5, 5, 2
LO, HI = 15, EP - 16
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
BTN = [OBJECTS.index('button0'), OBJECTS.index('button1')]
BODY = [OBJECTS.index('drawer'), OBJECTS.index('window')]


def targets(S, j):
    """→ {name: object-index mask [6]} for transition j."""
    o = S['TO'][j]
    if S['TK'][j] == 'lock':
        b, h = o.copy(), o.copy()
        b[BODY] = False; h[BTN] = False
        return {'lock: button': b, 'lock: handle': h}
    return {str(S['TK'][j]): o}


def summarise(ranks, k):
    out = {}
    for name, d in ranks.items():
        v = np.array(list(d.values()))
        out[name] = dict(n=int(len(v)), top_patch_exact=round(float((v == 0).mean()), 4),
                         found_top1pct=round(float((v < k).mean()), 4), median_rank=float(np.median(v)))
    return out


def otsu(v):
    c = np.sort(v)
    n = len(c)
    cs = np.cumsum(c)
    i = np.arange(1, n)
    ma, mb = cs[i - 1] / i, (cs[-1] - cs[i - 1]) / (n - i)
    b = int(np.argmax(i * (n - i) * (ma - mb) ** 2))
    return (c[b] + c[b + 1]) / 2


def grow(sim, seed, thr, g):
    lab, _ = ndimage.label((sim >= thr).reshape(g, g))
    lab = lab.reshape(-1)
    if not lab[seed]:
        out = np.zeros(g * g, bool); out[seed] = True
        return out
    return lab == lab[seed]


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def main():
    D = ROOT / arg('--dir', 'runs/122')
    g = int(arg('--grid', '32'))
    Pn = g * g
    K1 = int(np.ceil(0.01 * Pn))
    T = np.load(D / 'computed_tail.npz', allow_pickle=True)
    S = dict(np.load(D / 'computed_val.npz', allow_pickle=True))
    S['masks'] = S['masks'].astype(bool)
    res = dict(grid=g, top1pct_patches=K1, touches=int(len(S['st'])))
    arm = OBJECTS.index('arm')

    # touch-bracketed
    eps = float(np.median(T['tden']))
    z = S['tnum'] / (S['tden'] + eps)
    rng = np.random.default_rng(0)
    variants = {'touch': z, 'touch_arm_masked': np.where(S['share'] < 0.5, z, -np.inf),
                'chance': rng.permuted(z, axis=1)}
    pairs = [(i, j) for i, (s_, e_) in enumerate(zip(S['st'], S['en']))
             for j in np.where((S['A'] <= e_ + TOL) & (S['B'] >= s_ - TOL) & (S['A'] // EP == s_ // EP))[0]]
    for nm, zz in variants.items():
        rk = np.argsort(np.argsort(-zz, 1, kind='stable'), 1)
        ranks = {}
        for i, j in pairs:
            m0, m1 = S['masks'][S['st'][i] - CLEAR], S['masks'][S['en'][i] + CLEAR]
            for name, o in targets(S, j).items():
                m = m0[o].any(0) | m1[o].any(0)
                if m.any():
                    d = ranks.setdefault(name, {})
                    d[j] = min(d.get(j, Pn), int(rk[i][m].min()))
        res[nm] = summarise(ranks, K1)
        if nm == 'touch':
            top = np.argmax(zz, 1)
            at = S['masks'][S['en'] + CLEAR, arm] | S['masks'][S['st'] - CLEAR, arm]
            res['touch_top_patch_on_arm'] = round(float(at[np.arange(len(top)), top].mean()), 4)

    # card 119's fixed-window signal
    den_t = np.asarray(T['den'], np.float32)
    eps_f = float(np.nanmedian(den_t))

    def fixed(X):
        zf = np.asarray(X['num'], np.float32) / (np.asarray(X['den'], np.float32) + eps_f)
        loc = np.arange(len(zf)) % EP
        zf = np.where(((loc >= LO) & (loc <= HI))[:, None], zf, np.nan)
        sc = frame_score(np.where(np.isnan(zf), -np.inf, zf))
        return zf, sc

    zv, scv = fixed(S)
    ranks = {}
    for j in sorted(set(j for _, j in pairs)):
        lo, hi = S['A'][j] - TOL, S['B'][j] + TOL
        seg = scv[lo:hi + 1]
        if not np.isfinite(seg).any():
            continue
        t = lo + int(np.argmax(seg))
        rk = np.argsort(np.argsort(-np.nan_to_num(zv[t], nan=-np.inf), kind='stable'))
        for name, o in targets(S, j).items():
            m = S['masks'][t][o].any(0) | S['masks'][t - SKIP][o].any(0)
            if m.any():
                ranks.setdefault(name, {})[j] = int(rk[m].min())
    res['fixed_best_frame'] = summarise(ranks, K1)

    # thresholded transitions F1, strict place
    def trans_f1(X, sc, zf, thr):
        pk = peaks(sc)
        bd = pk[sc[pk] > thr]
        top = np.argmax(np.nan_to_num(zf[bd], nan=-np.inf), 1)
        A, B, masks = X['A'], X['B'], X['masks']
        found, right = np.zeros(len(A), bool), np.zeros(len(bd), bool)
        for q, t in enumerate(bd):
            for i in np.where((A - TOL <= t) & (t <= B + TOL) & (A // EP == t // EP))[0]:
                o = X['TO'][i]
                if (masks[t][o].any(0) | masks[t - SKIP][o].any(0))[top[q]]:
                    found[i] = right[q] = True
        p, r = (right.mean() if len(bd) else 0.0), found.mean()
        return (2 * p * r / (p + r) if p + r else 0.0), float(p), float(r), found

    Td = dict(T)
    Td['masks'] = Td['masks'].astype(bool)
    zt, sct = fixed(Td)
    pkt = peaks(sct)
    cands = np.quantile(sct[pkt], np.linspace(0, 0.995, 40))
    thr = max((trans_f1(Td, sct, zt, c)[0], c) for c in cands)[1]
    F, Pp, R, found = trans_f1(S, scv, zv, thr)
    res['fixed_thresholded_transitions'] = dict(
        f1=round(F, 4), precision=round(Pp, 4), recall=round(R, 4),
        recall_by_kind={str(k): round(float(found[S['TK'] == k].mean()), 4) for k in np.unique(S['TK'])})

    # extent from a true seed (card 121's gate)
    if 's12' in S:
        ext = {}
        for blk, key in [(2, 's2'), (12, 's12')]:
            acc = {k: {o: [] for o in OBJECTS} for k in ['grown', 'oracle_threshold', 'kmeans', 'seed_alone']}
            rs = np.random.default_rng(0)
            for n, fr in enumerate(S['sidx']):
                Fm = S[key][n].astype(np.float32)
                f = Fm - Fm.mean(0)
                f /= np.linalg.norm(f, axis=1, keepdims=True)
                kl = KMeans(8, n_init=2, random_state=0).fit_predict(Fm)
                for oi, o in enumerate(OBJECTS):
                    m = S['masks'][fr, oi]
                    if not m.any():
                        continue
                    sd = int(rs.choice(np.where(m)[0]))
                    sim = f @ f[sd]
                    acc['grown'][o].append(iou(grow(sim, sd, otsu(sim), g), m))
                    acc['oracle_threshold'][o].append(max(iou(grow(sim, sd, t, g), m)
                                                          for t in np.linspace(-0.2, 0.95, 24)))
                    acc['kmeans'][o].append(max(iou(kl == c, m) for c in range(8)))
                    acc['seed_alone'][o].append(1 / m.sum())
            ext[f'block{blk}'] = {k: {o: round(float(np.mean(v)), 3) for o, v in d.items()} for k, d in acc.items()}
        res['extent_true_seed'] = ext
    print(json.dumps(res, indent=1))
    json.dump(res, open(D / 'score122.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
