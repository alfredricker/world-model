"""Card 125.1: card 125's contrast and chain, with a condition read only where its patches are steady.

Memory: training episodes 600-699 (set m600). Test: 500-599 (set t500). Neither was used before. Places, outcomes
and settings as in card 125 (10 places, 10 candidate patches). New:
- steady: a patch is steady before a touch if its variance over the 10-frame before window (steady.py) is under
  Otsu's threshold on the log variance over all the memory's touches and patches.
- contrast: each patch's class means and variance use only the touches where it is steady; a patch is a candidate
  only if it is steady in at least half of the place's touches and in at least 10 touches of each outcome.
- state: for a slide place's candidate patches, each touch's reading is the mean over its steady candidate patches
  of the before-features, each patch centred on its mean over memory. Readings are split in two by 2-means. A
  touch with fewer than 3 steady candidate patches has no reading and keeps the state of the previous touch of
  its episode (none if there is no earlier reading).
- chain: a touch sets the condition if the state before the next touch of its episode differs from the state
  before it (both known).
Written before the run:
1. Card 125's criteria 1 and 2 hold: slide places (a slide moves in >= 30% of touches): >= 6 of 10 candidate
   patches on the locking button, contrast >= 5 times shuffled; quiet places (< 5% moves, < 5% flips): under 5
   times; candidate patches predict "the slide moved" on test touches within 0.03 of the true lock.
2. The arm is not a condition: at button places (a button flips in >= 30% of touches), at most 1 of the 10
   candidate patches lies on the arm's usual patches there. (Card 125: 8 of 10 at one such place.)
3. Chain, per slide place: >= 80% of the setting touches are presses of the locking button; >= 80% of that
   button's presses (those with a known state before and after) are caught; >= 80% of the setting touches lie in
   one place.

  bin/prun python tools/card125.1/main.py [--mem m600 --first 600 --test t500 --tfirst 500] → runs/125.1/main.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card125'))
from gate import EP, OBJECTS, ROOT, SLIDE, TOP, bal, load, ncm, otsu  # noqa: E402

K = 10
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def fisher_steady(X, y, w):
    """X [n, P, F], y [n] bool, w [n, P] bool (steady) → score [P], -inf where the patch cannot be a candidate."""
    out = np.full(X.shape[1], -np.inf, np.float32)
    wa, wb = w & y[:, None], w & ~y[:, None]
    na, nb = wa.sum(0), wb.sum(0)
    ok = (w.mean(0) >= 0.5) & (na >= 10) & (nb >= 10)
    for p in np.where(ok)[0]:
        a, b = X[wa[:, p], p], X[wb[:, p], p]
        den = (a.var(0) * len(a) + b.var(0) * len(b)) / (len(a) + len(b)) + 1e-4
        out[p] = (((a.mean(0) - b.mean(0)) ** 2) / den).sum()
    return out


def main():
    mem, first = arg('--mem', 'm600'), int(arg('--first', 600))
    test, tfirst = arg('--test', 't500'), int(arg('--tfirst', 500))
    rng = np.random.default_rng(0)
    M, V = load(mem, first), load(test, tfirst)
    vm = np.load(ROOT / f'runs/125.1/var_{mem}.npy')
    cutv = otsu(np.log(vm.reshape(-1)[::7] + 1e-6))
    steady = np.log(vm + 1e-6) < cutv                                         # [touches, P]
    eps = np.median(M['tden'])
    score = lambda d: np.log(np.sort(d['tnum'] / (d['tden'] + eps), 1)[:, -4:].mean(1))
    sm, sv = score(M), score(V)
    cut = otsu(sm)
    ym, yv = sm > cut, sv > cut
    km = KMeans(K, n_init=5, random_state=0).fit(M['pos'])
    pm, pv = km.labels_, km.predict(V['pos'])
    n = len(sm)
    nxt = np.where(M['st'][1:] // EP == M['st'][:-1] // EP)[0]
    res = dict(touches=dict(memory=n, test=len(sv)), steady_share=round(float(steady.mean()), 3), places=[])
    ok1, ok2, ok3 = [], [], []
    for c in range(K):
        im, iv = np.where(pm == c)[0], np.where(pv == c)[0]
        moved = {s: float(M[f'{s}_moved'][im].mean()) for s in SLIDE}
        flip = float(M['flip'][im].mean())
        slide = next((s for s in SLIDE if moved[s] >= 0.3), None)
        kind = slide or ('button' if flip >= 0.3 else 'quiet' if max(moved.values()) < 0.05 and flip < 0.05 else 'other')
        e = dict(place=c, kind=kind, memory=len(im), test=len(iv), changed=round(float(ym[im].mean()), 3),
                 moved={s: round(v, 3) for s, v in moved.items()}, flip=round(flip, 3))
        y = ym[im]
        if min(int(y.sum()), int((~y).sum())) >= 15 and len(iv) >= 20:
            X = np.asarray(M['before'][im], np.float32)
            w = steady[im]
            f = fisher_steady(X, y, w)
            top = np.sort(np.argsort(f)[-TOP:])
            sh = float(np.mean([np.sort(fisher_steady(X, rng.permutation(y), w))[-TOP:].mean() for _ in range(3)]))
            ratio = float(f[top].mean() / sh)
            mk = M['masks'][im].mean(0) > 0.5
            e['contrast_ratio'] = round(ratio, 1)
            e['candidates_on'] = {o: int(mk[oi][top].sum()) for oi, o in enumerate(OBJECTS)}
            e['candidate_patches_allowed'] = int(np.isfinite(f).sum())
            if kind == 'quiet':
                ok1.append(ratio < 5)
            if kind == 'button':
                ok2.append(bool(mk[0][top].sum() <= 1))
            if slide:
                qi, bi, _, bmask = SLIDE[slide]
                Xv = np.asarray(V['before'][iv], np.float32)
                sel = lambda A, idx: A[:, idx].reshape(len(A), -1)
                t = V[f'{slide}_moved'][iv]
                p = dict(candidate=bal(ncm(sel(X, top), y, sel(Xv, top)), t),
                         true_lock=bal(~V[f'{slide}_locked'][iv], t))
                e['predict_moved_test'] = p
                ok1.append(bool(mk[bmask][top].sum() >= 6 and ratio >= 5 and p['candidate'] >= p['true_lock'] - 0.03))
                # the state of the condition before every touch of the memory
                F = np.concatenate([np.asarray(M['before'][i:i + 256][:, top], np.float32) for i in range(0, n, 256)])
                ws = steady[:, top]                                             # [n, TOP]
                centre = (F * ws[..., None]).sum(0) / np.maximum(ws.sum(0), 1)[:, None]
                known = ws.sum(1) >= 3
                read = ((F - centre) * ws[..., None]).sum(1) / np.maximum(ws.sum(1), 1)[:, None]
                lab = KMeans(2, n_init=5, random_state=0).fit_predict(read[known])
                state = np.full(n, -1)
                state[known] = lab
                for i in range(1, n):                                           # carry over touches with no reading
                    if state[i] < 0 and M['st'][i] // EP == M['st'][i - 1] // EP:
                        state[i] = state[i - 1]
                locked = M[f'{slide}_locked']
                kk = state >= 0
                agree = max((state[kk] == locked[kk]).mean(), (state[kk] != locked[kk]).mean())
                both = (state[nxt] >= 0) & (state[nxt + 1] >= 0)
                setter = both & (state[nxt + 1] != state[nxt])
                press = M['flipb'][nxt, bi]
                places = np.bincount(pm[nxt][setter], minlength=K)
                ch = dict(states_match_true_lock=round(float(agree), 3), readings_known=round(float(known.mean()), 3),
                          setters=int(setter.sum()), are_presses=round(float(press[setter].mean()), 3),
                          presses=int((press & both).sum()),
                          presses_caught=round(float(setter[press & both].mean()), 3),
                          in_one_place=round(float(places.max() / max(setter.sum(), 1)), 3),
                          modal_place=int(places.argmax()))
                e['chain'] = ch
                ok3.append(bool(ch['are_presses'] >= 0.8 and ch['presses_caught'] >= 0.8 and ch['in_one_place'] >= 0.8))
        elif slide:
            ok1.append(False); ok3.append(False)
        res['places'].append(e)
    res['criteria'] = dict(contrast_holds=bool(ok1 and all(ok1)), arm_not_a_condition=bool(ok2 and all(ok2)),
                           chain=bool(ok3 and all(ok3)), each=dict(c1=ok1, c2=ok2, c3=ok3))
    print(json.dumps(res, indent=1))
    json.dump(res, open(ROOT / f'runs/125.1/main_{mem}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
