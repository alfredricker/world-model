"""Card 125, main run: the gate's procedure on episodes not seen before, and the chain.

Memory: training episodes 800-899 (set m800). Test: 700-799 (set t700). Settings as in gate.py (10 places, 10
candidate patches). Truth is used only for scoring. Written before the run:
1. Found. Slide places: places where a slide moves in >= 30% of the memory's touches. Pass: in each, >= 6 of the 10
   candidate patches on the button that locks that slide (its usual patches), and the contrast >= 5 times the
   shuffled level (mean of 5 shuffles). Quiet places (each slide moves in < 5% and a button flips in < 5%, and the
   contrast could be computed): the contrast under 5 times.
2. Predicts. In each slide place, nearest class mean on the candidate patches, fitted on memory, predicts "the
   slide moved" on test touches with balanced accuracy within 0.03 of the true lock's.
3. Chain. For each slide place: the memory's touches (any place) whose change map has >= 3 of the 10 candidate
   patches among its 10 highest patches (top 1%). Pass: >= 80% of them are presses of the locking button (it
   flips across the touch), and >= 80% lie in one place. Reported: the share of that button's presses caught.

  bin/prun python tools/card125/main.py → runs/125/main.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import OBJECTS, ROOT, SLIDE, TOP, bal, fisher, load, ncm, otsu  # noqa: E402

K = 10


def main():
    rng = np.random.default_rng(0)
    M, V = load('m800', 800), load('t700', 700)
    eps = np.median(M['tden'])
    zmap = lambda d: d['tnum'] / (d['tden'] + eps)
    score = lambda d: np.log(np.sort(zmap(d), 1)[:, -4:].mean(1))
    sm, sv = score(M), score(V)
    cut = otsu(sm)
    ym, yv = sm > cut, sv > cut
    km = KMeans(K, n_init=5, random_state=0).fit(M['pos'])
    pm, pv = km.labels_, km.predict(V['pos'])
    top10 = np.argsort(zmap(M), 1)[:, -10:]                                   # each memory touch's changed patches
    res = dict(touches=dict(memory=len(sm), test=len(sv)), changed_share=round(float(ym.mean()), 3), places=[])
    ok1, ok2, ok3 = [], [], []
    for c in range(K):
        im, iv = np.where(pm == c)[0], np.where(pv == c)[0]
        moved = {s: float(M[f'{s}_moved'][im].mean()) for s in SLIDE}
        flip = float(M['flip'][im].mean())
        e = dict(place=c, memory=len(im), test=len(iv), changed=round(float(ym[im].mean()), 3),
                 moved={s: round(v, 3) for s, v in moved.items()}, flip=round(flip, 3))
        slide = next((s for s in SLIDE if moved[s] >= 0.3), None)
        quiet = max(moved.values()) < 0.05 and flip < 0.05
        e['kind'] = slide or ('quiet' if quiet else 'other')
        y = ym[im]
        if min(int(y.sum()), int((~y).sum())) >= 15 and len(iv) >= 20:
            X = np.asarray(M['before'][im], np.float32)
            f = fisher(X, y)
            top = np.argsort(f)[-TOP:]
            sh = float(np.mean([np.sort(fisher(X, rng.permutation(y)))[-TOP:].mean() for _ in range(5)]))
            ratio = float(f[top].mean() / sh)
            mk = M['masks'][im].mean(0) > 0.5
            e['contrast_ratio'] = round(ratio, 1)
            e['candidates_on'] = {o: int(mk[oi][top].sum()) for oi, o in enumerate(OBJECTS)}
            if quiet:
                ok1.append(ratio < 5)
            if slide:
                qi, bi, _, bmask = SLIDE[slide]
                ok1.append(bool(mk[bmask][top].sum() >= 6 and ratio >= 5))
                Xv = np.asarray(V['before'][iv], np.float32)
                sel = lambda A, idx: A[:, idx].reshape(len(A), -1)
                t = V[f'{slide}_moved'][iv]
                p = dict(candidate=bal(ncm(sel(X, top), y, sel(Xv, top)), t),
                         all_patches=bal(ncm(X.mean(1), y, Xv.mean(1)), t),
                         position=bal(ncm(M['pos'][im], y, V['pos'][iv]), t),
                         true_lock=bal(~V[f'{slide}_locked'][iv], t))
                e['predict_moved_test'] = p
                ok2.append(bool(p['candidate'] >= p['true_lock'] - 0.03))
                hit = np.isin(top10, top).sum(1) >= 3
                press = M['flipb'][:, bi]
                places = np.bincount(pm[hit], minlength=K)
                ch = dict(setters=int(hit.sum()),
                          are_presses=round(float(press[hit].mean()), 3) if hit.any() else None,
                          in_one_place=round(float(places.max() / max(hit.sum(), 1)), 3),
                          modal_place=int(places.argmax()),
                          presses_caught=round(float(hit[press].mean()), 3), presses=int(press.sum()),
                          other_button_presses_among_setters=round(float(M['flipb'][hit, 1 - bi].mean()), 3) if hit.any() else None)
                e['chain'] = ch
                ok3.append(bool(hit.any() and ch['are_presses'] >= 0.8 and ch['in_one_place'] >= 0.8))
        elif slide:
            ok1.append(False); ok2.append(False); ok3.append(False)
        res['places'].append(e)
    res['criteria'] = dict(found=bool(ok1 and all(ok1)), predicts=bool(ok2 and all(ok2)), chain=bool(ok3 and all(ok3)),
                           found_each=ok1, predicts_each=ok2, chain_each=ok3)
    print(json.dumps(res, indent=1))
    json.dump(res, open(ROOT / 'runs/125/main.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
