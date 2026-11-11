"""Card 127: what the planner knows, read from the memory of touches (episodes 400-499) with no truth.

Per place (k-means over gripper positions, card 125):
- effect patches: what touches there change and touches elsewhere do not. Per touch, per patch, the squared
  distance between the steady reading before it and the steady reading before the next touch of its episode;
  per place, the mean over its touches divided by the mean over all touches. The TOP highest are the place's
  variable.
  A place has a variable if that ratio, over its TOP patches, is at least 5.
- condition (cards 125.1, 125.2): candidate patches by contrast of the before-features between outcomes, their two
  states (2-means over steady readings), which state lets a touch here change something, and the cause place.
  A place has a condition if the contrast is at least 5 times the shuffled level, the cause place is another
  place and at least 80% of the changed intervals hold a touch there.
The truth (object masks) is used only to print where the patches lie.

  bin/prun python tools/card127/know.py → runs/127/know.npz, runs/127/know.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card125'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card125.1'))
from main import K, fisher_steady  # noqa: E402
from gate import EP, OBJECTS, ROOT, TOP, load, otsu  # noqa: E402

MEM, FIRST = 'm400', 400


def main():
    rng = np.random.default_rng(0)
    M = load(MEM, FIRST)
    st = M['st']
    n = len(st)
    ep = st // EP
    vm = np.log(np.load(ROOT / f'runs/125.1/var_{MEM}.npy') + 1e-6)
    steady = vm < otsu(vm.reshape(-1)[::7])
    eps = np.median(M['tden'])
    sm = np.log(np.sort(M['tnum'] / (M['tden'] + eps), 1)[:, -4:].mean(1))
    ym = sm > otsu(sm)
    km = KMeans(K, n_init=5, random_state=0).fit(M['pos'])
    pm = km.labels_
    nxt = np.where(np.r_[ep[1:] == ep[:-1], False])[0]                     # touches with a next touch in the episode
    # lasting change between steady readings
    D = np.zeros((n, 1024), np.float32)
    W = np.zeros((n, 1024), bool)
    for i in range(0, n - 1, 128):
        a = np.asarray(M['before'][i:i + 129], np.float32)
        m = len(a) - 1
        D[i:i + m] = ((a[1:] - a[:-1]) ** 2).sum(-1)
        W[i:i + m] = steady[i:i + m] & steady[i + 1:i + m + 1]
    W[np.setdiff1d(np.arange(n), nxt)] = False
    base = (D * W).sum(0) / np.maximum(W.sum(0), 1)
    mk_all = M['masks'].mean(0) > 0.5
    on = lambda idx, mk: {o: int(mk[oi][idx].sum()) for oi, o in enumerate(OBJECTS) if mk[oi][idx].sum()}
    out, rep = {}, []
    for c in range(K):
        im = np.where(pm == c)[0]
        w = W[im]
        ok = w.sum(0) >= max(10, 0.3 * len(im))
        ratio = np.where(ok, (D[im] * w).sum(0) / np.maximum(w.sum(0), 1) / (base + 1e-3), 0)
        eff = np.sort(np.argsort(ratio)[-TOP:])
        e = dict(place=c, touches=len(im), changed=round(float(ym[im].mean()), 2),
                 centre=km.cluster_centers_[c].round(2).tolist(),
                 effect_ratio=round(float(ratio[eff].mean()), 1), effect_on=on(eff, mk_all))
        if e['effect_ratio'] >= 5:
            out[f'eff{c}'] = eff
            out[f'ratio{c}'] = ratio.astype(np.float32)
        y = ym[im]
        if min(y.sum(), (~y).sum()) >= 15:
            X = np.asarray(M['before'][im], np.float32)
            f = fisher_steady(X, y, steady[im])
            top = np.sort(np.argsort(f)[-TOP:])
            sh = np.mean([np.sort(fisher_steady(X, rng.permutation(y), steady[im]))[-TOP:].mean() for _ in range(3)])
            del X
            F = np.concatenate([np.asarray(M['before'][i:i + 256][:, top], np.float32) for i in range(0, n, 256)])
            ws = steady[:, top]
            centre = (F * ws[..., None]).sum(0) / np.maximum(ws.sum(0), 1)[:, None]
            known = ws.sum(1) >= 3
            read = ((F - centre) * ws[..., None]).sum(1) / np.maximum(ws.sum(1), 1)[:, None]
            k2 = KMeans(2, n_init=5, random_state=0).fit(read[known])
            state = np.full(n, -1)
            state[known] = k2.labels_
            ki = np.where(known)[0]
            iv = [(a, b) for a, b in zip(ki[:-1], ki[1:]) if ep[a] == ep[b]]
            said = np.array([state[a] != state[b] for a, b in iv])
            holds = np.stack([[bool((pm[a:b] == p).any()) for a, b in iv] for p in range(K)])
            cause = int(holds[:, said].sum(1).argmax())
            here = im[known[im]]
            rate = [float(ym[here][state[here] == s].mean()) for s in (0, 1)]
            cond = dict(contrast=round(float(f[top].mean() / sh), 1), on=on(top, mk_all), cause_place=cause,
                        changed_with_cause=round(float(holds[cause][said].mean()), 2),
                        unchanged_with_cause=round(float(holds[cause][~said].mean()), 2),
                        changed_rate_by_state=[round(r, 2) for r in rate])
            cond['valid'] = bool(cond['contrast'] >= 5 and cause != c and cond['changed_with_cause'] >= 0.8)
            e['condition'] = cond
            if cond['valid']:
                # presses: single touches at the cause place across which the state changed
                one = [(a, b) for (a, b), s in zip(iv, said) if s and (pm[a:b] == cause).sum() == 1]
                press = np.array([a + int(np.argmax(pm[a:b] == cause)) for a, b in one])
                pfrom = np.array([state[a] for a, b in one])
                out.update({f'cond{c}': top, f'centre{c}': centre, f'states{c}': k2.cluster_centers_,
                            f'enabled{c}': int(np.argmax(rate)), f'cause{c}': cause, f'press{c}': press,
                            f'pressfrom{c}': pfrom})
                e['condition']['presses'] = len(press)
        rep.append(e)
        print(json.dumps(e), flush=True)
    # per touch: readings usable for recall (the next touch's before is its after)
    out.update(place=pm, changed=ym, has_next=np.isin(np.arange(n), nxt), steady=steady, st=st, en=M['en'])
    np.savez(ROOT / 'runs/127/know.npz', **out)
    json.dump(rep, open(ROOT / 'runs/127/know.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
