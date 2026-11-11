"""Card 124, gate step 1: does "an event changes few tokens" prefer the true objects over other groupings?

From card 122's before/after-touch signal (block 2, 32 x 32 patches, validation touches). For a grouping of the
1,024 patches into tokens, a token's change is the summed signal of its patches, and an event's effective number
of changed tokens is (sum e)^2 / sum e^2 (1 when one token holds all the change). Lower is sparser. Groupings:
- true objects (arm, cube, 2 buttons, drawer, window as covered before or after the touch; the rest background), 7;
- k-means on the mean block-12 patch features of 400 sampled frames, 7 and 12 clusters (a fixed partition);
- a regular grid of 3 x 3 blocks, 9; random partitions of 7 equal parts.
Also the share of the change held by the single largest token, and the same with the background token left out.

  bin/prun python tools/card124/sparsity.py → runs/124/sparsity.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card115'))
from masks import OBJECTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
G, CLEAR = 32, 2


def eff(e):
    e = np.asarray(e, np.float64)
    return float(e.sum() ** 2 / (e ** 2).sum()), float(e.max() / e.sum())


def main():
    T = np.load(ROOT / 'runs/122/computed_tail.npz', allow_pickle=True)
    S = np.load(ROOT / 'runs/122/computed_val.npz', allow_pickle=True)
    masks = S['masks'].astype(bool)
    z = S['tnum'] / (S['tden'] + float(np.median(T['tden'])))
    f12 = S['s12'].astype(np.float32).mean(0)
    rng = np.random.default_rng(0)
    fixed = {'k-means 7': KMeans(7, n_init=5, random_state=0).fit_predict(f12),
             'k-means 12': KMeans(12, n_init=5, random_state=0).fit_predict(f12),
             'grid 3 x 3': ((np.arange(G)[:, None] * 3 // G) * 3 + (np.arange(G)[None, :] * 3 // G)).reshape(-1),
             'random 7': rng.permutation(np.arange(G * G) % 7)}
    acc = {k: [] for k in list(fixed) + ['true objects', 'true objects, background left out']}
    order = [OBJECTS.index(o) for o in ['cube', 'button0', 'button1', 'drawer', 'window', 'arm']]
    for i, (s, e) in enumerate(zip(S['st'], S['en'])):
        for k, lab in fixed.items():
            acc[k].append(eff(np.bincount(lab, weights=z[i])))
        m = masks[s - CLEAR] | masks[e + CLEAR]
        lab = np.full(G * G, 6)
        for oi in reversed(order):                      # small objects win where masks overlap
            lab[m[oi]] = oi
        en = np.bincount(lab, weights=z[i], minlength=7)
        acc['true objects'].append(eff(en))
        acc['true objects, background left out'].append(eff(en[:6] + 1e-12))
    res = {k: dict(effective_tokens_changed=round(float(np.mean([a for a, _ in v])), 2),
                   share_in_largest_token=round(float(np.mean([b for _, b in v])), 3)) for k, v in acc.items()}
    print(json.dumps(res, indent=1))
    json.dump(res, open(ROOT / 'runs/124/sparsity.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
