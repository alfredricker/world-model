"""Card 125, after the main run (a diagnostic, not judged): the chain read as a change of state.
For each slide place, its 10 candidate patches' before-features over all the memory's touches are split into two
states by 2-means (no labels). A touch sets the condition if the state before the NEXT touch of the same episode
differs from the state before it. Reported: the share of such touches that are presses of the locking button, the
share of those presses caught, the share lying in one place, and how well the two states match the true lock.

  bin/prun python tools/card125/chain_state.py → runs/125/chain_state.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import EP, ROOT, SLIDE, TOP, fisher, load, otsu  # noqa: E402

M = load('m800', 800)
eps = np.median(M['tden'])
z = M['tnum'] / (M['tden'] + eps)
sm = np.log(np.sort(z, 1)[:, -4:].mean(1))
ym = sm > otsu(sm)
pm = KMeans(10, n_init=5, random_state=0).fit(M['pos']).labels_
nxt = np.where(M['st'][1:] // EP == M['st'][:-1] // EP)[0]            # touch i has a next touch i+1 in its episode
out = []
for c in range(10):
    im = np.where(pm == c)[0]
    slide = next((s for s in SLIDE if M[f'{s}_moved'][im].mean() >= 0.3), None)
    if not slide:
        continue
    bi = SLIDE[slide][1]
    top = np.sort(np.argsort(fisher(np.asarray(M['before'][im], np.float32), ym[im]))[-TOP:])
    F = np.stack([np.asarray(M['before'][i:i + 256][:, top], np.float32).reshape(-1, TOP * 768)
                  for i in range(0, len(sm), 256)]) if False else \
        np.concatenate([np.asarray(M['before'][i:i + 256][:, top], np.float32).reshape(-1, TOP * 768)
                        for i in range(0, len(sm), 256)])
    state = KMeans(2, n_init=5, random_state=0).fit_predict(F)
    locked = M[f'{slide}_locked']
    agree = max((state == locked).mean(), (state != locked).mean())
    setter = state[nxt + 1] != state[nxt]
    press = M['flipb'][nxt, bi]
    places = np.bincount(pm[nxt][setter], minlength=10)
    out.append(dict(place=c, slide=slide, states_match_true_lock=round(float(agree), 3), setters=int(setter.sum()),
                    are_presses=round(float(press[setter].mean()), 3), presses=int(press.sum()),
                    presses_caught=round(float(setter[press].mean()), 3),
                    in_one_place=round(float(places.max() / max(setter.sum(), 1)), 3), modal_place=int(places.argmax())))
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / 'runs/125/chain_state.json', 'w'), indent=1)
