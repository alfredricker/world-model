"""Card 125.1, after the main run (diagnostic, not judged): the chain rescored with a corrected truth.
main.py counted a touch as a press if the button's state differed between contact and 2 frames after release.
24% of flips happen 1-5 frames BEFORE contact is sensed, so those presses were not counted. Here a touch is a press
if the button flips between 5 frames before contact and 2 frames after release. Reported per slide place, with
main.py's states: the touch-level chain (criterion 3's three numbers), and the span-level chain (diag.py).

  bin/prun python tools/card125.1/diag2.py → runs/125.1/diag2.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import K, fisher_steady  # noqa: E402
from gate import DATA, EP, ROOT, SLIDE, TOP, load, otsu  # noqa: E402

M = load('m600', 600)
btn = np.load(DATA / 'scene-play-v0.npz')['button_states'][600 * EP:700 * EP]
lo = np.maximum(M['st'] - 5, (M['st'] // EP) * EP)
press2 = btn[np.minimum(M['en'] + 2, len(btn) - 1)] != btn[lo]
vm = np.log(np.load(ROOT / 'runs/125.1/var_m600.npy') + 1e-6)
steady = vm < otsu(vm.reshape(-1)[::7])
eps = np.median(M['tden'])
sm = np.log(np.sort(M['tnum'] / (M['tden'] + eps), 1)[:, -4:].mean(1))
ym = sm > otsu(sm)
pm = KMeans(K, n_init=5, random_state=0).fit(M['pos']).labels_
n = len(sm)
ep = M['st'] // EP
nxt = np.where(ep[1:] == ep[:-1])[0]
out = []
for c in range(K):
    im = np.where(pm == c)[0]
    slide = next((s for s in SLIDE if M[f'{s}_moved'][im].mean() >= 0.3), None)
    if not slide:
        continue
    bi = SLIDE[slide][1]
    top = np.sort(np.argsort(fisher_steady(np.asarray(M['before'][im], np.float32), ym[im], steady[im]))[-TOP:])
    F = np.concatenate([np.asarray(M['before'][i:i + 256][:, top], np.float32) for i in range(0, n, 256)])
    ws = steady[:, top]
    centre = (F * ws[..., None]).sum(0) / np.maximum(ws.sum(0), 1)[:, None]
    known = ws.sum(1) >= 3
    read = ((F - centre) * ws[..., None]).sum(1) / np.maximum(ws.sum(1), 1)[:, None]
    state = np.full(n, -1)
    state[known] = KMeans(2, n_init=5, random_state=0).fit_predict(read[known])
    direct = state.copy()
    for i in range(1, n):
        if state[i] < 0 and ep[i] == ep[i - 1]:
            state[i] = state[i - 1]
    press = press2[:, bi]
    both = (state[nxt] >= 0) & (state[nxt + 1] >= 0)
    setter = both & (state[nxt + 1] != state[nxt])
    places = np.bincount(pm[nxt][setter], minlength=K)
    ki = np.where(known)[0]
    spans = [(a, b) for a, b in zip(ki[:-1], ki[1:]) if ep[a] == ep[b] and direct[a] != direct[b]]
    inside, between = np.zeros(n, bool), np.zeros(n, bool)
    for a, b in spans:
        inside[a:b] = True
    for a, b in zip(ki[:-1], ki[1:]):
        if ep[a] == ep[b]:
            between[a:b] = True
    # presses that change the state seen at the next direct reading: an odd number of flips inside the interval
    odd = np.zeros(n, bool)
    for a, b in zip(ki[:-1], ki[1:]):
        if ep[a] == ep[b] and press[a:b].sum() % 2 == 1:
            odd[a:b] = True
    out.append(dict(place=c, slide=slide,
                    touch_level=dict(setters=int(setter.sum()), are_presses=round(float(press[nxt][setter].mean()), 3),
                                     presses_caught=round(float(setter[press[nxt] & both].mean()), 3),
                                     in_one_place=round(float(places.max() / setter.sum()), 3)),
                    span_level=dict(spans=len(spans), with_a_press=round(float(np.mean([press[a:b].any() for a, b in spans])), 3),
                                    touches_per_span=round(float(np.mean([b - a for a, b in spans])), 2),
                                    all_touches_in_one_place=round(float(np.mean([len(set(pm[a:b])) == 1 for a, b in spans])), 3),
                                    presses=int((press & between).sum()),
                                    presses_inside_a_span=round(float(inside[press & between].mean()), 3),
                                    presses_with_a_net_change=int((press & odd).sum()),
                                    of_those_inside_a_span=round(float(inside[press & odd].mean()), 3))))
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / 'runs/125.1/diag2.json', 'w'), indent=1)
