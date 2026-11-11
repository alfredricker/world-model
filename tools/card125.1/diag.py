"""Card 125.1, after the main run (diagnostics, not judged): where do the chain's errors come from?
For each slide place of main.py: (a) how often the state matches the true lock, for direct readings and carried
ones, and for touches at the modal (button) place against elsewhere; (b) the chain scored by spans: between two
consecutive direct readings of one episode that differ, the touches from the first up to before the second form
a span; is a press of the locking button inside it, and is each press inside such a span; (c) the same with a
stricter steady rule (the lower half of the steady variances).

  bin/prun python tools/card125.1/diag.py → runs/125.1/diag.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import K, fisher_steady  # noqa: E402
from gate import EP, ROOT, SLIDE, TOP, load, otsu  # noqa: E402

M = load('m600', 600)
vm = np.log(np.load(ROOT / 'runs/125.1/var_m600.npy') + 1e-6)
cutv = otsu(vm.reshape(-1)[::7])
eps = np.median(M['tden'])
sm = np.log(np.sort(M['tnum'] / (M['tden'] + eps), 1)[:, -4:].mean(1))
ym = sm > otsu(sm)
pm = KMeans(K, n_init=5, random_state=0).fit(M['pos']).labels_
n = len(sm)
ep = M['st'] // EP
out = []
for c in range(K):
    im = np.where(pm == c)[0]
    slide = next((s for s in SLIDE if M[f'{s}_moved'][im].mean() >= 0.3), None)
    if not slide:
        continue
    bi = SLIDE[slide][1]
    steady = vm < cutv
    top = np.sort(np.argsort(fisher_steady(np.asarray(M['before'][im], np.float32), ym[im], steady[im]))[-TOP:])
    F = np.concatenate([np.asarray(M['before'][i:i + 256][:, top], np.float32) for i in range(0, n, 256)])
    locked, press = M[f'{slide}_locked'], M['flipb'][:, bi]
    e = dict(place=c, slide=slide)
    for rule, cv in [('steady', cutv), ('stricter', np.median(vm[vm < cutv]))]:
        ws = vm[:, top] < cv
        centre = (F * ws[..., None]).sum(0) / np.maximum(ws.sum(0), 1)[:, None]
        known = ws.sum(1) >= 3
        read = ((F - centre) * ws[..., None]).sum(1) / np.maximum(ws.sum(1), 1)[:, None]
        lab = KMeans(2, n_init=5, random_state=0).fit_predict(read[known])
        state = np.full(n, -1)
        state[known] = lab
        flipit = (state[known] == locked[known]).mean() < 0.5
        right = lambda idx: round(float(((state[idx] == locked[idx]) != flipit).mean()), 3)
        ki = np.where(known)[0]
        spans, inside = [], np.zeros(n, bool)
        for a, b in zip(ki[:-1], ki[1:]):
            if ep[a] == ep[b] and state[a] != state[b]:
                spans.append((a, b))
                inside[a:b] = True
        has = [bool(press[a:b].any()) for a, b in spans]
        one = [len(set(pm[a:b])) == 1 for a, b in spans]
        modal = np.bincount(np.concatenate([pm[a:b] for a, b in spans]), minlength=K).argmax() if spans else -1
        at = [bool((pm[a:b] == modal).any()) for a, b in spans]
        between = np.zeros(n, bool)                       # presses with a direct reading before and after in the episode
        for a, b in zip(ki[:-1], ki[1:]):
            if ep[a] == ep[b]:
                between[a:b] = True
        e[rule] = dict(direct_readings=round(float(known.mean()), 3), direct_right=right(ki),
                       direct_right_at_button_place=right(ki[pm[ki] == modal]) if (pm[ki] == modal).any() else None,
                       direct_right_elsewhere=right(ki[pm[ki] != modal]),
                       spans=len(spans), spans_with_a_press=round(float(np.mean(has)), 3),
                       spans_touching_the_modal_place=round(float(np.mean(at)), 3),
                       mean_touches_per_span=round(float(np.mean([b - a for a, b in spans])), 2),
                       presses=int((press & between).sum()),
                       presses_inside_a_span=round(float(inside[press & between].mean()), 3))
    out.append(e)
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / 'runs/125.1/diag.json', 'w'), indent=1)
