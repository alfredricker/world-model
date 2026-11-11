"""Card 125, after the main run (a diagnostic, not judged): the chain read directly at the candidate patches.
For each slide place: each memory touch is scored by the median change signal over that place's 10 candidate
patches (not by whether they are among the touch's 10 most changed patches, as criterion 3 did). Reported: the
area under the ROC curve for "this touch is a press of the locking button", and with Otsu's threshold on the
log score: the share of touches above it that are presses, the share of presses caught, the share in one place.

  bin/prun python tools/card125/chain_diag.py → runs/125/chain_diag.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import ROOT, SLIDE, TOP, fisher, load, otsu  # noqa: E402

M = load('m800', 800)
eps = np.median(M['tden'])
z = M['tnum'] / (M['tden'] + eps)
sm = np.log(np.sort(z, 1)[:, -4:].mean(1))
ym = sm > otsu(sm)
pm = KMeans(10, n_init=5, random_state=0).fit(M['pos']).labels_
out = []
for c in range(10):
    im = np.where(pm == c)[0]
    slide = next((s for s in SLIDE if M[f'{s}_moved'][im].mean() >= 0.3), None)
    if not slide:
        continue
    bi = SLIDE[slide][1]
    top = np.argsort(fisher(np.asarray(M['before'][im], np.float32), ym[im]))[-TOP:]
    sc = np.log(np.median(z[:, top], 1) + 1e-6)
    press = M['flipb'][:, bi]
    hit = sc > otsu(sc)
    places = np.bincount(pm[hit], minlength=10)
    out.append(dict(place=c, slide=slide, auc=round(float(roc_auc_score(press, sc)), 3), setters=int(hit.sum()),
                    are_presses=round(float(press[hit].mean()), 3), presses_caught=round(float(hit[press].mean()), 3),
                    in_one_place=round(float(places.max() / hit.sum()), 3),
                    other_button=round(float(M['flipb'][hit, 1 - bi].mean()), 3)))
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / 'runs/125/chain_diag.json', 'w'), indent=1)
