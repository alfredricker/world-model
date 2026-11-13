"""Card 129.1: for which groups of places is the read-out's aim informative?

Per group, on held-out touches (episodes 80-99 of the three sets): the median error of the group-asked read-out
(card 129's readout_q.pt) against where the gripper then closed, and the same for the mean contact point of the
touch's own place (the finest summary memory has without looking). The aim is informative for a group if its
error is under half that baseline. Only contact points are used; no truth.

  bin/prun python tools/card129.1/aim.py → runs/129.1/aim.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card129'))
from readout import OUT, ROOT  # noqa: E402
from readout_q import ReadOutQ  # noqa: E402

X, M = np.load(OUT / 'feats.npy', mmap_mode='r'), np.load(OUT / 'meta.npy')
ck = torch.load(OUT / 'readout_q.pt', weights_only=False)
G = ck['group']
model = ReadOutQ().cuda()
model.load_state_dict(ck['state'])
model.eval()
te, tr = np.where(M[:, 10] >= 80)[0], M[:, 10] < 80
g = G[M[:, 9].astype(int)]
P = []
with torch.no_grad():
    for a in range(0, len(te), 64):
        P.append(model(torch.from_numpy(np.asarray(X[te[a:a + 64]])).cuda().float()).cpu().numpy())
P = np.concatenate(P)
err = np.linalg.norm(P[np.arange(len(te)), g[te]] - M[te, :3], axis=1) * 10
base = np.array([np.linalg.norm(M[tr & (M[:, 9] == M[i, 9]), :3].mean(0) - M[i, :3]) * 10 for i in te])
res = {}
for c in np.unique(g[te]):
    m = g[te] == c
    if m.sum() >= 20:
        e, b = float(np.median(err[m])), float(np.median(base[m]))
        res[int(c)] = dict(n=int(m.sum()), aim_median_cm=round(e, 2), aim_p90_cm=round(float(np.quantile(err[m], 0.9)), 2), place_mean_median_cm=round(b, 2), informative=bool(e < 0.5 * b))
print(json.dumps(res, indent=1))
json.dump(res, open(ROOT / 'runs/129.1/aim.json', 'w'), indent=1)
