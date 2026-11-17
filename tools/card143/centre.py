"""Card 143, diagnostic: does a fit's weight sit on the cube? As tools/card142/check_open.py, for any fit.

The centre of the read-out's weights in the image (row, column) is turned into a position by a second-order fit,
separately for the table and the open drawer, made (a) on the bed's learning frames with the true cube position
(scores only: what the centre holds) and (b) on the agent's own labels only: runs/129 contact points and runs/143
release points of the training episodes, split by the height of the label (no truth).

  bin/prun python tools/card143/centre.py runs/143/fit_upper_0.pt → runs/143/centre_<name>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

GRP = 3
ck = torch.load(ROOT / sys.argv[1], weights_only=False)
model = C.read_out().cuda(); model.load_state_dict(ck['state']); model.eval()


def centre(f):
    with torch.no_grad():
        h = model.trunk(f); s = model.score(h)[:, :, GRP]; w = torch.softmax(s, 1)
        ans = torch.einsum('bp,bpd->bd', w, model.where(model.grid) + model.offset(h))
        return (w[:, :, None] * model.grid[None]).sum(1).cpu().numpy(), ans.cpu().numpy() / 10 + C.CEN


def bed(sets):
    t, X = C.truth(sets), C.frames(sets)
    cen, ans = [], []
    for i0 in range(0, len(t['ep']), 64):
        f = torch.from_numpy(np.stack([np.asarray(X[t['set'][i]][t['row'][i]], np.float32) for i in range(i0, min(i0 + 64, len(t['ep'])))])).cuda()
        c, a = centre(f); cen.append(c); ans.append(a)
    return t, np.concatenate(cen), np.concatenate(ans)


def own():
    cen, lab = [], []
    for d, cols in (('129', slice(0, 3)), ('143', slice(0, 3))):
        X, M = np.load(ROOT / f'runs/{d}/feats.npy', mmap_mode='r'), np.load(ROOT / f'runs/{d}/meta.npy')
        g = ck['group'][M[:, 9].astype(int)] if d == '129' else M[:, 9].astype(int)
        idx = np.where((M[:, 10] < 80) & (g == GRP))[0]
        for a in range(0, len(idx), 64):
            cen.append(centre(torch.from_numpy(np.asarray(X[idx[a:a + 64]], np.float32)).cuda())[0])
        lab.append(M[idx, cols] / 10 + C.CEN)
    return np.concatenate(cen), np.concatenate(lab)


def quad(c):
    return np.stack([np.ones(len(c)), c[:, 0], c[:, 1], c[:, 0] ** 2, c[:, 1] ** 2, c[:, 0] * c[:, 1]], 1)


tl, cl, _ = bed(C.LEARN); tt, ct, at = bed(C.TEST)
oc, ol = own()
low = ol[:, 2] < np.median(ol[:, 2]) + 0.03                                 # the agent's own split: labels at the usual height, and above it
print('own labels', len(ol), 'above the usual height', int((~low).sum()), 'height quantiles', np.quantile(ol[:, 2], [.05, .5, .95]).round(3))
out = {}
for wh, own_k in (('table', low), ('open', ~low)):
    kl, kt = tl['where'] == wh, tt['where'] == wh
    r = dict(n=int(kt.sum()), read_out_answer_cm=round(float(np.median(np.linalg.norm(at[kt][:, :2] - tt['cube'][kt][:, :2], axis=1))) * 100, 2))
    for name, cc, yy in (('centre_fit_on_truth', cl[kl], tl['cube'][kl][:, :2]), ('centre_fit_on_own_labels', oc[own_k], ol[own_k][:, :2])):
        B = np.linalg.lstsq(quad(cc), yy, rcond=None)[0]
        e = np.linalg.norm(quad(ct[kt]) @ B - tt['cube'][kt][:, :2], axis=1) * 100
        r[name] = dict(fit_points=int(len(cc)), median_cm=round(float(np.median(e)), 2), within_1cm=round(100 * float((e < 1).mean()), 1), within_3cm=round(100 * float((e < 3).mean()), 1))
    out[wh] = r
json.dump(out, open(ROOT / f'runs/143/centre_{Path(sys.argv[1]).stem}.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
