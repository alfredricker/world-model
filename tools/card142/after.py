"""Card 142, the map fitted afterwards (the form of the card's check 2; added during the run, not in the amendment).

The weights are those of an og-form fit (runs/142/fit_og_<a|w>_<seed>.pt, trained on card 143's labels, with or
without the upper third of the table). The centre of the cube group's weights in the image is turned into a position
by least squares on the agent's own training labels (the same rows the fit saw; no truth):
  own     second order in (row, column), one map per group and per height cluster of its labels (the table; above it)
  shared  second order in (row, column, height), one map for every group; the height is the label's own when fitting,
          the group's usual height when answering
Output in the format of fit.py, so score.py reads it: runs/142/fit_ownls_<a|w>_<seed>.npz, fit_sharedls_<a|w>_<seed>.npz

  bin/prun python tools/card142/after.py a 0
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

GRP, YCUT, K = 3, 1.02, 10
wh, seed = sys.argv[1], int(sys.argv[2])
ck = torch.load(ROOT / f'runs/142/fit_og_{wh}_{seed}.pt', weights_only=False)
model = C.read_out().cuda(); model.load_state_dict(ck['state']); model.eval()
G = ck['group']


def centres(X, rows):
    out, top = [], []
    with torch.no_grad():
        for a in range(0, len(rows), 64):
            f = torch.from_numpy(np.stack([np.asarray(X[i], np.float32) for i in rows[a:a + 64]])).cuda()
            s = model.score(model.trunk(f))
            out.append(torch.einsum('bpk,pd->bkd', torch.softmax(s, 1), model.grid).cpu().numpy()); top.append(s[:, :, GRP].max(1).values.cpu().numpy())
    return np.concatenate(out), np.concatenate(top)                         # [n, K, 2]


def q2(c):
    return np.stack([np.ones(len(c)), c[:, 0], c[:, 1], c[:, 0] ** 2, c[:, 1] ** 2, c[:, 0] * c[:, 1]], 1)


def q3(c, h):
    r, q = c[:, 0], c[:, 1]
    return np.stack([np.ones(len(r)), r, q, h, r * r, q * q, h * h, r * q, r * h, q * h], 1)


X1, M1 = np.load(ROOT / 'runs/129/feats.npy', mmap_mode='r'), np.load(ROOT / 'runs/129/meta.npy')
X2, M2 = np.load(ROOT / 'runs/143/feats4.npy', mmap_mode='r'), np.load(ROOT / 'runs/143/meta4.npy')
Y = np.concatenate([M1[:, :3], M2[:, :3]])
Q = np.concatenate([G[M1[:, 9].astype(int)], M2[:, 9].astype(int)])
ep = np.concatenate([M1[:, 10], M2[:, 10]])
Call = np.concatenate([centres(X1, np.arange(len(X1)))[0], centres(X2, np.arange(len(X2)))[0]])
Cown = Call[np.arange(len(Q)), Q]                                           # each row's own group's centre
keep = ~((Q == GRP) & (Y[:, 1] > YCUT)) if wh == 'w' else np.ones(len(Y), bool)
tr, te = (ep < 80) & keep, ep >= 80
usual = np.array([np.median(Y[tr & (Q == k), 2]) if (tr & (Q == k)).any() else 0.0 for k in range(K)])
cube = Q == GRP
low = Y[:, 2] < usual[GRP] + 0.3
high = float(np.median(Y[tr & cube & ~low, 2]))
tt, Xt = C.truth(C.TEST), C.frames(C.TEST)
ct, top = [], []
for s in range(len(Xt)):
    c, t_ = centres(Xt[s], tt['row'][tt['set'] == s]); ct.append(c[:, GRP]); top.append(t_)
ct, top = np.concatenate(ct), np.concatenate(top)                           # truth() lists the sets in order
# own: a map per height cluster
B_low = np.linalg.lstsq(q2(Cown[tr & cube & low]), Y[tr & cube & low], rcond=None)[0]
B_high = np.linalg.lstsq(q2(Cown[tr & cube & ~low]), Y[tr & cube & ~low], rcond=None)[0]
k = te & cube
np.savez(ROOT / f'runs/142/fit_ownls_{wh}_{seed}.npz', aim=q2(ct) @ B_low / 10 + C.CEN, aim_high=q2(ct) @ B_high / 10 + C.CEN, top=top,
         own_err_cm=np.linalg.norm((q2(Cown[k]) @ B_low - Y[k])[:, :2], axis=1) * 10, own_group=Q[k], own_y=Y[k, 1], own_z=Y[k, 2])
# shared: one map for every group
B = np.linalg.lstsq(q3(Cown[tr], Y[tr, 2]), Y[tr], rcond=None)[0]
n = len(ct)
np.savez(ROOT / f'runs/142/fit_sharedls_{wh}_{seed}.npz', aim=q3(ct, np.full(n, usual[GRP])) @ B / 10 + C.CEN, aim_high=q3(ct, np.full(n, high)) @ B / 10 + C.CEN, top=top,
         own_err_cm=np.linalg.norm((q3(Cown[k], np.full(k.sum(), usual[GRP])) @ B - Y[k])[:, :2], axis=1) * 10, own_group=Q[k], own_y=Y[k, 1], own_z=Y[k, 2])
print(wh, seed, 'cube labels fitted', int((tr & cube & low).sum()), 'on the table,', int((tr & cube & ~low).sum()), 'above; all groups', int(tr.sum()), flush=True)
