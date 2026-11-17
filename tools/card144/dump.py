"""Card 144, step 1: the cube group's score for every patch, from an og-form fit of card 142 (trained on card 143's
labels, with or without the upper third of the table), for the cube group's own labelled frames and every test-bed
frame. Everything after this works on these score maps.

  bin/prun python tools/card144/dump.py a 0 → runs/144/scores_a_0.npz
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

GRP = 3
wh, seed = sys.argv[1], int(sys.argv[2])
ck = torch.load(ROOT / f'runs/142/fit_og_{wh}_{seed}.pt', weights_only=False)
model = C.read_out().cuda(); model.load_state_dict(ck['state']); model.eval()
G = ck['group']


def scores(X, rows):
    out = []
    with torch.no_grad():
        for a in range(0, len(rows), 64):
            f = torch.from_numpy(np.stack([np.asarray(X[i], np.float32) for i in rows[a:a + 64]])).cuda()
            out.append(model.score(model.trunk(f))[:, :, GRP].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


X1, M1 = np.load(ROOT / 'runs/129/feats.npy', mmap_mode='r'), np.load(ROOT / 'runs/129/meta.npy')
X2, M2 = np.load(ROOT / 'runs/143/feats4.npy', mmap_mode='r'), np.load(ROOT / 'runs/143/meta4.npy')
r1 = np.where(G[M1[:, 9].astype(int)] == GRP)[0]
r2 = np.where(M2[:, 9].astype(int) == GRP)[0]
S = np.concatenate([scores(X1, r1), scores(X2, r2)])
Y = np.concatenate([M1[r1, :3], M2[r2, :3]])
ep = np.concatenate([M1[r1, 10], M2[r2, 10]])
src = np.concatenate([np.zeros(len(r1), int), np.ones(len(r2), int)])      # 0 contact label, 1 release label
row = np.concatenate([r1, r2])
tt, Xt = C.truth(C.TEST), C.frames(C.TEST)
St = np.concatenate([scores(Xt[s], tt['row'][tt['set'] == s]) for s in range(len(Xt))])
np.savez(ROOT / f'runs/144/scores_{wh}_{seed}.npz', S=S, Y=Y, ep=ep, src=src, row=row, St=St)
print(wh, seed, 'cube labels', len(S), 'test frames', len(St), flush=True)
