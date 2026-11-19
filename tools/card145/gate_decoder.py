"""Card 145, gate step 1: can a frame be explained from a small state?

Two decoders (train.py --mode truth and --mode pose): one given the true scene state and the arm's pose, one the pose
only. On validation episodes, for every stretch of change of a variable: the frame just before and the frame just
after it; the 20 patches that differ most between the two once the pose-only decoder's picture is taken away (so the
arm's own motion does not count); the squared error of each decoder on those patches. The gate asks the true-state
decoder to leave at most half of the pose-only decoder's error there. Also each decoder's error over whole frames
(1 = always answering the mean frame).

  bin/prun python tools/card145/gate_decoder.py → runs/145/gate_decoder.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shared as C  # noqa: E402
from model import Decoder  # noqa: E402
from train import norms, true_state  # noqa: E402

nz = norms()
X, M = np.load(C.OUT / 'feat_val.npy', mmap_mode='r'), np.load(C.OUT / 'meta_val.npz')
Mt = np.load(C.OUT / 'meta_train.npz')
pose = torch.from_numpy(((M['pose'] - nz['pmean']) / nz['pstd']).astype(np.float32)).cuda()
s_tr = np.concatenate([Mt['cube'], Mt['drawer'][..., None], Mt['window'][..., None], Mt['locks']], -1).reshape(-1, 11)
s = np.concatenate([M['cube'], M['drawer'][..., None], M['window'][..., None], M['locks']], -1).astype(np.float32)
truth = torch.from_numpy(((s - s_tr.mean(0)) / (s_tr.std(0) + 1e-6)).astype(np.float32)).cuda()   # standardised as in training
mean, scale = torch.from_numpy(nz['mean']).cuda(), float(nz['scale'])
dec = {}
for k in ('truth', 'pose'):
    dec[k] = Decoder(11).cuda(); dec[k].load_state_dict(torch.load(C.OUT / f'gate_{k}.pt')['state']); dec[k].eval()


def run(e, t):
    x = (torch.from_numpy(np.stack([np.asarray(X[a, b], np.float32) for a, b in zip(e, t)])).cuda() - mean) / scale
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return x, dec['truth'](truth[e, t], pose[e, t]).float(), dec['pose'](truth[e, t] * 0, pose[e, t]).float()


ch = C.changes(M)
rng = np.random.default_rng(0)
out = {}
e, t = rng.integers(0, 100, 600), rng.integers(0, 201, 600)
o = np.lexsort((t, e)); e, t = e[o], t[o]
ft, fp = [], []
for i in range(0, 600, 100):
    x, q, p = run(e[i:i + 100], t[i:i + 100])
    ft.append(((x - q) ** 2).sum(-1).mean(1).cpu().numpy()); fp.append(((x - p) ** 2).sum(-1).mean(1).cpu().numpy())
out['whole frame'] = dict(true_state=round(float(np.concatenate(ft).mean()), 4), pose_only=round(float(np.concatenate(fp).mean()), 4), mean_frame=1.0)
for vi, name in enumerate(C.NAMES):
    ev = []
    for ep in range(100):
        xch = ch[ep, :, vi].astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, xch, 0]))
        ev += [(ep, a, b) for a, b in zip(edges[::2], edges[1::2])]
    ev = [ev[i] for i in rng.permutation(len(ev))[:300]]
    et, ep_ = [], []
    for i in range(0, len(ev), 50):
        part = ev[i:i + 50]
        ee = np.array([v[0] for v in part])
        xa, qa, pa = run(ee, np.array([v[1] for v in part]))
        xb, qb, pb = run(ee, np.array([v[2] for v in part]))
        top = ((xb - pb) - (xa - pa)).norm(dim=-1).topk(20, dim=1).indices
        for x, q, p in ((xa, qa, pa), (xb, qb, pb)):
            et.append(((x - q) ** 2).sum(-1).gather(1, top).mean(1).cpu().numpy()); ep_.append(((x - p) ** 2).sum(-1).gather(1, top).mean(1).cpu().numpy())
    et, ep_ = np.concatenate(et).mean(), np.concatenate(ep_).mean()
    out[name] = dict(events=len(ev), true_state=round(float(et), 4), pose_only=round(float(ep_), 4), share_left=round(float(et / ep_), 3))
json.dump(out, open(C.OUT / 'gate_decoder.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
