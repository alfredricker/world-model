"""Card 131, data check 2: a belief carried along recorded episodes the read-out never trained on (set t500).

The stream is the frame before each touch of an episode (runs/125/before_t500.npy). For the cube's group:
- reading valid = largest patch score at least memory's threshold (5.69; check.py: 5th percentile over held-out
  frames that preceded a cube touch);
- belief: the last valid reading; when the reading is not valid, the belief is kept, except that after a touch
  whose release point is more than 5 cm from the belief and whose contact point was within 3 cm of it (the agent
  itself carried the believed thing away), the belief becomes that release point (proprioception).
Truth (scores only): the cube on the table, in the open drawer, or in the closed drawer (drawer under 2 cm open).

  bin/prun python tools/card131/stream.py → runs/131/stream.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'card129'))
from readout_q import ReadOutQ  # noqa: E402

ROOT = HERE.parent.parent
EP, CEN, GRP, CUT, FIRST = 1001, np.array([0.425, 0, 0]), 3, 5.69, 500
z = np.load(Path.home() / '.ogbench/data/scene-play-v0.npz')
sl = slice(FIRST * EP, (FIRST + 100) * EP)
obs, qpos = z['observations'][sl], z['qpos'][sl]
c = np.load(ROOT / 'runs/122/computed_t500.npz')
st, en = c['st'], c['en']
X = np.load(ROOT / 'runs/125/before_t500.npy', mmap_mode='r')
print('touches', len(st), 'features', X.shape, 'qpos ranges', qpos.min(0)[14:].round(2), qpos.max(0)[14:].round(2), flush=True)
ro = ReadOutQ().cuda()
ro.load_state_dict(torch.load(ROOT / 'runs/129/readout_q.pt', weights_only=False)['state'])
ro.eval()
score, read = [], []
with torch.no_grad():
    for i in range(0, len(st), 64):
        f = torch.from_numpy(np.asarray(X[i:i + 64], dtype=np.float32)).cuda()
        score += ro.score(ro.trunk(f))[:, :, GRP].max(1).values.tolist()
        read += ro(f)[:, GRP].cpu().numpy().tolist()
score, read = np.array(score), np.array(read) / 10 + CEN
t0 = np.maximum(st - 1, 0)
cube, drawer = qpos[t0, 14:17], qpos[t0, 23]
inside = (np.abs(cube[:, 0] - 0.33) < 0.1) & (cube[:, 1] < -0.15) & (cube[:, 2] > 0.04)
kind = np.where(~inside, 'table', np.where(drawer > -0.02, 'closed drawer', 'open drawer'))
valid = score >= CUT
con, rel = obs[st + 2, 12:15] / 10 + CEN, obs[en, 12:15] / 10 + CEN
bel = np.full((len(st), 3), np.nan)
b = None
for i in range(len(st)):
    if i == 0 or st[i] // EP != st[i - 1] // EP:
        b = None
    elif b is not None and np.linalg.norm(con[i - 1] - b) < 0.03 and np.linalg.norm(rel[i - 1] - b) > 0.05:
        b = rel[i - 1]
    if valid[i]:
        b = read[i]
    if b is not None:
        bel[i] = b
err_read, err_bel = np.linalg.norm(read - cube, axis=1) * 100, np.linalg.norm(bel - cube, axis=1) * 100
res = {}
for k in ('table', 'open drawer', 'closed drawer'):
    m = kind == k
    q = lambda x: [round(float(v), 1) for v in np.nanquantile(x, [0.5, 0.9])]
    res[k] = dict(readings=int(m.sum()), read_valid=round(float(valid[m].mean()), 3), reading_error_cm_when_valid=q(err_read[m & valid]) if (m & valid).any() else None,
                  belief_held=round(float(np.isfinite(err_bel[m]).mean()), 3), belief_error_cm=q(err_bel[m]),
                  belief_within_4cm=round(float((err_bel[m] <= 4).mean()), 3), belief_within_10cm=round(float((err_bel[m] <= 10).mean()), 3))
    print(k, res[k], flush=True)
ch = np.where((kind[1:] != kind[:-1]) & (st[1:] // EP == st[:-1] // EP))[0]
from collections import Counter
res['changes'] = {f'{a} → {b}': n for (a, b), n in Counter(zip(kind[ch], kind[ch + 1])).items()}
print(res['changes'])
json.dump(res, open(ROOT / 'runs/131/stream.json', 'w'), indent=1)
