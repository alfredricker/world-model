"""Card 129, second read-out: asked about one place at a time.

readout.py's mixture over all targets failed its gate (a peak within 1 cm of the cube in 40% of frames): the
targets share one softmax, and the one about to be touched takes the weight. Here the question names a place of
the agent's memory of touches (k-means over contact points, card 125): "if I touched at this place, where would
the gripper close?" Per patch, one score per place (768 → 128 → 10) and one offset; softmax over patches for the
asked place; the answer is the weighted mean of (patch position + offset). Loss: squared error to the contact
point, asked with the touch's own place. Same data as readout.py.
Score on held-out touches (episodes 80-99; the truth only scores):
- own place asked: error against where the gripper then closed, by kind of place; baseline the place's mean;
- every frame, each free place asked (no variable, no cause role): error against the true cube.

Third version (--groups 1): the question names a group of places, not a place. Two places are joined if remembered
moves run both ways between them (at least 15% of each one's touches with a seen change start there and are
released in the other); groups are the connected sets. Asked per place, a free place answered well only when the
cube was in its own region (72% within 1 cm).

  bin/prun python tools/card129/readout_q.py train [--groups 1] → runs/129/readout_q.json, readout_q.pt
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readout import OUT, ROOT, arg  # noqa: E402

K = 10


class ReadOutQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU())
        self.score, self.offset = nn.Linear(128, K), nn.Linear(128, 3)
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
        self.where = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
        g = torch.linspace(-1, 1, 32)
        self.register_buffer('grid', torch.stack(torch.meshgrid(g, g, indexing='ij'), -1).reshape(1024, 2))

    def forward(self, f):                                                  # [B, 1024, 768] → [B, K, 3]: an answer per place
        h = self.trunk(f)
        w = torch.softmax(self.score(h), 1)                                # [B, 1024, K]
        return torch.einsum('bpk,bpd->bkd', w, self.where(self.grid) + self.offset(h))


def groups(M=None):
    """→ group id per place: connected sets of places joined by moves both ways (m400's touches only)."""
    M = np.load(OUT / 'meta.npy') if M is None else M
    M = M[M[:, 11] == 400]
    k = np.load(ROOT / 'runs/127/know.npz')
    from readout import DATA, EP
    o = np.load(DATA / 'scene-play-v0.npz')['observations'][400 * EP:500 * EP]
    cen = np.stack([o[k['st'][k['place'] == c] + 2, 12:15].mean(0) for c in range(K)])
    ca, cb = M[:, 9].astype(int), np.linalg.norm(M[:, None, 3:6] - cen[None], axis=-1).argmin(1)
    T = np.zeros((K, K))
    np.add.at(T, (ca, cb), 1)
    S = T / np.maximum(T.sum(1, keepdims=True), 1)
    link = (S >= 0.15) & (S.T >= 0.15)
    g = np.arange(K)
    for _ in range(K):
        for a in range(K):
            for b in range(K):
                if link[a, b]:
                    g[a] = g[b] = min(g[a], g[b])
    return g


def train():
    X, M = np.load(OUT / 'feats.npy'), np.load(OUT / 'meta.npy')
    G = groups(M) if int(arg('--groups', 0)) else np.arange(K)
    print('group of each place', G.tolist(), flush=True)
    M = M.copy(); M[:, 9] = G[M[:, 9].astype(int)]
    tr = np.where(M[:, 10] < 80)[0]
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ReadOutQ().cuda()
    steps = int(arg('--steps', 6000))
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    Y, C = torch.from_numpy(M[:, :3]).float().cuda(), torch.from_numpy(M[:, 9]).long().cuda()
    pw = np.ones(len(tr))
    if int(arg('--balance', 0)):                                           # each group drawn equally often
        cnt = np.bincount(M[tr, 9].astype(int), minlength=K)
        pw = 1.0 / cnt[M[tr, 9].astype(int)]
    pw = pw / pw.sum()
    for s in range(steps):
        b = rng.choice(tr, 64, replace=False, p=pw)
        out = model(torch.from_numpy(X[b]).cuda().float())
        loss = ((out[torch.arange(64), C[b]] - Y[b]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 1000 == 0:
            print(s, round(loss.item(), 4), flush=True)
    torch.save(dict(state=model.state_dict(), group=G), OUT / 'readout_q.pt')
    model.eval()
    k = np.load(ROOT / 'runs/127/know.npz')
    slide = {int(n[3:]) for n in k.files if n.startswith('eff')}
    cause = {int(k[n]) for n in k.files if n.startswith('cause')}
    slide, cause = {int(G[c]) for c in slide}, {int(G[c]) for c in cause}
    free = sorted({int(G[c]) for c in range(K)} - slide - cause)
    kind = lambda c: 'slide place' if c in slide else 'cause place' if c in cause else 'free place'
    te, trm = np.where(M[:, 10] >= 80)[0], M[:, 10] < 80
    P = []
    with torch.no_grad():
        for a in range(0, len(te), 64):
            P.append(model(torch.from_numpy(X[te[a:a + 64]]).cuda().float()).cpu().numpy())
    P = np.concatenate(P)                                                  # [n, K, 3]
    own = np.linalg.norm(P[np.arange(len(te)), M[te, 9].astype(int)] - M[te, :3], axis=1) * 10
    base = np.array([np.linalg.norm(M[trm & (M[:, 9] == M[i, 9]), :3].mean(0) - M[i, :3]) * 10 for i in te])
    res = dict(train=int(len(tr)), test=int(len(te)), own_place={}, free_places=free)
    for kd in ('cause place', 'slide place', 'free place'):
        m = np.array([kind(int(c)) == kd for c in M[te, 9]])
        res['own_place'][kd] = dict(n=int(m.sum()), median_cm=round(float(np.median(own[m])), 2),
                                    within_1cm=round(float((own[m] <= 1).mean()), 3), within_3cm=round(float((own[m] <= 3).mean()), 3),
                                    baseline_median_cm=round(float(np.median(base[m])), 2))
    cube = np.linalg.norm(P[:, free, :2] - M[te, None, 6:8], axis=-1) * 10            # [n, free]
    res['every_frame_free_place_vs_true_cube'] = {
        str(c): dict(median_cm=round(float(np.median(cube[:, j])), 2), within_1cm=round(float((cube[:, j] <= 1).mean()), 3))
        for j, c in enumerate(free)}
    best = cube.min(1)
    res['every_frame_nearest_free_answer'] = dict(median_cm=round(float(np.median(best)), 2), within_1cm=round(float((best <= 1).mean()), 3))
    json.dump(res, open(OUT / 'readout_q.json', 'w'), indent=1)
    sys.stdout.write(json.dumps(res, indent=1) + '\n'); sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    train()
