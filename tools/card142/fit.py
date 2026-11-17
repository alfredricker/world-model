"""Card 142: three ways of turning the read-out's weights into a position, each trained on card 143's labels.

Training rows as tools/card143/fit.py --per 4: frame before a touch → contact point (runs/129), frame after a touch →
release point (runs/143, cube group). The score that gives the weights, the features and the schedule are og4's.
  --arm og      position = sum of weight × (place(row, column) + offset(patch features))         (og4's form)
  --arm own     centre = sum of weight × (row, column);  position = Q(centre, height) · B[group]  (a map per group)
  --arm shared  the same with one B for every group
Q is second order in (row, column, height). The height is the hand's own height at the label when training, and the
group's usual height (the median of its training labels) when answering; no offset from features in own and shared.
--withhold 1: no cube-group label whose y is over 0.102 m (the upper third of the table), as card 141.
Saved for every test-bed frame: the cube group's answer at its usual height, the answer at the height of its labels
above the table (the open drawer; reported without a bar), and the largest patch score; and the error on the arm's
own held-out labels.

  bin/prun python tools/card142/fit.py --arm own --withhold 1 --seed 0 → runs/142/fit_own_w_0.npz, .pt
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

GRP, STEPS, YCUT, K = 3, 6000, 1.02, 10
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def quad(c, h):                                                            # c [..., 2], h [...] → [..., 10]
    r, q = c[..., 0], c[..., 1]
    return torch.stack([torch.ones_like(r), r, q, h, r * r, q * q, h * h, r * q, r * h, q * h], -1)


class Mapped(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU())
        self.score = nn.Linear(128, K)
        self.B = nn.Parameter(torch.zeros(1 if shared else K, 10, 3))
        g = torch.linspace(-1, 1, 32)
        self.register_buffer('grid', torch.stack(torch.meshgrid(g, g, indexing='ij'), -1).reshape(1024, 2))

    def centre(self, f):
        return torch.einsum('bpk,pd->bkd', torch.softmax(self.score(self.trunk(f)), 1), self.grid)   # [B, K, 2]

    def forward(self, f, h):                                               # h [B, K] → [B, K, 3]
        return torch.einsum('bkq,kqd->bkd', quad(self.centre(f), h), self.B.expand(K, -1, -1))


def main():
    arm, seed, wh, t0 = arg('--arm', 'own'), int(arg('--seed', 0)), int(arg('--withhold', 0)), time.time()
    X1, M1 = np.load(ROOT / 'runs/129/feats.npy'), np.load(ROOT / 'runs/129/meta.npy')
    G = torch.load(ROOT / 'runs/129/readout_q.pt', weights_only=False)['group']
    X2, M2 = np.load(ROOT / 'runs/143/feats4.npy', mmap_mode='r'), np.load(ROOT / 'runs/143/meta4.npy')
    n1 = len(X1)
    Yn = np.concatenate([M1[:, :3], M2[:, :3]])
    Qn = np.concatenate([G[M1[:, 9].astype(int)], M2[:, 9].astype(int)])
    ep = np.concatenate([M1[:, 10], M2[:, 10]])
    keep = ~((Qn == GRP) & (Yn[:, 1] > YCUT)) if wh else np.ones(len(Yn), bool)
    tr, te = np.where((ep < 80) & keep)[0], np.where(ep >= 80)[0]
    usual = np.array([np.median(Yn[tr][Qn[tr] == k, 2]) if (Qn[tr] == k).any() else 0.0 for k in range(K)])
    cz = Yn[tr][Qn[tr] == GRP, 2]
    high = float(np.median(cz[cz > usual[GRP] + 0.3])) if (cz > usual[GRP] + 0.3).any() else float(usual[GRP])
    print(arm, 'withheld' if wh else 'all', 'seed', seed, 'rows', len(tr), 'cube rows', int((Qn[tr] == GRP).sum()),
          'usual heights', usual.round(2).tolist(), 'cube labels above the table at', round(high, 2), flush=True)
    Y, Q = torch.from_numpy(Yn).float().cuda(), torch.from_numpy(Qn).long().cuda()
    U = torch.from_numpy(usual).float().cuda()

    def feats(b):
        return torch.from_numpy(np.stack([X1[i] if i < n1 else np.asarray(X2[i - n1]) for i in b])).cuda().float()

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if arm == 'og':
        model = C.read_out().cuda()
        opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
        ans = lambda f, h: model(f)
    else:
        model = Mapped(arm == 'shared').cuda()
        opt = torch.optim.AdamW([dict(params=[p for n, p in model.named_parameters() if n != 'B'], lr=1e-3, weight_decay=1e-4),
                                 dict(params=[model.B], lr=1e-2, weight_decay=0.0)])
        ans = lambda f, h: model(f, h)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [2e-3] if arm == 'og' else [2e-3, 2e-2], total_steps=STEPS)
    for s in range(STEPS):
        b = np.sort(rng.choice(tr, 64, replace=False))
        h = Y[b, 2][:, None].expand(-1, K)                                 # the hand's own height at the label
        out = ans(feats(b), h)
        loss = ((out[torch.arange(64), Q[b]] - Y[b]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 1000 == 0:
            print(s, round(loss.item(), 4), f'{time.time() - t0:.0f}s', flush=True)
    model.eval()
    name = f'fit_{arm}_{"w" if wh else "a"}_{seed}'
    torch.save(dict(state=model.state_dict(), group=G, usual=usual, high=high), ROOT / f'runs/142/{name}.pt')
    own = []
    with torch.no_grad():
        for a in range(0, len(te), 64):
            b = te[a:a + 64]
            o = ans(feats(b), U[None].expand(len(b), -1))
            own.append((o[torch.arange(len(b)), Q[b]] - Y[b])[:, :2].norm(dim=-1).cpu().numpy() * 10)   # cm, in the plane
    tt, Xt = C.truth(C.TEST), C.frames(C.TEST)
    P, PH, top, n = [], [], [], len(tt['ep'])
    Uh = U.clone(); Uh[GRP] = high
    with torch.no_grad():
        for i0 in range(0, n, 64):
            f = torch.from_numpy(np.stack([np.asarray(Xt[tt['set'][i]][tt['row'][i]], np.float32) for i in range(i0, min(i0 + 64, n))])).cuda()
            P.append(ans(f, U[None].expand(len(f), -1))[:, GRP].cpu().numpy())
            PH.append(ans(f, Uh[None].expand(len(f), -1))[:, GRP].cpu().numpy())
            top.append(model.score(model.trunk(f))[:, :, GRP].max(1).values.cpu().numpy())
    np.savez(ROOT / f'runs/142/{name}.npz', aim=np.concatenate(P) / 10 + C.CEN, aim_high=np.concatenate(PH) / 10 + C.CEN, top=np.concatenate(top),
             own_err_cm=np.concatenate(own), own_group=Qn[te], own_y=Yn[te, 1], own_z=Yn[te, 2])
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
