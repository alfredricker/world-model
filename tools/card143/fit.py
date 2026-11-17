"""Card 143: the aim read-out trained on contact labels plus release labels, answering the test bed's frames.

og4's read-out and training (tools/card141/cover.py's fit: 6,000 steps, batch 64, the touch's group as the
question). Training rows: og4's (runs/129, episodes under 80: frame before a touch → contact point) and, added,
runs/143's (frame after a touch → release point; --arm rel) or the same frames with the true cube position as the
target (--arm upper, the gate's upper bound; the truth is used only there). --arm og4 refits og4 alone.
Saved: the answer and the largest patch score for the cube's group on every test-bed frame, and the read-out's error
on its own held-out labels (episodes 80 to 99), from which og4's rule sets the "differs" cut (twice the 90th percentile).

  bin/prun python tools/card143/fit.py --arm rel --seed 0 → runs/143/fit_rel_0.npz, fit_rel_0.pt
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

GRP, STEPS = 3, 6000
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
PER = arg('--per', '')                                                     # '4': the build with up to 4 frames per release


def main():
    arm, seed, t0 = arg('--arm', 'rel'), int(arg('--seed', 0)), time.time()
    X1, M1 = np.load(ROOT / 'runs/129/feats.npy'), np.load(ROOT / 'runs/129/meta.npy')
    G = torch.load(ROOT / 'runs/129/readout_q.pt', weights_only=False)['group']
    Y1, Q1, E1 = M1[:, :3], G[M1[:, 9].astype(int)], M1[:, 10]
    if arm == 'og4':
        X2, Y2, Q2, E2 = np.zeros((0, 1024, 768), np.float16), np.zeros((0, 3)), np.zeros(0, int), np.zeros(0)
    else:
        X2, M2 = np.load(ROOT / f'runs/143/feats{PER}.npy', mmap_mode='r'), np.load(ROOT / f'runs/143/meta{PER}.npy')
        Y2, Q2, E2 = (M2[:, 6:9] if arm == 'upper' else M2[:, :3]), M2[:, 9].astype(int), M2[:, 10]
    n1 = len(X1)
    Y = torch.from_numpy(np.concatenate([Y1, Y2])).float().cuda()
    Q = torch.from_numpy(np.concatenate([Q1, Q2])).long().cuda()
    ep = np.concatenate([E1, E2])
    tr, te = np.where(ep < 80)[0], np.where(ep >= 80)[0]
    print(arm, 'seed', seed, 'training rows', len(tr), 'of them added', int((tr >= n1).sum()), flush=True)

    def feats(b):
        return torch.from_numpy(np.stack([X1[i] if i < n1 else np.asarray(X2[i - n1]) for i in b])).cuda().float()

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = C.read_out().cuda()
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
    for s in range(STEPS):
        b = np.sort(rng.choice(tr, 64, replace=False))
        out = model(feats(b))
        loss = ((out[torch.arange(64), Q[b]] - Y[b]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 1000 == 0:
            print(s, round(loss.item(), 4), f'{time.time() - t0:.0f}s', flush=True)
    model.eval()
    torch.save(dict(state=model.state_dict(), group=G), ROOT / f'runs/143/fit_{arm}{PER}_{seed}.pt')
    own = []
    with torch.no_grad():
        for a in range(0, len(te), 64):
            b = te[a:a + 64]
            o = model(feats(b))
            own.append((o[torch.arange(len(b)), Q[b]] - Y[b]).norm(dim=-1).cpu().numpy() * 10)   # cm
    own = np.concatenate(own)
    tt, Xt = C.truth(C.TEST), C.frames(C.TEST)
    P, top, n = [], [], len(tt['ep'])
    with torch.no_grad():
        for i0 in range(0, n, 64):
            f = torch.from_numpy(np.stack([np.asarray(Xt[tt['set'][i]][tt['row'][i]], np.float32) for i in range(i0, min(i0 + 64, n))])).cuda()
            P.append(model(f)[:, GRP].cpu().numpy())
            top.append(model.score(model.trunk(f))[:, :, GRP].max(1).values.cpu().numpy())
    np.savez(ROOT / f'runs/143/fit_{arm}{PER}_{seed}.npz', aim=np.concatenate(P) / 10 + C.CEN, top=np.concatenate(top),
             own_err_cm=own, own_group=Q[te].cpu().numpy(), own_added=te >= n1)
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
