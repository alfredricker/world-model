"""Card 141, criterion 3: og4's aim read-out retrained with a third of the table withheld from its training signal.

The read-out's own data and training (tools/card129/readout_q.py: runs/129/feats.npy, the gripper's position at
contact as the target, the touch's group as the question, 6,000 steps). Two fits: all touches; and without the cube
group's touches whose contact point has y over 0.102 m (the upper third). Each answers the test bed's frames.

  bin/prun python tools/card141/cover.py → runs/141/cover.npz
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

GRP, STEPS, YCUT = 3, 6000, 1.02


def fit(X, M, tr, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = C.read_out().cuda()
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
    Y, Q = torch.from_numpy(M[:, :3]).float().cuda(), torch.from_numpy(M[:, 9]).long().cuda()
    for s in range(STEPS):
        b = np.sort(rng.choice(tr, 64, replace=False))
        out = model(torch.from_numpy(X[b]).cuda().float())
        loss = ((out[torch.arange(64), Q[b]] - Y[b]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model.eval()


def main():
    t0 = time.time()
    X, M = np.load(C.ROOT / 'runs/129/feats.npy'), np.load(C.ROOT / 'runs/129/meta.npy')
    G = torch.load(C.ROOT / 'runs/129/readout_q.pt', weights_only=False)['group']
    M = M.copy(); M[:, 9] = G[M[:, 9].astype(int)]
    base = M[:, 10] < 80
    arms = dict(all=np.where(base)[0], withheld=np.where(base & ~((M[:, 9] == GRP) & (M[:, 1] > YCUT)))[0])
    tt, Xt = C.truth(C.TEST), C.frames(C.TEST)
    out = dict(train_cube_xy=M[base & (M[:, 9] == GRP)][:, 6:8] / 10 + C.CEN[:2])
    for name, tr in arms.items():
        model = fit(X, M, tr)
        print(name, 'touches', len(tr), 'of the cube group', int((M[tr, 9] == GRP).sum()), f'{time.time() - t0:.0f}s', flush=True)
        P, n = [], len(tt['ep'])
        with torch.no_grad():
            for i0 in range(0, n, 64):
                f = torch.from_numpy(np.stack([np.asarray(Xt[tt['set'][i]][tt['row'][i]], np.float32) for i in range(i0, min(i0 + 64, n))])).cuda()
                P.append(model(f)[:, GRP].cpu().numpy())
        out[name] = np.concatenate(P) / 10 + C.CEN
    np.savez(C.OUT / 'cover.npz', **out)
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
