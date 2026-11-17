"""Card 144: the second stage. The cube group's position is read as  position = map_height(centre),
centre = sum over patches of weight × (patch location + shift).

Stage 1 is an og-form fit of card 142 (runs/142/fit_og_<a|w>_<seed>.pt, trained on card 143's labels, with or
without the upper third of the table). Its trunk is frozen. The map is second order in (row, column), one per height
cluster of the cube group's labels (the table; above it), fitted by least squares on the agent's own labels.
Arms of stage 2 (3,000 steps on the cube group's training labels, the map fixed, then the map fitted once more):
  map    nothing trained (card 142's fitted-afterwards arm)
  off    a shift per patch from its features, bounded to one patch by tanh; the score frozen
  tune   the cube group's score trained on; no shift
  both   both
One answer per frame: the height cluster is that of most of the 5 training labels whose centres lie nearest in the
image; the answer is that cluster's map.

  bin/prun python tools/card144/fit.py a 0 → runs/144/fit_<arm>_a_0.npz
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

GRP, YCUT, STEPS, BOUND = 3, 1.02, 3000, 2 / 31
wh, seed, t0 = sys.argv[1], int(sys.argv[2]), time.time()
ck = torch.load(ROOT / f'runs/142/fit_og_{wh}_{seed}.pt', weights_only=False)
model = C.read_out().cuda(); model.load_state_dict(ck['state']); model.eval()
G, GRID = ck['group'], model.grid


def trunk(X, rows):
    out = []
    with torch.no_grad():
        for a in range(0, len(rows), 64):
            f = torch.from_numpy(np.stack([np.asarray(X[i], np.float32) for i in rows[a:a + 64]])).cuda()
            out.append(model.trunk(f).half())
    return torch.cat(out)                                                   # [n, 1024, 128] on the GPU, fp16


def q2(c):
    return torch.stack([torch.ones_like(c[:, 0]), c[:, 0], c[:, 1], c[:, 0] ** 2, c[:, 1] ** 2, c[:, 0] * c[:, 1]], 1)


X1, M1 = np.load(ROOT / 'runs/129/feats.npy', mmap_mode='r'), np.load(ROOT / 'runs/129/meta.npy')
X2, M2 = np.load(ROOT / 'runs/143/feats4.npy', mmap_mode='r'), np.load(ROOT / 'runs/143/meta4.npy')
r1, r2 = np.where(G[M1[:, 9].astype(int)] == GRP)[0], np.where(M2[:, 9].astype(int) == GRP)[0]
Yn = np.concatenate([M1[r1, :3], M2[r2, :3]])
ep = np.concatenate([M1[r1, 10], M2[r2, 10]])
usual = np.median(Yn[ep < 80, 2])
lown = Yn[:, 2] < usual + 0.3
keep = (Yn[:, 1] <= YCUT) if wh == 'w' else np.ones(len(Yn), bool)
tr, te = np.where((ep < 80) & keep)[0], np.where(ep >= 80)[0]
H = torch.cat([trunk(X1, r1), trunk(X2, r2)])
tt, Xt = C.truth(C.TEST), C.frames(C.TEST)
Ht = torch.cat([trunk(Xt[s], tt['row'][tt['set'] == s]) for s in range(len(Xt))])
Y, low = torch.from_numpy(Yn).float().cuda(), torch.from_numpy(lown).cuda()
TR = torch.from_numpy(tr).cuda()
print(wh, seed, 'cube labels', len(tr), 'of them above the table', int((~lown[tr]).sum()), 'held out', len(te), 'test frames', len(Ht), f'{time.time() - t0:.0f}s', flush=True)


def run(arm):
    torch.manual_seed(seed)
    sc = torch.nn.Linear(128, 1).cuda()
    sc.weight.data, sc.bias.data = model.score.weight.data[GRP:GRP + 1].clone(), model.score.bias.data[GRP:GRP + 1].clone()
    sh = torch.nn.Linear(128, 2).cuda()
    torch.nn.init.zeros_(sh.weight); torch.nn.init.zeros_(sh.bias)

    def centre(h):                                                          # h [b, 1024, 128] fp16 → [b, 2], largest score, mean shift in patches
        h = h.float()
        s = sc(h)[..., 0]
        w = torch.softmax(s, 1)
        d = torch.tanh(sh(h)) * BOUND if arm in ('off', 'both') else torch.zeros(1, 1, 2, device=h.device)
        return w @ GRID + (w[..., None] * d).sum(1), s.max(1).values, (w[..., None] * d).sum(1).norm(dim=-1) / BOUND

    def allc(Hx):
        with torch.no_grad():
            o = [centre(Hx[a:a + 256]) for a in range(0, len(Hx), 256)]
        return torch.cat([x[0] for x in o]), torch.cat([x[1] for x in o]), torch.cat([x[2] for x in o])

    def maps():
        c = allc(H)[0]
        return torch.stack([torch.linalg.lstsq(q2(c[TR][m]), Y[TR][m]).solution for m in (~low[TR], low[TR])])   # [2 (above, table), 6, 3]

    B = maps()
    train = [p for n, m in (('tune', sc), ('off', sh)) if arm in (n, 'both') for p in m.parameters()]
    if train:
        opt = torch.optim.Adam(train, 1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
        for s in range(STEPS):
            b = TR[torch.randint(len(TR), (64,), device='cuda')]
            out = torch.einsum('bq,bqd->bd', q2(centre(H[b])[0]), B[low[b].long()])
            loss = ((out - Y[b]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if s % 1000 == 0:
                print(arm, s, round(loss.item(), 5), flush=True)
        B = maps()
    c, _, _ = allc(H)
    ct, top, shift = allc(Ht)

    def answer(cx):                                                         # the height cluster of the nearest labelled centres
        near = torch.cdist(cx, c[TR]).topk(5, largest=False).indices
        lo = low[TR][near].float().mean(1) > 0.5
        a_lo, a_hi = q2(cx) @ B[1], q2(cx) @ B[0]
        return torch.where(lo[:, None], a_lo, a_hi), a_lo, a_hi, lo
    a, a_lo, a_hi, lo = answer(ct)
    TE = torch.from_numpy(te).cuda()
    ao = answer(c[TE])[0]
    cpu = lambda x: x.cpu().numpy()
    np.savez(ROOT / f'runs/144/fit_{arm}_{wh}_{seed}.npz', aim=cpu(a) / 10 + C.CEN, aim_low=cpu(a_lo) / 10 + C.CEN, aim_high=cpu(a_hi) / 10 + C.CEN,
             chose_table=cpu(lo), top=cpu(top), shift_patches=cpu(shift), own_err_cm=cpu((ao - Y[TE])[:, :2].norm(dim=-1)) * 10, own_low=lown[te])
    torch.save(dict(score=sc.state_dict(), shift=sh.state_dict(), B=B.cpu(), centres=c[TR].cpu(), low=low[TR].cpu()), ROOT / f'runs/144/fit_{arm}_{wh}_{seed}.pt')
    print(arm, 'done', f'{time.time() - t0:.0f}s', flush=True)


for arm in (sys.argv[3:] or ['map', 'off', 'tune', 'both']):
    run(arm)
