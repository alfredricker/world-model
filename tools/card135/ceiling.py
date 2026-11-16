"""Card 135, gate: what can be had from the agent's own information? On the store only, a fifth of the episodes held
out in turn, pulls with the lock touched earlier. (1) recall with the plain record and with the time item;
(2) a small network trained on the slide's record, its lock's record and the contact point, with and without the
two time items, to predict whether the pull moves (a reference for what the information allows).

  bin/prun python tools/card135/ceiling.py → runs/135/ceiling.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift as D
from drift import B, C, E, F

mem = C.load(C.MEM)
tgm, chm, _ = C.targets(mem)
lk, _ = B.lock_of(mem, tgm, chm)
out = {}


def net(Xa, ya, Xq, seed):
    torch.manual_seed(seed)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    A, Q, y = torch.tensor((Xa - mu) / sd), torch.tensor((Xq - mu) / sd), torch.tensor(ya, dtype=torch.float32)
    f = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(f.parameters(), 3e-3, weight_decay=1e-2)
    for _ in range(400):
        opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(f(A)[:, 0], y).backward(); opt.step()
    return torch.sigmoid(f(Q)[:, 0]).detach().numpy()


for p in D.PS[1:]:
    R, T, Y, tg, m = D.events(mem, tgm, chm, lk, p, 0)
    o = {}
    for a in ('plain', 'time'):
        r = D.arm(R, T, a)
        pull, cube, prob = np.zeros(len(tg), bool), np.zeros(len(tg), bool), np.zeros(len(tg))
        for f in range(D.FOLDS):
            tr, te = np.where(~m['mixed'] & (m['ep'] % D.FOLDS != f))[0], np.where(m['ep'] % D.FOLDS == f)[0]
            (a_, b_, c_), _ = D.run(r, tg, Y, m['ep'], tr, {k: v[te] for k, v in r.items()}, tg[te], m['slide'][te])
            pull[te], cube[te], prob[te] = a_, b_, c_
        s = D.score(pull, cube, prob, m)
        o['recall_' + a] = {k: s[k] for k in ('lock_touched', 'lock_touched_logloss', 'pull_only', 'reopen', 'by_elapsed')}
    q = np.where(m['pull'] & (m['group'] == 0))[0]
    own = np.where((m['slide'][q] == 1)[:, None], R['drawer'][q], R['window'][q])
    li = np.where(m['slide'][q] == 1, lk[1], lk[2])
    lock = np.where((li == 0)[:, None], R['l0'][q], R['l1'][q])
    X = np.c_[m['slide'][q] == 1, R['contact'][q], own, lock].astype(np.float32)
    notime = np.ones(X.shape[1], bool); notime[[4 + own.shape[1] - 1, X.shape[1] - 1]] = False
    y = m['moved'][q]
    for name, cols in (('net_plain', notime), ('net_time', np.ones(X.shape[1], bool))):
        ps = []
        for seed in range(3):
            pr = np.zeros(len(q))
            for f in range(D.FOLDS):
                te = m['ep'][q] % D.FOLDS == f
                pr[te] = net(X[~te][:, cols], y[~te], X[te][:, cols], seed)
            ps.append(pr)
        pr = np.mean(ps, 0)
        o[name] = dict(logloss=round(E.logloss(y, pr), 3), bal=round(100 * B.bal(pr > 0.5, y), 1),
                       by_elapsed={f'{lo}-{hi}': round(E.logloss(y[(m['el'][q] >= lo) & (m['el'][q] <= hi)], pr[(m['el'][q] >= lo) & (m['el'][q] <= hi)]), 3) for lo, hi in D.BINS})
    base = y.mean(); o['base_rate_logloss'] = round(E.logloss(y, np.full(len(y), base)), 3); o['n'] = [int(len(y)), int((~y).sum())]
    out[str(p)] = o
    print(p, json.dumps(o), flush=True)
json.dump(out, open(C.ROOT / 'runs/135/ceiling.json', 'w'), indent=1)
