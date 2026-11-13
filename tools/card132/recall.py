"""Card 132, second gate: memory first, the network as the prior (version 27's rule carried to prediction).

To predict the frame before the next touch:
- recall: the 5 remembered steps (training sets' events, kept as they are) nearest the present one. The key is
  the touch (gripper position at contact and at release, in cm) and, with --hist 1, what the agent has left where:
  the set of points where its hand released something after carrying it at least 5 cm, each kept until the hand
  next makes contact within 3 cm of it (version 27's "a place keeps the last token seen there", from the agent's
  own actions). Sets are compared by the mean nearest distance both ways (capped at 30 cm; one empty: 30 cm).
- the recalled outcome: the patches that changed in at least 3 of the 5 recalled steps (each step's 50 most
  changed) take the mean of the recalled next frames; every other patch stays as it is now.
- blend with the network (gate.py --tag net, no carried state): P = (n * recalled + BETA * network) / (n + BETA),
  n = how many of the 5 lie within the usual nearest-neighbour distance in memory (median, leave-one-out).
Scores as gate.py: error on each step's 20 most changed patches; the reopened drawer.

  bin/prun python tools/card132/recall.py --hist 1 [--net 1] → runs/132/recall_hist1.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('g132', HERE / 'gate.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
OUT, arg = G.OUT, G.arg
BETA, KN, CAP = 1.0, 5, 3.0                                                 # positions are in units of 10 cm


def left_behind(a):
    """a [T,9] raw touches → for each touch, the points left before it [T,P,3] and a mask."""
    pts, out = [], []
    for t in range(len(a)):
        out.append(list(pts))
        con, rel = a[t, :3], a[t, 3:6]
        pts = [p for p in pts if np.linalg.norm(p - con) > 0.3]
        if np.linalg.norm(rel - con) > 0.5:
            pts.append(rel)
    P = max(1, max(len(o) for o in out))
    S, M = np.zeros((len(a), P, 3), np.float32), np.zeros((len(a), P), bool)
    for t, o in enumerate(out):
        for j, p in enumerate(o):
            S[t, j], M[t, j] = p, True
    return S, M


def flat(eps):
    """Episodes → every step (t → t+1) as flat arrays."""
    x, y, a, S, M, lab = [], [], [], [], [], []
    P = max(left_behind(b['a'].cpu().numpy())[0].shape[1] for b in eps)
    for b in eps:
        raw = b['a'].cpu().numpy()
        s, m = left_behind(raw)
        s, m = np.pad(s, ((0, 0), (0, P - s.shape[1]), (0, 0))), np.pad(m, ((0, 0), (0, P - m.shape[1])))
        n = len(raw) - 1
        x.append(b['z'][:-1]); y.append(b['z'][1:]); a.append(raw[:n, :6]); S.append(s[:n]); M.append(m[:n])
        d, ins = b['drawer'], G.inside(b['cube'])
        lab.append(np.where((d[:-1] > -0.03) & (d[1:] < -0.10), ins[1:].astype(int), -1))     # -1: not a reopening
    c = lambda v: torch.from_numpy(np.concatenate(v)).cuda()
    return torch.cat(x), torch.cat(y), c(a), c(S), c(M), np.concatenate(lab)


def setdist(S, M, s, m):
    """Memory sets [N,P,3], masks [N,P] against one set s [Q,3], m [Q] → [N]."""
    D = (S[:, :, None] - s[None, None]).norm(dim=-1).clamp(max=CAP)         # [N,P,Q]
    big = torch.full_like(D, CAP)
    a = torch.where(m[None, None], D, big).min(2).values                    # each memory point to its nearest present point
    b = torch.where(M[:, :, None], D, big).min(1).values                    # each present point to its nearest memory point
    na, nb = M.sum(1), m.sum()
    da = torch.where(na > 0, (a * M).sum(1) / na.clamp(min=1), torch.zeros_like(na, dtype=a.dtype))
    db = (b * m[None]).sum(1) / nb.clamp(min=1) if nb > 0 else torch.zeros(len(S), device=S.device)
    return (da + db) / 2


def main():
    hist, usenet = int(arg('--hist', 1)), int(arg('--net', 0))
    tr, te = G.load(G.TRAIN), G.load(G.TEST)
    X, Y, A, S, M, _ = flat(tr)
    x, y, a, s, m, lab = flat(te)
    print('remembered steps', len(X), 'test steps', len(x), 'reopenings', int((lab >= 0).sum()), flush=True)
    dist = lambda i, A2, S2, M2, a2, s2, m2: ((A2[:, :3] - a2[i, :3]).norm(dim=1) + (A2[:, 3:] - a2[i, 3:]).norm(dim=1)
                                             + (setdist(S2, M2, s2[i], m2[i]) if hist else 0))
    nn1 = []
    for i in np.random.default_rng(0).choice(len(X), 400, replace=False):
        d = dist(i, A, S, M, A, S, M); d[i] = 1e9
        nn1.append(float(d.min()))
    radius = float(np.median(nn1))
    net = None
    if usenet:
        ck = torch.load(OUT / 'model_net.pt')
        net = G.Model(ck['mem']).cuda(); net.load_state_dict(ck['state']); net.eval()
        tmu, tsd = ck['mu'], ck['sd']
        raw = torch.cat([b['a'][:-1] for b in te])
    # the rule for "the cube shows in the open drawer", as gate.py
    zs = torch.cat([b['z'] for b in tr]).float()
    cube, drawer = np.concatenate([b['cube'] for b in tr]), np.concatenate([b['drawer'] for b in tr])
    w = torch.from_numpy((drawer < -0.10) & G.inside(cube)).cuda(); wo = torch.from_numpy((drawer < -0.10) & ~G.inside(cube)).cuda()
    cw, co = zs[w].mean(0), zs[wo].mean(0)
    top = (cw - co).norm(dim=1).topk(20).indices
    says = lambda f: float(((f[top] - co[top]) ** 2).sum() - ((f[top] - cw[top]) ** 2).sum())
    del zs
    names = ['recall'] + (['net', 'blend'] if usenet else [])
    err, sc, ns = {k: [] for k in names + ['copy']}, {k: [] for k in names + ['real', 'copy']}, []
    with torch.no_grad():
        for i in range(len(x)):
            d = dist(i, A, S, M, a, s, m)
            dv, nb = d.topk(KN, largest=False)
            xi, yi = x[i].float(), y[i].float()
            Xn, Yn = X[nb].float(), Y[nb].float()
            votes = torch.zeros(1024, device='cuda')
            votes.scatter_add_(0, ((Yn - Xn) ** 2).mean(2).topk(50, dim=1).indices.reshape(-1), torch.ones(KN * 50, device='cuda'))
            pr = torch.where((votes >= 3)[:, None], Yn.mean(0), xi)
            pred = dict(recall=pr)
            if usenet:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pn, _ = net.step(xi[None], ((raw[i] - tmu) / tsd)[None], torch.zeros(1, G.K, G.W, device='cuda'))
                n = float((dv <= radius).sum())
                pred.update(net=pn[0].float(), blend=(n * pr + BETA * pn[0].float()) / (n + BETA))
                ns.append(n)
            k = ((yi - xi) ** 2).mean(1).topk(20).indices
            for name, p in pred.items():
                err[name].append(float(((p[k] - yi[k]) ** 2).mean()))
            err['copy'].append(float(((xi[k] - yi[k]) ** 2).mean()))
            if lab[i] >= 0:
                for name, p in list(pred.items()) + [('real', yi), ('copy', xi)]:
                    sc[name].append(says(p))
    L = lab[lab >= 0].astype(bool)
    res = dict(hist=hist, radius_cm=round(radius * 10, 1), reopenings=dict(with_cube=int(L.sum()), without=int((~L).sum())),
               changed_patches_mse={k: round(float(np.mean(v)), 3) for k, v in err.items()},
               mean_matches_within_radius=round(float(np.mean(ns)), 2) if ns else None)
    for k, v in sc.items():
        v = np.array(v)
        res[f'reopen_{k}'] = dict(auc=round(float((v[L][:, None] > v[~L][None]).mean()), 3), says_cube_when_there=round(float((v[L] > 0).mean()), 3),
                                  says_cube_when_not=round(float((v[~L] > 0).mean()), 3), score_with=round(float(v[L].mean()), 1), score_without=round(float(v[~L].mean()), 1))
    json.dump(res, open(OUT / f'recall_hist{hist}{"_net" if usenet else ""}.json', 'w'), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
