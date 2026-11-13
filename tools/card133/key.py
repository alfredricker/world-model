"""Card 133: a recall key learned from the error of what recall returned.

Test bed of card 132 (true visible state, narrow view), with the touch's effect as what is recalled: how much the
touch changed the cube's place, each slide and each lock. A pull "moved" if the recalled change of a slide is over
1 cm; the cube "was inside" if the recalled change of the cube on a reopening is over 5 cm. Recall: 5 nearest
stored events, the median of their effects. A variable the touch changes is seen after it.
Arms (what the distance is taken over):
  fixed    card 132's weights over [contact point, touched object, last seen values and their "seen" flags]
  truth    contact point, touched object, the true locks and whether the cube is truly in the drawer
  weights  one learned weight per item of the fixed key (starts at the fixed weights)
  network  a small network over the same items
Training (store only): each stored event's effect is predicted as the softmax(-distance^2)-weighted mean of the
effects of events in other episodes; squared error; Adam.

  bin/prun python tools/card133/key.py [--episodes 400] [--steps 3000] → runs/133/key_<episodes>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card132'))
import bed as B
import check as C

ROOT, U, K = C.ROOT, 0.02, 5
W_FIXED = np.r_[np.ones(3), 10 * np.ones(6), 3, 1, 1, 1, 3, 1, 3, 1, 3, 3, 3, 3]          # card 132's key
LOCK, CUBE = slice(17, 21), slice(9, 13)                                                    # places of the remembered locks and cube in the key


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def events(eps, tg, ch, lk):
    X, T, Y, meta = [], [], [], dict(pull=[], moved=[], reopen=[], cube_in=[], ep=[], mixed=[], seen_cube=[], slide=[])
    for e, ((s, h), t, c) in enumerate(zip(eps, tg, ch)):
        ins = C.inside(s)
        b = dict(cube=None, drawer=None, window=None, l0=None, l1=None)
        for i in range(len(t)):
            k = t[i]
            head = np.r_[h[i] / U, 1.0 * (np.arange(-1, 5) == k)]
            f = []
            for name, n in (('cube', 3), ('drawer', 1), ('window', 1)):
                f += [float(b[name] is not None), *(np.zeros(n) if b[name] is None else np.atleast_1d(b[name]) / U)]
            for name in ('l0', 'l1'):
                f += [float(b[name] is not None), 0.0 if b[name] is None else float(b[name])]
            X.append(np.r_[head, f]); T.append(np.r_[head, s[i, 5:7], float(ins[i])])
            Y.append(np.r_[(s[i + 1, :3] - s[i, :3]) / U, (s[i + 1, 3:5] - s[i, 3:5]) / U, 3 * (s[i + 1, 5:7] - s[i, 5:7])])
            mixed = k in (1, 2) and bool(c[i, 3] or c[i, 4])
            meta['mixed'].append(mixed); meta['pull'].append(k in (1, 2) and not mixed); meta['moved'].append(bool(c[i, 1] or c[i, 2]))
            meta['reopen'].append(abs(s[i, 3]) < 0.03 and abs(s[i + 1, 3]) > 0.10); meta['cube_in'].append(bool(ins[i]))
            meta['ep'].append(e); meta['seen_cube'].append(b['cube'] is not None); meta['slide'].append(k if k in (1, 2) else 1)
            if k == 0 or c[i, 0]:
                b['cube'] = s[i + 1, :3]
            if k == 1 or c[i, 1]:
                b['drawer'] = s[i + 1, 3]; b[f'l{lk[1]}'] = s[i + 1, 5 + lk[1]]
            if k == 2 or c[i, 2]:
                b['window'] = s[i + 1, 4]; b[f'l{lk[2]}'] = s[i + 1, 5 + lk[2]]
            for j in (0, 1):
                if k == 3 + j or c[i, 3 + j]:
                    b[f'l{j}'] = s[i + 1, 5 + j]
    f32 = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    return f32(X), f32(T), f32(Y), {k: np.array(v) for k, v in meta.items()}


def recall(Es, Ys, Eq):
    idx = torch.cdist(Eq, Es).topk(K, largest=False).indices
    return Ys[idx].median(1).values                                                     # [Q,7]


def score(R, m, sel=None):
    """Balanced accuracy on pulls and reopenings from recalled effects R (one row per test event)."""
    R = R.numpy()
    slide = np.abs(R[np.arange(len(R)), 2 + m['slide']]) > 0.5                           # over 1 cm
    cube = np.linalg.norm(R[:, :3], axis=1) > 2.5                                       # over 5 cm
    out = {}
    for name, q, pred, y in (('pull', m['pull'], slide, m['moved']), ('reopen', m['reopen'], cube, m['cube_in']),
                             ('reopen_seen', m['reopen'] & m['seen_cube'], cube, m['cube_in']),
                             ('reopen_not_seen', m['reopen'] & ~m['seen_cube'], cube, m['cube_in'])):
        v = B.bal(pred[q], y[q])
        out[name] = None if v is None else round(100 * v, 1)
    return out


def train(X, Y, ep, kind, steps, seed=0):
    torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, Y, ep = X.to(dev), Y.to(dev), torch.tensor(ep, device=dev)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    if kind == 'weights':
        logw = torch.tensor(np.log(W_FIXED), dtype=torch.float32, device=dev, requires_grad=True)
        params, f = [logw], lambda x: x * logw.exp()
    else:
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 16)).to(dev)
        params, f = list(net.parameters()), lambda x: net((x - mu) / sd)
    opt = torch.optim.Adam(params, lr=3e-2 if kind == 'weights' else 1e-3)
    for step in range(steps):
        q = torch.randint(0, len(X), (512,), device=dev)
        d = torch.cdist(f(X[q]), f(X)) ** 2
        d = d.masked_fill(ep[q][:, None] == ep[None], float('inf'))
        loss = ((torch.softmax(-d, 1) @ Y - Y[q]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(kind, step, round(loss.item(), 3), flush=True)
    g = lambda x: f(x.to(dev)).detach().cpu()
    return g, (logw.exp().detach().cpu().numpy().round(2).tolist() if kind == 'weights' else None)


def main():
    n_ep, steps = arg('--episodes', 400), arg('--steps', 3000)
    mem, test = C.load(C.MEM), C.load(C.TEST)
    tgm, chm, _ = C.targets(mem)
    tgt, cht, _ = C.targets(test)
    lk, _ = B.lock_of(mem, tgm, chm)
    Xm, Tm, Ym, mm = events(mem, tgm, chm, lk)
    Xt, Tt, Yt, mt = events(test, tgt, cht, lk)
    keep = ~mm['mixed'] & np.isin(mm['ep'], np.random.default_rng(0).choice(400, n_ep, replace=False))
    Xm, Tm, Ym, epm = Xm[keep], Tm[keep], Ym[keep], mm['ep'][keep]
    # the gate's check: how far the cube moves on a reopening with the cube inside
    q = mt['reopen'] & mt['cube_in']
    out = dict(episodes=n_ep, stored_events=int(keep.sum()), cube_change_on_full_reopen_cm=np.round(2 * np.percentile(Yt[q, :3].norm(dim=1).numpy(), [0, 10, 50]), 1).tolist(),
               n={k: int(mt[k].sum()) for k in ('pull', 'reopen')} | dict(reopen_seen=int((mt['reopen'] & mt['seen_cube']).sum())), arms={})
    wf = torch.tensor(W_FIXED, dtype=torch.float32)
    wt = torch.tensor(np.r_[np.ones(3), 10 * np.ones(6), 3, 3, 3], dtype=torch.float32)
    out['arms']['fixed'] = score(recall(Xm * wf, Ym, Xt * wf), mt)
    out['arms']['truth'] = score(recall(Tm * wt, Ym, Tt * wt), mt)
    print(json.dumps(out, indent=1), flush=True)
    if '--gate' not in sys.argv:
        rng = np.random.default_rng(1)
        for kind in ('weights', 'network'):
            g, w = train(Xm, Ym, epm, kind, steps)
            Es = g(Xm)
            r = score(recall(Es, Ym, g(Xt)), mt)
            for name, sl in (('lock_shuffled', LOCK), ('cube_shuffled', CUBE)):
                Xs = Xt.clone(); Xs[:, sl] = Xt[torch.tensor(rng.permutation(len(Xt))), sl]
                r[name] = score(recall(Es, Ym, g(Xs)), mt)['pull']
            if w:
                r['learned_weights'] = w
            out['arms'][kind] = r
            print(kind, json.dumps(r), flush=True)
    json.dump(out, open(ROOT / f'runs/133/key_{n_ep}{"_gate" if "--gate" in sys.argv else ""}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
