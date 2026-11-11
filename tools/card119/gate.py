"""Card 119, feasibility gate: lasting change per patch, on features and masks already stored (no training).

Signal, for patch p at frame tau (every frame, per episode):
  before = frames tau-G-W .. tau-G-1, after = frames tau+G+1 .. tau+G+W   (W = 10 stable frames, G = 5 frames of gap)
  s = |mean(after) - mean(before)|^2 / (var(after) + var(before) + eps)
  var = mean squared distance to the window's own mean; eps = the median of var(after) + var(before) over episodes
  900-999. A patch the arm crosses and leaves scores low (before = after); a patch the arm is moving on scores low
  (unstable windows); a patch whose content switched between two steady states scores high.

Detectors: lasting change; lasting change with the arm's patches left out by the true mask (upper bound); card
115's plain change, with and without the true arm mask; random. Frame score, peaks and thresholds as in card 115
(mean of the 4 largest patches, highest within +-5 frames, threshold by best F1 on episodes 900-999).

Two scorings on validation episodes, both in time and place (the boundary's top patch on the changing object, one
patch of slack):
  events      card 115's list (lock flips; starts and stops of drawer, window, cube), one to one, +-5 frames;
  transitions each run of motion is one switch from rest to rest (a lock flip is one too): a boundary is right if
              it falls inside some transition's interval (+-5) on its object; a transition is found if any does.

  bin/prun python tools/card119/gate.py [--features dino|pixels] [--eps-quantile q] [--detectors a,b]
  → runs/119/gate_<features>_q<q>.json (signals cached in runs/119/signal_*.npz)

--features pixels is the control: the same signal on the raw 64 x 64 frame, resized to 56 x 56, one 4 x 4 x 3 block per
patch of the same 14 x 14 grid.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card115'))
from masks import OBJECTS  # noqa: E402
from score import EP, SKIP, TOL, frame_score, peaks, runs, true_events, zscore  # noqa: E402
from diagnose import dilated  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/119'
W, G = 10, 5
LO, HI = G + W, EP - 1 - G - W          # frames of an episode where the signal exists
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def lasting(feat, arm):
    """feat [n, 196, 384] (memmap), arm [n, 196] bool → num, den [n, 196] (nan where no signal), arm share [n, 196]."""
    n = len(feat)
    num = np.full((n, 196), np.nan, np.float32)
    den = np.full((n, 196), np.nan, np.float32)
    share = np.ones((n, 196), np.float32)
    tau = torch.arange(LO, HI + 1, device=DEV)
    for e in range(n // EP):
        x = torch.from_numpy(np.asarray(feat[e * EP:(e + 1) * EP], np.float32)).to(DEV)
        if x.ndim == 4:      # raw 64 x 64 frames: resize to 56 x 56, one 4 x 4 x 3 block per patch of the 14 x 14 grid
            x = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=56, mode='bilinear', antialias=True)
            x = torch.nn.functional.unfold(x, 4, stride=4).transpose(1, 2)
        x = x.double()
        S = torch.cat([torch.zeros_like(x[:1]), x.cumsum(0)])
        Q = torch.cat([torch.zeros_like(x[:1, :, 0]), (x * x).sum(-1).cumsum(0)])
        mb = (S[tau - G] - S[tau - G - W]) / W
        ma = (S[tau + G + W + 1] - S[tau + G + 1]) / W
        vb = (Q[tau - G] - Q[tau - G - W]) / W - (mb * mb).sum(-1)
        va = (Q[tau + G + W + 1] - Q[tau + G + 1]) / W - (ma * ma).sum(-1)
        sl = slice(e * EP + LO, e * EP + HI + 1)
        num[sl] = ((ma - mb) ** 2).sum(-1).float().cpu().numpy()
        den[sl] = (va + vb).clamp(min=0).float().cpu().numpy()
        a = torch.from_numpy(arm[e * EP:(e + 1) * EP].astype(np.float64)).to(DEV)
        A = torch.cat([torch.zeros_like(a[:1]), a.cumsum(0)])
        sb = (A[tau - G] - A[tau - G - W]) / W
        sa = (A[tau + G + W + 1] - A[tau + G + 1]) / W
        share[sl] = torch.maximum(sa, sb).float().cpu().numpy()
    return num, den, share


def transitions(s):
    """→ start [m], stop [m], kind [m], objects [m, 6]: one row per switch from rest to rest."""
    n = len(s['buttons'])
    ep = np.arange(n) // EP
    same = ep[1:] == ep[:-1]
    A, B, K, O = [], [], [], []
    for f in np.where((s['buttons'][1:] != s['buttons'][:-1]).any(1) & same)[0] + 1:
        b = int(np.argmax(s['buttons'][f] != s['buttons'][f - 1]))
        o = np.zeros(6, bool)
        o[OBJECTS.index(f'button{b}')] = o[OBJECTS.index('drawer' if b == 0 else 'window')] = True
        A.append(f); B.append(f); K.append('lock'); O.append(o)
    for name, v in [('drawer', s['drawer'][:, None]), ('window', s['window'][:, None]), ('cube', s['cube'])]:
        moving = (np.linalg.norm(np.diff(v, axis=0), axis=1) > 0.002) & same
        a, b = runs(moving)
        st, en = [], []
        for x, y in zip(a, b):
            if st and x - en[-1] < 3 and x // EP == st[-1] // EP:
                en[-1] = y
            else:
                st.append(x); en.append(y)
        for x, y in zip(st, en):
            if y - x < 3:
                continue
            o = np.zeros(6, bool)
            o[OBJECTS.index(name)] = True
            A.append(x + 1); B.append(y); K.append(name); O.append(o)
    order = np.argsort(A)
    return np.array(A)[order], np.array(B)[order], np.array(K)[order], np.array(O)[order]


def in_range(t):
    return (t % EP >= LO) & (t % EP <= HI)


def score_events(S, bd, top):
    """Card 115's diagnostic: one to one, +-TOL, top patch on the event's object."""
    ev = S['T']
    pairs = []
    lo, hi = np.searchsorted(bd, ev - TOL), np.searchsorted(bd, ev + TOL, side='right')
    for i, (a, b) in enumerate(zip(lo, hi)):
        for j in range(a, b):
            t = bd[j]
            if t // EP == ev[i] // EP and dilated(S['masks'][t], S['masks'][t - SKIP], S['O'][i])[top[j]]:
                pairs.append((abs(int(t) - int(ev[i])), i, j))
    pairs.sort()
    me, mb = np.zeros(len(ev), bool), np.zeros(len(bd), bool)
    for _, i, j in pairs:
        if not me[i] and not mb[j]:
            me[i] = mb[j] = True
    p, r = (mb.mean() if len(bd) else 0.0), me.mean()
    return (2 * p * r / (p + r) if p + r else 0.0), p, r, me


def score_trans(S, bd, top):
    a, b = S['A'], S['B']
    found, right = np.zeros(len(a), bool), np.zeros(len(bd), bool)
    for j, t in enumerate(bd):
        cand = np.where((a - TOL <= t) & (t <= b + TOL) & (a // EP == t // EP))[0]
        for i in cand:
            if dilated(S['masks'][t], S['masks'][t - SKIP], S['TO'][i])[top[j]]:
                found[i] = right[j] = True
    p, r = (right.mean() if len(bd) else 0.0), found.mean()
    return (2 * p * r / (p + r) if p + r else 0.0), p, r, found


def main():
    OUT.mkdir(exist_ok=True)
    arm = OBJECTS.index('arm')
    sets = {}
    for name, sfile, cut in [('tail', 'state_train.npz', 900 * EP), ('val', 'state_val.npz', 0)]:
        s = dict(np.load(ROOT / 'runs/114' / sfile))
        s = {k: v[cut:cut + 100 * EP] for k, v in s.items() if len(v) == len(s['buttons'])}
        masks = np.load(ROOT / f'runs/115/masks_{name}.npy').astype(bool)
        PIX = arg('--features', 'dino') == 'pixels'
        if PIX:
            obs = np.load(ROOT / ('runs/114/obs_visual-scene-play-v0' + ('-val' if name == 'val' else '') + '.npy'), mmap_mode='r')
            feat = obs[cut:cut + 100 * EP]
        else:
            feat = np.load(ROOT / f'runs/114/feat_{name}.npy', mmap_mode='r')
        ft = arg('--features', 'dino')
        cache = OUT / f"signal_{'pix_' if PIX else '' if ft == 'dino' else ft + '_'}{name}.npz"
        if cache.exists():
            c = np.load(cache); num, den, share = c['num'], c['den'], c['share']
        else:
            num, den, share = lasting(feat, masks[:, arm])
            np.savez(cache, num=num, den=den, share=share)
        T, Kd, O = true_events(s)
        ok = in_range(T)
        A, B, TK, TO = transitions(s)
        ok2 = in_range(A) & in_range(B)
        chg = np.load(ROOT / f'runs/115/surprise_{name}.npz')['chg'].astype(np.float32)
        sets[name] = dict(masks=masks, num=num, den=den, share=share, chg=chg, T=T[ok], K=Kd[ok], O=O[ok],
                          A=A[ok2], B=B[ok2], TK=TK[ok2], TO=TO[ok2])
        print(name, 'events', int(ok.sum()), 'transitions', int(ok2.sum()), flush=True)
    tail, val = sets['tail'], sets['val']
    q = float(arg('--eps-quantile', '0.5'))
    eps = float(np.nanquantile(tail['den'], q))
    print('den quantiles', np.nanquantile(tail['den'], [.1, .5, .9, .99, .999]), 'num', np.nanquantile(tail['num'], [.5, .9, .99, .999]), flush=True)
    res = dict(settings=dict(W=W, G=G, eps=eps),
               events={k: int((val['K'] == k).sum()) for k in np.unique(val['K'])},
               transitions={k: int((val['TK'] == k).sum()) for k in np.unique(val['TK'])})

    def signal(S, d):
        if d.startswith('lasting'):
            z = S['num'] / (S['den'] + eps)
            if d.endswith('masked'):
                z = np.where(S['share'] < 0.5, z, np.nan)
        else:
            z = zscore(S['chg'], tail['chg'])
            if d.endswith('masked'):
                am = S['masks'][:, arm]
                z = np.where(am | np.r_[np.zeros((SKIP, 196), bool), am[:-SKIP]], np.nan, z)
        z = np.where(in_range(np.arange(len(z)))[:, None], z, np.nan)
        sc = frame_score(np.where(np.isnan(z), -np.inf, z))
        pk = peaks(sc)
        top = np.argmax(np.nan_to_num(z[pk], nan=-np.inf), 1)
        return z, sc, pk, top

    for d in arg('--detectors', 'lasting,lasting_arm_masked,change,change_arm_masked').split(','):
        zt, sct, pkt, topt = signal(tail, d)
        zv, scv, pkv, topv = signal(val, d)
        cands = np.quantile(sct[pkt], np.linspace(0, 0.995, 80))
        entry = {}
        for tag, fn, kinds in [('events', score_events, val['K']), ('transitions', score_trans, val['TK'])]:
            thr = max((fn(tail, pkt[sct[pkt] > c], topt[sct[pkt] > c])[0], c) for c in cands)[1]
            sel = scv[pkv] > thr
            F, P, R, hit = fn(val, pkv[sel], topv[sel])
            entry[tag] = dict(f1=round(F, 4), precision=round(float(P), 4), recall=round(float(R), 4),
                              boundaries=int(sel.sum()),
                              recall_by_kind={k: round(float(hit[kinds == k].mean()), 4) for k in np.unique(kinds)})
            if d == 'lasting':
                rng = np.random.default_rng(0)
                valid = np.where(np.isfinite(scv))[0]
                n = int(sel.sum())
                rf = [fn(val, np.sort(rng.choice(valid, n, replace=False)), rng.integers(0, 196, n))[0]
                      for _ in range(3)]
                res[f'random_{tag}_f1'] = round(float(np.mean(rf)), 4)
                on_arm = val['masks'][pkv[sel], arm, topv[sel]]
                entry[tag]['boundaries_top_patch_on_arm'] = round(float(on_arm.mean()), 4)
        res[d] = entry
        print(d, json.dumps(entry), flush=True)
    json.dump(res, open(OUT / f"gate_{arg('--features', 'dino')}_q{q}.json", 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k.startswith('random')}))


if __name__ == '__main__':
    main()
