"""Card 120: what differs between just before a touch and just after it, on DINOv2 block 2.

Data: OGBench's state dataset scene-play-v0 (contact = observation 18), frames and masks rendered by render.py.
A touch: a run of frames with contact > 0.05, gaps under 3 frames merged, inside one episode.
Signal per touch and patch: |mean(after) - mean(before)|^2 / (var(after) + var(before) + eps), with
before = the 10 frames ending 2 before the touch begins, after = the 10 frames starting 2 after it ends;
eps = the median denominator over episodes 900-999's touches.

Scored on validation episodes, per true transition inside a touch (overlapping it, +-5 frames): found if one of the
touch's 4 highest patches lies on its object (the object's patches just before or just after the touch, one patch of
slack). Compared: the same with the arm's patches removed by the true mask (upper bound); card 119's fixed-window
signal at its best frame inside the transition; the touch's windows moved to a random time in the episode; the
touch with 4 random patches. Gate step 1: card 119's thresholded transitions F1 on this dataset.

  bin/prun python tools/card120/touch.py → runs/120/touch.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card115'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card119'))
from masks import OBJECTS  # noqa: E402
from score import frame_score, peaks, runs  # noqa: E402
from diagnose import dilated  # noqa: E402
from gate import EP, HI, LO, SKIP, TOL, in_range, score_trans, transitions  # noqa: E402
from blocks import BLOCKS, DEV, encode, lasting_ep  # noqa: E402
from encode import load_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
OUT = ROOT / 'runs/120'
W, CLEAR, TOPK = 10, 2, 4
ARM = OBJECTS.index('arm')


def touches(c):
    a, b = runs(c > 0.05)                       # [a, b) frames in contact
    st, en = [], []
    for x, y in zip(a, b):
        if st and x - en[-1] < 3 and x // EP == st[-1] // EP:
            en[-1] = y
        else:
            st.append(x); en.append(y)
    st, en = np.array(st), np.array(en) - 1     # inclusive
    ok = (st // EP == en // EP) & (st % EP >= W + CLEAR - 1) & (en % EP <= EP - 1 - W - CLEAR + 1)
    return st[ok], en[ok], int((~ok).sum())


def bracket(x, s, e):
    """x [1001, 196, d] one episode; s, e local frames → num, den [196]."""
    b = x[s - CLEAR - W + 1:s - CLEAR + 1].double()
    a = x[e + CLEAR:e + CLEAR + W].double()
    mb, ma = b.mean(0), a.mean(0)
    den = ((b - mb) ** 2).sum(-1).mean(0) + ((a - ma) ** 2).sum(-1).mean(0)
    return ((ma - mb) ** 2).sum(-1).cpu().numpy(), den.cpu().numpy()


def arm_share(arm, s, e):
    return np.maximum(arm[s - CLEAR - W + 1:s - CLEAR + 1].mean(0), arm[e + CLEAR:e + CLEAR + W].mean(0))


def compute(name, enc):
    z = np.load(DATA / ('scene-play-v0-val.npz' if name == 'val' else 'scene-play-v0.npz'))
    cut = 0 if name == 'val' else 900 * EP
    c = z['observations'][cut:cut + 100 * EP, 18]
    s = dict(np.load(ROOT / f"runs/114/sstate_{'val' if name == 'val' else 'train'}.npz"))
    s = {k: v[cut:cut + 100 * EP] for k, v in s.items() if len(v) == len(s['buttons'])}
    rgb = np.load(OUT / f'rgb_{name}.npy', mmap_mode='r')
    masks = np.load(OUT / f'masks_{name}.npy').astype(bool)
    st, en, dropped = touches(c)
    n = len(c)
    num = np.full((n, 196), np.nan, np.float32)
    den = np.full((n, 196), np.nan, np.float32)
    T = dict(tnum=np.zeros((len(st), 196), np.float32), tden=np.zeros((len(st), 196), np.float32),
             rnum=np.zeros((len(st), 196), np.float32), rden=np.zeros((len(st), 196), np.float32),
             share=np.zeros((len(st), 196), np.float32))
    rng = np.random.default_rng(0)
    for ep in range(n // EP):
        x = encode(enc, rgb[ep * EP:(ep + 1) * EP])[BLOCKS.index(2)]
        sl = slice(ep * EP + LO, ep * EP + HI + 1)
        num[sl], den[sl] = lasting_ep(x)
        for i in np.where(st // EP == ep)[0]:
            ls, le = st[i] % EP, en[i] % EP
            T['tnum'][i], T['tden'][i] = bracket(x, ls, le)
            T['share'][i] = arm_share(masks[ep * EP:(ep + 1) * EP, ARM], ls, le)
            rs = int(rng.integers(W + CLEAR - 1, EP - 1 - W - CLEAR + 1 - (le - ls) + 1))
            T['rnum'][i], T['rden'][i] = bracket(x, rs, rs + le - ls)
    A, B, K, O = transitions(s)
    ok = in_range(A) & in_range(B)
    return dict(c=c, masks=masks, st=st, en=en, dropped=dropped, num=num, den=den, A=A[ok], B=B[ok], TK=K[ok],
                TO=O[ok], **T)


def fixed_signal(S, eps):
    z = S['num'] / (S['den'] + eps)
    z = np.where(in_range(np.arange(len(z)))[:, None], z, np.nan)
    sc = frame_score(np.where(np.isnan(z), -np.inf, z))
    pk = peaks(sc)
    return z, sc, pk, np.argmax(np.nan_to_num(z[pk], nan=-np.inf), 1)


def main():
    enc = load_encoder(DEV)
    sets = {}
    for nm in ['tail', 'val']:
        cache = OUT / f'computed_{nm}.npz'
        if cache.exists():
            sets[nm] = dict(np.load(cache, allow_pickle=True))
            sets[nm]['dropped'] = int(sets[nm]['dropped'])
        else:
            sets[nm] = compute(nm, enc)
            np.savez(cache, **sets[nm])
    tail, val = sets['tail'], sets['val']
    res = dict(touches=int(len(val['st'])), touches_dropped_at_episode_ends=val['dropped'])

    # gate step 1: card 119's signal, thresholded, on this dataset
    eps_f = float(np.nanmedian(tail['den']))
    zt, sct, pkt, topt = fixed_signal(tail, eps_f)
    zv, scv, pkv, topv = fixed_signal(val, eps_f)
    cands = np.quantile(sct[pkt], np.linspace(0, 0.995, 80))
    thr = max((score_trans(tail, pkt[sct[pkt] > c], topt[sct[pkt] > c])[0], c) for c in cands)[1]
    sel = scv[pkv] > thr
    F, P, R, hit = score_trans(val, pkv[sel], topv[sel])
    res['card119_on_this_dataset'] = dict(f1=round(F, 4), precision=round(float(P), 4), recall=round(float(R), 4),
                                          recall_by_kind={k: round(float(hit[val['TK'] == k].mean()), 4)
                                                          for k in np.unique(val['TK'])})

    # touch-bracketed
    eps = float(np.median(tail['tden']))
    S = val
    zt_ = S['tnum'] / (S['tden'] + eps)
    variants = {'touch': zt_, 'touch_arm_masked': np.where(S['share'] < 0.5, zt_, -np.inf),
                'random_time': S['rnum'] / (S['rden'] + eps)}
    rng = np.random.default_rng(1)
    kinds = np.unique(S['TK'])
    pairs = []                                   # (touch, transition)
    for i, (s_, e_) in enumerate(zip(S['st'], S['en'])):
        for j in np.where((S['A'] <= e_ + TOL) & (S['B'] >= s_ - TOL) & (S['A'] // EP == s_ // EP))[0]:
            pairs.append((i, j))
    inside = np.array(sorted(set(j for _, j in pairs)))
    res['transitions_inside_a_touch'] = {k: int((S['TK'][inside] == k).sum()) for k in kinds}
    res['transitions_all'] = {k: int((S['TK'] == k).sum()) for k in kinds}
    obj_mask = {}
    for i, j in pairs:
        obj_mask[(i, j)] = dilated(S['masks'][S['en'][i] + CLEAR], S['masks'][S['st'][i] - CLEAR], S['TO'][j])

    def found_by(top4):
        """top4 [touches, 4] patch ids → per transition: found by any touch it lies in."""
        f = {}
        for i, j in pairs:
            f[j] = f.get(j, False) or bool(obj_mask[(i, j)][top4[i]].any())
        js = np.array(sorted(f))
        v = np.array([f[j] for j in js])
        out = {k: round(float(v[S['TK'][js] == k].mean()), 4) for k in kinds}
        out['all'] = round(float(v.mean()), 4)
        return out

    for nm, zz in variants.items():
        top4 = np.argsort(-zz, 1)[:, :TOPK]
        entry = dict(found=found_by(top4))
        if nm == 'touch':
            top1 = top4[:, 0]
            has = np.zeros(len(S['st']), bool)
            right = np.zeros(len(S['st']), bool)
            for i, j in pairs:
                has[i] = True
                right[i] |= bool(obj_mask[(i, j)][top1[i]])
            entry['top_patch_on_a_changed_object'] = round(float(right[has].mean()), 4)
            arm_at = S['masks'][S['en'] + CLEAR, ARM] | S['masks'][S['st'] - CLEAR, ARM]
            entry['top_patch_on_arm'] = round(float(arm_at[np.arange(len(top1)), top1].mean()), 4)
            sc = np.sort(zz, 1)[:, -TOPK:].mean(1)
            pos, neg = sc[has], sc[~has]
            entry['touches_without_transition'] = int((~has).sum())
            entry['auc_with_vs_without_transition'] = round(float((pos[:, None] > neg[None, :]).mean()), 4)
        res[nm] = entry
    res['touch_random_patches'] = dict(found=found_by(rng.integers(0, 196, (len(S['st']), TOPK))))

    # card 119's signal at its best frame inside each transition (no threshold), same top-4 rule
    f = {}
    for j in sorted(set(j for _, j in pairs)):
        lo, hi = S['A'][j] - TOL, S['B'][j] + TOL
        seg = scv[lo:hi + 1]
        if not np.isfinite(seg).any():
            f[j] = False
            continue
        t = lo + int(np.argmax(seg))
        top4 = np.argsort(-np.nan_to_num(zv[t], nan=-np.inf))[:TOPK]
        f[j] = bool(dilated(S['masks'][t], S['masks'][t - SKIP], S['TO'][j])[top4].any())
    js = np.array(sorted(f)); v = np.array([f[j] for j in js])
    out = {k: round(float(v[S['TK'][js] == k].mean()), 4) for k in kinds}
    out['all'] = round(float(v.mean()), 4)
    res['card119_best_frame_no_threshold'] = dict(found=out)
    # after the run (diagnostic, not a criterion): the single top patch, exactly on the object (no slack; for a lock
    # flip the button itself, not the drawer or window whose handle also changes colour)
    def strict_mask(i, j, t0, t1):
        o = S['TO'][j].copy()
        if S['TK'][j] == 'lock':
            o[OBJECTS.index('drawer')] = o[OBJECTS.index('window')] = False
        return S['masks'][t0][o].any(0) | S['masks'][t1][o].any(0)

    def by_kind(f):
        js = np.array(sorted(f)); v = np.array([f[j] for j in js], float)
        out = {k: round(float(v[S['TK'][js] == k].mean()), 4) for k in kinds}
        out['all'] = round(float(v.mean()), 4)
        return out

    strict = {}
    for nm, zz in variants.items():
        top1 = np.argmax(zz, 1)
        f = {}
        for i, j in pairs:
            f[j] = f.get(j, False) or bool(strict_mask(i, j, S['st'][i] - CLEAR, S['en'][i] + CLEAR)[top1[i]])
        strict[nm] = by_kind(f)
    f = {}
    for i, j in pairs:                                           # chance: the mask's share of the 196 patches
        f[j] = max(f.get(j, 0.0), float(strict_mask(i, j, S['st'][i] - CLEAR, S['en'][i] + CLEAR).mean()))
    strict['touch_random_patch_expected'] = by_kind(f)
    f = {}
    for j in sorted(set(j for _, j in pairs)):
        lo, hi = S['A'][j] - TOL, S['B'][j] + TOL
        seg = scv[lo:hi + 1]
        if not np.isfinite(seg).any():
            f[j] = False
            continue
        t = lo + int(np.argmax(seg))
        f[j] = bool(strict_mask(None, j, t - SKIP, t)[int(np.nanargmax(zv[t]))])
    strict['card119_best_frame'] = by_kind(f)
    res['strict_top1_diagnostic'] = strict
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / 'touch.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
