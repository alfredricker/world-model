"""Card 123: read an event as its connected regions of change, and score the regions as objects.

Input: card 122's before/after-touch signal on DINOv2 block 2, 32 x 32 patches (runs/122/computed_<set>.npz).
z = tnum / (tden + eps), eps the median tden over episodes 900-999 (as cards 120 and 122).
Threshold: one number for all events, Otsu's split of log z pooled over every touch and patch of episodes 900-999
(no labels). A region is a 4-connected set of patches above it.

Scoring per true transition inside a touch, by IoU of the best region with the true patches of
- lock: the button; cube: where it was just before the touch, and where it is just after; drawer, window: the body
  (reported: card 119 found that a slide changes only the strip it sweeps).
Also: at a lock flip, are the button and the handle in different regions (one region mostly on the button, another
mostly on the drawer's or window's body)? How many regions per touch, and what share lie mostly on the arm?
Compared: the top patch alone; regions with the arm's patches removed by the true mask (bound); the best single
threshold chosen with labels on the scored set (bound).

  bin/prun python tools/card123/regions.py --set tail|val → runs/123/regions_<set>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card115'))
from masks import OBJECTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/123'
EP, TOL, CLEAR, G = 1001, 5, 2, 32
ARM, CUBE = OBJECTS.index('arm'), OBJECTS.index('cube')
BTN = [OBJECTS.index('button0'), OBJECTS.index('button1')]
BODY = [OBJECTS.index('drawer'), OBJECTS.index('window')]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def otsu(v):
    c = np.sort(v)
    n = len(c)
    cs = np.cumsum(c)
    i = np.arange(1, n)
    ma, mb = cs[i - 1] / i, (cs[-1] - cs[i - 1]) / (n - i)
    b = int(np.argmax(i * (n - i) * (ma - mb) ** 2))
    return (c[b] + c[b + 1]) / 2


def regions(z, thr):
    lab, n = ndimage.label((z > thr).reshape(G, G))
    lab = lab.reshape(-1)
    return [lab == k for k in range(1, n + 1)]


def best_iou(regs, m):
    return max(((r & m).sum() / (r | m).sum() for r in regs), default=0.0)


def evaluate(S, z, thr, pairs, arm_masked=False):
    thr_of = thr if callable(thr) else (lambda zi: thr)
    out = {k: [] for k in ['button', 'cube left', 'cube arrived', 'drawer', 'window']}
    top = {k: [] for k in out}
    two, nreg, narm = [], [], []
    nobj = {}
    cache = {}
    for i, j in pairs:
        m0, m1 = S['masks'][S['st'][i] - CLEAR], S['masks'][S['en'][i] + CLEAR]
        arm = m0[ARM] | m1[ARM]
        if i not in cache:
            zi = np.where(arm, -np.inf, z[i]) if arm_masked else z[i]
            regs = regions(zi, thr_of(zi))
            cache[i] = regs
            nreg.append(len(regs))
            narm.append(sum((r & arm).sum() > r.sum() / 2 for r in regs))
        regs = cache[i]
        t1 = np.zeros(G * G, bool)
        t1[int(np.argmax(z[i]))] = True
        kind, o = S['TK'][j], S['TO'][j]
        mo = m0[o].any(0) | m1[o].any(0)
        nobj[i] = nobj.get(i, np.zeros(G * G, bool)) | mo
        if kind == 'lock':
            b, h = o.copy(), o.copy()
            b[BODY] = False; h[BTN] = False
            mb, mh = m0[b].any(0) | m1[b].any(0), m0[h].any(0) | m1[h].any(0)
            out['button'].append(best_iou(regs, mb)); top['button'].append(best_iou([t1], mb))
            on_b = [r for r in regs if (r & mb).sum() > r.sum() / 2]
            on_h = [r for r in regs if (r & mh & ~mb).sum() > r.sum() / 2]
            two.append(bool(on_b) and bool(on_h))
        elif kind == 'cube':
            for nm, m in [('cube left', m0[CUBE]), ('cube arrived', m1[CUBE])]:
                if m.any():
                    out[nm].append(best_iou(regs, m)); top[nm].append(best_iou([t1], m))
        else:
            m = m0[o].any(0) | m1[o].any(0)
            out[kind].append(best_iou(regs, m)); top[kind].append(best_iou([t1], m))
    res = dict(best_region_iou={k: round(float(np.mean(v)), 3) for k, v in out.items()},
               share_iou_at_least_half={k: round(float((np.array(v) >= 0.5).mean()), 3) for k, v in out.items()},
               top_patch_alone_iou={k: round(float(np.mean(v)), 3) for k, v in top.items()},
               lock_button_and_handle_in_separate_regions=round(float(np.mean(two)), 3),
               regions_per_touch=dict(mean=round(float(np.mean(nreg)), 2), median=float(np.median(nreg)),
                                      none=round(float((np.array(nreg) == 0).mean()), 3)),
               share_of_regions_mostly_on_arm=round(float(np.sum(narm) / max(np.sum(nreg), 1)), 3))
    on = sum(sum((r & m).sum() > r.sum() / 2 for r in cache[i]) for i, m in nobj.items())
    res['share_of_regions_mostly_on_a_changed_object'] = round(float(on / max(np.sum(nreg), 1)), 3)
    res['n'] = {k: len(v) for k, v in out.items()}
    return res


def main():
    name = arg('--set', 'tail')
    T = np.load(ROOT / 'runs/122/computed_tail.npz', allow_pickle=True)
    eps = float(np.median(T['tden']))
    zt = T['tnum'] / (T['tden'] + eps)
    thr = float(np.exp(otsu(np.log(zt.reshape(-1) + 1e-6)[::7])))
    S = dict(np.load(ROOT / f'runs/122/computed_{name}.npz', allow_pickle=True))
    S['masks'] = S['masks'].astype(bool)
    z = S['tnum'] / (S['tden'] + eps)
    pairs = [(i, j) for i, (s_, e_) in enumerate(zip(S['st'], S['en']))
             for j in np.where((S['A'] <= e_ + TOL) & (S['B'] >= s_ - TOL) & (S['A'] // EP == s_ // EP))[0]]
    res = dict(set=name, threshold=round(thr, 3), share_of_patches_above=round(float((z > thr).mean()), 4),
               touches=int(len(S['st'])))
    res['regions'] = evaluate(S, z, thr, pairs)
    fin = lambda zi: zi[np.isfinite(zi)]
    res['rule_half_of_top'] = evaluate(S, z, lambda zi: 0.5 * fin(zi).max(), pairs)
    res['rule_tenth_of_top'] = evaluate(S, z, lambda zi: 0.1 * fin(zi).max(), pairs)
    res['rule_otsu_per_event'] = evaluate(S, z, lambda zi: float(np.exp(otsu(np.log(fin(zi) + 1e-6)))), pairs)
    res['rule_max_of_global_and_tenth'] = evaluate(S, z, lambda zi: max(thr, 0.1 * fin(zi).max()), pairs)
    res['regions_arm_masked'] = evaluate(S, z, thr, pairs, arm_masked=True)
    sweep = []
    for c in np.exp(np.linspace(np.log(thr) - 3, np.log(thr) + 3, 13)):
        e = evaluate(S, z, c, pairs)['best_region_iou']
        sweep.append((round(float(c), 3), e['button'], round((e['cube left'] + e['cube arrived']) / 2, 3)))
    res['threshold_sweep_threshold_button_cube'] = sweep
    print(json.dumps(res, indent=1))
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / f'regions_{name}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
