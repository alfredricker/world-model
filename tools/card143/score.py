"""Card 143: scores of every fitted arm on the test bed's cube cells, and the gate's check of the added labels.

  bin/prun python tools/card143/score.py → runs/143/score.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

GRP = 3


def where(cube, drawer):
    ins = (np.abs(cube[:, 0] - 0.33) < 0.12) & (cube[:, 2] > 0.04) & (cube[:, 1] < -0.1)
    return np.where(~ins, 'table', np.where(drawer >= 0.03, 'open', 'hidden'))


def selection():
    M = np.load(ROOT / 'runs/143/meta.npy')
    cube = M[:, 6:9] / 10 + C.CEN
    wh = where(cube, M[:, 12])
    e = np.linalg.norm(M[:, :2] / 10 + C.CEN[:2] - cube[:, :2], axis=1) * 100
    tr = M[:, 10] < 80
    out = dict(added_frames=int(len(M)), in_training=int(tr.sum()), within_1cm_of_true_cube=round(100 * float((e < 1).mean()), 1))
    for w in ('table', 'open', 'hidden'):
        k = wh == w
        out[w] = dict(frames=int(k.sum()), in_training=int((k & tr).sum()), median_cm=round(float(np.median(e[k])), 2) if k.any() else None,
                      within_1cm=round(100 * float((e[k] < 1).mean()), 1) if k.any() else None)
    return out


def one(d, t, a, b, cut=None):
    aim, top = d['aim'], d['top']
    if cut is None:
        g = d['own_group'] == GRP
        cut = 2 * float(np.quantile(d['own_err_cm'][g], 0.9)) / 100
    err = np.linalg.norm(aim[:, :2] - t['cube'][:, :2], axis=1) * 100
    res = dict(cut_cm=round(cut * 100, 2))
    for w in ('table', 'open', 'hidden'):
        k = t['where'] == w
        res[w] = dict(n=int(k.sum()), median_cm=round(float(np.median(err[k])), 2), within_1cm=round(100 * float((err[k] < 1).mean()), 1),
                      within_3cm=round(100 * float((err[k] < 3).mean()), 1), largest_score_median=round(float(np.median(top[k])), 2))
    p = C.score_pairs(t, a, b, dict(cube=np.linalg.norm(aim[a] - aim[b], axis=1) > cut))
    res['pairs'] = {k: v['bal'] for k, v in p.items()}
    p = C.score_pairs(t, a, b, dict(cube=np.linalg.norm(aim[a] - aim[b], axis=1) > 0.024))
    res['pairs_at_og4_cut_2.4cm'] = {k: v['bal'] for k, v in p.items()}
    if 'own_added' in d and d['own_added'].any():
        g = d['own_group'] == GRP
        res['own_held_out_cm'] = dict(contact=round(float(np.median(d['own_err_cm'][g & ~d['own_added']])), 2),
                                      release=round(float(np.median(d['own_err_cm'][g & d['own_added']])), 2))
    # error in the open drawer by distance from the nearest added training label
    M = np.load(ROOT / 'runs/143/meta.npy')
    lab = (M[M[:, 10] < 80, :2] / 10 + C.CEN[:2])
    k = t['where'] == 'open'
    dist = np.linalg.norm(t['cube'][k][:, None, :2] - lab[None], axis=-1).min(1) * 100
    res['open_by_cm_from_nearest_release_label'] = {
        name: dict(n=int(m.sum()), median_cm=round(float(np.median(err[k][m])), 2) if m.any() else None)
        for name, m in (('under 1', dist < 1), ('1 to 3', (dist >= 1) & (dist < 3)), ('over 3', dist >= 3))}
    return res


def main():
    t = C.truth(C.TEST)
    a, b = C.pairs(t)
    out = dict(selection=selection(), arms={})
    c = np.load(ROOT / 'runs/141/current.npz')
    out['arms']['og4 (card 141, stored)'] = one(dict(aim=c['aim'], top=c['top']), t, a, b, cut=0.024)
    for f in sorted(glob.glob(str(ROOT / 'runs/143/fit_*.npz'))):
        out['arms'][Path(f).stem[4:]] = one(dict(np.load(f)), t, a, b)
    rel = [v for k, v in out['arms'].items() if k.startswith('rel4_')]
    if rel:
        m = lambda f: round(float(np.mean([f(v) for v in rel])), 2)
        out['rel_mean'] = dict(seeds=len(rel), open_median_cm=m(lambda v: v['open']['median_cm']), table_median_cm=m(lambda v: v['table']['median_cm']),
                               pairs_open=m(lambda v: v['pairs']['cube open drawer']), pairs_table=m(lambda v: v['pairs']['cube table']),
                               pairs_hidden=m(lambda v: v['pairs']['cube hidden']))
    json.dump(out, open(ROOT / 'runs/143/score.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
