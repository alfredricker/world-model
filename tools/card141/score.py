"""Card 141: every arm in every cell → runs/141/bed.json.

  bin/prun python tools/card141/score.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

YCUT = 0.102
med = lambda x: round(float(np.median(x)), 2) if len(x) else None


def probe_arm(name, t, tl, a, b):
    p = np.load(C.OUT / f'{name}.npz')
    cube, five = p['cube'], p['five']
    r = dict(thresholds_cm={})
    differs = {}
    view = five[:, 4] > 0
    differs['cube'] = np.where(view[a] & view[b], np.linalg.norm(cube[a] - cube[b], axis=1) > 0.02, view[a] != view[b])
    dv = p['dev_rows']
    for j, s in enumerate(('drawer', 'window')):
        dt = np.abs(tl[s][dv][:, None] - tl[s][dv][None])[np.triu_indices(len(dv), 1)]
        dp = np.abs(p['dev_five'][:, j][:, None] - p['dev_five'][:, j][None])[np.triu_indices(len(dv), 1)] / 10
        cuts = np.arange(0.002, 0.04, 0.001)
        thr = cuts[np.argmax([C.bal(dp > c, dt >= 0.01) for c in cuts])]
        r['thresholds_cm'][s] = round(float(thr * 100), 1)
        differs[s] = np.abs(five[a, j] - five[b, j]) / 10 > thr
    for j in (0, 1):
        differs[f'lock{j}'] = (five[a, 2 + j] > 0) != (five[b, 2 + j] > 0)
    r['pairs'] = C.score_pairs(t, a, b, differs)
    err = np.linalg.norm(cube - t['cube'], axis=1) * 100
    tab = t['where'] == 'table'
    r['cube_value'] = {w: dict(n=int((t['where'] == w).sum()), median_cm=med(err[t['where'] == w]), within_2cm=round(float((err[t['where'] == w] <= 2).mean()), 3))
                       for w in ('table', 'open')}
    r['cube_table_by_y'] = dict(kept=dict(n=int((tab & (t['cube'][:, 1] <= YCUT)).sum()), median_cm=med(err[tab & (t['cube'][:, 1] <= YCUT)])),
                                withheld_part=dict(n=int((tab & (t['cube'][:, 1] > YCUT)).sum()), median_cm=med(err[tab & (t['cube'][:, 1] > YCUT)])))
    r['in_view'] = C.bal(view, t['where'] != 'hidden')
    r['slide_value'] = {}
    for j, s in enumerate(('drawer', 'window')):
        e = np.abs(five[:, j] / 10 - t[s]) * 100
        r['slide_value'][s] = dict(median_cm=med(e), **{nm: med(e[(t[s] >= lo) & (t[s] < hi)]) for nm, lo, hi in (('under1', 0, 0.01), ('1-4', 0.01, 0.04), ('4-10', 0.04, 0.10), ('over10', 0.10, 9))})
    r['lock_value'] = [round(100 * float(((five[:, 2 + j] > 0) == t['locks'][:, j]).mean()), 1) for j in (0, 1)]
    return r


def main():
    t, tl = C.truth(C.TEST), C.truth(C.LEARN)
    a, b = C.pairs(t)
    out = dict(frames=dict(learn=len(tl['ep']), test=len(t['ep'])), pairs=len(a))
    zero = np.zeros(len(a), bool)
    out['base_rate'] = dict(pairs=C.score_pairs(t, a, b, dict(cube=zero, drawer=zero, window=zero, lock0=zero, lock1=zero)),
                            cube_table_median_cm=med(np.linalg.norm(t['cube'][t['where'] == 'table'] - tl['cube'][tl['where'] == 'table'].mean(0), axis=1) * 100),
                            slide_median_cm={s: med(np.abs(t[s] - tl[s].mean()) * 100) for s in ('drawer', 'window')})
    out['current'] = json.load(open(C.OUT / 'current.json'))
    out['current_withheld'] = json.load(open(C.OUT / 'current_withheld.json'))
    out['probe'] = probe_arm('probe', t, tl, a, b)
    out['probe_withheld'] = probe_arm('probe_withheld', t, tl, a, b)
    c = np.load(C.OUT / 'cover.npz')
    tab = t['where'] == 'table'
    y = t['cube'][:, 1]
    near = np.array([np.linalg.norm(c['train_cube_xy'] - p, axis=1).min() for p in t['cube'][:, :2]]) * 100
    out['read_out'] = {}
    for name in ('all', 'withheld'):
        e = np.linalg.norm(c[name][:, :2] - t['cube'][:, :2], axis=1) * 100
        out['read_out'][name] = dict(kept=dict(n=int((tab & (y <= YCUT)).sum()), median_cm=med(e[tab & (y <= YCUT)])),
                                     withheld_part=dict(n=int((tab & (y > YCUT)).sum()), median_cm=med(e[tab & (y > YCUT)])),
                                     by_distance_past_the_cut={f'{lo}-{hi}cm': med(e[tab & (y > YCUT + lo / 100) & (y <= YCUT + hi / 100)]) for lo, hi in ((0, 2), (2, 5), (5, 12))},
                                     open_drawer=med(e[t['where'] == 'open']))
    e = np.linalg.norm(c['all'][:, :2] - t['cube'][:, :2], axis=1) * 100
    out['read_out']['natural_coverage'] = {k: dict(n=int((tab & m).sum()), median_cm=med(e[tab & m])) for k, m in
                                           (('within 1 cm of a training touch', near <= 1), ('1 to 3 cm', (near > 1) & (near <= 3)), ('over 3 cm', near > 3))}
    json.dump(out, open(C.OUT / 'bed.json', 'w'), indent=1)
    cells = list(out['probe']['pairs'])
    print(f'{"cell":22s} {"n same/diff":>14s} base  current  probe  probe_withheld  current_withheld')
    for k in cells:
        g = lambda arm: (out[arm]['pairs'].get(k) or {}).get('bal')
        cur = out['current']['pairs_cube_with_in_view'][k]['bal'] if k.startswith('cube') else g('current')
        print(f'{k:22s} {str(out["probe"]["pairs"][k]["n"]):>14s} {g("base_rate")}  {g("current")} ({cur})  {g("probe")}  {g("probe_withheld")}  {g("current_withheld")}')
    for arm in ('probe', 'probe_withheld'):
        print(arm, {k: out[arm][k] for k in ('thresholds_cm', 'cube_value', 'cube_table_by_y', 'in_view', 'slide_value', 'lock_value')})
    print('current', out['current']['cube_value'], out['current']['in_view'], out['current']['lock_value'])
    print('read_out', json.dumps(out['read_out']))
    print('base', out['base_rate']['cube_table_median_cm'], out['base_rate']['slide_median_cm'])


if __name__ == '__main__':
    main()
