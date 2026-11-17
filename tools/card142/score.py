"""Card 142: the two repairs of card 141's bed, and every fitted arm on the cube's table cells.

Repairs (the labelled probe over its seeds, runs/141/probe.npz and probe_s<seed>.npz):
- the smallest slide cell with a gap: same under 0.5 cm, different 2 to 4 cm, threshold 1.25 cm (card 141, posthoc.py);
- the probe's "can the cube be located" answer: its balanced accuracy, how often it says "locatable" with the cube
  inside and the drawer shut (under 1 cm), and the hidden-cube pair cell, per seed.
Arms (runs/142/fit_<arm>_<a|w>_<seed>.npz; a = all labels, w = the upper third of the table withheld): median error
on the table in the kept part, the withheld part and over 5 cm past its edge; the pair cell "both on the table" with
og4's rule for the cut (twice the 90th percentile of the arm's own held-out error for the cube group at table height);
and, without a bar, the open drawer answered at the height of the cube's labels above the table.

  bin/prun python tools/card142/score.py → runs/142/score.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

GRP, YCUT = 3, 0.102
med = lambda x: round(float(np.median(x)), 2) if len(x) else None


def repairs(t, a, b):
    out = {}
    for f in sorted(glob.glob(str(ROOT / 'runs/141/probe.npz')) + glob.glob(str(ROOT / 'runs/141/probe_s*.npz'))):
        p = np.load(f)
        r = {}
        for j, s in enumerate(('drawer', 'window')):
            d, dp = np.abs(t[s][a] - t[s][b]), np.abs(p['five'][a, j] - p['five'][b, j]) / 10
            k = (d < 0.005) | ((d >= 0.02) & (d < 0.04))
            r[f'{s} gap cell'] = C.bal(dp[k] > 0.0125, d[k] >= 0.02)
        view = p['five'][:, 4] > 0
        shut = (t['where'] != 'table') & (t['drawer'] < 0.01)
        r['locatable, balanced accuracy'] = C.bal(view, t['where'] != 'hidden')
        r['said locatable with the drawer shut, %'] = round(100 * float(view[shut].mean()), 1)
        differs = np.where(view[a] & view[b], np.linalg.norm(p['cube'][a] - p['cube'][b], axis=1) > 0.02, view[a] != view[b])
        pr = C.score_pairs(t, a, b, dict(cube=differs))
        r.update({f'pair cell {k}': v['bal'] for k, v in pr.items()})
        e = np.linalg.norm((p['cube'] - t['cube'])[:, :2], axis=1) * 100
        r['cube on the table, cm'], r['cube in the open drawer, cm'] = med(e[t['where'] == 'table']), med(e[t['where'] == 'open'])
        out[Path(f).stem] = r
    keys = list(next(iter(out.values())))
    out['spread'] = {k: dict(mean=round(float(np.mean([v[k] for v in out.values()])), 2), lowest=min(v[k] for v in out.values()),
                             highest=max(v[k] for v in out.values())) for k in keys}
    return out


def arm(d, t, a, b):
    tab, y = t['where'] == 'table', t['cube'][:, 1]
    e = np.linalg.norm(d['aim'][:, :2] - t['cube'][:, :2], axis=1) * 100
    eh = np.linalg.norm(d['aim_high'][:, :2] - t['cube'][:, :2], axis=1) * 100
    g = (d['own_group'] == GRP) & (d['own_z'] < np.median(d['own_z'][d['own_group'] == GRP]) + 0.3)
    cut = 2 * float(np.quantile(d['own_err_cm'][g], 0.9)) / 100
    pr = C.score_pairs(t, a, b, dict(cube=np.linalg.norm(d['aim'][a, :2] - d['aim'][b, :2], axis=1) > cut))
    return dict(table=med(e[tab]), kept=med(e[tab & (y <= YCUT)]), withheld_part=med(e[tab & (y > YCUT)]),
                past_0_2=med(e[tab & (y > YCUT) & (y <= YCUT + 0.02)]), past_2_5=med(e[tab & (y > YCUT + 0.02) & (y <= YCUT + 0.05)]),
                past_over_5=med(e[tab & (y > YCUT + 0.05)]), cut_cm=round(cut * 100, 2), pair_table=pr['cube table']['bal'],
                open_drawer_at_its_height=med(eh[t['where'] == 'open']), open_drawer_within_1cm=round(100 * float((eh[t['where'] == 'open'] < 1).mean()), 1),
                open_drawer_at_table_height=med(e[t['where'] == 'open']),
                own_held_out_cm=med(d['own_err_cm'][g]), largest_score_table=med(d['top'][tab]), largest_score_open=med(d['top'][t['where'] == 'open']))


def main():
    t = C.truth(C.TEST)
    a, b = C.pairs(t)
    out = dict(n=dict(table=int((t['where'] == 'table').sum()), withheld_part=int(((t['where'] == 'table') & (t['cube'][:, 1] > YCUT)).sum()),
                      past_over_5=int(((t['where'] == 'table') & (t['cube'][:, 1] > YCUT + 0.05)).sum()), open=int((t['where'] == 'open').sum())))
    if '--no-repairs' not in sys.argv:
        out['repairs'] = repairs(t, a, b)
    out['fits'], by = {}, {}
    for f in sorted(glob.glob(str(ROOT / 'runs/142/fit_*.npz'))):
        name = Path(f).stem[4:]
        out['fits'][name] = arm(dict(np.load(f)), t, a, b)
        by.setdefault(name.rsplit('_', 1)[0], []).append(out['fits'][name])
    out['mean'] = {k: dict(seeds=len(v), **{m: round(float(np.mean([x[m] for x in v])), 2) for m in v[0]},
                           withheld_part_per_seed=[x['withheld_part'] for x in v]) for k, v in by.items()}
    json.dump(out, open(ROOT / 'runs/142/score.json', 'w'), indent=1)
    if 'repairs' in out:
        print(json.dumps(out['repairs']['spread'], indent=1))
    for k, v in out['mean'].items():
        print(k, json.dumps(v))


if __name__ == '__main__':
    main()
