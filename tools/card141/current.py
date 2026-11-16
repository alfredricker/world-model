"""Card 141, arm 1: og4's readings on the test bed, as they are (nothing trained here).

  aim         runs/129/readout_q.pt, the cube's group (3): a position; "differs" if two aims are over 2.4 cm apart
              (og4's cut: twice the read-out's 90th-percentile error); "in view" if the largest patch score is at
              least 5.69 (card 131's threshold, not part of og4)
  appearance  per place with an effect (card 127's knowledge, runs/127/know.npz): the pair (now, goal) differs if it
              is nearer a remembered change than a remembered non-change, over the place's 10 patches
  lock state  per condition: the nearer of two remembered states over the condition's 10 patches
A place is matched to the slide, and a condition to the lock, that its answers agree with best (bookkeeping).
--withhold: the appearance reading with remembered pairs dropped where the slide (before or after) is 4 to 10 cm open.

  bin/prun python tools/card141/current.py → runs/141/current.json, current.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
otsu = C.otsu

GRP, CUT_AIM, CUT_VIEW = 3, 0.024, 5.69


def main():
    withhold = '--withhold' in sys.argv
    t = C.truth(C.TEST)
    X = C.frames(C.TEST)
    k = np.load(C.ROOT / 'runs/127/know.npz')
    before = np.load(C.ROOT / 'runs/125/before_m400.npy', mmap_mode='r')
    tm = C.truth(['m400'])
    places = sorted(int(n[3:]) for n in k.files if n.startswith('eff'))
    idx0 = np.where(k['has_next'])[0]
    eff = {}
    for c in places:                                                        # as tools/card127/plan.py, Agent.__init__
        e = k[f'eff{c}']
        B = np.stack([np.asarray(before[i][e], np.float32) for i in idx0])
        A = np.stack([np.asarray(before[i + 1][e], np.float32) for i in idx0])
        w = k['steady'][idx0][:, e] & k['steady'][idx0 + 1][:, e]
        keep = w.sum(1) >= 3
        idx, B, A, w = idx0[keep], B[keep], A[keep], w[keep].astype(np.float32)
        d = (((A - B) ** 2).sum(-1) * w).sum(1) / w.sum(1)
        ch = d > float(np.exp(otsu(np.log(d[k['place'][idx] == c]))))
        eff[c] = dict(e=torch.tensor(e).cuda(), idx=idx, B=torch.tensor(B).cuda(), A=torch.tensor(A).cuda(), w=torch.tensor(w).cuda(), ch=ch)
    ro = C.read_out().cuda()
    ro.load_state_dict(torch.load(C.ROOT / 'runs/129/readout_q.pt', weights_only=False)['state'])
    ro.eval()
    n = len(t['ep'])
    aim, top = np.zeros((n, 3)), np.zeros(n)
    DB, DA = {c: np.zeros((n, len(eff[c]['idx'])), np.float32) for c in places}, {c: np.zeros((n, len(eff[c]['idx'])), np.float32) for c in places}
    state = {c: np.zeros(n, int) for c in places}
    cond = {c: (torch.tensor(k[f'cond{c}']).cuda(), torch.tensor(k[f'centre{c}']).cuda(), torch.tensor(k[f'states{c}']).cuda()) for c in places}
    with torch.no_grad():
        for i0 in range(0, n, 32):
            rows = np.arange(i0, min(i0 + 32, n))
            f = torch.from_numpy(np.stack([np.asarray(X[t['set'][i]][t['row'][i]], np.float32) for i in rows])).cuda()
            aim[rows] = ro(f)[:, GRP].cpu().numpy() / 10 + C.CEN
            top[rows] = ro.score(ro.trunk(f))[:, :, GRP].max(1).values.cpu().numpy()
            for c in places:
                v = eff[c]
                fe = f[:, v['e']]                                           # [b, 10, 768]
                for D, S in ((DB, v['B']), (DA, v['A'])):
                    d = ((fe[:, None] - S[None]) ** 2).sum(-1)              # [b, m, 10]
                    D[c][rows] = ((d * v['w'][None]).sum(-1) / v['w'].sum(1)[None]).cpu().numpy()
                top_, cen, sts = cond[c]
                r = (f[:, top_] - cen[None]).mean(1)
                state[c][rows] = ((sts[None] - r[:, None]) ** 2).sum(-1).argmin(1).cpu().numpy()
    a, b = C.pairs(t)

    def appearance(c, drop=None):
        ch = eff[c]['ch']
        ok = np.ones(len(ch), bool) if drop is None else ~drop
        s = DB[c][a] + DA[c][b]
        return s[:, ok & ~ch].min(1) / s[:, ok & ch].min(1) > 1

    res, differs = {}, {}
    differs['cube'] = np.linalg.norm(aim[a] - aim[b], axis=1) > CUT_AIM
    app = {c: appearance(c) for c in places}
    slide_of = {}
    for c in places:
        agree = {s: C.bal(app[c], np.abs(t[s][a] - t[s][b]) >= 0.01) for s in ('drawer', 'window')}
        slide_of[c] = max(agree, key=agree.get)
    lock_of = {}
    for c in places:
        agree = {j: abs(np.corrcoef(state[c], t['locks'][:, j])[0, 1]) for j in (0, 1)}
        lock_of[c] = max(agree, key=agree.get)
    res['place_is'] = {str(c): dict(slide=slide_of[c], lock=int(lock_of[c])) for c in places}
    for s in ('drawer', 'window'):
        cs = [c for c in places if slide_of[c] == s]
        differs[s] = np.any([app[c] for c in cs], 0) if cs else None
        if withhold and cs:
            dr = []
            for c in cs:
                i = eff[c]['idx']
                dr.append(((tm[s][i] >= 0.04) & (tm[s][i] < 0.10)) | ((tm[s][i + 1] >= 0.04) & (tm[s][i + 1] < 0.10)))
            differs[s] = np.any([appearance(c, d) for c, d in zip(cs, dr)], 0)
            res[f'{s}_remembered_pairs_dropped'] = [int(d.sum()) for d in dr]
    for j in (0, 1):
        cs = [c for c in places if lock_of[c] == j]
        differs[f'lock{j}'] = np.any([state[c][a] != state[c][b] for c in cs], 0) if cs else None
    res['pairs'] = C.score_pairs(t, a, b, differs)
    va, vb = top[a] >= CUT_VIEW, top[b] >= CUT_VIEW                             # with card 131's "in view" (not in og4)
    res['pairs_cube_with_in_view'] = C.score_pairs(t, a, b, dict(cube=np.where(va & vb, differs['cube'], va != vb)))
    res['per_place_pairs'] = {str(c): C.score_pairs(t, a, b, {slide_of[c]: app[c]}) for c in places}
    err = np.linalg.norm(aim[:, :2] - t['cube'][:, :2], axis=1) * 100
    valid = top >= CUT_VIEW
    res['cube_value'] = {w: dict(n=int((t['where'] == w).sum()), median_cm=round(float(np.median(err[t['where'] == w])), 2),
                                 within_2cm=round(float((err[t['where'] == w] <= 2).mean()), 3), read_in_view=round(float(valid[t['where'] == w].mean()), 3))
                         for w in ('table', 'open', 'hidden')}
    res['in_view'] = dict(locatable_vs_hidden=C.bal(valid, t['where'] != 'hidden'), table_vs_rest=C.bal(valid, t['where'] == 'table'))
    res['lock_value'] = {str(c): round(100 * max((state[c] == t['locks'][:, lock_of[c]]).mean(), (state[c] != t['locks'][:, lock_of[c]]).mean()), 1) for c in places}
    name = 'current_withheld' if withhold else 'current'
    json.dump(res, open(C.OUT / f'{name}.json', 'w'), indent=1)
    if not withhold:
        np.savez(C.OUT / 'current.npz', aim=aim, top=top)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
