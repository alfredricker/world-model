"""Card 142, check before drafting: which part of the aim read-out fails outside its training range.

The read-out answers with  sum over patches of  weight(patch) * (place(patch) + offset(patch)):
  weight  which patches hold the cube (softmax of a score from the patch's features)
  place   a smooth map from a patch's row and column to a position
  offset  a correction from the patch's features
Two fits as in card 141 (all touches; the cube's touches in the upper third of the table withheld). On test frames
with the cube on the table, each part is taken from one fit or the other, in all eight combinations, and the median
error is given in the kept part and in the withheld part. Also: how often the two fits put their largest weight on
the same patch.

  bin/prun python tools/card142/check.py → runs/142/check.json
"""
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

spec = importlib.util.spec_from_file_location('cover', ROOT / 'tools/card141/cover.py')
cover = importlib.util.module_from_spec(spec); spec.loader.exec_module(cover)
GRP, YCUT = cover.GRP, cover.YCUT


def parts(model, f):
    h = model.trunk(f)
    w = torch.softmax(model.score(h), 1)[:, :, GRP]
    return w.cpu().numpy(), model.where(model.grid).cpu().numpy(), model.offset(h).cpu().numpy()


def main():
    t0 = time.time()
    X, M = np.load(ROOT / 'runs/129/feats.npy'), np.load(ROOT / 'runs/129/meta.npy')
    G = torch.load(ROOT / 'runs/129/readout_q.pt', weights_only=False)['group']
    M = M.copy(); M[:, 9] = G[M[:, 9].astype(int)]
    base = M[:, 10] < 80
    arms = dict(all=np.where(base)[0], withheld=np.where(base & ~((M[:, 9] == GRP) & (M[:, 1] > YCUT)))[0])
    tt, Xt = C.truth(C.TEST), C.frames(C.TEST)
    idx = np.where(tt['where'] == 'table')[0]
    true = (tt['cube'][idx] - C.CEN) * 10
    P = {}
    for name, tr in arms.items():
        model = cover.fit(X, M, tr)
        print(name, f'{time.time() - t0:.0f}s', flush=True)
        W, O = [], []
        with torch.no_grad():
            for i0 in range(0, len(idx), 64):
                f = torch.from_numpy(np.stack([np.asarray(Xt[tt['set'][i]][tt['row'][i]], np.float32) for i in idx[i0:i0 + 64]])).cuda()
                w, wh, off = parts(model, f)
                W.append(w.astype(np.float32)); O.append(off.astype(np.float16))
        P[name] = dict(w=np.concatenate(W), wh=wh, off=np.concatenate(O))
    held = true[:, 1] > YCUT
    out = dict(n_kept=int((~held).sum()), n_withheld=int(held.sum()), combos={})
    for a, b, c in itertools.product(('all', 'withheld'), repeat=3):
        pred = np.einsum('bp,bpd->bd', P[a]['w'], P[b]['wh'][None] + P[c]['off'].astype(np.float32))
        e = np.linalg.norm(pred[:, :2] - true[:, :2], axis=1) * 10           # cm
        out['combos'][f'weight={a} place={b} offset={c}'] = dict(kept=round(float(np.median(e[~held])), 2),
                                                               withheld=round(float(np.median(e[held])), 2))
    pa, pw = P['all']['w'].argmax(1), P['withheld']['w'].argmax(1)
    d = np.hypot(pa // 32 - pw // 32, pa % 32 - pw % 32)
    out['largest weight on the same patch or a neighbour, %'] = dict(kept=round(100 * float((d[~held] <= 1.5).mean()), 1),
                                                                    withheld=round(100 * float((d[held] <= 1.5).mean()), 1))
    out['largest weight, median'] = {k: dict(kept=round(float(np.median(P[k]['w'].max(1)[~held])), 3),
                                             withheld=round(float(np.median(P[k]['w'].max(1)[held])), 3)) for k in P}
    for k in ('wh', 'off'):                                                 # how far the two fits' parts differ, cm
        if k == 'wh':
            row = np.arange(1024) // 32
            dd = np.linalg.norm(P['all']['wh'][:, :2] - P['withheld']['wh'][:, :2], axis=1) * 10
            out['place map, median difference between fits by image row band (top to bottom), cm'] = [
                round(float(np.median(dd[(row >= r) & (row < r + 8)])), 2) for r in (0, 8, 16, 24)]
    (ROOT / 'runs/142').mkdir(exist_ok=True)
    json.dump(out, open(ROOT / 'runs/142/check.json', 'w'), indent=1)
    print(json.dumps(out, indent=1), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
