"""Card 134, diagnostic after the run (not a declared arm). (1) Is the last pull's outcome in the contents? (2) A key
made of the touched object's own record only: touched object, its last seen value, the effect of the last touch on
it, and the two locks as last seen; equal weight, nothing trained.

  bin/prun python tools/card134/own.py → runs/134/own.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import effects as E
from effects import B, C

mem, test = C.load(C.MEM), C.load(C.TEST)
tgm, chm, _ = C.targets(mem)
tgt, cht, _ = C.targets(test)
lk, _ = B.lock_of(mem, tgm, chm)
Xm, Ym, mm = E.events(mem, tgm, chm, lk)
Xt, Yt, mt = E.events(test, tgt, cht, lk)
SEEN = {0: slice(9, 13), 1: slice(13, 15), 2: slice(15, 17), 3: slice(17, 19), 4: slice(19, 21)}
EFF = {0: slice(21, 25), 1: slice(25, 27), 2: slice(27, 29), 3: slice(29, 31), 4: slice(31, 33)}


def own(X, with_contact):
    X = X.numpy()
    k = X[:, 3:9].argmax(1) - 1
    out = np.zeros((len(X), 3 + 6 + 4 + 4 + 4), dtype=np.float32)
    out[:, :3] = X[:, :3] if with_contact else 0
    out[:, 3:9] = X[:, 3:9]
    for o in range(5):
        q = k == o
        a, b = X[q][:, SEEN[o]], X[q][:, EFF[o]]
        out[q, 9:9 + a.shape[1]] = a; out[q, 13:13 + b.shape[1]] = b
    out[:, 17:21] = X[:, 17:21]
    return torch.tensor(out)


q = mt['pull'] & (mt['group'] == 1)
slide_eff = np.where(mt['slide'] == 1, np.abs(Xt['effects'][:, 26].numpy()), np.abs(Xt['effects'][:, 28].numpy())) > 0.5
res = dict(last_effect_item_agrees_with_last_pull=round(float((slide_eff[q] == mt['last'][q]).mean()), 3),
           rule_from_item=round(100 * B.bal(slide_eff[q], mt['moved'][q]), 1))
keep = ~mm['mixed']
for name, wc in (('own_record', False), ('own_record_and_contact', True)):
    A, Q = own(Xm['effects'][keep], wc), own(Xt['effects'], wc)
    mu, sd = A.mean(0), A.std(0) + 1e-6
    res[name] = E.score(*E.answer((A - mu) / sd, Ym[keep], (Q - mu) / sd, mt['slide']), mt, None)
    print(name, json.dumps(res[name]), flush=True)
print({k: v for k, v in res.items() if not isinstance(v, dict)})
json.dump(res, open(C.ROOT / 'runs/134/own.json', 'w'), indent=1)
