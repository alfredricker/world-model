"""Gate for card 126: does a lock's state, read on one button, carry to the other button, on frozen features?
Memory m600 (card 125.1). Each touch gives one reading per button: the mean before-feature over the button's
usual patches (truth masks, for this gate only) that are steady. A nearest-class-mean reader of locked/unlocked
is fitted on one button and tested on the other, and on the same button in the test set's stead (split in half).
Also: the same after centring each button's readings on its own mean (removes what differs between the places).

  bin/prun python tools/card125.2/transfer_gate.py → runs/125.2/transfer_gate.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card125'))
from gate import ROOT, bal, load, ncm, otsu  # noqa: E402

M = load('m600', 600)
vm = np.log(np.load(ROOT / 'runs/125.1/var_m600.npy') + 1e-6)
steady = vm < otsu(vm.reshape(-1)[::7])
n = len(M['st'])
R, Y, OK = {}, {}, {}
for b, mi in [(0, 2), (1, 3)]:
    patches = np.where(M['masks'][:, mi].mean(0) > 0.5)[0]
    F = np.concatenate([np.asarray(M['before'][i:i + 256][:, patches], np.float32) for i in range(0, n, 256)])
    w = steady[:, patches]
    OK[b] = w.sum(1) >= 3
    R[b] = ((F * w[..., None]).sum(1) / np.maximum(w.sum(1), 1)[:, None])[OK[b]]
    Y[b] = (M['btn'][:, b] == 0)[OK[b]]
res = {}
for nm, sl in [('block2', slice(0, 384)), ('block12', slice(384, 768)), ('both', slice(0, 768))]:
    e = {}
    for a, b in [(0, 1), (1, 0)]:
        e[f'fit_button{a}_test_button{b}'] = bal(ncm(R[a][:, sl], Y[a], R[b][:, sl]), Y[b])
        ca, cb = R[a][:, sl] - R[a][:, sl].mean(0), R[b][:, sl] - R[b][:, sl].mean(0)
        e[f'centred_fit_button{a}_test_button{b}'] = bal(ncm(ca, Y[a], cb), Y[b])
    h = len(R[0]) // 2
    e['same_button_split'] = bal(ncm(R[0][:h, sl], Y[0][:h], R[0][h:, sl]), Y[0][h:])
    res[nm] = e
res['readings'] = {b: int(OK[b].sum()) for b in OK}
res['locked_share'] = {b: round(float(Y[b].mean()), 3) for b in Y}
print(json.dumps(res, indent=1))
json.dump(res, open(ROOT / 'runs/125.2/transfer_gate.json', 'w'), indent=1)
