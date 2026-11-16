"""Card 142, second check before drafting: with the cube in the open drawer, does og4's read-out still put its weight
on the cube? The weight's centre in the image (row, column) is turned into a position by a quadratic fit made on
learning frames (true cube position, scores only), separately for the table and for the open drawer. If the centre
tracks the cube, the fit's error on test frames is small; the read-out's own answer is given beside it.

  bin/prun python tools/card142/check_open.py → runs/142/check_open.json
"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

ck = torch.load(ROOT / 'runs/129/readout_q.pt', weights_only=False)
print({k: (type(v).__name__) for k, v in ck.items()})
model = C.read_out().cuda(); model.load_state_dict(ck['model'] if 'model' in ck else ck['state']); model.eval()
GRP = 3

def run(sets):
    t, X = C.truth(sets), C.frames(sets)
    cen, peak, ans = [], [], []
    with torch.no_grad():
        for i0 in range(0, len(t['ep']), 64):
            f = torch.from_numpy(np.stack([np.asarray(X[t['set'][i]][t['row'][i]], np.float32) for i in range(i0, min(i0 + 64, len(t['ep'])))])).cuda()
            h = model.trunk(f); s = model.score(h)[:, :, GRP]; w = torch.softmax(s, 1)
            cen.append((w[:, :, None] * model.grid[None]).sum(1).cpu().numpy()); peak.append(s.max(1).values.cpu().numpy())
            ans.append(torch.einsum('bp,bpd->bd', w, model.where(model.grid) + model.offset(h)).cpu().numpy())
    return t, np.concatenate(cen), np.concatenate(peak), np.concatenate(ans) / 10 + C.CEN

def quad(c):
    return np.stack([np.ones(len(c)), c[:, 0], c[:, 1], c[:, 0] ** 2, c[:, 1] ** 2, c[:, 0] * c[:, 1]], 1)

tl, cl, pl, _ = run(C.LEARN); tt, ct, pt, at = run(C.TEST)
out = {}
for wh in ('table', 'open', 'hidden'):
    kl, kt = tl['where'] == wh, tt['where'] == wh
    B = np.linalg.lstsq(quad(cl[kl]), tl['cube'][kl][:, :2], rcond=None)[0]
    e = np.linalg.norm(quad(ct[kt]) @ B - tt['cube'][kt][:, :2], axis=1) * 100
    e0 = np.linalg.norm(tt['cube'][kt][:, :2] - tl['cube'][kl][:, :2].mean(0), axis=1) * 100
    ea = np.linalg.norm(at[kt][:, :2] - tt['cube'][kt][:, :2], axis=1) * 100
    out[wh] = dict(n=int(kt.sum()), weight_centre_fit_cm=round(float(np.median(e)), 2), within_3cm=round(100 * float((e < 3).mean()), 1),
                   mean_position_cm=round(float(np.median(e0)), 2), read_out_answer_cm=round(float(np.median(ea)), 2),
                   largest_score_median=round(float(np.median(pt[kt])), 2))
json.dump(out, open(ROOT / 'runs/142/check_open.json', 'w'), indent=1); print(json.dumps(out, indent=1))
