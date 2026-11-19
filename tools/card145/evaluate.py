"""Card 145: the three criteria for a trained state layer, and the baselines for events.

The state is run over whole episodes (201 steps): 100 training episodes, on which dimensions are grouped and named,
and the 100 validation episodes, on which everything is scored.
1. Events: steps with any gate open against steps where a true variable changes (one step of slack): precision, and
   recall per variable. Baselines: the contact sensor; plain change of the features (threshold set on training
   episodes for the best F1).
2. Values: for pairs of validation steps from different episodes (every fourth step), the distance between the two
   states within a variable's dimensions, as a score for "this variable differs": area under the curve per cell.
3. Variables: dimensions that open in at least 0.2% of steps are grouped when their openings overlap (Jaccard over
   0.3, connected sets; no truth). For scoring, each group is named after the variable whose steps of change its
   openings fall on most. Per variable: the share of its steps of change at which one of its dimensions opens, and
   the share of its dimensions' openings that fall on its steps of change (one step of slack).
Reported: gates open per step, false alarms while only the arm moves, reconstruction error, values read from the
state by ridge regression on 100 labelled steps, and when a slide's gates close against when it stops.

  bin/prun python tools/card145/evaluate.py <name> → runs/145/eval_<name>.json   (name 'baselines' for those alone)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shared as C  # noqa: E402
from model import State, run_filter  # noqa: E402
from train import norms  # noqa: E402

name = sys.argv[1]
nz = norms()
mean, scale = torch.from_numpy(nz['mean']).cuda(), float(nz['scale'])
sets = {k: (np.load(C.OUT / f'feat_{k}.npy', mmap_mode='r'), np.load(C.OUT / f'meta_{k}.npz')) for k in ('train', 'val')}
CH = {k: C.changes(m)[:100] for k, (x, m) in sets.items()}
out = {}


def frames(k, e):
    return (torch.from_numpy(np.asarray(sets[k][0][e])).cuda().float() - mean) / scale        # [201, 1024, 64]


if name == 'baselines':
    plain = {}
    for k in sets:
        d = []
        for e in range(100):
            x = frames(k, e)
            d.append(((x[1:] - x[:-1]) ** 2).sum(-1).mean(1).cpu().numpy())
        plain[k] = np.stack(d)
    any_tr = CH['train'].any(-1)
    best = max(np.quantile(plain['train'], np.linspace(0.05, 0.95, 91)),
               key=lambda c: 2 * ((plain['train'] > c) & any_tr).sum() / ((plain['train'] > c).sum() + any_tr.sum()))
    out['plain change of features'] = C.events(plain['val'] > best, CH['val'])
    out['contact sensor'] = C.events(sets['val'][1]['contact'][:100], CH['val'])
    out['contact sensor, no slack'] = C.events(sets['val'][1]['contact'][:100], CH['val'], slack=False)
    out['always'] = C.events(np.ones_like(any_tr), CH['val'])
    json.dump(out, open(C.OUT / 'eval_baselines.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
    sys.exit()

ck = torch.load(C.OUT / f'{name}.pt')
model = State(ck['S'], use_pose=not ck['nopose']).cuda()
model.load_state_dict(ck['state']); model.eval()
Z, G = {}, {}
for k, (X, M) in sets.items():
    pose = torch.from_numpy(((M['pose'][:100] - nz['pmean']) / nz['pstd']).astype(np.float32)).cuda()
    zs, gs = [], []
    for e0 in range(0, 100, 10):
        x = torch.stack([frames(k, e) for e in range(e0, e0 + 10)])
        z, g = run_filter(model, x, pose[e0:e0 + 10])
        zs.append(z.cpu().numpy()); gs.append(g.cpu().numpy())
    Z[k], G[k] = np.concatenate(zs), np.concatenate(gs)[:, 1:] > 0             # gates of the 200 transitions
S = Z['val'].shape[-1]
np.savez(C.OUT / f'states_{name}.npz', z_train=Z['train'], z_val=Z['val'], g_train=G['train'], g_val=G['val'])
M = sets['val'][1]
ch, opn = CH['val'], G['val']
quiet = ~C.wide(ch.any(-1))
out['gates'] = dict(open_per_step=round(float(opn.sum(-1).mean()), 3), steps_with_a_gate_open=round(float(opn.any(-1).mean()), 3),
                    open_while_only_the_arm_moves=round(float(opn.any(-1)[quiet].mean()), 3), share_of_steps_with_a_true_change=round(float(ch.any(-1).mean()), 3),
                    dimensions_ever_used=int((G['train'].mean((0, 1)) >= 0.002).sum()))
# 1. events
out['1 events'] = C.events(opn.any(-1), ch)
# 3. variables: groups by co-opening on training episodes, named by truth for scoring
gt, ct = G['train'].reshape(-1, S), CH['train']
active = np.where(gt.mean(0) >= 0.002)[0]
inter = (gt[:, active].T.astype(np.float32) @ gt[:, active].astype(np.float32))
union = gt[:, active].sum(0)[:, None] + gt[:, active].sum(0)[None] - inter
J = inter / np.maximum(union, 1)
lab = -np.ones(len(active), int)
for i in range(len(active)):
    if lab[i] < 0:
        lab[i] = lab.max() + 1
        stack = [i]
        while stack:
            a = stack.pop()
            for b in np.where((J[a] > 0.3) & (lab < 0))[0]:
                lab[b] = lab[i]; stack.append(b)
groups, dims = [], {n: [] for n in C.NAMES}
for g in range(lab.max() + 1 if len(active) else 0):
    d = active[lab == g]
    o = G['train'][..., d].any(-1)
    prec = [float((o & C.wide(ct[..., i])).sum() / max(o.sum(), 1)) for i in range(5)]
    v = int(np.argmax(prec))
    groups.append(dict(dims=[int(x) for x in d], open_share=round(float(o.mean()), 4), named=C.NAMES[v], on_its_steps=round(prec[v], 3),
                       on_each={n: round(p, 2) for n, p in zip(C.NAMES, prec)}))
    if prec[v] >= 1.5 * float(C.wide(ct[..., v]).mean()):                # named only if its openings fall on that variable's steps 1.5 times more often than chance
        dims[C.NAMES[v]] += [int(x) for x in d]
out['groups'] = groups
out['3 variables'] = {}
for i, n in enumerate(C.NAMES):
    o = opn[..., dims[n]].any(-1) if dims[n] else np.zeros(opn.shape[:2], bool)
    out['3 variables'][n] = dict(dims=dims[n], recall=round(float((ch[..., i] & C.wide(o)).sum() / max(ch[..., i].sum(), 1)), 3),
                                 precision=round(float((o & C.wide(ch[..., i])).sum() / max(o.sum(), 1)), 3) if o.any() else None)
# 2. values
ee, tt = np.meshgrid(np.arange(100), np.arange(0, 201, 4), indexing='ij')
ee, tt = ee.reshape(-1), tt.reshape(-1)
z = Z['val'][ee, tt]
out['2 values'] = {}
for cell, (a, b, y) in C.pair_cells(M, (ee, tt)).items():
    d = dims[cell.split()[0]]
    out['2 values'][cell] = dict(auc=C.auc(np.linalg.norm(z[a][:, d] - z[b][:, d], axis=1), y) if d else None,
                                 auc_whole_state=C.auc(np.linalg.norm(z[a] - z[b], axis=1), y), same=int((~y).sum()), differ=int(y.sum()))
# reported: reconstruction, values from 100 labels, slide stop
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    rng = np.random.default_rng(0)
    e, t = rng.integers(0, 100, 300), rng.integers(0, 201, 300)
    pose = torch.from_numpy(((M['pose'][e, t] - nz['pmean']) / nz['pstd']).astype(np.float32)).cuda()
    x = (torch.from_numpy(np.stack([np.asarray(sets['val'][0][a, b]) for a, b in zip(e, t)])).cuda().float() - mean) / scale
    err = [((model.dec(torch.from_numpy(Z['val'][e[i:i + 100], t[i:i + 100]]).cuda(), pose[i:i + 100], model.use_pose).float() - x[i:i + 100]) ** 2).sum(-1).mean().item() for i in range(0, 300, 100)]
out['reconstruction error (1 = the mean frame)'] = round(float(np.mean(err)), 4)
Mt = sets['train'][1]
rng = np.random.default_rng(1)
e, t = rng.integers(0, 100, 100), rng.integers(0, 201, 100)
A = np.c_[Z['train'][e, t], np.ones(100)]
w = C.where(M)[ee, tt]
vals = {}
for n, y_tr, y_te, m in (('drawer', Mt['drawer'][e, t], M['drawer'][ee, tt], np.ones(len(ee), bool)), ('window', Mt['window'][e, t], M['window'][ee, tt], np.ones(len(ee), bool)),
                         ('cube', Mt['cube'][e, t, :2], M['cube'][ee, tt, :2], w == 'table')):
    W = np.linalg.solve(A.T @ A + 1e-2 * np.eye(S + 1), A.T @ y_tr)
    p = np.c_[z, np.ones(len(z))] @ W
    r = np.abs(p - y_te) if p.ndim == 1 else np.linalg.norm(p - y_te, axis=1)
    vals[f'{n}, cm'] = round(float(np.median(r[m])) * 100, 2)
out['values from the state, ridge on 100 labelled steps'] = vals
lag = {}
for i, n in ((1, 'drawer'), (2, 'window')):
    lg, lc = [], []
    for ep in range(100):
        xch = ch[ep, :, i].astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, xch, 0]))
        for a, b in zip(edges[::2], edges[1::2]):                           # the slide changes in transitions a .. b-1
            for sig, acc in ((opn[ep][:, dims[n]].any(-1) if dims[n] else np.zeros(200, bool), lg), (M['contact'][ep], lc)):
                on = np.where(sig[max(a - 2, 0):min(b + 3, 200)])[0]              # up to 3 steps after the stop
                if len(on):
                    acc.append(max(a - 2, 0) + on[-1] - (b - 1))
    lag[n] = dict(gates_last_open_minus_true_stop_steps=float(np.median(lg)) if lg else None, contact_last_on_minus_true_stop_steps=float(np.median(lc)) if lc else None,
                  stretches=len(lc), stretches_with_a_gate=len(lg))
out['when a slide stops'] = lag
json.dump(out, open(C.OUT / f'eval_{name}.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'groups'}, indent=1))
print('groups', json.dumps(groups))
