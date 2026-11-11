"""Card 126, feasibility gate: what does a touch change, and how well can memory alone say so without training?

Task. Given the scene before a touch and the gripper's position at contact and at release, give every patch a
score for "this patch will have changed lastingly". Scored on fresh touches against the truth (scoring only):
the patches of each object that truly moved, before and after (cube: 1 cm; drawer, window: 1 cm), and of a button
that flipped. Measures:
- place: average precision of the patch scores against the true patches, for touches with a true change, by kind;
- whether: area under the ROC curve of the touch's highest... mean of its 4 highest scores, for touches with a
  true change against touches without, by place of the touch.
Arms, none trained:
- oracle: the touch's own observed change signal (card 122's, block 2). The ceiling for anything trained on it.
- recall by motion: the mean change signal of the 8 remembered touches whose gripper start and end are nearest.
  It never looks at the scene.
- nothing: a constant map (average precision = the true patches' share).

  bin/prun python tools/card126/gate.py [--mem m600 --first 600 --test t500 --tfirst 500] → runs/126/gate.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
EP = 1001
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def load(name, first):
    z = np.load(DATA / 'scene-play-v0.npz')
    sl = slice(first * EP, (first + 100) * EP)
    qpos, btn, obs = z['qpos'][sl], z['button_states'][sl], z['observations'][sl]
    c = np.load(ROOT / f'runs/122/computed_{name}.npz')
    st, en, masks = c['st'], c['en'], c['masks']
    a, b = st - 2, np.minimum(en + 2, len(qpos) - 1)
    lo = np.maximum(st - 5, (st // EP) * EP)
    moved = dict(cube=np.linalg.norm(qpos[b, 14:17] - qpos[a, 14:17], axis=1) > 0.01,
                 drawer=np.abs(qpos[b, 23] - qpos[a, 23]) > 0.01, window=np.abs(qpos[b, 24] - qpos[a, 24]) > 0.01,
                 button0=btn[b, 0] != btn[lo, 0], button1=btn[b, 1] != btn[lo, 1])
    idx = dict(cube=1, button0=2, button1=3, drawer=4, window=5)
    truth = np.zeros((len(st), 1024), bool)
    for o, m in moved.items():
        truth[m] |= (masks[a[m], idx[o]] | masks[b[m], idx[o]]).astype(bool)
    return dict(st=st, z=c['tnum'], den=c['tden'], motion=np.c_[obs[st, 12:15], obs[b, 12:15]], truth=truth,
                moved=moved, arm=(masks[a, 0] | masks[b, 0]).astype(bool))


def main():
    M = load(arg('--mem', 'm600'), int(arg('--first', 600)))
    V = load(arg('--test', 't500'), int(arg('--tfirst', 500)))
    eps = np.median(M['den'])
    zm, zv = M['z'] / (M['den'] + eps), V['z'] / (V['den'] + eps)
    mu, sd = M['motion'].mean(0), M['motion'].std(0)
    a, b = (M['motion'] - mu) / sd, (V['motion'] - mu) / sd
    d = ((b[:, None] - a[None]) ** 2).sum(-1)
    nn = np.argsort(d, 1)[:, :8]
    recall = np.log1p(zm)[nn].mean(1)
    arms = dict(oracle=np.log1p(zv), recall_by_motion=recall)
    any_change = V['truth'].any(1)
    kinds = {k: m & (sum(V['moved'].values()) == 1) for k, m in V['moved'].items()}       # touches with one true change
    res = dict(touches=dict(memory=len(zm), test=len(zv)), with_a_true_change=round(float(any_change.mean()), 3),
               kinds={k: int(m.sum()) for k, m in kinds.items()}, place={}, whether={})
    for k, m in kinds.items():
        e = dict(nothing=round(float(V['truth'][m].mean()), 3))
        for nm, s in arms.items():
            e[nm] = round(float(np.mean([average_precision_score(V['truth'][i], s[i]) for i in np.where(m)[0]])), 3)
            e[nm + '_arm_set_aside'] = round(float(np.mean([average_precision_score(
                V['truth'][i][~V['arm'][i] | V['truth'][i]], s[i][~V['arm'][i] | V['truth'][i]]) for i in np.where(m)[0]])), 3)
        res['place'][k] = e
    for nm, s in arms.items():
        top = np.sort(s, 1)[:, -4:].mean(1)
        res['whether'][nm] = round(float(roc_auc_score(any_change, top)), 3)
    print(json.dumps(res, indent=1))
    json.dump(res, open(ROOT / 'runs/126/gate.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
