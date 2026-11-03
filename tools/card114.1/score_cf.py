"""Card 114.1: score card 114's trained predictors on the factual and counterfactual replays.

Truth: the replayed slide position at t + 5k (k = 1..4); "opened" at k when it has moved >= 4 cm in the commanded
direction. A changed-outcome event is a counterfactual whose "opened at any k <= 4" differs from its factual twin.
- Model: card 114's rule (right at k*, the first true opened step; with none, right if no step is forecast >= 4 cm).
- Action-only baseline: gradient-boosted trees on the 6 action chunks (the ones the model sees), trained on the pulls of
  training episodes 0-899 of the same dataset; right if its "opened" equals the truth.
- Lock-reading rule: the read-out's lock state on the start frame t; locked → stays shut, unlocked → opens.
Kinds: newly locked (counterfactual of an unlocked pull) and newly unlocked, drawer and window apart.

  bin/prun python tools/card114.1/score_cf.py --data visual|state → runs/114.1/score_<data>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card114'))
from encode import encode, load_encoder  # noqa: E402
from events import find  # noqa: E402
from model import H, Predictor  # noqa: E402
from probes import NAMES, targets, load as load_probe  # noqa: E402
from score import LinearRead  # noqa: E402
from train import load, state_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/114.1'
EP, SKIP, K, MOVE = 1001, 5, 4, 0.04
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def action_baseline(pre):
    """Train on pulls of episodes 0-899: the 6 action chunks around the start → opened within 20 steps."""
    st = dict(np.load(ROOT / f'runs/114/{pre}_train.npz'))
    st = {k: v[:900 * EP] for k, v in st.items()}
    ev = find(st)
    models = {}
    for slide in ['drawer', 'window']:
        e = ev[slide]
        t, d = e['t'], e['dir']
        X = np.stack([st['actions'][t - 2 * SKIP + j] for j in range(6 * SKIP)], 1).reshape(len(t), -1)
        q = st[slide]
        y = np.stack([(q[t + SKIP * k] - q[t]) * d for k in range(1, K + 1)], 1).max(1) >= MOVE
        models[slide] = GradientBoostingClassifier(random_state=0).fit(X, y)
    return models


def main():
    which = arg('--data', 'visual')
    dev = 'cuda'
    r = dict(np.load(OUT / f'replay_{which}.npz'))
    n = len(r['t'])
    d = r['dir']
    move = (r['pos'] - r['pos'][:, :1]) * d[:, None]              # [n, K+1], index 0 = start
    opened = move[:, 1:] >= MOVE
    has = opened.any(1)
    res = {'replays': n}
    fact, cf = ~r['cf'], r['cf']
    # gate 1: replay fidelity (factual)
    rec_move = (r['pos_rec'] - r['pos_rec'][:, :1]) * d[:, None]
    within = (np.abs(r['pos'] - r['pos_rec'])[:, 1:] <= 0.01).all(1)
    res['fidelity'] = dict(slide_within_1cm=round(float(within[fact].mean()), 4),
                           outcome_agrees=round(float(((rec_move[:, 1:] >= MOVE).any(1) == has)[fact].mean()), 4))
    if which == 'visual':
        diff = np.abs(r['hist'].astype(np.int16) - r['hist_rec'].astype(np.int16)).mean((1, 2, 3, 4))
        res['fidelity']['frame_abs_diff_factual'] = round(float(diff[fact].mean()), 3)
        res['fidelity']['frame_abs_diff_factual_max'] = round(float(diff[fact].max()), 3)
    # model inputs
    if which == 'visual':
        enc = load_encoder(dev)
        hist = np.concatenate([encode(enc, r['hist'][i:i + 512].reshape(-1, 64, 64, 3), dev)
                               for i in range(0, n, 512)]).reshape(n, H, 196, 384)
        N, D = 196, 384
        reader = load_probe(dev)
    else:
        mu, sd = state_stats()
        hist = ((r['hist'] - mu) / sd)[:, :, None, :].astype(np.float32)
        N, D = 1, 40
        feat, frames, _ = load('state', 'train')
        reader = LinearRead(np.asarray(feat), targets(np.load(ROOT / 'runs/114/sstate_train.npz'), frames))
    model = Predictor(N, D).to(dev)
    model.load_state_dict(torch.load(ROOT / f"runs/114/pred_{'patch' if which == 'visual' else 'state'}.pt",
                                     map_location=dev))
    model.eval()
    r0, rk = [], []
    for i in range(0, n, 64):
        z = torch.from_numpy(np.asarray(hist[i:i + 64])).float().to(dev)
        a = torch.from_numpy(r['act'][i:i + 64]).to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model.rollout(z, a).float()
            r0.append(reader.read(z[:, -1]).float().cpu().numpy())
            rk.append(reader.read(pred).float().cpu().numpy())
    r0, rk = np.concatenate(r0), np.concatenate(rk)
    j = np.where(r['slide'] == 'drawer', NAMES.index('drawer'), NAMES.index('window'))
    pmove = (rk[np.arange(n), :, j] - r0[np.arange(n), j][:, None]) * d[:, None]
    kstar = opened.argmax(1)
    right = np.where(has, pmove[np.arange(n), kstar] >= MOVE, (pmove < MOVE).all(1))
    lenient = (pmove >= MOVE).any(1) == has
    # baselines
    bi = np.where(r['slide'] == 'drawer', 0, 1)
    lock_read_locked = r0[np.arange(n), bi] < 0.5
    rule_right = (~lock_read_locked) == has
    models = action_baseline('state' if which == 'visual' else 'sstate')
    act_open = np.zeros(n, bool)
    for slide, m in models.items():
        s = r['slide'] == slide
        act_open[s] = m.predict(r['act'][s].reshape(s.sum(), -1))
    act_right = act_open == has
    # pair each counterfactual with its factual twin (stored one after the other)
    changed = np.zeros(n, bool)
    changed[1::2] = has[1::2] != has[0::2]
    res['lock_read_on_start_frame_right'] = round(float(((r0[np.arange(n), bi] >= 0.5) == (
        r['locked'] ^ r['cf'] == False)).mean()), 4)  # noqa: E712  (true lock = original lock flipped if cf)
    for slide in ['drawer', 'window']:
        for kind, sel in [('factual_locked', fact & r['locked']), ('factual_unlocked', fact & ~r['locked']),
                          ('newly_locked', cf & ~r['locked']), ('newly_unlocked', cf & r['locked'])]:
            m = sel & (r['slide'] == slide)
            entry = dict(n=int(m.sum()), model_right=round(float(right[m].mean()), 4),
                         model_lenient=round(float(lenient[m].mean()), 4),
                         actions_only_right=round(float(act_right[m].mean()), 4),
                         lock_rule_right=round(float(rule_right[m].mean()), 4))
            if kind.startswith('newly'):
                mc = m & changed
                entry['changed'] = dict(n=int(mc.sum()), model_right=round(float(right[mc].mean()), 4),
                                        model_lenient=round(float(lenient[mc].mean()), 4),
                                        actions_only_right=round(float(act_right[mc].mean()), 4),
                                        lock_rule_right=round(float(rule_right[mc].mean()), 4),
                                        model_forecast_opened=round(float((pmove[mc] >= MOVE).any(1).mean()), 4))
            res[f'{slide}_{kind}'] = entry
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / f'score_{which}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
