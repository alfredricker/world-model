"""Card 114: score a trained predictor on the held-out events (validation episodes and training episodes 900-999).

From a real frame t (history t-10, t-5, t) and the real actions, the predictor rolls out k = 1..4 model steps open
loop. The read-out (probe for patches, a linear least-squares map for the state input, both fitted on real training
frames) gives the predicted slide positions and lock states.
- Pull: scored at k*, the first model step whose true slide position has moved >= 4 cm in the commanded direction:
  right if the prediction has moved >= 4 cm at k*. With no such step (it did not move), right if the prediction moves
  < 4 cm at every k <= 4. Accuracy is reported for locked and unlocked pulls, drawer and window apart.
- Press (rollout from 10 steps before the flip): right if the predicted lock at k* (the first step showing the flip)
  equals the true one.
- Quiet windows (no flip in the next 20 steps): a false flip is a predicted lock state that differs from the one a
  step before (the read of the real frame for k = 1). Rate over model steps, and over windows.
Baselines from the truth: "no change" (nothing moves, nothing flips) and "lock-blind" (every pull opens).

  bin/prun python tools/card114/score.py --input patch|state → runs/114/score_<input>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import H, Predictor  # noqa: E402
from probes import NAMES, targets, load as load_probe  # noqa: E402
from train import OUT, SKIP, load  # noqa: E402

K, MOVE = 4, 0.04
SLIDE = {'drawer': (NAMES.index('drawer'), 0), 'window': (NAMES.index('window'), 1)}
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


class LinearRead:
    """For the state input: least squares from the standardised 40 numbers to the evaluator's state."""
    def __init__(self, feat, y):
        X = np.c_[feat.reshape(len(feat), -1), np.ones(len(feat))]
        self.W = torch.tensor(np.linalg.lstsq(X, y, rcond=None)[0], dtype=torch.float32)

    def read(self, f):
        x = f.flatten(-2).float()
        out = torch.cat([x, torch.ones(*x.shape[:-1], 1, device=x.device)], -1) @ self.W.to(x.device)
        return torch.cat([out[..., :2].clamp(0, 1), out[..., 2:]], -1)


def run(model, reader, feat, acts, starts, dev, B):
    """→ reads of the real start frame [n, 7] and of the K predicted frames [n, K, 7]."""
    r0, rk = [], []
    for i in range(0, len(starts), B):
        t = starts[i:i + B]
        hist = t[:, None] + SKIP * np.arange(-(H - 1), 1)
        aidx = t[:, None] + SKIP * np.arange(-(H - 1), K)
        z = torch.from_numpy(np.asarray(feat[hist.ravel()]).reshape(len(t), H, *feat.shape[1:])).float().to(dev)
        a = torch.from_numpy(acts[aidx.ravel()].reshape(len(t), H - 1 + K, 25)).to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model.rollout(z, a).float()
            r0.append(reader.read(z[:, -1]).float().cpu().numpy())
            rk.append(reader.read(pred).float().cpu().numpy())
    return np.concatenate(r0), np.concatenate(rk)


def score_set(model, reader, which, split, ev, st, dev, B):
    feat, frames, acts = load(which, split)
    res = {}
    for name, (j, _) in SLIDE.items():
        t, d, locked = ev[f'{split}/{name}/t'], ev[f'{split}/{name}/dir'], ev[f'{split}/{name}/locked']
        q = st[name]
        true_move = np.stack([(q[t + SKIP * k] - q[t]) * d for k in range(1, K + 1)], 1)
        r0, rk = run(model, reader, feat, acts, t, dev, B)
        pred_move = (rk[:, :, j] - r0[:, None, j]) * d[:, None]
        opened = true_move >= MOVE
        has = opened.any(1)
        kstar = opened.argmax(1)
        right = np.where(has, pred_move[np.arange(len(t)), kstar] >= MOVE, (pred_move < MOVE).all(1))
        lenient = (pred_move >= MOVE).any(1) == has
        res[name] = dict(locked=locked, right=right, lenient=lenient, has=has)
    t, b = ev[f'{split}/press/t'], ev[f'{split}/press/button']
    btn = st['buttons']
    truth = np.stack([btn[t + SKIP * k, b] for k in range(K + 1)], 1)
    kstar = (truth[:, 1:] != truth[:, :1]).argmax(1) + 1
    r0, rk = run(model, reader, feat, acts, t, dev, B)
    pred = np.concatenate([r0[:, None, :2], rk[:, :, :2]], 1)[np.arange(len(t)), :, b] > 0.5
    res['press'] = dict(right=pred[np.arange(len(t)), kstar] == truth[np.arange(len(t)), kstar].astype(bool),
                        button=b)
    tq = ev[f'{split}/quiet/t']
    tq = tq[np.random.default_rng(0).choice(len(tq), min(4000, len(tq)), replace=False)]
    r0, rk = run(model, reader, feat, acts, tq, dev, B)
    s = np.concatenate([r0[:, None, :2], rk[:, :, :2]], 1) > 0.5
    flips = s[:, 1:] != s[:, :-1]                                   # [n, K, 2]
    cube_true = np.stack([st['cube'][tq + SKIP * k] for k in range(1, K + 1)], 1)
    res['quiet'] = dict(step_flip=flips.reshape(-1), window_flip=flips.any((1, 2)),
                        cube_err=np.linalg.norm(rk[:, :, 4:7] - cube_true, axis=-1))
    return res


def main():
    which = arg('--input', 'patch')
    dev = 'cuda'
    feat, frames, _ = load(which, 'train')
    N, D = feat.shape[1:]
    model = Predictor(N, D).to(dev)
    model.load_state_dict(torch.load(OUT / f'pred_{which}.pt', map_location=dev))
    model.eval()
    if which == 'patch':
        reader = load_probe(dev)
        ev = dict(np.load(OUT / 'events.npz'))
        pre = 'state'
    else:
        st = np.load(OUT / 'sstate_train.npz')
        reader = LinearRead(np.asarray(feat), targets(st, frames))
        ev = dict(np.load(OUT / 'sevents.npz'))
        pre = 'sstate'
    sv = dict(np.load(OUT / f'{pre}_val.npz'))
    stt = dict(np.load(OUT / f'{pre}_train.npz'))
    tail = {k: v[900 * 1001:] for k, v in stt.items()}
    B = 64 if which == 'patch' else 4096
    parts = [score_set(model, reader, which, 'val', ev, sv, dev, B),
             score_set(model, reader, which, 'tail', ev, tail, dev, B)]
    cat = lambda key, f: np.concatenate([p[key][f] for p in parts])
    out = {}
    for name in SLIDE:
        locked, right, lenient, has = (cat(name, f) for f in ['locked', 'right', 'lenient', 'has'])
        for lk, tag in [(True, 'locked'), (False, 'unlocked')]:
            m = locked == lk
            out[f'{name}_{tag}'] = dict(n=int(m.sum()), right=round(float(right[m].mean()), 4),
                                        lenient=round(float(lenient[m].mean()), 4),
                                        no_change=round(float((~has[m]).mean()), 4),
                                        lock_blind=round(float(has[m].mean()), 4))
    pr, pb = cat('press', 'right'), cat('press', 'button')
    out['press'] = dict(n=int(len(pr)), right=round(float(pr.mean()), 4),
                        button0=round(float(pr[pb == 0].mean()), 4), button1=round(float(pr[pb == 1].mean()), 4),
                        no_change=0.0)
    sf, wf, ce = cat('quiet', 'step_flip'), cat('quiet', 'window_flip'), cat('quiet', 'cube_err')
    out['quiet'] = dict(n_windows=int(len(wf)), no_false_flip_per_step=round(float(1 - sf.mean()), 4),
                        no_false_flip_per_window=round(float(1 - wf.mean()), 4),
                        cube_err_cm_by_k=[round(float(x) * 100, 2) for x in ce.mean(0)])
    print(json.dumps(out, indent=1))
    json.dump(out, open(OUT / f'score_{which}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
