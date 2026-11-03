"""Card 115: per-patch surprise of card 114's predictor, and per-patch plain change, for every held-out frame.

For each frame t with t - 10 and t + 5 in one episode, the predictor gets the real frames t - 10, t - 5, t and their
action chunks and predicts frame t + 5 (one model step). Stored at index tau = t + 5 (the end of the interval in which a
change happened):
- err[tau, p]: mean squared error over the 384 numbers of patch p, predicted against real;
- chg[tau, p]: mean squared difference between patch p at t + 5 and at t (change with no prediction).
Frames without a full window are NaN.

  bin/prun python tools/card115/surprise.py --set val|tail → runs/115/surprise_<set>.npz
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card114'))
from model import Predictor  # noqa: E402
from train import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/115'
EP, SKIP = 1001, 5
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def main():
    which = arg('--set', 'val')
    dev = 'cuda'
    feat, frames, acts = load('patch', which)                  # every frame; acts[i] = the 5 actions after frame i
    n = len(frames)
    model = Predictor(196, 384).to(dev)
    model.load_state_dict(torch.load(ROOT / 'runs/114/pred_patch.pt', map_location=dev))
    model.eval()
    local = np.arange(n) % EP                                   # position inside the episode
    t = np.where((local >= 2 * SKIP) & (local + SKIP <= EP - 1))[0]
    err = np.full((n, 196), np.nan, np.float16)
    chg = np.full((n, 196), np.nan, np.float16)
    t0, B = time.time(), 256
    for i in range(0, len(t), B):
        tb = t[i:i + B]
        hist = tb[:, None] + SKIP * np.arange(-2, 1)
        z = torch.from_numpy(np.asarray(feat[hist.ravel()]).reshape(len(tb), 3, 196, 384)).float().to(dev)
        a = torch.from_numpy(acts[hist.ravel()].reshape(len(tb), 3, 25)).to(dev)
        real = torch.from_numpy(np.asarray(feat[tb + SKIP])).float().to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(z, a)[:, -1].float()
        err[tb + SKIP] = ((pred - real) ** 2).mean(-1).cpu().numpy()
        chg[tb + SKIP] = ((real - z[:, -1]) ** 2).mean(-1).cpu().numpy()
        if i % (B * 100) == 0:
            print(f'{which} {i}/{len(t)} {time.time() - t0:.0f}s', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f'surprise_{which}.npz', err=err, chg=chg)
    print(f'{which}: {len(t)} windows in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
