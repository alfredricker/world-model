"""Card 124: store the frozen encoder's features of the training frames, so training does not run the encoder.

Blocks 2 and 12 (each through the final norm), joined: fp16 [episodes, 201, 1024, 768], 142 GB for 450 episodes.
Temporary (the user, 2026-11-10: "we can temporarily store 150GB"); delete after the card's runs.

  bin/prun python tools/card124/encode_train.py [--part i/n] [--from E --pause seconds] → runs/124/feat_train.npy
  (--pause flushes and waits after each episode: a 142 GB write without it stalled the whole machine's disk)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import FEAT, N, OUT, features  # noqa: E402
from encode import load_encoder  # noqa: E402

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def main():
    rgb = np.load(OUT / 'rgb_train.npy', mmap_mode='r')
    neps, steps = rgb.shape[:2]
    path = OUT / 'feat_train.npy'
    i, n = (int(v) for v in arg('--part', '0/1').split('/'))
    mode = 'r+' if path.exists() else 'w+'
    out = np.lib.format.open_memmap(path, mode, np.float16, (neps, steps, N, FEAT)) if mode == 'w+' else \
        np.load(path, mmap_mode='r+')
    enc = load_encoder('cuda')
    t0 = time.time()
    first, pause = int(arg('--from', -1)), float(arg('--pause', 0))
    for e in (range(first, neps) if first >= 0 else range(neps * i // n, neps * (i + 1) // n)):
        fr = torch.from_numpy(np.ascontiguousarray(rgb[e])).cuda()
        out[e] = torch.cat([features(enc, fr[k:k + 67]) for k in range(0, steps, 67)]).half().cpu().numpy()
        if pause:
            out.flush()
            time.sleep(pause)
        if e % 25 == 0:
            print(f'episode {e}/{neps} {time.time() - t0:.0f}s', flush=True)
    out.flush()
    print(f'part {i}/{n} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
