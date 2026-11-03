"""Card 116.1: score the trained slot model against the pixel truth.

- Frames: each of the 2,000 truth frames is the last of 5 model steps (4 of history) run through the slot model; its
  masks are upsampled and scored (FG-ARI, mBO per object).
- Clips: each of the 100 clips runs 4 lead-in steps then 21 scored steps. For each object and step where it is visible,
  its best slot is the one with the highest IoU; consistency is the share of visible steps where the best slot is the
  object's most frequent best slot in that clip.
- Arm slots: in each truth frame, how many slots have most of their pixels on the arm.

  bin/prun python tools/card116/score116.py [--model slots.pt] → runs/116/score116[_<model>].json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from metrics import NAMES, scores, upsample  # noqa: E402
from slots import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/116'
SKIP = 5


@torch.no_grad()
def masks_for(model, feat, rows, dev='cuda', B=32):
    """rows: [n, T] feature rows → masks of every step [n, T, K, 1024]."""
    out = []
    for i in range(0, len(rows), B):
        r = rows[i:i + B]
        x = torch.from_numpy(np.asarray(feat[r.ravel()]).reshape(*r.shape, 1024, 384)).float().to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, _, m = model(x)
        out.append(m.float().cpu().numpy())
    return np.concatenate(out)


def main():
    dev = 'cuda'
    model_name = sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv else 'slots.pt'
    model = load_model(OUT / model_name, device=dev)
    t = np.load(OUT / 'truth.npz')
    feat = np.load(OUT / 'feat448_val.npy', mmap_mode='r')
    fidx = np.load(OUT / 'frames448_val.npy')
    row = lambda fr: np.searchsorted(fidx, fr)
    hist = t['frames'][:, None] + SKIP * np.arange(-4, 1)
    m = masks_for(model, feat, row(hist))[:, -1]
    seg = upsample(m)
    res = {'frames': scores(seg, t['labels'])}
    lab = t['labels']
    arm_slots = [sum(((s == k) & (l == 1)).sum() > 0.5 * max((s == k).sum(), 1) for k in np.unique(s))
                 for s, l in zip(seg, lab)]
    res['arm_slots_mean'] = round(float(np.mean(arm_slots)), 3)
    clips = t['clips']
    lead = clips[:, :1] - SKIP * np.arange(4, 0, -1)
    cm = masks_for(model, feat, row(np.concatenate([lead, clips], 1)), B=8)[:, 4:]
    cons = {k: [] for k in NAMES}
    for c in range(len(clips)):
        segs = upsample(cm[c])
        labs = t['clip_labels'][c]
        for j, name in enumerate(NAMES, start=1):
            best = []
            for s, l in zip(segs, labs):
                obj = l == j
                if not obj.any():
                    continue
                ids = np.unique(s)
                best.append(ids[np.argmax([((s == i) & obj).sum() / ((s == i) | obj).sum() for i in ids])])
            if best:
                vals, counts = np.unique(best, return_counts=True)
                cons[name].append(counts.max() / len(best))
    res['consistency'] = {k: round(float(np.mean(v)), 4) for k, v in cons.items()}
    res['consistency_min'] = min(res['consistency'].values())
    print(json.dumps(res, indent=1))
    tag = '' if model_name == 'slots.pt' else '_' + model_name.replace('.pt', '')
    json.dump(res, open(OUT / f'score116{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
