"""Card 121, feasibility gate: can an object's extent be grown from one seed patch by similarity in a deep block?

For 400 validation frames of the visual dataset and each object in view: the seed is a random patch on the object
(true mask, so only the growing step is tested). In block b, every patch's cosine similarity to the seed's vector
is computed after subtracting the frame's mean patch vector; the threshold is Otsu's on those 196 similarities (no
labels); the region is the patches above it that are connected to the seed (4-neighbours). Scored by IoU with the
object's true patches. Baselines on the same frames: the seed alone; the best k-means cluster (8 clusters) of the
same block; the oracle threshold per case (upper bound for this rule).

  bin/prun python tools/card121/gate.py → runs/121/gate.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card115'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card119'))
from masks import OBJECTS  # noqa: E402
from blocks import BLOCKS, DEV, encode  # noqa: E402
from encode import load_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/121'


def otsu(v):
    c = np.sort(v)
    best, thr = -1, c[0]
    for i in range(1, len(c)):
        a, b = c[:i], c[i:]
        s = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if s > best:
            best, thr = s, (c[i - 1] + c[i]) / 2
    return thr


def grow(sim, seed, thr):
    lab, _ = ndimage.label((sim >= thr).reshape(14, 14))
    lab = lab.reshape(-1)
    return (lab == lab[seed]) if lab[seed] else np.eye(196, dtype=bool)[seed]


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def main():
    masks = np.load(ROOT / 'runs/115/masks_val.npy', mmap_mode='r')
    obs = np.load(ROOT / 'runs/114/obs_visual-scene-play-v0-val.npy', mmap_mode='r')
    idx = np.arange(125, len(obs), 250)
    feats = [f.cpu().numpy() for f in encode(load_encoder(DEV), obs[idx])]
    rng = np.random.default_rng(0)
    seeds = {(n, oi): int(rng.choice(np.where(np.asarray(masks[i, oi]))[0]))
             for n, i in enumerate(idx) for oi in range(6) if np.asarray(masks[i, oi]).any()}
    res = {}
    for b, F in zip(BLOCKS, feats):
        acc = {k: {o: [] for o in OBJECTS} for k in ['grown', 'oracle_threshold', 'kmeans', 'seed_alone']}
        for n, i in enumerate(idx):
            f = F[n] - F[n].mean(0)
            f = f / np.linalg.norm(f, axis=1, keepdims=True)
            kl = KMeans(8, n_init=3, random_state=0).fit_predict(F[n])
            for oi, o in enumerate(OBJECTS):
                if (n, oi) not in seeds:
                    continue
                m = np.asarray(masks[i, oi]).astype(bool)
                sd = seeds[(n, oi)]
                sim = f @ f[sd]
                acc['grown'][o].append(iou(grow(sim, sd, otsu(sim)), m))
                acc['oracle_threshold'][o].append(max(iou(grow(sim, sd, t), m) for t in np.linspace(-0.2, 0.95, 24)))
                acc['kmeans'][o].append(max(iou(kl == g, m) for g in range(8)))
                acc['seed_alone'][o].append(1 / m.sum())
        res[f'block{b}'] = {k: {o: round(float(np.mean(v)), 3) for o, v in d.items()} for k, d in acc.items()}
        print(f'block {b}', json.dumps(res[f'block{b}']), flush=True)
    json.dump(res, open(OUT / 'gate.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
