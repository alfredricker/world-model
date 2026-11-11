"""Card 119, main run step 2: do patches that change together group into the scene's objects?

On one block's signal (default block 2), episodes 900-999, no labels:
- boundaries: peaks of the frame score in its top 3% (card 115's label-free rule);
- a patch takes part in a boundary if its signal is at least half the top patch's;
- patches taking part in at least 10 boundaries are grouped by average-linkage clustering on 1 - Jaccard overlap of
  the boundaries they take part in; the number of groups (4..32) by silhouette. Other patches are in no group.
Scoring on 200 validation frames (every 500th): for each object, the best group's IoU with the object's true
patches in that frame, averaged over frames where the object covers a patch. Baseline: k-means on that frame's own
196 patch vectors of the same block, with the same number of clusters.

  bin/prun python tools/card119/units.py [--block 2] → runs/119/units_b<k>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card115'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from masks import OBJECTS  # noqa: E402
from score import frame_score, peaks  # noqa: E402
from blocks import BLOCKS, DEV, EP, HI, LO, encode  # noqa: E402
from encode import load_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/119'
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def best_iou(labels, mask):
    """labels [196] (-1 = no group), mask [196] bool → the best group's IoU."""
    best = 0.0
    for g in np.unique(labels[labels >= 0]):
        m = labels == g
        best = max(best, (m & mask).sum() / (m | mask).sum())
    return best


def main():
    b = int(arg('--block', '2'))
    c = np.load(OUT / f'signal_b{b}_tail.npz')
    eps = float(np.nanmedian(c['den']))
    z = c['num'] / (c['den'] + eps)
    loc = np.arange(len(z)) % EP
    z = np.where(((loc >= LO) & (loc <= HI))[:, None], z, np.nan)
    sc = frame_score(np.where(np.isnan(z), -np.inf, z))
    pk = peaks(sc)
    bd = pk[sc[pk] > np.quantile(sc[np.isfinite(sc)], 0.97)]
    zb = z[bd]
    part = zb >= 0.5 * zb.max(1, keepdims=True)
    used = np.where(part.sum(0) >= 10)[0]
    P = part[:, used].astype(np.float64)
    inter = P.T @ P
    cnt = np.diag(inter)
    D = 1 - inter / (cnt[:, None] + cnt[None, :] - inter)
    np.fill_diagonal(D, 0)
    best = None
    for k in range(4, min(32, len(used) - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average').fit_predict(D)
        s = silhouette_score(D, lab, metric='precomputed')
        if best is None or s > best[0]:
            best = (s, k, lab)
    sil, k, lab = best
    labels = np.full(196, -1)
    labels[used] = lab
    print(f'block {b}: {len(bd)} boundaries, {len(used)} patches grouped, {k} groups (silhouette {sil:.2f})')
    grid = labels.reshape(14, 14)
    for r in grid:
        print(' '.join('.' if v < 0 else chr(ord('a') + v) if v < 26 else chr(ord('A') + v - 26) for v in r))

    masks = np.load(ROOT / 'runs/115/masks_val.npy', mmap_mode='r')
    obs = np.load(ROOT / 'runs/114/obs_visual-scene-play-v0-val.npy', mmap_mode='r')
    idx = np.arange(250, len(obs), 500)
    enc = load_encoder(DEV)
    feats = encode(enc, obs[idx])[BLOCKS.index(b)].cpu().numpy()
    ours = {o: [] for o in OBJECTS}
    km = {o: [] for o in OBJECTS}
    for f, i in zip(feats, idx):
        kl = KMeans(k, n_init=3, random_state=0).fit_predict(f)
        for oi, o in enumerate(OBJECTS):
            m = np.asarray(masks[i, oi]).astype(bool)
            if m.any():
                ours[o].append(best_iou(labels, m))
                km[o].append(best_iou(kl, m))
    res = dict(block=b, boundaries=int(len(bd)), patches_grouped=int(len(used)), groups=int(k),
               silhouette=round(float(sil), 3), labels=labels.tolist(),
               best_group_iou={o: round(float(np.mean(v)), 3) for o, v in ours.items()},
               kmeans_best_cluster_iou={o: round(float(np.mean(v)), 3) for o, v in km.items()},
               object_patches={o: round(float(np.mean([np.asarray(masks[i, oi]).sum() for i in idx])), 1)
                               for oi, o in enumerate(OBJECTS)})
    print(json.dumps({k_: v for k_, v in res.items() if k_ != 'labels'}, indent=1))
    json.dump(res, open(OUT / f'units_b{b}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
