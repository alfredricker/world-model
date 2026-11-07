"""Card 116.3: SAM 2 parts on the 256-px renders, the parts score, and the gate (teacher, k-means, ceiling).

- Teacher: SAM 2.1-small's automatic masks (16 x 16 points, no prompts, no labels) on the 256 x 256 render, painted
  smaller on top; pixels no mask covers form one more segment. Each patch of the 32 x 32 grid (8 x 8 render pixels)
  takes the segment covering most of its pixels. Segments of fewer than 2 patches are unsupervised (-1); beyond K = 32
  segments the largest are unsupervised too.
- Parts score: for each object, the union of the segments (slots) with most of their pixels on it, IoU with the
  object; 0 if no segment is mostly on it. Averaged over frames where the object is visible.

  bin/prun python tools/card116/parts.py --gate          → runs/116/parts_gate.json
  bin/prun python tools/card116/parts.py --label --minutes 25 → runs/116/parts_train.npz (rows, labels [n, 4, 1024])
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from metrics import NAMES, onehot, patch_of_pixels, scores, upsample  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/116'
CKPT = Path.home() / '.cache/wm-sam2/sam2/checkpoints/sam2.1_hiera_small.pt'
EP, T, K, POINTS = 1001, 4, 32, 16
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def generator(points=POINTS):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    model = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', str(CKPT), device='cuda')
    return SAM2AutomaticMaskGenerator(model, points_per_side=points, pred_iou_thresh=0.7, stability_score_thresh=0.85,
                                      min_mask_region_area=0)


def teacher(gen, img):
    """256 x 256 x 3 render → patch labels [1024]: 0..M-1, -1 unsupervised."""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        masks = gen.generate(np.ascontiguousarray(img))
    lab = np.zeros((256, 256), np.int16)
    for k, m in enumerate(sorted(masks, key=lambda m: -m['area']), start=1):
        lab[m['segmentation']] = k
    blocks = lab.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3).reshape(1024, 64)
    patch = np.array([np.bincount(b).argmax() for b in blocks], np.int16)
    ids, counts = np.unique(patch, return_counts=True)
    keep = ids[counts >= 2][np.argsort(counts[counts >= 2], kind='stable')][:K]       # the K smallest
    out = np.full(1024, -1, np.int16)
    for j, i in enumerate(np.sort(keep)):
        out[patch == i] = j
    return out


def parts_score(seg, truth):
    """seg, truth: [n, 64, 64] ids → {object: parts score}, mean."""
    res = {n: [] for n in NAMES}
    for s, t in zip(seg, truth):
        ids, inv = np.unique(s, return_inverse=True)
        maj = np.array([np.bincount(t.ravel()[inv.ravel() == k], minlength=7).argmax() for k in range(len(ids))])
        for j, n in enumerate(NAMES, start=1):
            obj = t == j
            if not obj.any():
                continue
            u = np.isin(s, ids[maj == j])
            res[n].append((u & obj).sum() / (u | obj).sum())
    mbo = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    return dict(parts=mbo, mean_parts=round(float(np.mean(list(mbo.values()))), 4))


def seg_of_labels(lab):
    """Patch labels [n, 1024] (with -1) → pixel ids [n, 64, 64]; unsupervised patches as one more segment."""
    lab = lab.astype(np.int64).copy()
    lab[lab < 0] = lab.max() + 1
    return upsample(onehot(lab, lab.max() + 1))


def gate():
    t = np.load(OUT / 'truth.npz')
    frames, truth = t['frames'], t['labels']
    vidx = np.load(OUT / 'frames448_val.npy')
    res = {}
    # ceiling: each patch its true majority
    pid = patch_of_pixels(32).reshape(-1)
    maj = np.zeros((len(truth), 1024), np.int64)
    for i, lab in enumerate(truth.reshape(len(truth), -1)):
        c = np.zeros((1024, 7), np.int64)
        np.add.at(c, (pid, lab), 1)
        maj[i] = c.argmax(1)
    seg = upsample(onehot(maj, 7))
    res['ceiling'] = {**parts_score(seg, truth), **scores(seg, truth)}
    print('ceiling', res['ceiling'], flush=True)
    # k-means, 32 clusters, on the new features
    from sklearn.cluster import MiniBatchKMeans
    ftr = np.load(OUT / 'feat256_train.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    X = np.asarray(ftr[np.sort(rng.choice(len(ftr), 2000, replace=False))]).reshape(-1, 384).astype(np.float32)
    km = MiniBatchKMeans(K, random_state=0, batch_size=4096, n_init=3).fit(X[rng.choice(len(X), 200000, replace=False)])
    fva = np.load(OUT / 'feat256_val.npy', mmap_mode='r')
    ids = km.predict(np.asarray(fva[np.searchsorted(vidx, frames)]).reshape(-1, 384).astype(np.float32))
    seg = upsample(onehot(ids.reshape(len(frames), 1024), K))
    res['kmeans32'] = {**parts_score(seg, truth), **scores(seg, truth)}
    print('kmeans32', res['kmeans32'], flush=True)
    # teacher on 300 truth frames
    rgb = np.load(OUT / 'rgb256_val.npy', mmap_mode='r')
    gen, t0 = generator(), time.time()
    labs = np.stack([teacher(gen, rgb[r]) for r in np.searchsorted(vidx, frames[:300])])
    sec = (time.time() - t0) / 300
    seg = seg_of_labels(labs)
    nseg = labs.max(1) + 1
    res['teacher'] = {**parts_score(seg, truth[:300]), **scores(seg, truth[:300]), 'seconds_per_frame': round(sec, 3),
                      'segments_mean': round(float(nseg.mean()), 1), 'segments_p95': int(np.percentile(nseg, 95)),
                      'frames_over_32': round(float((nseg >= K).mean()), 3)}
    print('teacher', res['teacher'], flush=True)
    bar116 = {n: round(0.7 * res['ceiling']['mbo'][n], 4) for n in NAMES}
    res['bars'] = {n: round(min(bar116[n], 0.9 * res['teacher']['parts'][n]), 4) for n in NAMES}
    print('bars', res['bars'])
    json.dump(res, open(OUT / 'parts_gate.json', 'w'), indent=1)


def label():
    budget = float(arg('--minutes', 25)) * 60
    frames = np.load(OUT / 'frames448_train.npy')
    rgb = np.load(OUT / 'rgb256_train.npy', mmap_mode='r')
    i = np.arange(len(frames) - T + 1)
    starts = np.random.default_rng(0).permutation(i[frames[i] // EP == frames[i + T - 1] // EP])
    path = OUT / 'parts_train.npz'
    rows, labs = ([*np.load(path)['rows']], [*np.load(path)['labels']]) if path.exists() else ([], [])
    done = set(rows)
    gen, t0 = generator(), time.time()
    for s in starts:
        if time.time() - t0 > budget:
            break
        if s in done:
            continue
        labs.append(np.stack([teacher(gen, rgb[s + k]) for k in range(T)]))
        rows.append(s)
        if len(rows) % 100 == 0:
            print(f'{len(rows)} segments {time.time() - t0:.0f}s', flush=True)
            np.savez(path, rows=np.array(rows), labels=np.stack(labs))
    np.savez(path, rows=np.array(rows), labels=np.stack(labs))
    print(f'saved {len(rows)} segments ({len(rows) * T} frames); {time.time() - t0:.0f}s this run', flush=True)


if __name__ == '__main__':
    gate() if '--gate' in sys.argv else label()
