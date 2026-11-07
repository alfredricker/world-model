"""Card 116.2: SAM 2.1-small's automatic masks as an unlabelled partition of the 32 x 32 patch grid (the teacher).

Each 64 x 64 frame is upsampled (bicubic) to 512 x 512 and SAM 2's automatic mask generator proposes masks (no prompts,
no labels). Masks are painted into one label image, either smaller on top ('fine') or larger on top ('coarse'); pixels
no mask covers form one more segment. The label image is read at 64 x 64, and each patch (exactly 2 x 2 frame pixels at
448 px) takes the segment covering most of its pixels. Segments of fewer than 2 patches are left unsupervised (-1).

  --gate:  300 truth frames, settings a (32 x 32 points, fine), b (32 x 32, coarse), c (16 x 16, the better order);
           scores each partition like the slots, counts segments per frame → runs/116/teacher_gate.json
  --label --setting a|b|c --minutes 9: whole 4-step training segments (random, seed 0), until time is up
           → runs/116/teacher_train.npz (start rows into feat448_train, labels [n, 4, 1024]); appends if present.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from metrics import scores, upsample  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/116'
CKPT = Path.home() / '.cache/wm-sam2/sam2/checkpoints/sam2.1_hiera_small.pt'
EP, T = 1001, 4
SETTINGS = {'a': (32, 'fine'), 'b': (32, 'coarse'), 'c': (16, None)}
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def generator(points):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    model = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', str(CKPT), device='cuda')
    return SAM2AutomaticMaskGenerator(model, points_per_side=points, pred_iou_thresh=0.7, stability_score_thresh=0.85,
                                      min_mask_region_area=0)


def sam_masks(gen, frame):
    x = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1)[None].float()
    big = F.interpolate(x, size=(512, 512), mode='bicubic', align_corners=False).clamp(0, 255)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return gen.generate(big[0].permute(1, 2, 0).byte().numpy())


def partition(masks, order):
    """SAM masks → patch labels [1024] (0..M-1, -1 unsupervised)."""
    lab = np.zeros((512, 512), np.int16)                           # 0 = covered by no mask
    key = (lambda m: -m['area']) if order == 'fine' else (lambda m: m['area'])
    for k, m in enumerate(sorted(masks, key=key), start=1):
        lab[m['segmentation']] = k
    px = lab[4::8, 4::8]                                           # 64 x 64
    blocks = px.reshape(32, 2, 32, 2).transpose(0, 2, 1, 3).reshape(1024, 4)
    patch = np.array([np.bincount(b).argmax() for b in blocks], np.int16)
    ids, counts = np.unique(patch, return_counts=True)
    keep = ids[counts >= 2]
    out = np.full(1024, -1, np.int16)
    for j, i in enumerate(keep):
        out[patch == i] = j
    return out


def as_segments(patch):
    """Patch labels → one-hot [M+1, 1024] (unsupervised patches as one more segment) for metrics.upsample."""
    ids = patch.copy()
    ids[ids < 0] = ids.max() + 1
    return np.eye(ids.max() + 1, dtype=np.float32)[ids].T


def score_partitions(parts, labels):
    K = max(len(p) for p in (as_segments(q) for q in parts))
    seg = np.stack([np.pad(as_segments(p), ((0, K - as_segments(p).shape[0]), (0, 0))) for p in parts])
    return scores(upsample(seg), labels), upsample(seg)


def own_segment(seg, labels, j, thr=0.5):
    """Share of frames with object j visible where some segment has IoU >= thr with it."""
    hits = []
    for s, l in zip(seg, labels):
        obj = l == j
        if obj.any():
            hits.append(max(((s == i) & obj).sum() / ((s == i) | obj).sum() for i in np.unique(s)) >= thr)
    return round(float(np.mean(hits)), 3)


def gate():
    t = np.load(OUT / 'truth.npz')
    frames, labels = t['frames'][:300], t['labels'][:300]
    obs = np.load(ROOT / 'runs/114/obs_visual-scene-play-v0-val.npy', mmap_mode='r')
    res = {}
    for points, orders in ((32, ('fine', 'coarse')), (16, ('fine', 'coarse'))):
        gen = generator(points)
        parts, n_masks, t0 = {o: [] for o in orders}, [], time.time()
        for f in frames:
            masks = sam_masks(gen, obs[f])
            n_masks.append(len(masks))
            for o in orders:
                parts[o].append(partition(masks, o))
        sec = (time.time() - t0) / len(frames)
        for o in orders:
            sc, seg = score_partitions(parts[o], labels)
            nseg = np.array([p.max() + 1 for p in parts[o]])
            sc.update(points=points, order=o, seconds_per_frame=round(sec, 3), sam_masks_per_frame=float(np.mean(n_masks)),
                      segments_mean=round(float(nseg.mean()), 2), segments_p95=int(np.percentile(nseg, 95)),
                      share_frames_over_7=round(float((nseg > 7).mean()), 3),
                      unsupervised_patches=round(float(np.mean([(p < 0).sum() for p in parts[o]])), 1),
                      own_segment={n: own_segment(seg, labels, j) for j, n in ((2, 'cube'), (3, 'button0'), (4, 'button1'))})
            res[f'{points}_{o}'] = sc
            print(f'{points}_{o}', json.dumps(sc), flush=True)
        del gen
        torch.cuda.empty_cache()
    json.dump(res, open(OUT / 'teacher_gate.json', 'w'), indent=1)


def label():
    points, order = SETTINGS[arg('--setting', 'a')][0], arg('--order', None) or SETTINGS[arg('--setting', 'a')][1]
    budget = float(arg('--minutes', 9)) * 60
    frames = np.load(OUT / 'frames448_train.npy')
    obs = np.load(ROOT / 'runs/114/obs_visual-scene-play-v0.npy', mmap_mode='r')
    i = np.arange(len(frames) - T + 1)
    starts = i[frames[i] // EP == frames[i + T - 1] // EP]
    starts = np.random.default_rng(0).permutation(starts)
    path = OUT / 'teacher_train.npz'
    done_rows, done_labels = (np.load(path)['rows'], np.load(path)['labels']) if path.exists() else (
        np.zeros(0, np.int64), np.zeros((0, T, 1024), np.int16))
    todo = [s for s in starts if s not in set(done_rows.tolist())]
    gen = generator(points)
    rows, labs, t0 = [], [], time.time()
    for s in todo:
        if time.time() - t0 > budget:
            break
        labs.append([partition(sam_masks(gen, obs[frames[s + k]]), order) for k in range(T)])
        rows.append(s)
        if len(rows) % 50 == 0:
            print(f'{len(rows)} segments {time.time() - t0:.0f}s', flush=True)
    rows = np.concatenate([done_rows, np.array(rows, np.int64)])
    labels = np.concatenate([done_labels, np.array(labs, np.int16).reshape(-1, T, 1024)])
    np.savez(path, rows=rows, labels=labels, points=points, order=order)
    print(f'saved {len(rows)} segments ({len(rows) * T} frames); {time.time() - t0:.0f}s this run')


if __name__ == '__main__':
    gate() if '--gate' in sys.argv else label()
