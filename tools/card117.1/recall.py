"""Card 117.1: the self per patch, by recall over pose.

Memory: the frames of 450 training episodes (every fifth), each with the agent's pose (12 numbers) and its patch
features (blocks 2 and 12). For a frame t: recall the K memory frames whose pose is nearest to pose_t, and K frames
at random. A patch belongs to the self where the frame differs from the patch's usual look (the median of 64
random memory frames) and agrees with its pose-neighbours:
  score(p) = ||x_p - usual_p||^2 - mean_near ||x_p - m_p||^2.
What the pose fixes looks the same whenever the pose recurs, whatever else the scene holds. The first version
compared with random frames instead of the usual look; it marked every patch the arm often covers, covered now or
not (IoU 0.30-0.32). No labels, no training. The part of the arm that never moves cannot be found this way.

  bin/prun python tools/card117.1/recall.py [--k 8] → runs/117.1/recall_gate.json
  (card 122's 4 stored frames per episode; threshold by Otsu on episodes 900-999, tested on validation)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selfmodel import DATA, EP, OBJECTS, OUT, POSE, ROOT, SKIP, arg, otsu  # noqa: E402


class PoseMemory:
    def __init__(self, k=8, seed=0):
        self.src = np.load(ROOT / 'runs/124/feat_train.npy', mmap_mode='r')        # [450, 201, 1024, 768]
        n = self.src.shape[0]
        obs = np.load(DATA / 'scene-play-v0.npz')['observations'][:n * EP].reshape(n, EP, -1)[:, ::SKIP][:, :, POSE]
        self.mu, self.sd = obs.reshape(-1, len(POSE)).mean(0), obs.reshape(-1, len(POSE)).std(0) + 1e-6
        self.pose = ((obs - self.mu) / self.sd).reshape(-1, len(POSE)).astype(np.float32)
        self.k, self.rng = k, np.random.default_rng(seed)
        far = np.sort(self.rng.integers(0, len(self.pose), 64))
        self.usual = np.median(np.stack([self.read(i) for i in far]), 0)           # [1024, 768]

    def read(self, i):
        return np.asarray(self.src[i // 201, i % 201], np.float32)

    def score(self, pose, x):
        """pose [12] raw, x [1024, 768] → score [6, 1024]: block 2, block 12, both, with the neighbours' distance
        averaged, then the same three with the closest neighbour per patch; and the neighbours' pose distance."""
        q = (pose - self.mu) / self.sd
        d = ((self.pose - q) ** 2).sum(1)
        near = np.argpartition(d, self.k)[:self.k]
        nb = np.stack([self.read(i) for i in np.sort(near)])
        dn, df = (x[None] - nb) ** 2, (x[None] - self.usual[None]) ** 2
        parts = [slice(0, 384), slice(384, 768), slice(0, 768)]
        mean = [df[..., s].sum(-1).mean(0) - dn[..., s].sum(-1).mean(0) for s in parts]
        best = [df[..., s].sum(-1).mean(0) - dn[..., s].sum(-1).min(0) for s in parts]     # the closest neighbour per patch
        return np.stack(mean + best), float(d[near].mean())


def main():
    mem = PoseMemory(int(arg('--k', 8)))
    tr = np.load(DATA / 'scene-play-v0.npz')['observations']
    S = {}
    for name in ['tail', 'val']:
        c = np.load(ROOT / f'runs/122/computed_{name}.npz')
        obs = tr[900 * EP:] if name == 'tail' else np.load(DATA / 'scene-play-v0-val.npz')['observations']
        sidx = c['sidx'][::2] if name == 'tail' else c['sidx']
        x = np.concatenate([c['s2'], c['s12']], -1).astype(np.float32)
        x = x[::2] if name == 'tail' else x
        sc = [mem.score(obs[i][POSE], x[j]) for j, i in enumerate(sidx)]
        S[name] = dict(score=np.stack([s[0] for s in sc]), masks=c['masks'][sidx].astype(bool),
                       dist=float(np.mean([s[1] for s in sc])))
    iou = lambda a, b: float((a & b).sum() / max((a | b).sum(), 1))
    v = S['val']
    arm = v['masks'][:, 0]
    static = S['tail']['masks'][:, 0].mean(0) > 0.95                      # arm patches that never move (scoring only)
    mov = arm & ~static[None]
    res = dict(frames=len(arm), k=mem.k, static_arm_share_of_arm=round(float((arm & static[None]).sum() / arm.sum()), 3), mean_pose_distance_of_neighbours=round(v['dist'], 3),
               arm_share_of_patches=round(float(arm.mean()), 3))
    for bi, b in enumerate(['block2', 'block12', 'both', 'block2_closest', 'block12_closest', 'both_closest']):
        cut = otsu(S['tail']['score'][:, bi].reshape(-1))
        sel = v['score'][:, bi] > cut
        cuts = np.quantile(v['score'][:, bi], np.linspace(0.5, 0.99, 50))
        res[b] = dict(iou_with_arm=round(float(np.mean([iou(sel[i], arm[i]) for i in range(len(sel))])), 3),
                      oracle_threshold_pooled_iou=round(max(iou(v['score'][:, bi] > c, arm) for c in cuts), 3),
                      iou_with_moving_arm=round(float(np.mean([iou(sel[i] & ~static, mov[i]) for i in range(len(sel))])), 3),
                      oracle_moving=round(max(iou((v['score'][:, bi] > c) & ~static[None], mov) for c in cuts), 3),
                      moving_arm_covered=round(float(sel[mov].mean()), 3),
                      arm_covered=round(float(sel[arm].mean()), 3), self_share=round(float(sel.mean()), 3),
                      other_objects_in_self={o: round(float(sel[v['masks'][:, oi] & ~arm].mean()), 3)
                                             for oi, o in enumerate(OBJECTS) if oi}, threshold=float(cut))
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / 'recall_gate.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
