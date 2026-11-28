"""Card 172, section 4 (data check). Stored truth and the arm's pose only; no frame is read, nothing is trained.

  bash tools/card171/run.sh 300 python tools/card172/datacheck.py   -> runs/172/datacheck.json

For pairs of moments of one episode (steps of 5 frames, as card 145): how many scene variables differ at each gap,
how well the pose alone picks the true near moment out of 16 others under three ways of choosing them, and which
variable is the one that tells a false moment from the true one.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
z = np.load(Path.home() / ".ogbench/data/scene-play-v0.npz")
EP, SKIP, N = 1001, 5, 450
steps = np.arange(0, EP, SKIP); T = len(steps)
rows = (np.arange(N)[:, None] * EP + steps[None]).reshape(-1)
o, q = z["observations"][rows], z["qpos"][rows]
pose = np.c_[o[:, 0:6], o[:, 12:18]].reshape(N, T, 12).astype(np.float64)
pose = (pose - pose.mean((0, 1))) / (pose.std((0, 1)) + 1e-9)
cube = q[:, 14:17].reshape(N, T, 3); drawer = np.abs(q[:, 23]).reshape(N, T); window = np.abs(q[:, 24]).reshape(N, T)
locks = z["button_states"][rows].reshape(N, T, 2)
NAMES = ["cube", "drawer", "window", "lock0", "lock1"]


def differ(e, a, b):
    """[.., 5] bool: which variables differ between moments a and b of episodes e (cube or a slide by over 1 cm)."""
    return np.stack([np.linalg.norm(cube[e, a] - cube[e, b], axis=-1) > 0.01, np.abs(drawer[e, a] - drawer[e, b]) > 0.01,
                     np.abs(window[e, a] - window[e, b]) > 0.01, locks[e, a, 0] != locks[e, b, 0], locks[e, a, 1] != locks[e, b, 1]], -1)


rng = np.random.default_rng(0)
res = {"gap": {}, "schemes": {}}
for d in (1, 2, 5, 10, 20, 40):
    e = rng.integers(0, N, 40000); a = rng.integers(0, T - d, 40000)
    D = differ(e, a, a + d); n = D.sum(-1)
    res["gap"][d] = dict(none=round(100 * float((n == 0).mean()), 1), one=round(100 * float((n == 1).mean()), 1), more=round(100 * float((n >= 2).mean()), 1),
                         **{k: round(100 * float(D[:, i].mean()), 1) for i, k in enumerate(NAMES)})
    print("gap", d, res["gap"][d], flush=True)

K = 16
for d in (2, 5, 10, 20):
    for scheme in ("any moment of the episode", "just outside the window", "just outside, pose distance matched to the true pair's", "true one at random, false ones the nearest in pose", "just outside, matched in pose"):
        hit, info, pos_same, cnt = [], np.zeros(5), [], 0
        for _ in range(6000):
            e = int(rng.integers(0, N)); a = int(rng.integers(4 * d, T - 4 * d))
            near = np.r_[np.arange(a - d, a), np.arange(a + 1, a + d + 1)]
            ring = np.r_[np.arange(a - 4 * d, a - d), np.arange(a + d + 1, a + 4 * d + 1)]
            dp_near = np.abs(pose[e, near] - pose[e, a]).sum(-1); dp_ring = np.abs(pose[e, ring] - pose[e, a]).sum(-1)
            if scheme == "any moment of the episode":
                p = int(rng.choice(near)); neg = rng.choice(np.setdiff1d(np.arange(T), np.r_[near, a]), K, replace=False)
            elif scheme == "just outside the window":
                p = int(rng.choice(near)); neg = rng.choice(ring, K, replace=len(ring) < K)
            elif scheme == "just outside, pose distance matched to the true pair's":
                j = int(rng.integers(len(near))); p = int(near[j]); neg = ring[np.argsort(np.abs(dp_ring - dp_near[j]))[:K]]
            elif scheme == "true one at random, false ones the nearest in pose":
                p = int(rng.choice(near)); neg = ring[np.argsort(dp_ring)[:K]]
            else:
                p = int(near[np.argmax(dp_near)]); neg = ring[np.argsort(dp_ring)[:K]]
            dpp = np.abs(pose[e, p] - pose[e, a]).sum(); dpn = np.abs(pose[e, neg] - pose[e, a]).sum(-1)
            hit.append(bool(dpp < dpn.min()))
            Dp = differ(e, a, p); Dn = differ(np.full(len(neg), e), np.full(len(neg), a), neg)
            pos_same.append(not Dp.any())
            tell = Dn & ~Dp[None]                      # a variable in which this false moment differs and the true one does not
            only = tell & (tell.sum(-1, keepdims=True) == 1)
            info += only.sum(0); cnt += len(neg)
        r = dict(pose_alone_picks_true_pct=round(100 * float(np.mean(hit)), 1), chance_pct=round(100.0 / (K + 1), 1),
                 true_pair_same_scene_pct=round(100 * float(np.mean(pos_same)), 1),
                 false_moments_told_by_one_variable_only_pct=round(100 * float(info.sum() / cnt), 1),
                 of_which={k: round(100 * float(info[i] / max(info.sum(), 1)), 1) for i, k in enumerate(NAMES)})
        res["schemes"][f"d={d} | {scheme}"] = r
        print(f"d={d} | {scheme}", r, flush=True)
json.dump(res, open(ROOT / "runs/172/datacheck.json", "w"), indent=1)
