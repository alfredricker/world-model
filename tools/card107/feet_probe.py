"""Card 107, exploratory (not a criterion): a creature's position from the ground depth where its feet stand.

The feet's pixel is the logged position projected by the pose (scoring-only stand-in for the bottom of a segment);
the depth is the median within 2 px just below it; the point is back-projected and its ground position kept.
Then belief_check's spread arm, scored as belief_eval.

  tools/card107/cprun tools/card107/feet_probe.py <clip> <tuning clip> <depth arm>...   → runs/107/feet_<clip>.json
"""
import json
import sys

import numpy as np

CLIP, TUNE, ARMS = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.argv = ["belief_check.py", CLIP, TUNE]
import belief_check as B  # noqa: E402
from truth import EYE, FOV_Y, project, truth  # noqa: E402

v = B.step_speed(TUNE)
rep = {"clip": CLIP}
for name in ARMS:
    d = truth(CLIP, B.OUT)
    D = np.load(B.OUT / f"{CLIP}_depth_{name}.npz")["D"].astype(np.float32)
    T, K = d["vis"].shape
    H, W = d["frames"].shape[1:3]
    DH, DW = D.shape[1:]
    f = (H / 2) / np.tan(np.radians(FOV_Y) / 2)
    est, err = np.full((T, K, 2), np.nan), []
    for t in range(T):
        q = d["poses"][max(t - 1, 0)]
        cam = np.r_[q[:3], np.radians(q[3]), np.radians(q[4])]
        yaw, p = cam[3], cam[4]
        fw = np.array([-np.sin(yaw) * np.cos(p), -np.sin(p), np.cos(yaw) * np.cos(p)])
        r = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        u = np.array([-np.sin(yaw) * np.sin(p), np.cos(p), np.cos(yaw) * np.sin(p)])
        C = np.array([q[0], q[1] + EYE, q[2]])
        for j in np.flatnonzero(d["vis"][t]):
            uvf, zf = project(cam, d["world"][t, j], W, H)
            if zf <= 0.5 or not (0 <= uvf[0] < W and 0 <= uvf[1] + 2 < H):
                continue
            x0, y0 = int(uvf[0]), int(uvf[1]) + 2
            win = D[t][max(y0 * DH // H - 1, 0):y0 * DH // H + 2, max(x0 * DW // W - 1, 0):x0 * DW // W + 2]
            e = float(np.median(win))
            X = C + e * (r * (x0 - W / 2) / f - u * (y0 - H / 2) / f + fw)
            est[t, j] = X[[0, 2]]
            err.append(np.linalg.norm(est[t, j] - d["world"][t, j, [0, 2]]) / max(zf, 0.5))
    B.EST = est
    r_ = B.run(d, "spread_est", v, np.random.default_rng(0))
    right, lab, mapped = B.run.last
    vis, kind = d["vis"], [d["names"][i] for i in d["ids"]]
    rows = []
    for j in range(K):
        if kind[j] != "mobs_mc:cow":
            continue
        s = vis[:, j].astype(int)
        on, off = np.flatnonzero(np.diff(s) == 1) + 1, np.flatnonzero(np.diff(s) == -1) + 1
        for a in on:
            pv = off[off < a]
            if len(pv):
                b = pv[-1] - 1
                rows.append((a - b, lab[a, j] == lab[b, j]))
    rows = np.array(rows, float)
    m = rows[:, 0] <= 100
    rep[name] = {"position error / distance: median, mean": [round(float(np.median(err)), 3), round(float(np.mean(err)), 3)],
                 "IDF1": r_["IDF1"], "IDF1, cows": r_["IDF1 by kind [creatures, IDF1]"].get("cow", [0, None])[1],
                 "cow returns within 100 steps that get their own track back": [round(float(rows[m, 1].mean()), 3), int(m.sum())]}
    print(name, json.dumps(rep[name]), flush=True)
json.dump(rep, open(B.OUT.parent / f"feet_{CLIP}.json", "w"), indent=1)
