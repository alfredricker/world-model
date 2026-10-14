"""Card 107, criterion 3: belief identity on the pasture clip with each depth source (exact, released, tuned).

The belief is belief_check's "spread" arm (perfect detections without identities; positions on the ground; a match
within 0.5 + v * steps unseen blocks, v from the tuning clip). With a depth model, each in-view cow's position comes
from its depth at the cow's projected centre (belief_check.estimates). Scored: IDF1, and for cows, the share of
returns to view within 100 steps unseen whose track is the one they had when last seen.

  tools/card107/cprun tools/card107/belief_eval.py <clip> <tuning clip> <depth arm>...   → runs/107/belief_<clip>.json
"""
import json
import sys

import numpy as np

CLIP, TUNE, ARMS = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.argv = ["belief_check.py", CLIP, TUNE]
import belief_check as B  # noqa: E402
from truth import truth  # noqa: E402

v = B.step_speed(TUNE)
B.CSQ = B.sqrt_spread(TUNE)
rep = {"clip": CLIP, "v (blocks per step, from " + TUNE + ")": round(v, 4)}
for name in ["exact"] + ARMS:
    d = truth(CLIP, B.OUT)
    arm = "spread"
    if name != "exact":
        B.EST, nb = B.estimates(d, B.OUT / f"{CLIP}_depth_{name}.npz")
        d["vis"] = d["vis"] & ~nb[:, None]
        arm = "spread_est"
    r = B.run(d, arm, v, np.random.default_rng(0))
    right, lab, mapped = B.run.last
    vis, kind = d["vis"], [d["names"][i] for i in d["ids"]]
    rows = []
    for j in range(vis.shape[1]):
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
    rep[name] = {"IDF1": r["IDF1"], "IDF1, cows": r["IDF1 by kind [creatures, IDF1]"].get("cow", [0, None])[1],
                 "cow returns within 100 steps that get their own track back": [round(float(rows[m, 1].mean()), 3), int(m.sum())],
                 "cow returns after more than 100 steps": [round(float(rows[~m, 1].mean()), 3) if (~m).any() else None, int((~m).sum())]}
    print(name, json.dumps(rep[name]), flush=True)
json.dump(rep, open(B.OUT.parent / f"belief_{CLIP}.json", "w"), indent=1)
