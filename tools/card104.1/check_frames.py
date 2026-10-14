"""Card 104.1, data check: in 105.9's held-out set, how many blocked moves the two frames alone tell apart. A blocked
move is told when frame t+1 differs from frame t by at least one intensity level less (mean over the 49 x 63 world
region, the agent's own tile left out) under no shift than under the one-tile shift the action would cause. The same
test on moves that succeeded, by day, is printed per source.

  bin/prun python tools/card104.1/check_frames.py
"""
import numpy as np
from collections import Counter
names = {1: "water", 2: "grass", 3: "stone", 4: "path", 5: "sand", 6: "tree", 7: "lava", 8: "coal", 9: "iron", 10: "diamond",
         11: "table", 12: "furnace", 13: "player", 14: "cow", 15: "zombie", 16: "skeleton", 17: "arrow", 18: "plant"}
srcs = [("curr", w, f"runs/102.1/curr_w{w}.npz") for w in (8, 9)] + [("play", w, f"runs/098.5/play_w{w}.npz") for w in (8, 9)]
tot = Counter(); tell = Counter(); kinds = Counter()
for kind, w, f in srcs:
    z = np.load(f)
    obs = z["obs"][:, :49, :63].astype(np.float32); act = z["act"].astype(int); pos = z["pos"].astype(int)
    ok = np.r_[(z["ep"][1:] == z["ep"][:-1]) & ~z["done"][:-1] & ~z["sleep"][:-1] & ~z["sleep"][1:], False]
    dp = np.r_[pos[1:] - pos[:-1], np.zeros((1, 2), int)]
    mv = ok & np.isin(act, (1, 2, 3, 4))
    blocked = mv & (dp == 0).all(1)
    night = z["light"] < 0.5
    # each move action's direction, from successful moves
    dirs = {a: tuple(Counter(map(tuple, dp[mv & (act == a) & ~blocked])).most_common(1)[0][0]) for a in (1, 2, 3, 4)}
    def diff(t, sy, sx):        # mean |next - cur shifted|, on the overlap, leaving out the agent's own tile
        a, b = obs[t], obs[t + 1]
        H, W = 49, 63
        ys, xs = slice(max(0, -sy), H - max(0, sy)), slice(max(0, -sx), W - max(0, sx))
        ys2, xs2 = slice(max(0, sy), H - max(0, -sy)), slice(max(0, sx), W - max(0, -sx))
        d = np.abs(b[ys, xs] - a[ys2, xs2]).mean(-1)
        m = np.ones_like(d, bool)
        cy, cx = 21 - max(0, -sy), 28 - max(0, -sx)
        m[max(0, cy):cy + 7, max(0, cx):cx + 7] = False
        return float(d[m].mean())
    tot[(kind, "move steps")] += int(mv.sum())
    for t in np.flatnonzero(blocked):
        dx, dy = dirs[act[t]]
        d0, d1 = diff(t, 0, 0), diff(t, 7 * dy, 7 * dx)   # no shift, or the view moved one tile with the agent
        key = (kind, "night" if night[t] else "day")
        tot[key] += 1
        tell[key] += d0 + 1.0 < d1
        kinds[names.get(int(z["sem"][t, 3 + dy, 4 + dx]), "?")] += 1
    # sanity: on successful moves the moved shift explains better
    s = np.flatnonzero(mv & ~blocked & ~night)[:300]
    good = np.mean([diff(t, 7 * dirs[act[t]][1], 7 * dirs[act[t]][0]) + 1.0 < diff(t, 0, 0) for t in s])
    print(kind, w, "dirs", dirs, "| day successful moves told by the frames:", round(float(good), 3))
for k in sorted(tot):
    print(k, tot[k], "" if "move steps" in k else f"told by the frames: {tell[k] / tot[k]:.3f}")
print("blockers:", kinds.most_common(10))
