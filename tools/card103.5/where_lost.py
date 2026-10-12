"""Card 103.5: where identity is lost on a test worker. Each creature frame is classed by what reports it: the file the
creature is mostly reported by (its own), another file, a tentative file or a kept record only, or nothing; by day or
night, and whether the creature has moved since it came into view. Files per creature followed.

  bin/prun python tools/card103.5/where_lost.py ORIGINS W MEM   → runs/103.5/where_lost_<ORIGINS>_w<W>.json
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
O, W, MEM = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.argv = ["files.py", "--origins", O, "--workers", str(W)]
sys.path.insert(0, str(ROOT / "tools" / "card103.5"))
import files as FL                                     # noqa: E402
import __main__                                        # noqa: E402
__main__.Memory = FL.Memory
mem = pickle.load(open(MEM, "rb"))
cam = np.load(ROOT / "runs" / "103.1" / "pi_crafter_world_learned.npy")[:FL.HW, :FL.WW] > 0.5
T = FL.Files(cam, mem)
z = np.load(ROOT / "runs" / "098.5" / f"play_w{W}.npz")
d = {k: z[k] for k in ("obs", "act", "done", "pos", "sleep", "light", "ep", "obj")}
by_t = defaultdict(list)
for row in d["obj"]:
    if row[2] in (1, 2, 3, 4):
        by_t[int(row[0])].append(row)
seg = FL.TR.segments(d)
org = FL.origins(d, seg, W)


def centre(F, o):
    ys, xs, ok = T.placed(F, o, (0, 0))
    if not ok.any():
        return None
    return np.array([(xs[ok].mean() - FL.CX) / FL.UNIT, (ys[ok].mean() - FL.CY) / FL.UNIT])


frames, first = [], {}
s2, last = 1e-4, None
for t in range(len(seg)):
    if seg[t] < 0:
        continue
    f = d["obs"][t, :FL.HW, :FL.WW].astype(np.float32) / 255
    if seg[t] != last:
        T.reset()
    last = seg[t]
    if T.known.any():
        sy, sx = T.view(org[t])
        Kn = T.known[sy, sx] & ~cam
        if Kn.sum() > 50:
            s2 = max(float(np.median(((f - T.canvas[sy, sx]) ** 2).sum(-1)[Kn])) / 2.366, 1e-4)
    seen = T.step(t, f, org[t], s2)
    for r in by_t.get(t, []):
        g = (int(d["ep"][t]), int(r[1]))
        wpos = (int(d["pos"][t][0] + r[3]), int(d["pos"][t][1] + r[4]))
        st = first.setdefault(g, [wpos, False])
        if wpos != st[0]:
            st[1] = True
        q = np.array([r[3], r[4]], float)
        rep = [fid for fid, p in seen if np.abs(np.array(p) - q).max() < 0.5]
        other = None
        if not rep:
            other = "nothing"
            for F in T.files:
                c = centre(F, org[t])
                if c is not None and np.abs(c - q).max() < 0.5:
                    other = "confirmed, not found this step" if F["confirmed"] else "tentative file"
                    break
            if other == "nothing":
                for D in T.dormant:
                    c = centre(D, org[t])
                    if c is not None and np.abs(c - q).max() < 0.5:
                        other = "kept record only"
                        break
        frames.append((g, int(r[2]), "night" if d["light"][t] < 0.5 else "day",
                       "moved" if st[1] else "still since seen", rep[0] if rep else None, other))
own = {}
cnt = defaultdict(Counter)
for g, k, n, m, fid, other in frames:
    if fid is not None:
        cnt[g][fid] += 1
for g, c in cnt.items():
    own[g] = c.most_common(1)[0][0]
C = Counter()
for g, k, n, m, fid, other in frames:
    s = ("its own file" if fid == own.get(g) else "another file") if fid is not None else other
    C[(n, m, s)] += 1
tot = len(frames)
rep = {"origins": O, "worker": W, "creature frames": tot,
       "by day or night, moved or still since seen, status": {" / ".join(k): [v, round(v / tot, 4)] for k, v in
                                                                 sorted(C.items())},
       "files per creature followed (median, mean)": [float(np.median([len(c) for c in cnt.values()])),
                                                      round(float(np.mean([len(c) for c in cnt.values()])), 2)],
       "creatures followed / seen": [len(cnt), len({f[0] for f in frames})]}
(ROOT / "runs" / "103.5" / f"where_lost_{O}_w{W}.json").write_text(json.dumps(rep, indent=1) + "\n")
print(json.dumps(rep, indent=1))
