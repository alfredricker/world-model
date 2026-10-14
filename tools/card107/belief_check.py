"""Card 107 (from the smoke tests of 2026-10-13): belief identity — perfect detections without identities.

Each frame the tracker receives the creatures in view (not hidden behind a nearer one) as (kind, world position on the
ground), i.e. a perfect detector with perfect depth, and no identities. Kinds are given (appearance tells a cow from a
chicken); look-alikes of one kind must be told apart by belief. The belief keeps one track per creature it has seen;
each frame detections are matched to tracks of the same kind by the Hungarian method (scipy), then matched tracks move
to their detection and unmatched detections start new tracks. Arms:
  none        no memory: a track not seen in the last frame is dropped
  stay        belief: an unseen track stays where it was last seen; a match within 1.5 blocks
  spread      belief: an unseen track may have moved v blocks per step; a match within r0 + v * steps unseen, cost the
              distance over that radius (v measured on the tuning clip cows8: the 95th percentile of a step's move)
  spread_sqrt spread, but the radius grows with the square root of the steps unseen: r0 + c sqrt(steps) (mostly
              standing, sometimes wandering; c measured on the tuning clip)
  spread_d10  spread, with each detection's depth off by a factor 1 + e, e ~ N(0, 0.1) (stand-in for learned depth)
  view_est    with a depth model's estimates: in view as hybrid; across gaps, the track's 3D estimate is projected
              into this frame by the pose and compared with the detection's pixel, within the radius its spread
              (r0 + v * steps unseen, in blocks) makes at its estimated depth: depth errors along the ray barely move it
  hybrid_d10  as spread_d10 across gaps, but a track seen in the last frame is matched in the image: its last pixel
              moved by the camera's turn (no depth needed), within 8 px at 320 wide (an in-view tracker's job)
Scored with IDF1 (one-to-one mapping of tracks to creatures over the clip), and the share of in-view frames on the
right track before a creature first left the view and after it had left and come back.

  tools/card107/cprun tools/card107/belief_check.py <clip> [<tuning clip> [<depth .npz>]]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from truth import truth, EYE  # noqa: E402,F401

CLIP, TUNE = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "cows8")
OUT = HERE.parents[1] / "runs" / "107" / "clips"
R0, BIG = 0.5, 1e6


GATE_PX = 8


def rotate_uv(uv, P, t, W, H):
    """Where a pixel seen in frame t - 1 appears in frame t if only the camera's look changed (frames t - 1 and t are
    drawn from poses t - 2 and t - 1)."""
    from truth import FOV_Y
    f = (H / 2) / np.tan(np.radians(FOV_Y) / 2)

    def basis(q):
        yaw, p = np.radians(q[3]), np.radians(q[4])
        fw = np.array([-np.sin(yaw) * np.cos(p), -np.sin(p), np.cos(yaw) * np.cos(p)])
        r = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        u = np.array([-np.sin(yaw) * np.sin(p), np.cos(p), np.cos(yaw) * np.sin(p)])
        return r, u, fw
    r0, u0, f0 = basis(P[max(t - 2, 0)])
    r1, u1, f1 = basis(P[max(t - 1, 0)])
    ray = r0 * (uv[0] - W / 2) / f - u0 * (uv[1] - H / 2) / f + f0
    z = ray @ f1
    return np.array([W / 2 + f * (ray @ r1) / z, H / 2 - f * (ray @ u1) / z])


CSQ = None


def sqrt_spread(stem):
    """95th percentile of how far a creature is from where it was g steps before, over sqrt(g) (g = 1 .. 600)."""
    w = truth(stem, OUT)["world"][:, :, [0, 2]]
    r = []
    for g in (1, 3, 10, 30, 100, 300, 600):
        if g < len(w):
            m = np.linalg.norm(w[g:] - w[:-g], axis=-1).reshape(-1)
            r.append(m[np.isfinite(m)] / np.sqrt(g))
    return float(np.percentile(np.concatenate(r), 95))


def step_speed(stem):
    d = truth(stem, OUT)
    w = d["world"][:, :, [0, 2]]
    m = np.linalg.norm(np.diff(w, axis=0), axis=-1)
    m = m[np.isfinite(m) & (m > 1e-4)]
    return float(np.percentile(m, 95))


def run(d, arm, v, rng):
    vis, world, names, ids, P = d["vis"], d["world"], d["names"], d["ids"], d["poses"]
    T, K = vis.shape
    H, W = d["frames"].shape[1:3]
    kind = [names[i] for i in ids]
    tracks = []                                   # dicts: kind, pos (x, z), last (frame last seen)
    lab = np.full((T, K), -1)                     # the track each in-view creature was given
    for t in range(T):
        js = np.flatnonzero(vis[t])
        det = world[t, js][:, [0, 2]].copy()
        if arm.endswith("_est") and len(js):
            det = EST[t, js]
        if arm in ("spread_d10", "hybrid_d10") and len(js):
            cam = P[max(t - 1, 0), [0, 2]]
            det = cam + (det - cam) * (1 + rng.normal(0, 0.1, (len(js), 1)))
        got = {}
        if arm in ("hybrid_d10", "hybrid_est", "view_est") and len(js):
            det_uv = d["uv"][t, js]
            prev = [k for k, tr in enumerate(tracks) if tr["last"] == t - 1]
            C = np.full((len(js), len(prev)), BIG)
            for a, j in enumerate(js):
                for b, k in enumerate(prev):
                    if tracks[k]["kind"] == kind[j]:
                        e = np.linalg.norm(det_uv[a] - rotate_uv(tracks[k]["uv"], P, t, W, H))
                        if e <= GATE_PX * W / 320:
                            C[a, b] = e
            if len(prev):
                ra, cb = linear_sum_assignment(C)
                got = {a: prev[b] for a, b in zip(ra, cb) if C[a, b] < BIG}
        taken = set(got.values())
        live = [k for k, tr in enumerate(tracks) if (arm != "none" or tr["last"] == t - 1) and k not in taken]
        rest = [a for a in range(len(js)) if a not in got]
        C = np.full((len(js), len(live)), BIG)
        if arm == "view_est":
            from truth import project, FOV_Y
            fpx = (H / 2) / np.tan(np.radians(FOV_Y) / 2)
            q = P[max(t - 1, 0)]
            cam = np.r_[q[:3], np.radians(q[3]), np.radians(q[4])]
            for a, j in enumerate(js):
                if a not in rest:
                    continue
                for b, k in enumerate(live):
                    tr = tracks[k]
                    if tr["kind"] != kind[j] or not np.isfinite(tr["p3"]).all():
                        continue
                    uvk, zk = project(cam, tr["p3"], W, H)
                    if zk <= 0.3:
                        continue
                    gate = max(GATE_PX * W / 320, fpx * (R0 + v * (t - tr["last"])) / zk)
                    e = np.linalg.norm(d["uv"][t, j] - uvk)
                    if e <= gate:
                        C[a, b] = e / gate
            live_cost_done = True
        else:
            live_cost_done = False
        for a, j in enumerate(js):
            if a not in rest or live_cost_done:
                continue
            for b, k in enumerate(live):
                tr = tracks[k]
                if tr["kind"] != kind[j]:
                    continue
                dist = np.linalg.norm(det[a] - tr["pos"])
                gap = t - tr["last"]
                if arm in ("none", "stay"):
                    if dist <= 1.5:
                        C[a, b] = dist
                else:
                    noise = 0.0 if arm in ("spread", "spread_sqrt") else 0.3 * np.linalg.norm(det[a] - P[max(t - 1, 0), [0, 2]])
                    r = R0 + noise + (v * gap if arm != "spread_sqrt" else CSQ * np.sqrt(gap))
                    if dist <= r:
                        C[a, b] = dist / r
        if len(js) and len(live):
            ra, cb = linear_sum_assignment(C)
            got.update({a: live[b] for a, b in zip(ra, cb) if C[a, b] < BIG and a in rest})
        for a, j in enumerate(js):
            if a in got:
                tr = tracks[got[a]]
                alpha = 1.0 if arm not in ("spread_d10", "hybrid_d10", "spread_est", "hybrid_est") or tr["last"] != t - 1 else 0.3
                tr["pos"] = (1 - alpha) * tr["pos"] + alpha * det[a]
                tr["last"] = t
                tr["uv"] = d["uv"][t, j]
                if EST3 is not None:
                    tr["p3"] = EST3[t, j] if tr.get("last3") != t - 1 or not np.isfinite(tr["p3"]).all() \
                        else 0.7 * tr["p3"] + 0.3 * EST3[t, j]
                    tr["last3"] = t
                lab[t, j] = got[a]
            else:
                tracks.append({"kind": kind[j], "pos": det[a], "last": t, "uv": d["uv"][t, j],
                               "p3": EST3[t, j] if EST3 is not None else np.full(3, np.nan), "last3": t})
                lab[t, j] = len(tracks) - 1
    # IDF1: one-to-one mapping of tracks to creatures maximising frames in common
    n = len(tracks)
    co = np.zeros((K, max(n, 1)))
    for t in range(T):
        for j in np.flatnonzero(vis[t]):
            co[j, lab[t, j]] += 1
    r, c = linear_sum_assignment(-co)
    mapped = dict(zip(r, c))
    gt = int(vis.sum())
    idtp = int(sum(co[j, mapped[j]] for j in mapped))
    right = np.zeros((T, K), bool)
    for t in range(T):
        for j in np.flatnonzero(vis[t]):
            right[t, j] = mapped.get(j, -2) == lab[t, j]
    left = np.zeros((T, K), bool)
    for j in range(K):
        if vis[:, j].any():
            gone = False
            for t in range(np.flatnonzero(vis[:, j])[0], T):
                gone |= not vis[t, j]
                left[t, j] = gone
    before, after = vis & ~left, vis & left
    run.last = (right, lab, mapped)
    per_kind = {}
    for kd in sorted(set(kind)):
        js_ = [j for j in mapped if kind[j] == kd]
        g_ = int(vis[:, [j for j in range(K) if kind[j] == kd]].sum())
        per_kind[kd.split(":")[1]] = [sum(1 for j in range(K) if kind[j] == kd and vis[:, j].any()),
                                      round(sum(co[j, mapped[j]] for j in js_) / max(g_, 1), 3)]
    return {"IDF1": round(2 * idtp / (2 * gt), 3), "IDF1 by kind [creatures, IDF1]": per_kind, "tracks made": n, "creatures seen": int(vis.any(0).sum()),
            "on the right track, in view since first seen": round(float(right[before].mean()), 3),
            "on the right track, after leaving view and returning": round(float(right[after].mean()), 3),
            "frames: before / after first leaving": [int(before.sum()), int(after.sum())]}


EST = None
EST3 = None


def estimates(d, depth_file):
    """Each in-view creature's position on the ground from a depth model: its projected centre's pixel, the median
    depth within half its apparent radius, back-projected from the camera (frame t is drawn from pose t - 1)."""
    from truth import FOV_Y
    z = np.load(depth_file)
    D = z["D"].astype(np.float32)
    nb = z["no_base"] if "no_base" in z else np.zeros(len(D), bool)
    H, W = d["frames"].shape[1:3]
    DH, DW = D.shape[1:]
    f = (H / 2) / np.tan(np.radians(FOV_Y) / 2)
    T, K = d["vis"].shape
    est = np.full((T, K, 2), np.nan)
    global EST3
    EST3 = np.full((T, K, 3), np.nan)
    for t in range(T):
        q = d["poses"][max(t - 1, 0)]
        yaw, p = np.radians(q[3]), np.radians(q[4])
        fw = np.array([-np.sin(yaw) * np.cos(p), -np.sin(p), np.cos(yaw) * np.cos(p)])
        r = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        u = np.array([-np.sin(yaw) * np.sin(p), np.cos(p), np.cos(yaw) * np.sin(p)])
        C = np.array([q[0], q[1] + EYE, q[2]])
        for j in np.flatnonzero(d["vis"][t]):
            uu, vv, rr = d["uv"][t, j, 0], d["uv"][t, j, 1], max(d["rad"][t, j] / 2, 2)
            ys, xs = np.mgrid[max(int(vv - rr), 0):min(int(vv + rr) + 1, H), max(int(uu - rr), 0):min(int(uu + rr) + 1, W)]
            inn = (xs - uu) ** 2 + (ys - vv) ** 2 <= rr * rr
            e = np.median(D[t][(ys[inn] * DH / H).astype(int), (xs[inn] * DW / W).astype(int)]) if inn.any() else np.nan
            X = C + e * (r * (uu - W / 2) / f - u * (vv - H / 2) / f + fw)
            est[t, j] = X[[0, 2]]
            EST3[t, j] = X
    return est, nb


def main():
    global CSQ
    v, CSQ = step_speed(TUNE), sqrt_spread(TUNE)
    d = truth(CLIP, OUT)
    w = d["world"][:, :, [0, 2]]
    gaps = []                                     # how long creatures were out of view before they came back
    for j in range(d["vis"].shape[1]):
        s = d["vis"][:, j].astype(int)
        on = np.flatnonzero(np.diff(s) == 1) + 1
        off = np.flatnonzero(np.diff(s) == -1) + 1
        for a in on:
            prev = off[off < a]
            if len(prev):
                gaps.append(a - prev[-1])
    gaps = np.array(gaps)
    rep = {"clip": CLIP, "frames": int(len(w)), "v (blocks per step, 95th percentile, from " + TUNE + ")": round(v, 4),
           "returns to view": int(len(gaps)),
           "steps out of view before returning: median, 90th percentile": [float(np.median(gaps)),
                                                                           float(np.percentile(gaps, 90))] if len(gaps) else None}
    rep["c (blocks per sqrt step, 95th percentile, from " + TUNE + ")"] = round(CSQ, 4)
    arms = ["none", "stay", "spread", "spread_sqrt", "spread_d10", "hybrid_d10"]
    global EST
    if len(sys.argv) > 3:                                   # a depth model's estimates: belief_check.py clip tune depth.npz
        EST, nb = estimates(d, sys.argv[3])
        d["vis"] = d["vis"] & ~nb[:, None]                  # frames whose depth has no scale: nothing detected
        arms = ["spread", "spread_est", "hybrid_est", "view_est"]
        rep["depth estimates"] = sys.argv[3]
    for arm in arms:
        rep[arm] = run(d, arm, v, np.random.default_rng(0))
        print(arm, json.dumps(rep[arm]), flush=True)
    json.dump(rep, open(OUT / (f"{CLIP}_belief.json" if EST is None else f"{CLIP}_belief_{Path(sys.argv[3]).stem}.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in list(rep)[:5]}))


if __name__ == "__main__":
    main()
