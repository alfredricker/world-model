"""Card 107: depth from a pretrained geometry backbone, scored against the game's own voxels.

  tools/card107/dprun tools/card107/depth_test.py gt <clip>                     → <clip>_gt.npz: true depth on every 10th frame (scoring only)
  tools/card107/dprun tools/card107/depth_test.py model <clip> mono|posed [--views 8 --gap 4 --model depth-anything/DA3-BASE]
        mono:  DA3METRIC-LARGE, one frame at a time, metric depth = focal * output / 300 (its README)
        posed: a DA3 any-view model on frame t and up to 7 earlier keyframes (frames whose camera moved >= 0.5 blocks
               from the last keyframe), given their poses and the intrinsics; depth scaled to the poses (blocks) when
               the views' cameras span >= 0.5 blocks, else the model's own scale
        → <clip>_depth_<arm>.npz: depth on every frame at 320 x 180
  tools/card107/dprun tools/card107/depth_test.py score <clip> <arm>...          → per-pixel error, scale drift, error at creatures' centres

True depth: each pixel's ray, from the eye (feet + 1.5) along the pose of the step before the frame, marched in 0.05
block steps through the voxel observation (41 x 21 x 41 blocks around the agent; [north, up, east]), stopping at the
first node drawn as a full block (normal, allfaces, glass, liquid). Pixels are left out when the ray passes a node drawn
otherwise first (plants, fences, torches), leaves the grid, lies on a creature, or on the hand and the item bar.
Depth is along the camera's forward axis.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from truth import EYE, FOV_Y, parse, truth  # noqa: E402

OUT = HERE.parents[1] / "runs" / "107" / "clips"
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
MODE, CLIP = sys.argv[1], sys.argv[2]
SOLID = {"normal", "allfaces", "allfaces_optional", "glasslike", "glasslike_framed", "glasslike_framed_optional",
         "liquid", "flowingliquid"}
GW, GH = 160, 90                                    # the scoring grid of rays (every 4th pixel at 640 x 360)
DW, DH = 320, 180                                   # model depth stored at this size


def basis(q):
    """Camera axes in Luanti's world (x east, y up, z north) for a pose (x, y, z, yaw deg, pitch deg)."""
    yaw, p = np.radians(q[3]), np.radians(q[4])
    f = np.array([-np.sin(yaw) * np.cos(p), -np.sin(p), np.cos(yaw) * np.cos(p)])
    r = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    u = np.array([-np.sin(yaw) * np.sin(p), np.cos(p), np.cos(yaw) * np.sin(p)])
    return r, u, f


def cam_of(P, t):
    q = P[max(t - 1, 0)]                            # frame t is drawn from the pose of the step before
    return np.array([q[0], q[1] + EYE, q[2]]), basis(q)


def world_map(clip, d):
    """One static map of node ids from the voxel observations of frames where the agent stood still."""
    V = np.load(OUT / f"{clip}_vox.npz")["vox"]
    rows = [r for r in parse(OUT / f"{clip}.log") if r[0] == 1]
    k = d["offset"]
    M = {}
    for t in range(len(V)):
        if k + t + 2 >= len(rows) or t < 2:
            continue
        ps = np.array([rows[k + t + s][1][:3] for s in (-2, -1, 0, 1, 2)])
        if np.abs(ps - ps[2]).max() > 1e-6:
            continue
        c = np.floor(ps[2] + 0.5).astype(int)       # Luanti's vector.round
        M[tuple(c)] = V[t]
    lo = np.min([np.array(c) - (20, 10, 20) for c in M], 0)
    hi = np.max([np.array(c) + (20, 10, 20) for c in M], 0)
    G = np.zeros(tuple(hi - lo + 1)[::-1], np.int32) - 1          # [z, y, x], -1 unknown
    for c, v in M.items():
        o = np.array(c) - (20, 10, 20) - lo
        G[o[2]:o[2] + 41, o[1]:o[1] + 21, o[0]:o[0] + 41] = v
    return G, lo


def gt(clip):
    d = truth(clip, OUT)
    F, P = d["frames"], d["poses"]
    H, W = F.shape[1:3]
    G, lo = world_map(clip, d)
    nodes = {int(l.split()[0]): (l.split()[1], l.split()[2]) for l in open(OUT / f"{clip}_nodes.txt")}
    ids = np.arange(G.max() + 2)
    solid = np.array([nodes.get(int(i), ("?", "airlike"))[1] in SOLID for i in ids])
    other = np.array([nodes.get(int(i), ("air", "airlike"))[1] not in SOLID | {"airlike"} for i in ids])
    f = (H / 2) / np.tan(np.radians(FOV_Y) / 2)
    us, vs = np.meshgrid((np.arange(GW) + 0.5) * W / GW, (np.arange(GH) + 0.5) * H / GH)
    ts = list(range(10, len(F), 10))
    D = np.full((len(ts), GH, GW), np.nan, np.float32)
    steps = np.arange(1, 501) * 0.05
    for n, t in enumerate(ts):
        C, (r, u, fw) = cam_of(P, t)
        dirs = (r[None, None] * ((us - W / 2) / f)[..., None] - u[None, None] * ((vs - H / 2) / f)[..., None]
                + fw[None, None])                                  # unit forward component: depth = s * (dir . f) = s
        pts = C + steps[:, None, None, None] * dirs[None]          # [S, GH, GW, 3]; depth along f equals steps
        idx = np.floor(pts + 0.5).astype(int) - lo
        inside = ((idx >= 0) & (idx < np.array(G.shape[::-1]))).all(-1)
        cell = np.full(inside.shape, -1)
        z, y, x = idx[..., 2], idx[..., 1], idx[..., 0]
        cell[inside] = G[z[inside], y[inside], x[inside]]
        known = cell >= 0
        hit = known & solid[np.clip(cell, 0, None)]
        bad = (~known) | (known & other[np.clip(cell, 0, None)])
        first = np.where(hit.any(0), hit.argmax(0), -1)
        first_bad = np.where(bad.any(0), bad.argmax(0), 10 ** 6)
        ok = (first >= 0) & (first < first_bad)
        D[n][ok] = steps[first[ok]]
    np.savez_compressed(OUT / f"{clip}_gt.npz", D=D, ts=np.array(ts))
    print(f"true depth on {len(ts)} frames; valid share {np.isfinite(D).mean():.2f}; median {np.nanmedian(D):.1f} blocks")


def model(clip, arm):
    import torch
    from depth_anything_3.api import DepthAnything3
    d = truth(clip, OUT)
    F, P = d["frames"], d["poses"]
    T, H, W = F.shape[:3]
    f = (H / 2) / np.tan(np.radians(FOV_Y) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
    name = arg("--model", "depth-anything/DA3METRIC-LARGE" if arm == "mono" else "depth-anything/DA3-BASE")
    net = DepthAnything3.from_pretrained(name).to("cuda").eval()

    def scale_to_poses(extrinsics, intrinsics, prediction, align_to_input_ext_scale=True, **_):
        """In place of DA3's Umeyama step (which fails when the cameras lie on a line): depth times the median ratio
        of distances between the given camera centres and the predicted ones."""
        if extrinsics is None:
            return prediction
        cen = lambda E: np.stack([-e[:3, :3].T @ e[:3, 3] for e in E])
        ci, cp = cen(extrinsics.numpy()), cen(prediction.extrinsics)
        i, j = np.triu_indices(len(ci), 1)
        di, dp = np.linalg.norm(ci[i] - ci[j], axis=1), np.linalg.norm(cp[i] - cp[j], axis=1)
        ok = (di >= 0.3) & (dp > 1e-6)
        prediction.depth = prediction.depth * float(np.median(di[ok] / dp[ok]))
        prediction.intrinsics = intrinsics.numpy()
        return prediction
    net._align_to_input_extrinsics_intrinsics = scale_to_poses
    out = np.zeros((T, DH, DW), np.float16)
    t0 = time.monotonic()

    def resize(x):
        return torch.nn.functional.interpolate(torch.as_tensor(x)[None, None].float(), size=(DH, DW),
                                               mode="bilinear", align_corners=False)[0, 0].numpy()

    if arm == "mono":
        for t in range(T):
            pr = net.inference([F[t]])
            fo = f * pr.depth.shape[-1] / W                 # the focal length at the size the model processed
            out[t] = resize(fo * pr.depth[0] / 300.0)
            if t == 0:
                print("processed size", pr.depth.shape, flush=True)
    else:
        views, gap = int(arg("--views", 8)), int(arg("--gap", 4))

        def w2c(t):
            # Luanti's world is left-handed (x east, y up, z north): mirror z for a right-handed frame, then OpenCV axes
            C, (r, u, fw) = cam_of(P, t)
            m = np.diag([1.0, 1.0, -1.0])
            R = np.stack([m @ r, -(m @ u), m @ fw])
            assert abs(np.linalg.det(R) - 1) < 1e-6
            E = np.eye(4)
            E[:3, :3], E[:3, 3] = R, -R @ (m @ C)
            return E.astype(np.float32)
        no_base = np.zeros(T, bool)
        keys, unaligned = [], 0                     # keyframes: frames whose camera is >= 0.5 blocks from the last one
        for t in range(T):
            C = cam_of(P, t)[0]
            if not keys or np.linalg.norm(C - cam_of(P, keys[-1])[0]) >= 0.5:
                keys.append(t)
            idx = [k for k in keys if k != t][-(views - 1):] + [t]
            ex, ix = np.stack([w2c(i) for i in idx]), np.stack([K] * len(idx))
            spread = np.ptp(np.stack([cam_of(P, i)[0] for i in idx]), 0).max()
            align = spread >= 0.5                     # scale from the poses needs the camera to have moved
            unaligned += int(not align)
            if align:
                pr = net.inference([F[i] for i in idx], extrinsics=ex, intrinsics=ix, ref_view_strategy="first")
            else:                                      # DA3 aligns to any poses it is given; without a baseline, none
                pr = net.inference([F[i] for i in idx], ref_view_strategy="first")
            no_base[t] = not align
            out[t] = resize(pr.depth[-1])
            if t == 0:
                print("processed size", pr.depth.shape, flush=True)
        print(f"frames without a baseline (model's own scale): {unaligned} of {T}", flush=True)
    np.savez_compressed(OUT / f"{clip}_depth_{arm}.npz", D=out, no_base=locals().get("no_base", np.zeros(T, bool)))
    print(f"{arm}: {T} frames in {time.monotonic() - t0:.0f} s", flush=True)


def score(clip, arms):
    d = truth(clip, OUT)
    g = np.load(OUT / f"{clip}_gt.npz")
    Dg, ts = g["D"], g["ts"]
    H, W = d["frames"].shape[1:3]
    us, vs = np.meshgrid((np.arange(GW) + 0.5) * W / GW, (np.arange(GH) + 0.5) * H / GH)
    keep = ~((vs > 0.80 * H) | ((us > 0.70 * W) & (vs > 0.45 * H)))   # the item bar and the hand
    rep = {"clip": clip, "frames scored": int(len(ts))}
    G, lo = world_map(clip, d)
    nodes = {int(l.split()[0]): (l.split()[1], l.split()[2]) for l in open(OUT / f"{clip}_nodes.txt")}
    solid = np.array([nodes.get(int(i), ("?", "airlike"))[1] in SOLID for i in range(G.max() + 2)])
    hidden = np.zeros(d["vis"].shape, bool)
    f = (H / 2) / np.tan(np.radians(FOV_Y) / 2)
    for t in range(len(d["frames"])):
        C, (r, u, fw) = cam_of(d["poses"], t)
        for j in np.flatnonzero(d["vis"][t]):
            dirn = r * (d["uv"][t, j, 0] - W / 2) / f - u * (d["uv"][t, j, 1] - H / 2) / f + fw
            ss = np.arange(0.3, max(d["dist"][t, j] - 0.5, 0.31), 0.05)
            idx = np.floor(C + ss[:, None] * dirn + 0.5).astype(int) - lo
            inside = ((idx >= 0) & (idx < np.array(G.shape[::-1]))).all(-1)
            cell = np.where(inside, G[np.clip(idx[:, 2], 0, G.shape[0] - 1), np.clip(idx[:, 1], 0, G.shape[1] - 1),
                                      np.clip(idx[:, 0], 0, G.shape[2] - 1)], -1)
            hidden[t, j] = bool((solid[np.clip(cell, 0, None)] & (cell >= 0)).any())
    for arm in arms:
        z = np.load(OUT / f"{clip}_depth_{arm}.npz")
        Dm, nb = z["D"].astype(np.float32), z["no_base"] if "no_base" in z else np.zeros(len(z["D"]), bool)
        rel, d125, scale, rel_s = [], [], [], []
        for n, t in enumerate(ts):
            if nb[t]:
                continue
            m = np.isfinite(Dg[n]) & keep & (Dg[n] > 0.3)
            for j in np.flatnonzero(d["vis"][t]):                    # not on a creature
                m &= np.hypot(us - d["uv"][t, j, 0], vs - d["uv"][t, j, 1]) > 1.5 * d["rad"][t, j]
            if m.sum() < 50:
                continue
            e = Dm[t][(vs[m] * DH / H).astype(int), (us[m] * DW / W).astype(int)]
            rel.append(np.abs(e - Dg[n][m]) / Dg[n][m])
            es = e / np.median(e / Dg[n][m])                         # the frame's scale fixed: the shape alone
            rel_s.append(np.abs(es - Dg[n][m]) / Dg[n][m])
            d125.append(np.maximum(e / Dg[n][m], Dg[n][m] / np.maximum(e, 1e-6)) < 1.25)
            scale.append(np.median(e / Dg[n][m]))
        cre, near, cow = [], [], []            # creatures: depth at the centre vs true centre and near surface
        for t in range(len(d["frames"])):
            if nb[t]:
                continue
            for j in np.flatnonzero(d["vis"][t]):
                if hidden[t, j]:
                    continue
                uu, vv, rr = d["uv"][t, j, 0], d["uv"][t, j, 1], max(d["rad"][t, j] / 2, 2)
                ys, xs = np.mgrid[max(int(vv - rr), 0):min(int(vv + rr) + 1, H), max(int(uu - rr), 0):min(int(uu + rr) + 1, W)]
                inn = (xs - uu) ** 2 + (ys - vv) ** 2 <= rr * rr
                if not inn.any():
                    continue
                e = np.median(Dm[t][(ys[inn] * DH / H).astype(int), (xs[inn] * DW / W).astype(int)])
                cre.append((e - d["dist"][t, j]) / d["dist"][t, j])
                near.append((e - (d["dist"][t, j] - 0.4)) / (d["dist"][t, j] - 0.4))
                if d["names"][d["ids"][j]] == "mobs_mc:cow":
                    cow.append(near[-1])
        cre = np.array(cre)
        rel, d125, scale = np.concatenate(rel), np.concatenate(d125), np.array(scale)
        rep[arm] = {"frames left out (no baseline)": int(nb.sum()), "terrain: mean |error| / depth": round(float(rel.mean()), 3),
                    "terrain: share within 25%": round(float(d125.mean()), 3),
                    "terrain, each frame's scale fixed: mean |error| / depth": round(float(np.concatenate(rel_s).mean()), 3),
                    "terrain: per-frame scale (median estimate / true): median, spread (std)":
                        [round(float(np.median(scale)), 3), round(float(scale.std()), 3)],
                    "creatures: (estimate - true centre depth) / true: median, mean |.|":
                        [round(float(np.median(cre)), 3), round(float(np.abs(cre).mean()), 3)],
                    "creatures: (estimate - near surface) / near surface, surface = centre - 0.4: median, mean |.|":
                        [round(float(np.median(near)), 3), round(float(np.abs(np.array(near)).mean()), 3)],
                    "cows only, against the near surface: median, mean |.|":
                        [round(float(np.median(cow)), 3), round(float(np.abs(np.array(cow)).mean()), 3)] if cow else None,
                    "creature samples (cows), hidden behind blocks left out": [int(len(cre)), int(len(cow)), int(hidden.sum())]}
        print(arm, json.dumps(rep[arm]), flush=True)
    json.dump(rep, open(OUT / f"{clip}_depth_score_{'_'.join(arms)}.json", "w"), indent=1)


if __name__ == "__main__":
    if MODE == "gt":
        gt(CLIP)
    elif MODE == "model":
        model(CLIP, sys.argv[3])
    else:
        score(CLIP, sys.argv[3:])
