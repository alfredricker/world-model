"""Scoring only: align the probe mod's log with the frames and project each creature into the image.

truth(stem) -> dict(frames, poses, acts, ids, names, uv [T, K, 2] pixel centre, vis [T, K] in view, dist [T, K])
"""
import numpy as np

EYE, FOV_Y, BODY = 1.5, 90.0, 0.7          # eye height above the feet; vertical field of view; creature centre above its feet


def parse(path):
    rows = []
    for line in open(path):
        parts = line.strip().split(" | ")
        s = parts[0].split()
        mobs = {}
        for m in parts[1:]:
            _, i, name, x, y, z, yaw = m.split()
            mobs[int(i)] = (name, float(x), float(y), float(z))
        rows.append((int(s[2]), np.array([float(v) for v in s[4:9]]), mobs))   # ready, (x, y, z, yaw, pitch), mobs
    return rows


def project(cam, X, W, H):
    """cam = (x, y, z, yaw, pitch) of the feet, radians (Luanti: yaw 0 looks along +z, pitch > 0 looks down)."""
    x, y, z, yaw, p = cam
    C = np.array([x, y + EYE, z])
    f = np.array([-np.sin(yaw) * np.cos(p), -np.sin(p), np.cos(yaw) * np.cos(p)])
    r = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    u = np.array([-np.sin(yaw) * np.sin(p), np.cos(p), np.cos(yaw) * np.sin(p)])
    v = X - C
    xc, yc, zc = v @ r, v @ u, v @ f
    fy = (H / 2) / np.tan(np.radians(FOV_Y) / 2)
    uv = np.stack([W / 2 + fy * xc / np.maximum(zc, 1e-6), H / 2 - fy * yc / np.maximum(zc, 1e-6)], -1)
    return uv, zc


def truth(stem, out):
    z = np.load(f"{out}/{stem}.npz")
    F, P = z["frames"], z["poses"]
    rows = [r for r in parse(f"{out}/{stem}.log") if r[0] == 1]
    lp = np.array([r[1] for r in rows])
    # align: frame t <-> log row t + k, chosen by matching the player's position and look (degrees in info, radians in the log)
    T = len(F)
    best = None
    for k in range(0, len(rows) - T + 1):
        d = np.abs(np.degrees(lp[k:k + T, 3]) % 360 - P[:, 3] % 360)   # the server hears the look only every ~4 steps
        d = np.minimum(d, 360 - d).mean() + np.abs(lp[k:k + T, :3] - P[:, :3]).sum(1).mean()
        if best is None or d < best[0]:
            best = (d, k)
    k = best[1]
    ids = sorted({i for r in rows for i in r[2]})
    names = {i: next(r[2][i][0] for r in rows if i in r[2]) for i in ids}
    H, W = F.shape[1:3]
    uv = np.full((T, len(ids), 2), np.nan)
    world = np.full((T, len(ids), 3), np.nan)              # each creature's feet in the world (scoring only)
    zc = np.full((T, len(ids)), np.nan)
    for t in range(T):
        r = rows[k + t]
        for j, i in enumerate(ids):
            if i in r[2]:
                world[t, j] = r[2][i][1:]
                X = np.array(r[2][i][1:]) + [0, BODY, 0]
                q = P[max(t - 1, 0)]                    # frame t is drawn from the pose reported one step earlier
                cam = np.r_[q[:3], np.radians(q[3]), np.radians(q[4])]
                uv[t, j], zc[t, j] = project(cam, X, W, H)
    rad = (H / 2) / np.tan(np.radians(FOV_Y) / 2) * 0.6 / np.maximum(zc, 0.5)   # a creature's apparent radius in pixels
    vis = (zc > 0.5) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
    for t in range(T):                                     # hidden behind a nearer creature: not in view
        for j in range(len(ids)):
            for i in range(len(ids)):
                if i != j and vis[t, i] and vis[t, j] and zc[t, i] < zc[t, j] and \
                        np.linalg.norm(uv[t, i] - uv[t, j]) < 0.6 * rad[t, i]:
                    vis[t, j] = False
    return dict(frames=F, poses=P, acts=z["acts"], ids=ids, names=names, uv=uv, vis=vis, dist=zc, rad=rad, world=world, offset=k,
                align_err=best[0])
