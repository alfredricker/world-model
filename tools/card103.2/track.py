"""Card 103.2: object files as sprite layers. Each frame, once placed in the world (its origin in world pixels), is
explained by three layers: the world (a pixel canvas of the episode), the camera (the pixels card 103.1 learned the
view carries), and object files. A file is a sprite: a template of its object's own pixels, a mask, a world position,
its last displacement and an existence count. Nothing about tiles, kinds or ground is declared.

Each step:
  1. Residual: pixels the world does not explain (robust: a global brightness offset removed, a threshold at four
     times this frame's noise), the camera's pixels left out.
  2. Propagation: each file searches displacements d up to K pixels around its position, scoring its template on its
     mask (robust) with a learned prior p(d) (counts of accepted displacements). Accepted if most visible mask pixels
     match and the match is better than the world's explanation of the same pixels. Files claim their pixels in order
     of hits; a file whose mask is mostly out of view is kept as it was (persistence) and not counted as a miss.
  3. Revealed: where a file moved from, the frame shows what it covered; those pixels are the world's (the canvas
     learns the ground), never a birth.
  4. Discovery: connected components of the residual that no file claimed. A component whose content matches the
     canvas at exactly one displacement (or two) other than none came from there: a mover is born, and its source is
     revealed. One that matches at many displacements (uniform content, the aperture problem) or at none (appeared,
     changed in place) is the world's; an appeared component becomes a tentative file that is reported once it moves.
  5. The canvas is written with every pixel no file claims and the camera does not carry.
A file is reported (for the evaluator) while it is seen and once it has moved: objects by what they do.

Origins: --origins true (the evaluator's, the gate's upper bound), 1031 (card 103.1's robust learned placing) or
1034 (card 103.4's tempered placing); the learned ones on worker 8 only.
Scored with card 098.5's IDF1 over cows, zombies, skeletons and arrows (Crafter, workers 8 and 9) or world 3's ball
and cats.

  bin/prun python tools/card103.2/track.py crafter|world3 [--origins true|1028] [--n N]   → runs/103.2/track_*.json
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card098.5"))
from assign import linear_sum_assignment               # noqa: E402

OUT = ROOT / "runs" / "103.2"
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
WORLD, ORIGINS = sys.argv[1], arg("--origins", "true")
HW, WW, K, UNIT = 49, 63, 9, 7
CX, CY = 31, 24                                        # the agent's tile centre in the view (evaluator's scoring only)
MIN_PIX, GOOD, HALF = 4, 0.6, 700
SEARCH = "--search" in sys.argv                       # a lost file searches the whole view (arm)
GROUP = "--group" in sys.argv                         # birth pieces within 2 px examined together (arm)
TOL = 0.06                                             # a pixel differs when a channel differs by more (15 of 255 levels),
                                                       # or by four times this frame's noise if that is larger


def load():
    if WORLD == "crafter":
        out = []
        for w in [int(x) for x in arg("--workers", "8,9").split(",")]:
            z = np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz")
            out.append({k: z[k] for k in ("obs", "act", "done", "pos", "sleep", "light", "ep", "obj")})
        return out
    z = np.load(ROOT / "runs" / "card102" / "world3_test.npz")
    return [{k: z[k] for k in ("obs", "act", "done", "pos", "sleep", "light", "ep", "obj")}]


def segments(d):
    """Episodes; sleeping frames are skipped (-1) but do not end what the files know: sleep moves nothing."""
    brk = np.r_[True, (d["ep"][1:] != d["ep"][:-1]) | d["done"][:-1]]
    seg = np.cumsum(brk)
    seg[d["sleep"]] = -1
    return seg


def origins(d, seg, wi):
    """The view's origin in world pixels: the evaluator's (true), or card 103.1's learned placing (robust arm, worker 8),
    whose segments restart at sleep; sleep moves nothing, so each restart is chained to the last origin before it."""
    if ORIGINS == "true":
        return UNIT * (d["pos"].astype(np.int64) - (4, 3))
    src = {"1031": ROOT / "runs" / "103.1" / "org_crafter_world_none_test.npy",
           "1034": ROOT / "runs" / "103.4" / "org_tau300_test.npy"}[ORIGINS]
    o = np.load(src)[:len(seg)].copy()
    brk = np.r_[True, (d["ep"][1:] != d["ep"][:-1]) | d["sleep"][1:] | d["sleep"][:-1] | d["done"][:-1]]
    s31 = np.cumsum(brk)
    s31[d["sleep"]] = -1
    last, shift = None, np.zeros(2, np.int64)
    for t in range(len(seg)):
        if s31[t] < 0:
            continue
        if t > 0 and seg[t] >= 0 and last is not None and s31[t] != s31[last] and seg[t] == seg[last]:
            if s31[t - 1] != s31[t]:
                shift = o_last - o[t]
        o[t] = o[t] + shift
        o_last, last = o[t].copy(), t
    return o


def thr(s2):
    return 3 * max(16 * s2, TOL ** 2)


def components(mask):
    """4-connected components of a boolean (H, W) array: list of (ys, xs)."""
    lab = -np.ones(mask.shape, np.int64)
    comps = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0] >= 0:
            continue
        stack, ys, xs = [(y0, x0)], [], []
        lab[y0, x0] = len(comps)
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and lab[yy, xx] < 0:
                    lab[yy, xx] = len(comps)
                    stack.append((yy, xx))
        comps.append((np.array(ys), np.array(xs)))
    return comps


class Tracker:
    def __init__(self, cam):
        self.cam = cam
        self.prior = np.ones((2 * K + 1, 2 * K + 1))               # p(d) over file displacements, counts
        self.next_id = 0
        self.stats = defaultdict(int)

    def reset(self):
        self.canvas = np.zeros((2 * HALF, 2 * HALF, 3), np.float32)
        self.known = np.zeros((2 * HALF, 2 * HALF), bool)
        self.files = []

    def view(self, o):
        x, y = int(o[0]) + HALF, int(o[1]) + HALF
        return slice(y, y + HW), slice(x, x + WW)

    def match(self, f, o, F, s2):
        """Robust cost of file F's template at each displacement: (2K+1, 2K+1) mean cost, inlier share, visible share."""
        th, tw = F["tmpl"].shape[:2]
        vy, vx = F["pos"][1] - o[1], F["pos"][0] - o[0]               # template's top-left in view pixels
        P = np.full((HW + 2 * (K + th), WW + 2 * (K + tw), 3), np.nan, np.float32)
        P[K + th:K + th + HW, K + tw:K + tw + WW] = f
        y0, x0 = vy + th, vx + tw                                      # in P, displacement 0 at (y0 + K, x0 + K)
        if not (0 <= y0 and y0 + 2 * K + th <= P.shape[0] and 0 <= x0 and x0 + 2 * K + tw <= P.shape[1]):
            return None
        R = P[y0:y0 + 2 * K + th, x0:x0 + 2 * K + tw]
        W = np.lib.stride_tricks.sliding_window_view(R, (th, tw), axis=(0, 1))   # 2K+1, 2K+1, 3, th, tw
        W = np.moveaxis(W, 2, -1)
        r2 = ((W - F["tmpl"]) ** 2).sum(-1)
        vis = ~np.isnan(r2) & F["mask"]
        nv = vis.sum((-1, -2))
        inl = ((r2 < thr(s2)) & vis).sum((-1, -2))
        return inl / np.maximum(nv, 1), nv / F["mask"].sum()

    def search(self, f, o, F, s2, res):
        """The best place in view for a lost file's template, among places where the world does not explain it."""
        th, tw = F["tmpl"].shape[:2]
        if th > HW or tw > WW:
            return None
        W = np.moveaxis(np.lib.stride_tricks.sliding_window_view(f, (th, tw), axis=(0, 1)), 2, -1)
        r2 = ((W - F["tmpl"]) ** 2).sum(-1)
        inl = ((r2 < thr(s2)) & F["mask"]).sum((-1, -2)) / F["mask"].sum()
        Rw = np.lib.stride_tricks.sliding_window_view(res, (th, tw), axis=(0, 1))
        unexpl = (Rw & F["mask"]).sum((-1, -2)) / F["mask"].sum()
        sc = np.where((inl >= GOOD) & (unexpl >= 0.5), inl, -1)
        if sc.max() < 0:
            return None
        y, x = np.unravel_index(int(np.argmax(sc)), sc.shape)
        return (int(x + o[0]), int(y + o[1]))

    def step(self, t, f, o, prev_o, s2):
        sy, sx = self.view(o)
        V, Kn = self.canvas[sy, sx], self.known[sy, sx]
        dif = f - V
        if Kn.sum() > 50:
            dif = dif - np.median(dif[Kn], 0)
        r2 = (dif ** 2).sum(-1)
        res = Kn & (r2 > thr(s2)) & ~self.cam
        claimed = np.zeros((HW, WW), bool)
        revealed = np.zeros((HW, WW), bool)
        lp = np.log(self.prior / self.prior.sum())
        seen = []
        for F in sorted(self.files, key=lambda F: -F["hits"]):
            m = self.match(f, o, F, s2)
            if SEARCH and (m is None or m[0].max() < GOOD) and F["moved"]:
                g = self.search(f, o, F, s2, res & ~claimed)              # lost: look for it anywhere in view
                if g is not None:
                    F["pos"] = g
                    m = self.match(f, o, F, s2)
                    self.stats["re-acquired by a search of the view"] += 1
            if m is None:
                continue
            inl, vis = m
            if vis[K, K] < 0.3 and vis.max() < 0.3:                   # out of view: kept as it was
                continue
            score = np.where(vis >= 0.3, inl + 0.02 * lp, -1)
            dy, dx = np.unravel_index(int(np.argmax(score)), score.shape)
            th, tw = F["tmpl"].shape[:2]
            ny, nx = F["pos"][1] + dy - K - o[1], F["pos"][0] + dx - K - o[0]
            ys, xs = np.nonzero(F["mask"])
            ys, xs = ys + ny, xs + nx
            ok = (ys >= 0) & (ys < HW) & (xs >= 0) & (xs < WW)
            ys, xs = ys[ok], xs[ok]
            free = ~claimed[ys, xs]
            world_too = (~res[ys, xs]).mean() if len(ys) else 1.0    # the world explains these pixels as well
            if inl[dy, dx] >= GOOD and free.mean() > 0.5 and not ((dy, dx) == (K, K) and world_too > 0.9
                                                                     and F["moved"] and F["still"] > 50):
                d = (dx - K, dy - K)
                if d != (0, 0):
                    oys, oxs = np.nonzero(F["mask"])
                    oys, oxs = oys + F["pos"][1] - o[1], oxs + F["pos"][0] - o[0]
                    ok2 = (oys >= 0) & (oys < HW) & (oxs >= 0) & (oxs < WW)
                    revealed[oys[ok2], oxs[ok2]] = True
                    F["moved"] = True
                    F["still"] = 0
                    self.prior[dy, dx] += 1
                else:
                    F["still"] += 1
                    self.prior[K, K] += 1
                F["pos"] = (F["pos"][0] + d[0], F["pos"][1] + d[1])
                claimed[ys, xs] = True
                F["hits"] += 1
                F["exist"] = min(F["exist"] + 1, 5)
                F["last"] = t
                ty, tx = np.nonzero(F["mask"])                          # the template follows slow changes (light)
                vy, vx = ty + ny, tx + nx
                ok3 = (vy >= 0) & (vy < HW) & (vx >= 0) & (vx < WW)
                F["tmpl"][ty[ok3], tx[ok3]] = 0.5 * F["tmpl"][ty[ok3], tx[ok3]] + 0.5 * f[vy[ok3], vx[ok3]]
                if F["moved"]:
                    seen.append((F["id"], ((np.mean(xs) - CX) / UNIT, (np.mean(ys) - CY) / UNIT)))
            elif vis[K, K] >= 0.5:
                F["exist"] -= 1
        revealed &= ~claimed
        self.files = [F for F in self.files if F["exist"] > -3 and t - F["last"] < 300
                      and (F["moved"] or t - F["born"] < 20)]
        # discovery
        free = res & ~claimed & ~revealed
        comps = components(free)
        if GROUP:                                # pieces within 2 px examined together (arm)
            grown = free.copy()
            for _ in range(2):
                g2 = grown.copy()
                g2[1:] |= grown[:-1]
                g2[:-1] |= grown[1:]
                g2[:, 1:] |= grown[:, :-1]
                g2[:, :-1] |= grown[:, 1:]
                grown = g2
            comps = []
            for gy, gx in components(grown):
                sel = free[gy, gx]
                if sel.any():
                    comps.append((gy[sel], gx[sel]))
        comps = sorted(comps, key=lambda c: -len(c[0]))
        for ys, xs in comps:
            if len(ys) < MIN_PIX or not free[ys[0], xs[0]]:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            msk = np.zeros((y1 - y0, x1 - x0), bool)
            msk[ys - y0, xs - x0] = True
            tmpl = f[y0:y1, x0:x1].copy()
            # where did it come from: the canvas at each displacement
            wy, wx = y0 + o[1] + HALF, x0 + o[0] + HALF
            C = self.canvas[wy - K:wy + K + (y1 - y0), wx - K:wx + K + (x1 - x0)]
            Kc = self.known[wy - K:wy + K + (y1 - y0), wx - K:wx + K + (x1 - x0)]
            Wc = np.moveaxis(np.lib.stride_tricks.sliding_window_view(C, (y1 - y0, x1 - x0), axis=(0, 1)), 2, -1)
            Wk = np.lib.stride_tricks.sliding_window_view(Kc, (y1 - y0, x1 - x0), axis=(0, 1))
            r = ((Wc - tmpl) ** 2).sum(-1)
            good = (((r < thr(s2)) & Wk & msk).sum((-1, -2)) / msk.sum()) >= 0.8
            good[K, K] = False
            ng = int(good.sum())
            if 1 <= ng <= 2:                                            # came from one place: a mover
                gy, gx = np.unravel_index(int(np.argmax(np.where(good, -np.abs(np.arange(-K, K + 1))[:, None]
                                                                  - np.abs(np.arange(-K, K + 1))[None], -99))),
                                          good.shape)
                d = (K - gx, K - gy)                                     # it moved by d from (x0 - d, y0 - d)
                sy0, sx0 = ys - d[1], xs - d[0]
                ok = (sy0 >= 0) & (sy0 < HW) & (sx0 >= 0) & (sx0 < WW)
                free[sy0[ok], sx0[ok]] = False
                revealed[sy0[ok], sx0[ok]] = True
                self.files.append({"id": self.next_id, "pos": (x0 + o[0], y0 + o[1]), "tmpl": tmpl, "mask": msk,
                                   "hits": 1, "exist": 1, "last": t, "born": t, "moved": True, "still": 0})
                self.next_id += 1
                self.stats["born moving"] += 1
                claimed[ys, xs] = True
                seen.append((self.next_id - 1, ((xs.mean() - CX) / UNIT, (ys.mean() - CY) / UNIT)))
            elif ng == 0:                                               # appeared or changed in place: tentative
                self.files.append({"id": self.next_id, "pos": (x0 + o[0], y0 + o[1]), "tmpl": tmpl, "mask": msk,
                                   "hits": 1, "exist": 1, "last": t, "born": t, "moved": False, "still": 0})
                self.next_id += 1
                self.stats["born appeared"] += 1
                claimed[ys, xs] = True
            else:
                self.stats["uniform, the world's"] += 1
        # the world layer
        wr = ~claimed & ~self.cam
        Vw = self.canvas[sy, sx]
        Vw[wr] = f[wr]
        self.known[sy, sx] |= wr
        return seen


def idf1(rec, obj, ep, kinds):
    """Card 098.5's IDF1 over the given kinds, with files placed relative to the agent."""
    gt = defaultdict(list)
    for t, i, k, dx, dy, *_ in obj:
        if k in kinds:
            gt[int(t)].append(((int(ep[t]), int(i)), int(k), np.array([dx, dy], np.float64)))
    co = defaultdict(lambda: defaultdict(int))
    ngt, npred, gkind, n_kind = 0, 0, {}, defaultdict(int)
    for t, seen in enumerate(rec):
        e = int(ep[t])
        for g, k, r in gt.get(t, []):
            ngt += 1
            gkind[g] = k
            n_kind[k] += 1
        npred += len(seen)
        for pid, p in seen:
            for g, k, r in gt.get(t, []):
                if np.abs(np.array(p) - r).max() < 0.5:
                    co[e][(g, pid)] += 1
    idtp, idtp_kind = 0, defaultdict(int)
    for e, pairs in co.items():
        G = sorted({g for g, p in pairs})
        Pd = sorted({p for g, p in pairs})
        gi, pi = {g: i for i, g in enumerate(G)}, {p: i for i, p in enumerate(Pd)}
        M = np.zeros((len(G), len(Pd)))
        for (g, p), c in pairs.items():
            M[gi[g], pi[p]] = c
        r, c = linear_sum_assignment(-M)
        for i, j in zip(r, c):
            idtp += M[i, j]
            idtp_kind[gkind[G[i]]] += M[i, j]
    return idtp, ngt, npred, idtp_kind, n_kind


def main():
    t0 = time.monotonic()
    data = load()
    cam = np.zeros((HW, WW), bool)
    pf = ROOT / "runs" / "103.1" / f"pi_{WORLD}_world_learned.npy"
    if pf.exists():
        cam = np.load(pf)[:HW, :WW] > 0.5
    kinds = {1: "cow", 2: "zombie", 3: "skeleton", 4: "arrow"} if WORLD == "crafter" else {5: "ball", 6: "cat"}
    S = defaultdict(float)
    byk = defaultdict(lambda: [0.0, 0])
    T = Tracker(cam)
    n_lim = int(arg("--n", 0)) or None
    for wi, d in enumerate(data):
        if n_lim:
            d = {k: (v[:n_lim] if k != "obj" else v[v[:, 0] < n_lim]) for k, v in d.items()}
        seg = segments(d)
        org = origins(d, seg, wi)
        rec = []
        s2 = 1e-4
        last_seg = None
        for t in range(len(seg)):
            if seg[t] < 0:
                rec.append([])
                continue
            f = d["obs"][t, :HW, :WW].astype(np.float32) / 255
            if t == 0 or seg[t] != last_seg:
                T.reset()
            last_seg = seg[t]
            if T.known.any():
                sy, sx = T.view(org[t])
                Kn = T.known[sy, sx] & ~cam
                if Kn.sum() > 50:
                    r = ((f - T.canvas[sy, sx]) ** 2).sum(-1)[Kn]
                    s2 = max(float(np.median(r)) / 2.366, 1e-4)
            rec.append(T.step(t, f, org[t], org[t - 1] if t else org[t], s2))
            if t % 5000 == 0:
                print(f"  world part {wi} step {t} files {len(T.files)} ({time.monotonic() - t0:.0f} s)", flush=True)
        if "--dump" in sys.argv:
            import pickle
            OUT.mkdir(exist_ok=True, parents=True)
            pickle.dump({"rec": rec, "n": len(seg)}, open(OUT / f"dump_{WORLD}_{ORIGINS}_{wi}.pkl", "wb"))
        idtp, ngt, npred, ik, nk = idf1(rec, d["obj"], d["ep"], kinds)
        S["idtp"] += idtp
        S["gt"] += ngt
        S["pred"] += npred
        for k in kinds:
            byk[kinds[k]][0] += ik.get(k, 0)
            byk[kinds[k]][1] += nk.get(k, 0)
    rep = {"world": WORLD, "origins": ORIGINS, "IDF1 (movers pooled)": round(2 * S["idtp"] / (S["gt"] + S["pred"]), 4),
           "ID recall / ID precision": [round(S["idtp"] / max(S["gt"], 1), 4), round(S["idtp"] / max(S["pred"], 1), 4)],
           "object frames / file detections": [int(S["gt"]), int(S["pred"])],
           "ID recall by kind": {k: round(a / max(b, 1), 4) for k, (a, b) in byk.items()},
           "file events": dict(T.stats), "learned p(d), likeliest five": [
               (int(i % (2 * K + 1) - K), int(i // (2 * K + 1) - K), round(float(T.prior.flat[i] / T.prior.sum()), 3))
               for i in np.argsort(-T.prior.ravel())[:5]],
           "camera pixels left out": int(cam.sum()), "seconds": round(time.monotonic() - t0, 1)}
    OUT.mkdir(exist_ok=True, parents=True)
    tag = f"{WORLD}_{ORIGINS}_w{arg('--workers', '8,9').replace(',', '')}" + (f"_n{n_lim}" if n_lim else "") + ("_search" if SEARCH else "") + ("_group" if GROUP else "")
    (OUT / f"track_{tag}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
