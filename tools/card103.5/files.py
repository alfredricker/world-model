"""Card 103.5: card 098.6's files level on pixel sprites. Card 103.2's files (a template of the object's own pixels, a
mask and a world position, followed by search, born from what the world leaves unexplained) get card 098.6's
bookkeeping in place of "reported once it has moved":

  existence (IPDA; the Bernoulli part of PMBM, as 098.6): each file has r. A placement is a hit with likelihood ratio
      L = own / other:  own = p(d | steps unseen), the displacement distribution of confirmed files (before any is
                        seen, the agent's own placements: 098.6's "a move is one of the agent's own steps");
                        other = where the canvas knows the world: the clutter density, how often the file's template
                        matches elsewhere in its gate in this frame (098.6's drift "as far as this"; a tracker's false
                        detections per unit of its gate); where the canvas does not know it: 1 - share (it is scenery
                        that looks like this);
      a placement the world explains as well (the canvas shows the same) is not a detection. A file not found
      where it could be seen: r <- r (1 - q) / (1 - r q), q = P_D x P(it is in view after its possible moves), P_D
      the confirmed files' hit rate. The template follows slow changes (light) on the pixels that still match it.
      Confirmed at 0.95, deleted below 0.05; a tentative file not confirmed within 3 steps is deleted (M-of-N).
      Only confirmed files are reported, and only they keep the canvas's ground under them; tentative ones are
      written into the canvas, as 098.6's belief takes a tentative file's token.
  records and the share (098.6): one record per file (its look, the steps it was seen, how many it moved by itself);
      records are grouped by look (leader clusters: at least GOOD of each mask's pixels match at the best alignment
      within 2 px); the share of records like a look that needed memory of their own is 098.6's two-part mixture fitted
      by EM, with records seen for few steps counted by how much they show, all records pooled joining with one
      record's weight.
  births: a residual piece that came from one place that has changed since (the pixel pair test): the look's share pi updated by the move's
      likelihood ratio L = p(d) / clutter (it moved, against "its content is found nearby by chance"); the record counts the move by
      L / (1 + L), as 098.6's file hits count theirs; a piece that appeared from nowhere: r0 = pi; looks whose share is at least 0.05 are searched for
      in pixels seen for the first time (098.6's births at a place first seen): r0 = pi, confirmed at once at 0.95.
  one placement for all files: each file's candidate displacements (template matches of at least GOOD), resolved
      jointly by ln r + ln L, not by the order of hits; a file not found carries a map of where it could be (098.5-098.6):
      moved by p(d) each step, with the places looked at and found empty lowered by P_D (negative information), out
      to G px.
  cohesion (the sprite's mask, Jojic and Frey 2001; card 102.2's common fate): residual pixels within 2 px of a file
      that moved this step, whose content the canvas shows at the same displacement where the frame has changed since,
      moved with it: they join its mask.

Declared (rule 8): one displacement distribution for all files (098.6's bank of predictors and recall per look are not
ported); no unseen-object density (098.6's lam); recognition searches at most the 32 looks with the largest share.

  bin/prun python tools/card103.5/files.py --origins true|1034 --workers 0,1 [--load MEM] [--save MEM] [--n N]
      [--through-sleep]  (arm: frames while asleep are tracked too, with the view held, not skipped)
      → runs/103.5/files_<origins>_w<workers>.json
"""
import json
import pickle
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
ORIGINS, WORKERS = arg("--origins", "true"), [int(x) for x in arg("--workers", "8,9").split(",")]
N_LIM, LOAD, SAVE, DUMP = int(arg("--n", 0)) or None, arg("--load", None), arg("--save", None), "--dump" in sys.argv
THROUGH = "--through-sleep" in sys.argv                # arm: frames while the agent sleeps are tracked too, the view held
sys.argv = ["track.py", "crafter"]
sys.path.insert(0, str(ROOT / "tools" / "card103.2"))
import track as TR                                     # noqa: E402

OUT = ROOT / "runs" / "103.5"
HW, WW, K, UNIT, CX, CY, HALF = TR.HW, TR.WW, TR.K, TR.UNIT, TR.CX, TR.CY, TR.HALF
MIN_PIX, GOOD, TOL = TR.MIN_PIX, TR.GOOD, TR.TOL
CONFIRM, DELETE, WINDOW = 0.95, 0.05, 3
G = 45                                                 # px: how far the map of an unseen file's places extends
RECENT, EM, B, NREC = 10, 5, 15, 32
LOOK_THR = 3 * TOL ** 2                                # two looks' pixels match (fixed: looks are compared across frames)
thr = TR.thr


def lgamma(x):
    y = np.array(x, np.float64)
    s = np.zeros_like(y)
    for _ in range(7):
        small = y < 7
        s -= np.where(small, np.log(y), 0)
        y = np.where(small, y + 1, y)
    return s + (y - 0.5) * np.log(y) - y + 0.5 * np.log(2 * np.pi) + 1 / (12 * y) - 1 / (360 * y ** 3) + \
        1 / (1260 * y ** 5)


def betaln(a, b):
    return lgamma(a) + lgamma(b) - lgamma(np.asarray(a) + np.asarray(b))


def look_of(tmpl, mask):
    """A fixed B x B look: the mask's pixels around its centroid."""
    ys, xs = np.nonzero(mask)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    T = np.zeros((B, B, 3), np.float32)
    M = np.zeros((B, B), bool)
    yy, xx = ys - cy + B // 2, xs - cx + B // 2
    ok = (yy >= 0) & (yy < B) & (xx >= 0) & (xx < B)
    T[yy[ok], xx[ok]] = tmpl[ys[ok], xs[ok]]
    M[yy[ok], xx[ok]] = True
    return T, M


class Memory:
    """Long-term: looks (leaders), records, the files' displacement distribution, detection and move rates."""

    def __init__(self, p_own):
        self.LT = np.zeros((0, B, B, 3), np.float32)
        self.LM = np.zeros((0, B, B), bool)
        self.Ln = np.zeros(0, np.int64)
        self.Lc = np.zeros((0, 3), np.float32)
        self.rk = []                                   # record -> look
        self.k = []                                    # steps seen
        self.mu = []                                   # steps it moved by itself (soft)
        self.pd = 10 * p_own + 1e-3                    # confirmed files' displacements per step; before any is seen,
                                                       # the agent's own placements with ten steps' weight (098.6's prior)
        self.det = [9.0, 1.0]                          # confirmed files visible: found, not found (Beta(9, 1))
        self.drift_long = [0.5, 1.0]                   # world pixels: changed, seen (long run)
        self.share_cache, self.share_t = {}, -1
        self.pi0 = 0.5

    # looks
    def like(self, T, M):
        """Similarity of a look to every leader: the smaller of the two masks' matched shares at the best alignment."""
        n = M.sum()
        if not len(self.Ln) or n == 0:
            return np.zeros(len(self.Ln))
        c = T[M].mean(0)
        cand = np.flatnonzero((self.Ln >= GOOD * n) & (n >= GOOD * self.Ln) & (((self.Lc - c) ** 2).sum(1) < 0.05))
        sim = np.zeros(len(self.Ln))
        if not len(cand):
            return sim
        LT, LM, Ln = self.LT[cand], self.LM[cand], self.Ln[cand]
        best = np.zeros(len(cand))
        for oy in range(-2, 3):
            for ox in range(-2, 3):
                Ts = np.zeros_like(T)
                Ms = np.zeros_like(M)
                ys = slice(max(oy, 0), B + min(oy, 0))
                yd = slice(max(-oy, 0), B + min(-oy, 0))
                xs = slice(max(ox, 0), B + min(ox, 0))
                xd = slice(max(-ox, 0), B + min(-ox, 0))
                Ts[ys, xs], Ms[ys, xs] = T[yd, xd], M[yd, xd]
                m = Ms[None] & LM & (((Ts[None] - LT) ** 2).sum(-1) < LOOK_THR)
                cnt = m.sum((1, 2))
                best = np.maximum(best, np.minimum(cnt / n, cnt / Ln))
        sim[cand] = best
        return sim

    def kind(self, tmpl, mask, add=True):
        T, M = look_of(tmpl, mask)
        sim = self.like(T, M)
        if len(sim) and sim.max() >= GOOD:
            return int(sim.argmax())
        if not add:
            return -1
        self.LT = np.concatenate([self.LT, T[None]])
        self.LM = np.concatenate([self.LM, M[None]])
        self.Ln = np.r_[self.Ln, M.sum()]
        self.Lc = np.concatenate([self.Lc, T[M].mean(0)[None]])
        return len(self.Ln) - 1

    def new_record(self, look):
        self.rk.append(look)
        self.k.append(0.0)
        self.mu.append(0.0)
        self.share_t = -1
        return len(self.rk) - 1

    # rates
    def m0(self):
        """How often a confirmed file moves in a step."""
        return float((self.pd.sum() - self.pd[K, K] + 0.5) / (self.pd.sum() + 1))

    def eps(self):
        """How often a piece of the world is seen to change by itself in a step (098.6: the rate any token is seen to
        move): the world's pixels' change rate, since records exist only for what changed."""
        return float(self.drift_long[0] / self.drift_long[1])

    def p_d(self):
        return float(self.det[0] / sum(self.det))

    def _evidence(self, k, mu):
        """098.6: log P(history | memory of its own) - log P(history | the map predicts it)."""
        m, e = np.clip(self.m0(), 1e-3, 1 - 1e-3), np.clip(self.eps(), 1e-4, 1 - 1e-4)
        stay = np.maximum(k - mu, 0)
        return betaln(m + mu, 1 - m + stay) - betaln(m, 1 - m) - mu * np.log(e) - stay * np.log1p(-e)

    def shares(self, t):
        """Per look, the share of its records that needed memory of their own (EM), all records pooled joining with
        one record's weight. Refreshed every 100 steps or when records were added."""
        if self.share_t >= 0 and t - self.share_t < 100:
            return self.share_cache
        self.share_t = t
        if not self.rk:
            self.share_cache = {}
            return {}
        rk, k, mu = np.array(self.rk), np.array(self.k), np.array(self.mu)
        ev = self._evidence(k, mu)
        pi = 0.5
        for _ in range(EM):
            z = 1 / (1 + np.exp(-np.clip(np.log(pi / (1 - pi)) + ev, -50, 50)))
            pi = float((z.sum() + 0.5) / (len(z) + 1))
        self.pi0 = pi
        out = {}
        order = np.argsort(rk, kind="stable")
        bounds = np.flatnonzero(np.r_[True, np.diff(rk[order]) != 0, True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            idx = order[a:b]
            p = pi
            for _ in range(EM):
                z = 1 / (1 + np.exp(-np.clip(np.log(p / (1 - p)) + ev[idx], -50, 50)))
                p = float(np.clip((z.sum() + self.pi0) / (len(idx) + 1), 1e-6, 1 - 1e-6))
            out[int(rk[idx[0]])] = p
        self.share_cache = out
        return out

    def share(self, look, t):
        return self.shares(t).get(look, self.pi0) if look >= 0 else self.pi0


class Files:
    def __init__(self, cam, mem):
        self.cam, self.mem = cam, mem
        self.next_id = 0
        self.stats = Counter()
        self.trace = None                                  # diagnostics: (t, file, d, ln L, no evidence, r) per placement

    def reset(self):
        self.canvas = np.zeros((2 * HALF, 2 * HALF, 3), np.float32)
        self.known = np.zeros((2 * HALF, 2 * HALF), bool)
        self.files = []
        self.dormant = []                  # records of tentative files no longer followed, kept at their place (098.6)
        self.recent = deque(maxlen=RECENT)

    def view(self, o):
        x, y = int(o[0]) + HALF, int(o[1]) + HALF
        return slice(y, y + HW), slice(x, x + WW)

    def drift(self):
        """This scan's rate of world pixels changing: the last RECENT frames pooled with one pseudo-frame of the long run."""
        ch = sum(c for c, n in self.recent)
        n = sum(n for c, n in self.recent)
        p_long = self.mem.drift_long[0] / self.mem.drift_long[1]
        frame = max(n / max(len(self.recent), 1), 1.0)
        return float(np.clip((ch + frame * p_long) / (n + frame), 1e-6, 1))

    def match(self, f, o, F, s2, R):
        """Inlier share and visible share of file F's template at each displacement up to R."""
        th, tw = F["tmpl"].shape[:2]
        vy, vx = F["pos"][1] - o[1], F["pos"][0] - o[0]
        y0, x0 = vy - R, vx - R
        if y0 + 2 * R + th <= 0 or y0 >= HW or x0 + 2 * R + tw <= 0 or x0 >= WW:
            return None
        P = np.full((2 * R + th, 2 * R + tw, 3), np.nan, np.float32)
        sy0, sy1 = max(y0, 0), min(y0 + 2 * R + th, HW)
        sx0, sx1 = max(x0, 0), min(x0 + 2 * R + tw, WW)
        P[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = f[sy0:sy1, sx0:sx1]
        cm = np.zeros((2 * R + th, 2 * R + tw), bool)
        cm[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.cam[sy0:sy1, sx0:sx1]
        P[cm] = np.nan                                            # the camera's pixels are not the file's
        W = np.moveaxis(np.lib.stride_tricks.sliding_window_view(P, (th, tw), axis=(0, 1)), 2, -1)
        r2 = ((W - F["tmpl"]) ** 2).sum(-1)
        vis = ~np.isnan(r2) & F["mask"]
        nv = vis.sum((-1, -2))
        inl = ((r2 < thr(s2)) & vis).sum((-1, -2))
        return inl / np.maximum(nv, 1), nv / F["mask"].sum()

    def predict(self, M, p):
        """A file's map of places after one more step: convolved with the per-step moves, kept on its grid."""
        n = M.shape[0] + p.shape[0] - 1
        out = np.fft.irfft2(np.fft.rfft2(M, (n, n)) * np.fft.rfft2(p, (n, n)), (n, n))[K:K + 2 * G + 1, K:K + 2 * G + 1]
        out = np.clip(out, 0, None)
        return out / out.sum() if out.sum() > 0 else out

    def clutter(self, f, tmpl, msk, y0, x0, s2):
        """How often a piece's content is found elsewhere in its gate in this frame (at least 0.8 of its pixels), as a
        rate per displacement: the density of false matches a tracker's gate sees (clutter)."""
        h, w = msk.shape
        P = np.full((2 * K + h, 2 * K + w, 3), np.nan, np.float32)
        ya, yb, xa, xb = max(y0 - K, 0), min(y0 + K + h, HW), max(x0 - K, 0), min(x0 + K + w, WW)
        P[ya - (y0 - K):yb - (y0 - K), xa - (x0 - K):xb - (x0 - K)] = f[ya:yb, xa:xb]
        W = np.moveaxis(np.lib.stride_tricks.sliding_window_view(P, (h, w), axis=(0, 1)), 2, -1)
        r2 = ((W - tmpl) ** 2).sum(-1)
        vis = (~np.isnan(r2) & msk).sum((-1, -2)) >= 0.5 * msk.sum()
        good = vis & ((((r2 < thr(s2)) & msk).sum((-1, -2)) / msk.sum()) >= 0.8)
        good[K - 2:K + 3, K - 2:K + 3] = False
        return float((good.sum() + 0.5) / (vis.sum() + 1))

    def placed(self, F, o, d):
        """View pixels of F's mask after displacement d, and which of them are in view."""
        ys, xs = np.nonzero(F["mask"])
        ys, xs = ys + F["pos"][1] + d[1] - o[1], xs + F["pos"][0] + d[0] - o[0]
        ok = (ys >= 0) & (ys < HW) & (xs >= 0) & (xs < WW)
        return ys, xs, ok

    def birth(self, t, o, tmpl, msk, y0, x0, r0, how, look, rec=None):
        if rec is None:
            rec = self.mem.new_record(look)
        else:
            look = self.mem.rk[rec]
        F = {"id": self.next_id, "pos": (x0 + o[0], y0 + o[1]), "tmpl": tmpl, "mask": msk, "r": float(r0),
             "born": t, "seen": t, "confirmed": False, "look": look, "rec": rec, "hits": 0}
        self.next_id += 1
        self.files.append(F)
        self.stats[f"born {how}"] += 1
        if r0 >= CONFIRM:
            self.confirm(F)
        return F

    def linked(self, wys, wxs):
        """The kept record whose place a moved piece came from (098.6: a pair continues the place's record)."""
        best, bi = 0.0, None
        if not len(wys) or not self.dormant:
            return None
        src = set(zip(wys.tolist(), wxs.tolist()))
        for i, D in enumerate(self.dormant):
            if D.get("gone") is None:
                continue
            ys, xs = np.nonzero(D["mask"])
            ys, xs = ys + D["pos"][1], xs + D["pos"][0]
            ov = sum((y, x) in src for y, x in zip(ys.tolist(), xs.tolist())) / len(ys)
            if ov > best:
                best, bi = ov, i
        if bi is None or best < 0.5:
            return None
        D = self.dormant.pop(bi)
        return D["rec"]

    def confirm(self, F):
        F["confirmed"] = True
        self.stats["confirmed"] += 1
        lk = self.mem.kind(F["tmpl"], F["mask"])           # the look it has now (its mask may have grown)
        if lk != F["look"]:
            F["look"] = lk
            self.mem.rk[F["rec"]] = lk
            self.mem.share_t = -1

    def grow(self, F, wys, wxs, cols):
        """Add world pixels (wys, wxs) with colours cols to F's mask."""
        x0, y0 = F["pos"]
        th, tw = F["tmpl"].shape[:2]
        ny0, nx0 = min(y0, wys.min()), min(x0, wxs.min())
        ny1, nx1 = max(y0 + th, wys.max() + 1), max(x0 + tw, wxs.max() + 1)
        T = np.zeros((ny1 - ny0, nx1 - nx0, 3), np.float32)
        M = np.zeros((ny1 - ny0, nx1 - nx0), bool)
        T[y0 - ny0:y0 - ny0 + th, x0 - nx0:x0 - nx0 + tw] = F["tmpl"]
        M[y0 - ny0:y0 - ny0 + th, x0 - nx0:x0 - nx0 + tw] = F["mask"]
        T[wys - ny0, wxs - nx0] = cols
        M[wys - ny0, wxs - nx0] = True
        F["tmpl"], F["mask"], F["pos"] = T, M, (int(nx0), int(ny0))

    def step(self, t, f, o, s2):
        mem = self.mem
        sy, sx = self.view(o)
        V, Kn = self.canvas[sy, sx], self.known[sy, sx]
        dif = f - V
        world = Kn & ~self.cam
        if world.sum() > 50:
            dif = dif - np.median(dif[world], 0)
        r2w = (dif ** 2).sum(-1)
        res = world & (r2w > thr(s2))
        new = ~Kn & ~self.cam
        delta = self.drift()
        shares = mem.shares(t)
        claimed = np.zeros((HW, WW), bool)
        revealed = np.zeros((HW, WW), bool)
        # 1. candidate placements of every file
        cands, vis_at = [], {}
        self.why = {}                                      # diagnostics: why a file was not placed
        pstep = mem.pd / mem.pd.sum()
        for F in self.files:
            if F.get("map") is None:                    # seen last step: one step's moves
                R, Pm = K, pstep
            else:                                       # unseen: its map of places, moved by one step's moves
                R, Pm = G, self.predict(F["map"], pstep)
                F["map"] = Pm
            F["P"], F["vis"] = Pm, None
            m = self.match(f, o, F, s2, R)
            if m is None:
                continue
            inl, vis = m
            lp = np.log(np.maximum(Pm, 1e-30))
            F["vis"] = vis
            vis_at[F["id"]] = float((Pm * (vis >= 0.5)).sum())        # P(it is in view after its possible moves)
            if vis.max() < 0.3:
                continue
            ok = (inl >= GOOD) & (vis >= 0.3)
            if not ok.any():
                self.why[F["id"]] = "no match"
                continue
            self.why[F["id"]] = "taken by another file"
            sc = np.where(ok, inl + 0.02 * lp, -9)
            pi = shares.get(F["look"], mem.pi0)
            nvis = max(int((vis >= 0.3).sum()), 1)
            for i in np.argsort(-sc.ravel())[:3]:
                if sc.flat[i] < -1:
                    break
                dy, dx = np.unravel_index(i, sc.shape)
                el = ok.copy()                       # clutter: how often its template matches elsewhere in its gate
                el[max(dy - 2, 0):dy + 3, max(dx - 2, 0):dx + 3] = False
                rho = (el.sum() + 0.5) / (nvis + 1)
                d = (int(dx - R), int(dy - R))
                ys, xs, okv = self.placed(F, o, d)
                ys, xs = ys[okv], xs[okv]
                ys, xs = ys[~self.cam[ys, xs]], xs[~self.cam[ys, xs]]
                if not len(ys):
                    continue
                f_res, f_unk = res[ys, xs].mean(), new[ys, xs].mean()
                f_exp = 1 - f_res - f_unk
                own = float(np.exp(lp[dy, dx]))
                if f_exp > 0.5 or (not F["confirmed"] and d == (0, 0)):
                    self.why.setdefault(-F["id"] - 1, "the world explains it")
                    continue              # the world explains it as well (under a tentative file the canvas holds the
                                          # file's own last look): not a detection of the file (098.6: an unchanged
                                          # place is not matched), so a visible file not found elsewhere is missed
                else:
                    other = (f_res * rho + f_unk * (1 - pi)) / max(f_res + f_unk, 1e-9)
                    lnL = float(np.log(own) - np.log(max(other, 1e-12)))
                if lnL != 0.0 and np.log(max(F["r"], 1e-12)) + lnL <= 0:
                    self.why.setdefault(-F["id"] - 1, "clutter")
                    continue                  # "not a file" explains this place better (098.6's assignment)
                cands.append((np.log(max(F["r"], 1e-12)) + lnL + 0.1 * inl[dy, dx], F["id"], d, lnL, lnL == 0.0,
                              ys, xs))
        # 2. one placement for all files
        byid = {F["id"]: F for F in self.files}
        done, seen = set(), []
        cands.sort(key=lambda c: -c[0])
        for score, fid, d, lnL, neutral, ys, xs in cands:
            if fid in done or claimed[ys, xs].mean() > 0.5:
                continue
            F = byid[fid]
            done.add(fid)
            found_again = t - F["seen"] > 1
            if not neutral:
                L = np.exp(np.clip(lnL, -50, 50))
                F["r"] = float(F["r"] * L / (1 - F["r"] + F["r"] * L))
            if self.trace is not None:
                self.trace.append((t, fid, d, round(lnL, 2), neutral, round(F["r"], 3)))
            mv = d != (0, 0)
            mem.k[F["rec"]] += 1
            if mv:
                mem.mu[F["rec"]] += float(L / (1 + L)) if not neutral else 0.5
                ys0, xs0, ok0 = self.placed(F, o, (0, 0))
                revealed[ys0[ok0], xs0[ok0]] = True
            if F["confirmed"]:
                mem.det[0] += 1
                if not found_again and max(abs(d[0]), abs(d[1])) <= K:
                    mem.pd[d[1] + K, d[0] + K] += 1
            F["pos"] = (F["pos"][0] + d[0], F["pos"][1] + d[1])
            F["seen"] = t
            F["map"] = None
            F["hits"] += 1
            F["moved_now"] = d if mv else None
            claimed[ys, xs] = True
            ty, tx = np.nonzero(F["mask"])                              # the template follows slow changes (light):
            vy, vx = ty + F["pos"][1] - o[1], tx + F["pos"][0] - o[0]  # only pixels that still match it (robust)
            ok3 = (vy >= 0) & (vy < HW) & (vx >= 0) & (vx < WW)
            ok3[ok3] &= ~self.cam[vy[ok3], vx[ok3]]
            ty, tx, vy, vx = ty[ok3], tx[ok3], vy[ok3], vx[ok3]
            inl3 = ((F["tmpl"][ty, tx] - f[vy, vx]) ** 2).sum(-1) < thr(s2)
            ty, tx, vy, vx = ty[inl3], tx[inl3], vy[inl3], vx[inl3]
            F["tmpl"][ty, tx] = 0.5 * F["tmpl"][ty, tx] + 0.5 * f[vy, vx]
            if not F["confirmed"] and F["r"] >= CONFIRM:
                self.confirm(F)
        # 3. misses: a file not found where it is visible
        for F in self.files:
            if F["id"] in done:
                continue
            F["moved_now"] = None
            if self.why.get(F["id"]) == "taken by another file" and not any(c[1] == F["id"] for c in cands):
                self.why[F["id"]] = self.why.get(-F["id"] - 1, "?")
            v = vis_at.get(F["id"], 0.0)
            if v <= 0:
                continue
            q = mem.p_d() * v
            F["r"] = float(F["r"] * (1 - q) / max(1 - F["r"] * q, 1e-12))
            M = F["P"]                                  # negative information (098.5): where it was looked for
            if M.shape[0] != 2 * G + 1:                 # and not found, it is less likely to be
                E = np.zeros((2 * G + 1, 2 * G + 1))
                E[G - K:G + K + 1, G - K:G + K + 1] = M
                M, Vv = E, np.zeros((2 * G + 1, 2 * G + 1))
                Vv[G - K:G + K + 1, G - K:G + K + 1] = F["vis"]
            else:
                Vv = F["vis"]
            M = M * (1 - mem.p_d() * (Vv >= 0.5))
            F["map"] = M / M.sum() if M.sum() > 0 else M
            if F["confirmed"]:                  # EM for P_D: the chance it was in view, given it was not found
                mem.det[1] += v * (1 - mem.p_d()) / max(1 - v * mem.p_d(), 1e-12)
        revealed &= ~claimed
        # 4. cohesion: residual pixels near a file that moved, shown by the canvas at the same displacement, join it
        free = res & ~claimed & ~revealed
        for F in self.files:
            d = F.get("moved_now")
            if d is None or F["id"] not in done or not free.any():
                continue
            ys, xs, ok = self.placed(F, o, (0, 0))
            near = np.zeros((HW, WW), bool)
            near[ys[ok], xs[ok]] = True
            for _ in range(2):
                g = near.copy()
                g[1:] |= near[:-1]
                g[:-1] |= near[1:]
                g[:, 1:] |= near[:, :-1]
                g[:, :-1] |= near[:, 1:]
                near = g
            cy, cx = np.nonzero(near & free)
            if not len(cy):
                continue
            wy, wx = cy + o[1] + HALF - d[1], cx + o[0] + HALF - d[0]
            sv_y, sv_x = cy - d[1], cx - d[0]
            inv = (sv_y >= 0) & (sv_y < HW) & (sv_x >= 0) & (sv_x < WW)
            gone = np.zeros(len(cy), bool)                              # and is no longer there (the pair test)
            gone[inv] = res[sv_y[inv], sv_x[inv]] | revealed[sv_y[inv], sv_x[inv]]
            same = self.known[wy, wx] & gone & (((self.canvas[wy, wx] - f[cy, cx]) ** 2).sum(-1) < thr(s2))
            if not same.any():
                continue
            cy, cx = cy[same], cx[same]
            self.grow(F, cy + o[1], cx + o[0], f[cy, cx])
            claimed[cy, cx] = True
            free[cy, cx] = False
            sy0, sx0 = cy - d[1], cx - d[0]
            ok2 = (sy0 >= 0) & (sy0 < HW) & (sx0 >= 0) & (sx0 < WW)
            revealed[sy0[ok2], sx0[ok2]] = True
            free[sy0[ok2], sx0[ok2]] = False
            self.stats["pixels joined by cohesion"] += len(cy)
        # 4b. records kept at their place: one more step where the place still shows them; otherwise they end
        alive = []
        for D in self.dormant:
            ys, xs, ok = self.placed(D, o, (0, 0))
            if ok.mean() < 0.5:
                alive.append(D)
                continue
            ys, xs = ys[ok], xs[ok]
            r2 = ((f[ys, xs] - D["tmpl"][D["mask"]][ok]) ** 2).sum(-1)
            if (r2 < thr(s2)).mean() >= GOOD and not claimed[ys, xs].any():
                mem.k[D["rec"]] += 1
                alive.append(D)
            else:
                D["gone"] = t
                alive.append(D)
        self.dormant = alive
        # 5. discovery: residual pieces no file explains
        free &= ~revealed
        pmove = mem.pd / mem.pd.sum()
        for ys, xs in sorted(TR.components(free), key=lambda c: -len(c[0])):
            if len(ys) < MIN_PIX or not free[ys[0], xs[0]]:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            msk = np.zeros((y1 - y0, x1 - x0), bool)
            msk[ys - y0, xs - x0] = True
            tmpl = f[y0:y1, x0:x1].copy()
            wy, wx = y0 + o[1] + HALF, x0 + o[0] + HALF
            C = self.canvas[wy - K:wy + K + (y1 - y0), wx - K:wx + K + (x1 - x0)]
            Kc = self.known[wy - K:wy + K + (y1 - y0), wx - K:wx + K + (x1 - x0)]
            Wc = np.moveaxis(np.lib.stride_tricks.sliding_window_view(C, (y1 - y0, x1 - x0), axis=(0, 1)), 2, -1)
            Wk = np.lib.stride_tricks.sliding_window_view(Kc, (y1 - y0, x1 - x0), axis=(0, 1))
            r = ((Wc - tmpl) ** 2).sum(-1)
            good = (((r < thr(s2)) & Wk & msk).sum((-1, -2)) / msk.sum()) >= 0.8
            good[K, K] = False
            ng = int(good.sum())
            if ng > 2:
                self.stats["uniform, the world's"] += 1
                continue
            look = mem.kind(tmpl, msk) if len(ys) <= B * B else -1
            pi = mem.share(look, t)
            vanished = False
            if ng >= 1:                 # the pair test (098.5): what is here was there, and it is no longer there
                gy, gx = np.unravel_index(int(np.argmax(np.where(good, -np.abs(np.arange(-K, K + 1))[:, None]
                                                                  - np.abs(np.arange(-K, K + 1))[None], -99))),
                                          good.shape)
                d = (K - gx, K - gy)
                sy0, sx0 = ys - d[1], xs - d[0]
                ok = (sy0 >= 0) & (sy0 < HW) & (sx0 >= 0) & (sx0 < WW)
                ok[ok] &= ~self.cam[sy0[ok], sx0[ok]]
                if ok.sum() >= 0.5 * len(ys):
                    piece = np.zeros((HW, WW), bool)
                    piece[ys, xs] = True
                    gone = res | revealed | piece
                    vanished = gone[sy0[ok], sx0[ok]].mean() >= 0.5
                if not vanished:
                    self.stats["one source, still there: appeared"] += 1
            if vanished:                                                # it moved by itself
                free[sy0[ok], sx0[ok]] = False
                revealed[sy0[ok], sx0[ok]] = True
                L = float(pmove[d[1] + K, d[0] + K]) / self.clutter(f, tmpl, msk, y0, x0, s2)   # against clutter
                r0 = pi * L / (1 - pi + pi * L)
                link = self.linked(sy0[ok] + o[1], sx0[ok] + o[0])
                if link is not None:
                    look = mem.rk[link]
                    pi = mem.share(look, t)
                    r0 = pi * L / (1 - pi + pi * L)
                    self.stats["moving births continuing a record"] += 1
                F = self.birth(t, o, tmpl, msk, y0, x0, r0, "moving", look, link)
                mem.k[F["rec"]] += 1
                mem.mu[F["rec"]] += L / (1 + L)
            else:                                                       # appeared, or changed in place
                F = self.birth(t, o, tmpl, msk, y0, x0, pi, "appeared", look)
                mem.k[F["rec"]] += 1
            F["hits"] = 1
            claimed[ys, xs] = True
        # 6. recognition in pixels seen for the first time: looks that have needed memory of their own
        if new.sum() >= MIN_PIX and shares:
            eligible = sorted(((p, lk) for lk, p in shares.items() if p >= DELETE), reverse=True)[:NREC]
            ny, nx = np.nonzero(new)
            by0, by1 = max(ny.min() - B + 1, -B + 1), min(ny.max() + 1, HW)
            bx0, bx1 = max(nx.min() - B + 1, -B + 1), min(nx.max() + 1, WW)
            py, px = np.mgrid[by0:by1, bx0:bx1]
            py, px = py.ravel(), px.ravel()
            hits = []
            for p, lk in eligible:
                my, mx = np.nonzero(mem.LM[lk])
                if len(my) < MIN_PIX:
                    continue
                cols = mem.LT[lk][my, mx]
                Y, X = py[:, None] + my[None], px[:, None] + mx[None]
                inside = (Y >= 0) & (Y < HW) & (X >= 0) & (X < WW)
                Yc, Xc = np.clip(Y, 0, HW - 1), np.clip(X, 0, WW - 1)
                inside &= ~self.cam[Yc, Xc]
                nv = inside.sum(1)
                fresh = (new[Yc, Xc] & inside).sum(1)
                r2 = ((f[Yc, Xc] - cols[None]) ** 2).sum(-1)
                inl = ((r2 < thr(s2)) & inside).sum(1)
                okp = (nv >= 0.5 * len(my)) & (fresh >= 0.5 * nv) & (inl >= GOOD * np.maximum(nv, 1))
                for i in np.flatnonzero(okp):
                    hits.append((p, inl[i] / nv[i], lk, int(py[i]), int(px[i])))
            hits.sort(key=lambda h: (-h[0], -h[1]))
            for p, s, lk, y, x in hits:
                M = mem.LM[lk]
                my, mx = np.nonzero(M)
                Y, X = y + my, x + mx
                ok = (Y >= 0) & (Y < HW) & (X >= 0) & (X < WW)
                if claimed[Y[ok], X[ok]].mean() > 0.2:
                    continue
                y0, x0 = y + my.min(), x + mx.min()
                msk = M[my.min():my.max() + 1, mx.min():mx.max() + 1].copy()
                tm = mem.LT[lk][my.min():my.max() + 1, mx.min():mx.max() + 1].copy()
                vy, vx = np.nonzero(msk)
                vy, vx = vy + y0, vx + x0
                okv = (vy >= 0) & (vy < HW) & (vx >= 0) & (vx < WW)
                tm[vy[okv] - y0, vx[okv] - x0] = f[vy[okv], vx[okv]]
                F = self.birth(t, o, tm, msk, y0, x0, p, "recognised", lk)
                mem.k[F["rec"]] += 1
                F["hits"] = 1
                claimed[Y[ok], X[ok]] = True
        # 7. deletion; a tentative file that ran out of its window keeps its record at its place
        keep = []
        for F in self.files:
            if F["r"] < DELETE or (not F["confirmed"] and t - F["born"] >= WINDOW) or t - F["seen"] > 300:
                self.stats["deleted"] += 1
                if not F["confirmed"] and F["r"] >= DELETE and F["seen"] == t:
                    self.dormant.append({"rec": F["rec"], "pos": F["pos"], "tmpl": F["tmpl"], "mask": F["mask"]})
                continue
            keep.append(F)
        self.files = keep
        self.dormant = [D for D in self.dormant if "gone" not in D]     # a changed place's record could be continued
                                                                        # by this step's births only
        # 8. the world: every pixel no confirmed file claims and the camera does not carry; this scan's drift
        under = np.zeros((HW, WW), bool)
        for F in self.files:
            if F["confirmed"] and F["seen"] == t:
                ys, xs, ok = self.placed(F, o, (0, 0))
                under[ys[ok], xs[ok]] = True
                seen.append((F["id"], ((xs[ok].mean() - CX) / UNIT, (ys[ok].mean() - CY) / UNIT)))
        wl = world & ~under & ~claimed
        self.recent.append((int((res & wl).sum()), int(wl.sum())))
        mem.drift_long[0] += int((res & wl).sum())
        mem.drift_long[1] += int(wl.sum())
        wr = ~under & ~self.cam
        Vw = self.canvas[sy, sx]
        Vw[wr] = f[wr]
        self.known[sy, sx] |= wr
        return seen


def agent_prior():
    """The agent's own placements (card 103.4's placing, fitted on training play), pooled over actions."""
    pr = np.load(OUT / "placing_tau300.npz")["pr"].sum(0)
    return (pr / pr.sum()).reshape(2 * K + 1, 2 * K + 1)


def origins(d, seg, w):
    if ORIGINS == "true":
        return UNIT * (d["pos"].astype(np.int64) - (4, 3))
    o = np.load(OUT / f"org_tau300_w{w}.npy")[:len(seg)].copy()
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
    if THROUGH:                                        # asleep, the agent does not move: its view is held
        for t in range(1, len(seg)):
            if d["sleep"][t] and seg[t] == seg[t - 1]:
                o[t] = o[t - 1]
    return o


def extra(rec, d, kinds):
    """Still creatures (before their first move in view) with a file on them; detections on no creature; night."""
    obj = d["obj"]
    by_t = defaultdict(list)
    for row in obj:
        if row[2] in kinds:
            by_t[int(row[0])].append(row)
    moved_since = {}
    still_n = still_hit = 0
    n_det = n_off = 0
    night_n = night_hit = 0
    for t, s in enumerate(rec):
        e = int(d["ep"][t])
        ps = [np.array(p) for _, p in s]
        n_det += len(ps)
        cre = [np.array([r[3], r[4]], float) for r in by_t.get(t, [])]
        for p in ps:
            if not any(np.abs(p - c).max() < 0.5 for c in cre):
                n_off += 1
        for r in by_t.get(t, []):
            g = (e, int(r[1]))
            wpos = (int(d["pos"][t][0] + r[3]), int(d["pos"][t][1] + r[4]))
            first = moved_since.setdefault(g, [wpos, False])
            if wpos != first[0]:
                first[1] = True
            hit = any(np.abs(p - np.array([r[3], r[4]], float)).max() < 0.5 for p in ps)
            if not first[1]:
                still_n += 1
                still_hit += hit
            if d["light"][t] < 0.5:
                night_n += 1
                night_hit += hit
    return {"still since seen: frames, with a file": [still_n, still_hit],
            "detections, on no creature": [n_det, n_off], "night frames, with a file": [night_n, night_hit]}


def main():
    t0 = time.monotonic()
    OUT.mkdir(exist_ok=True, parents=True)
    cam = np.load(ROOT / "runs" / "103.1" / "pi_crafter_world_learned.npy")[:HW, :WW] > 0.5
    mem = pickle.load(open(LOAD, "rb")) if LOAD else Memory(agent_prior())
    kinds = {1: "cow", 2: "zombie", 3: "skeleton", 4: "arrow"}
    S, byk, ex = defaultdict(float), defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0, 0])
    T = Files(cam, mem)
    n_lim = N_LIM
    for w in WORKERS:
        z = np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz")
        d = {k: z[k] for k in ("obs", "act", "done", "pos", "sleep", "light", "ep", "obj")}
        if n_lim:
            d = {k: (v[:n_lim] if k != "obj" else v[v[:, 0] < n_lim]) for k, v in d.items()}
        seg = TR.segments(d)
        if THROUGH:
            seg = np.cumsum(np.r_[True, (d["ep"][1:] != d["ep"][:-1]) | d["done"][:-1]])
        org = origins(d, seg, w)
        rec, s2, last_seg = [], 1e-4, None
        for t in range(len(seg)):
            if seg[t] < 0:
                rec.append([])
                continue
            f = d["obs"][t, :HW, :WW].astype(np.float32) / 255
            if seg[t] != last_seg:
                T.reset()
            last_seg = seg[t]
            if T.known.any():
                sy, sx = T.view(org[t])
                Kn = T.known[sy, sx] & ~cam
                if Kn.sum() > 50:
                    s2 = max(float(np.median(((f - T.canvas[sy, sx]) ** 2).sum(-1)[Kn])) / 2.366, 1e-4)
            rec.append(T.step(t, f, org[t], s2))
            if t % 5000 == 0:
                print(f"  worker {w} step {t} files {len(T.files)} looks {len(mem.Ln)} records {len(mem.rk)} "
                      f"({time.monotonic() - t0:.0f} s)", flush=True)
        idtp, ngt, npred, ik, nk = TR.idf1(rec, d["obj"], d["ep"], kinds)
        S["idtp"] += idtp
        S["gt"] += ngt
        S["pred"] += npred
        for k in kinds:
            byk[kinds[k]][0] += ik.get(k, 0)
            byk[kinds[k]][1] += nk.get(k, 0)
        for k, v in extra(rec, d, kinds).items():
            ex[k][0] += v[0]
            ex[k][1] += v[1]
        if DUMP:
            pickle.dump({"rec": rec}, open(OUT / f"dump_{ORIGINS}_w{w}.pkl", "wb"))
    sh = mem.shares(10 ** 9)
    top = sorted(((len([1 for r in mem.rk if r == lk]), lk, p) for lk, p in sh.items()), reverse=True)[:12]
    rep = {"origins": ORIGINS, "workers": WORKERS, "loaded memory": LOAD,
           "IDF1 (movers pooled)": round(2 * S["idtp"] / max(S["gt"] + S["pred"], 1), 4),
           "ID recall / ID precision": [round(S["idtp"] / max(S["gt"], 1), 4), round(S["idtp"] / max(S["pred"], 1), 4)],
           "object frames / file detections": [int(S["gt"]), int(S["pred"])], "ID true positives": float(S["idtp"]),
           "ID recall by kind": {k: round(a / max(b, 1), 4) for k, (a, b) in byk.items()},
           "still since seen: share with a file": round(ex["still since seen: frames, with a file"][1]
                                                        / max(ex["still since seen: frames, with a file"][0], 1), 4),
           "detections on no creature: share": round(ex["detections, on no creature"][1]
                                                     / max(ex["detections, on no creature"][0], 1), 4),
           "night creature frames: share with a file": round(ex["night frames, with a file"][1]
                                                             / max(ex["night frames, with a file"][0], 1), 4),
           "raw": {k: v for k, v in ex.items()},
           "file events": dict(T.stats), "looks": int(len(mem.Ln)), "records": len(mem.rk),
           "pooled share": round(mem.pi0, 4), "m0 (files' move rate)": round(mem.m0(), 4),
           "eps (records' move rate)": round(mem.eps(), 4), "P_D": round(mem.p_d(), 4),
           "drift (long run)": round(mem.drift_long[0] / mem.drift_long[1], 5),
           "largest looks: records, share": [(int(n), round(float(p), 3)) for n, lk, p in top],
           "seconds": round(time.monotonic() - t0, 1)}
    tag = f"{ORIGINS}_w{''.join(map(str, WORKERS))}" + (f"_n{n_lim}" if n_lim else "") + \
        (f"_from_{Path(LOAD).stem}" if LOAD else "") + ("_through_sleep" if THROUGH else "")
    (OUT / f"files_{tag}.json").write_text(json.dumps(rep, indent=1) + "\n")
    if SAVE:
        pickle.dump(mem, open(SAVE, "wb"))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
