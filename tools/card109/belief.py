"""Card 109: conditions from memory. The agent's belief that a place holds a code now, for places out of view, from
what it remembers (world 6, "searched caves"; tools/card102/worlds.py).

Memory: each place as last seen in this episode (code, vector, step), placed by the pose; the agent's own tile is
not written (the pose says where the body is). Events (the learning signal, no labels): whenever a place comes into
view, what is seen there is the target for a belief formed from memory as it stood one step before.

Arms (only the belief differs; everything else is shared):
  R (recall)    remembered place: recall over stored revisits, weights exp(-|v_r - v_x|^2 / s^2) exp(-(ln g_r - ln
                g_x)^2 / 2h^2), v the vector at the last sighting, g the gap; never seen: recall over stored first
                sightings, weights exp(-|c_r - c_x|^2 / s_c^2), c the mean vector of remembered places within 3 tiles
                (and whether there are any); b = (weighted counts + a / NC) / (weights + a)
  N (network)   card 105.6's transformer (bounded relative bias, positions only as differences) over the 9 x 9
                remembered places around x (vector, age; never seen absent), read at x: the code's likelihood (105.8)
Baselines: view only (the never-seen rule, its surroundings from the current view alone); static memory (version
27: the last sighting holds forever; never seen as R). Upper bound: the evaluator's probability given the history.

Episodes: training 0-399 build the stores and train N; 400-449 fit R's bandwidths; 450-499 choose the arm (held-out
log-likelihood at cave ends coming into view, CHARTER's rule for recall and networks). Test: 200 episodes.

  bin/prun python tools/card109/belief.py check                       → runs/109/data_check.json (section 4)
  bin/prun python tools/card109/belief.py run [--steps 8000] [--pairs 100]   → runs/109/run.json, probes.npz
  bin/prun python tools/card109/belief.py report                     → experiments/109-conditions-from-memory/results.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fn

ROOT = Path(__file__).resolve().parents[2]
ARGV = sys.argv[:]
arg = lambda k, d: next((ARGV[i + 1] for i in range(len(ARGV) - 1) if ARGV[i] == k), d)
CMD = sys.argv[1]
sys.argv = sys.argv[:1]
for d in ("card102", "card105", "card102.1"):
    sys.path.insert(0, str(ROOT / "tools" / d))
import worlds as W6                                       # noqa: E402
from model import TransitionModel                         # noqa: E402
import tokens as TK                                       # noqa: E402  (098.3's recipe: EN, nearest)

EN = TK.EN
dev = "cuda"
OUT = ROOT / "runs" / "109"
GH, GW = W6.GH6, W6.GW6
OFF = 8                                                   # map padding: view places reach 4 past the grid, windows 4 more
DX, DY = np.arange(63) % 9 - 4, np.arange(63) // 9 - 3   # view index -> offset from the agent
AGENT = 31
WX, WY = np.arange(81) % 9 - 4, np.arange(81) // 9 - 4   # 9 x 9 window index -> offset from the queried place
CENTRE = 40
NEAR3 = np.flatnonzero((np.abs(WX) <= 3) & (np.abs(WY) <= 3) & (np.arange(81) != CENTRE))
CLIP = 1e-3
STEPS = 300


def data(split):
    z = np.load(ROOT / "runs" / "card102" / f"world6_{split}.npz")
    d = {k: z[k] for k in ("act", "pos", "ep", "sem", "grid", "caves", "event")}
    d["code"] = np.load(OUT / f"world6_code_{split}.npy").astype(np.int64)
    d["V"] = torch.as_tensor(np.load(OUT / f"world6_vec_{split}.npy")).to(dev)
    return d


def codes_of(d):
    """Each evaluator kind's most common code in view (scoring and the goals as conditions)."""
    sem, code = d["sem"].reshape(-1, 63), d["code"][:, :63]
    return {k: int(np.bincount(code[sem == k]).argmax()) for k in range(8) if (sem == k).any()}


# ------------------------------------------------------------------------------------------------ memory replay
def replay(d, episodes, probe_every=0, probe_at=None):
    """Replays the episodes' memory. Returns the events (every place coming into view: its window of memory one step
    before, as (step, view index) pointers, and what was seen) and, for probes, the same for the six cave ends with the
    evaluator's truth and probability. probe_every: a probe every k steps while the agent is in the corridor;
    probe_at: {episode: step} (paired probes)."""
    ev = {k: [] for k in ("t", "win_t", "win_i", "code", "kind", "end", "x", "y")}
    pr = {k: [] for k in ("t", "ep", "win_t", "win_i", "truth", "ub", "dist", "ax", "ay")}
    caves_all = d["caves"]
    for e in episodes:
        t0 = e * STEPS
        mt = np.full((GH + 2 * OFF, GW + 2 * OFF), -1, np.int64)   # last seen: global step, view index, kind
        mi = np.zeros_like(mt)
        mk = np.full_like(mt, -1)
        caves = caves_all[e]
        ends = {(int(c[0]), int(c[1])): j for j, c in enumerate(caves)}
        eats = {}                                                   # end -> steps at which its berry was eaten
        for s in range(STEPS):
            t = t0 + s
            ax, ay = (int(v) for v in d["pos"][t])
            vx, vy = ax + DX, ay + DY
            seen_prev = mt[vy + OFF, vx + OFF] == t - 1
            enter = np.flatnonzero(~seen_prev & (np.arange(63) != AGENT))
            if len(enter):
                px, py = vx[enter], vy[enter]
                wy, wx = py[:, None] + WY[None] + OFF, px[:, None] + WX[None] + OFF
                ev["t"].append(np.full(len(enter), t))
                ev["win_t"].append(mt[wy, wx])
                ev["win_i"].append(mi[wy, wx].astype(np.int8))
                ev["code"].append(d["code"][t, enter])
                ev["kind"].append(d["sem"][t].reshape(63)[enter])
                ev["end"].append(np.array([(int(x), int(y)) in ends for x, y in zip(px, py)]))
                ev["x"].append(px)
                ev["y"].append(py)
            probe = (probe_every and s % probe_every == 0 and s > 0 and ay in W6.CORRIDOR6) or \
                (probe_at is not None and probe_at.get(e) == s)
            if probe:
                ex, ey = caves[:, 0].astype(int), caves[:, 1].astype(int)
                wy, wx = ey[:, None] + WY[None] + OFF, ex[:, None] + WX[None] + OFF
                pr["t"].append(t)
                pr["ep"].append(e)
                pr["win_t"].append(mt[wy, wx])
                pr["win_i"].append(mi[wy, wx].astype(np.int8))
                g = d["grid"][t]
                pr["truth"].append(np.stack([g[ey, ex] == W6.GEM, g[ey, ex] == W6.RIPE], 1))
                pr["ub"].append(upper_bound(caves, mt, mk, eats, s, t0))
                pr["dist"].append(np.abs(ex - ax) + np.abs(ey - ay))
                pr["ax"].append(ax)
                pr["ay"].append(ay)
            keep = np.arange(63) != AGENT
            mt[vy[keep] + OFF, vx[keep] + OFF] = t
            mi[vy[keep] + OFF, vx[keep] + OFF] = np.arange(63)[keep]
            mk[vy[keep] + OFF, vx[keep] + OFF] = d["sem"][t].reshape(63)[keep]
            if d["event"][t, 0] == 1:
                eats.setdefault((int(d["event"][t, 1]), int(d["event"][t, 2])), []).append(s)
    ev = {k: np.concatenate(v) if v else np.zeros(0) for k, v in ev.items()}
    pr = {k: np.array(v) for k, v in pr.items()}
    return ev, pr


def F(dd):
    """P(U <= dd), U uniform over the delays 30..60 (a bush ripens U steps after it is eaten or the episode starts)."""
    return np.clip((dd - 29) / 31.0, 0.0, 1.0)


def upper_bound(caves, mt, mk, eats, s, t0):
    """The evaluator's probability, given what the agent has seen, of a gem and of a ripe bush at each cave end now."""
    out = np.zeros((6, 2))
    for j, (ex, ey, kind, side) in enumerate(caves):
        ex, ey = int(ex), int(ey)
        tiles = [(ex, ey - side * d) for d in range(4)]
        known = any(mt[y + OFF, x + OFF] >= 0 for x, y in tiles)
        last = mk[ey + OFF, ex + OFF]
        if last >= 0:                                          # the end itself was seen
            out[j, 0] = float(last == W6.GEM)
            if last == W6.RIPE:
                out[j, 1] = 1.0
            elif last == W6.BARE:
                s0 = int(mt[ey + OFF, ex + OFF]) - t0
                se = max([x for x in eats.get((ex, ey), []) if x < s0 + 1], default=-1)   # last eaten, else the start
                a, b = F(s - se - 1), F(s0 - se - 1)
                out[j, 1] = (a - b) / max(1 - b, 1e-9)
        else:
            p_gem, p_ripe = 0.5, 0.7 + 0.3 * F(s)
            if known:
                out[j] = (p_gem, 0.0) if kind == 0 else (0.0, p_ripe)
            else:
                out[j] = (0.5 * p_gem, 0.5 * p_ripe)
    return out


# ------------------------------------------------------------------------------------------------ features
def gather(d, win_t, win_i, t):
    """Window pointers -> vectors (B, 81, 32), present (B, 81), age (B, 81) at query step t."""
    wt = torch.as_tensor(win_t, device=dev)
    present = wt >= 0
    V = d["V"][wt.clamp(min=0), torch.as_tensor(win_i.astype(np.int64), device=dev)].float() * present[..., None]
    age = (torch.as_tensor(t, device=dev)[:, None] - wt).clamp(min=0) * present
    return V, present, age


def recall_keys(d, win_t, win_i, t):
    """R's keys: the last sighting's vector and log gap (remembered), the surroundings (never seen)."""
    V, present, age = gather(d, win_t, win_i, t)
    seen = present[:, CENTRE]
    near = present[:, NEAR3]
    ctx = (V[:, NEAR3] * near[..., None]).sum(1) / near.sum(1, keepdim=True).clamp(min=1)
    ctx = torch.cat([ctx, (near.any(1, keepdim=True)).float()], 1)
    return V[:, CENTRE], torch.log(age[:, CENTRE].float().clamp(min=1)), seen, ctx


class Recall:
    """Arm R. Remembered: stores are revisits (the last sighting's vector and code, the log gap, what was then seen);
    a query recalls only stores whose last sighting had its code (card 042: codes first), weighted by vector similarity
    and gap. Never seen: stores are first sightings, weighted by their surroundings (the mean vector of remembered places
    within 3 tiles, and whether there are any)."""

    def __init__(self, d, ev, idx, nc):
        self.nc = nc
        v, lg, seen, ctx = [], [], [], []
        for s in range(0, len(idx), 20000):
            i = idx[s:s + 20000]
            a, b, c, e = recall_keys(d, ev["win_t"][i], ev["win_i"][i], ev["t"][i])
            v.append(a), lg.append(b), seen.append(c), ctx.append(e)
        v, lg, seen, ctx = torch.cat(v), torch.cat(lg), torch.cat(seen), torch.cat(ctx)
        y = torch.as_tensor(ev["code"][idx], device=dev)
        self.rv, self.rg, self.ry = v[seen], lg[seen], y[seen]
        self.rc = code_of(self.rv)
        self.fc, self.fy = ctx[~seen], y[~seen]
        self.s, self.h, self.sc, self.a = 0.3, 0.5, 0.3, 1.0

    def remembered(self, v, lg):
        qc, out = code_of(v), torch.full((len(v), self.nc), 1.0 / self.nc, device=dev)
        for c in torch.unique(qc).tolist():
            qi, si = torch.nonzero(qc == c)[:, 0], torch.nonzero(self.rc == c)[:, 0]
            if len(si) == 0:
                continue
            for s in range(0, len(qi), 256):
                q = qi[s:s + 256]
                w = torch.exp(-torch.cdist(v[q], self.rv[si]) ** 2 / self.s ** 2
                              - (lg[q][:, None] - self.rg[si][None]) ** 2 / (2 * self.h ** 2))
                out[q] = self.mix(w, self.ry[si])
        return out

    def never_seen(self, ctx):
        out = torch.zeros(len(ctx), self.nc, device=dev)
        for s in range(0, len(ctx), 256):
            w = torch.exp(-torch.cdist(ctx[s:s + 256], self.fc) ** 2 / self.sc ** 2)
            out[s:s + 256] = self.mix(w, self.fy)
        return out

    def probs(self, v, lg, seen, ctx, form="R"):
        """form: R; static (version 27: the last sighting holds); view only (every place gets the never-seen rule)."""
        out = torch.zeros(len(v), self.nc, device=dev)
        sv = seen if form != "view only" else torch.zeros_like(seen)
        if sv.any():
            if form == "static":
                out[sv] = fn.one_hot(code_of(v[sv]), self.nc).float() * (1 - CLIP) + CLIP / self.nc
            else:
                out[sv] = self.remembered(v[sv], lg[sv])
        if (~sv).any():
            out[~sv] = self.never_seen(ctx[~sv])
        return out

    def mix(self, w, y):
        cnt = torch.zeros(len(w), self.nc, device=dev).index_add_(1, y, w)
        return (cnt + self.a / self.nc) / (w.sum(1, keepdim=True) + self.a)

    def fit(self, d, ev, idx):
        """Bandwidths by held-out log-likelihood on the fitting episodes' events (every place coming into view)."""
        v, lg, seen, ctx = recall_keys(d, ev["win_t"][idx], ev["win_i"][idx], ev["t"][idx])
        y = torch.as_tensor(ev["code"][idx], device=dev)
        ll = lambda P, m: float(torch.log(P.gather(1, y[m][:, None]).clamp(min=CLIP)).mean())
        grid = {}
        for s_ in (0.1, 0.3, 1.0):
            for h in (0.125, 0.25, 0.5, 1.0):
                self.s, self.h = s_, h
                grid[("s", s_, h)] = ll(self.remembered(v[seen], lg[seen]), seen)
        best = max((k for k in grid if k[0] == "s"), key=grid.get)
        self.s, self.h = best[1], best[2]
        for sc in (0.03, 0.1, 0.3, 1.0):
            self.sc = sc
            grid[("sc", sc)] = ll(self.never_seen(ctx[~seen]), ~seen)
        self.sc = max((k for k in grid if k[0] == "sc"), key=grid.get)[1]
        return {"s": self.s, "h": self.h, "sc": self.sc,
                "grid": {" ".join(str(x) for x in k): round(v_, 4) for k, v_ in grid.items()}}


def code_of(v):
    return torch.cdist(v.float(), PROTOS).argmin(1)


def view_ctx(d, pr):
    """The view-only baseline's surroundings: remembered places within 3 tiles that are in view now (step t - 1)."""
    win_t = np.where(pr["win_t"] == (pr["t"][:, None, None] - 1), pr["win_t"], -1)
    return win_t


# ------------------------------------------------------------------------------------------------ arm N
def net_inputs(d, win_t, win_i, t):
    V, present, age = gather(d, win_t, win_i, t)
    feat = torch.log1p(age.float()) / math.log1p(STEPS) * present
    X = torch.cat([V, feat[..., None]], -1)
    P = torch.as_tensor(np.stack([WX, WY], 1), device=dev, dtype=torch.float32).expand(len(V), 81, 2)
    return X, P, torch.zeros(len(V), 81, dtype=torch.long, device=dev), present


def net_logp(net, log_tau, d, win_t, win_i, t):
    X, P, Fr, M = net_inputs(d, win_t, win_i, t)
    out, _ = net(X, P, Fr, M, torch.zeros(len(X), dtype=torch.long, device=dev))
    d2 = torch.cdist(out[:, CENTRE, :32].float(), PROTOS) ** 2
    return torch.log_softmax(-d2 / log_tau.exp(), -1)


def train_net(d, ev, idx, steps, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = TransitionModel(33, 1, arm="Rb").to(dev)
    log_tau = torch.nn.Parameter(torch.tensor(-2.3, device=dev))
    opt = torch.optim.AdamW([*net.parameters(), log_tau], lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    for step in range(steps):
        i = idx[rng.integers(0, len(idx), 256)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = net_logp(net, log_tau, d, ev["win_t"][i], ev["win_i"][i], ev["t"][i])
        loss = -lp.float().gather(1, torch.as_tensor(ev["code"][i], device=dev)[:, None]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"   N step {step} loss {float(loss.detach()):.4f}", flush=True)
    net.eval()
    return net, log_tau


@torch.no_grad()
def net_probs(net, log_tau, d, win_t, win_i, t):
    out = []
    for s in range(0, len(t), 2048):
        out.append(net_logp(net, log_tau, d, win_t[s:s + 2048], win_i[s:s + 2048], t[s:s + 2048]).exp().float())
    return torch.cat(out)


# ------------------------------------------------------------------------------------------------ scoring
def loglik(b, truth):
    b = np.clip(b, CLIP, 1 - CLIP)
    return np.where(truth, np.log(b), np.log(1 - b))


def choose(b, dist):
    """The cave to search per probe and goal: the highest belief, ties to the nearest."""
    key = np.round(b, 6) * 1000 - dist[..., None] * 1e-3
    return key.argmax(1)


def decidable(ub):
    top = ub.max(1)
    return (top >= 0.05) & ((ub >= top[:, None] - 1e-9).sum(1) < ub.shape[1])


def boot_p(x, ep, n=4000, seed=0):
    """Paired bootstrap over episodes: mean, 95% interval and one-sided p that the mean is not above 0."""
    eps, inv = np.unique(ep, return_inverse=True)
    sums, cnt = np.bincount(inv, x), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(eps), (n, len(eps)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(x.mean()), [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))], float((m <= 0).mean())


PROTOS = None


def beliefs(d, pr, R, net, log_tau, goal_codes):
    """b (P, 6, 2): gem and ripe bush, per form."""
    P = len(pr["t"])
    wt, wi = pr["win_t"].reshape(-1, 81), pr["win_i"].reshape(-1, 81)
    tq = np.repeat(pr["t"], 6)
    v, lg, seen, ctx = recall_keys(d, wt, wi, tq)
    vt = view_ctx(d, pr).reshape(-1, 81)
    _, _, _, ctx_view = recall_keys(d, vt, wi, tq)
    t0 = time.time()
    forms = {"R": R.probs(v, lg, seen, ctx)}
    torch.cuda.synchronize()
    cost = {"R": (time.time() - t0) / len(tq)}
    forms["static"] = R.probs(v, lg, seen, ctx, "static")
    forms["view only"] = R.probs(v, lg, seen, ctx_view, "view only")
    if net is not None:
        t0 = time.time()
        forms["N"] = net_probs(net, log_tau, d, wt, wi, tq)
        torch.cuda.synchronize()
        cost["N"] = (time.time() - t0) / len(tq)
    src = seen.cpu().numpy().reshape(P, 6)
    return {k: f[:, goal_codes].cpu().numpy().reshape(P, 6, 2) for k, f in forms.items()}, src, cost


def score(B, pr, src):
    ub, truth = pr["ub"], pr["truth"]
    dec = [decidable(ub[..., g]) for g in (0, 1)]
    unique = [((ub[..., g] >= ub[..., g].max(1, keepdims=True) - 1e-9).sum(1) == 1) for g in (0, 1)]
    rep = {"probes": int(len(ub)),
           "probes where not every cave ties (gem, berry)": [round(float(x.mean()), 4) for x in dec],
           "probes with one best cave (gem, berry)": [round(float(x.mean()), 4) for x in unique]}
    lls = {}
    for name, b in list(B.items()) + [("upper bound", ub)]:
        L = loglik(b, truth)                                   # (P, 6, 2)
        lls[name] = L.reshape(len(ub), -1).mean(1)
        ch = choose(b, pr["dist"])
        right = [(ub[np.arange(len(ub)), ch[:, g], g] >= ub[..., g].max(1) - 1e-9)[dec[g]] for g in (0, 1)]
        rep[name] = {"log-likelihood per place and condition": round(float(L.mean()), 4),
                     "by goal (gem, berry)": [round(float(L[..., g].mean()), 4) for g in (0, 1)],
                     "by source (remembered, never seen)": [round(float(L[src].mean()), 4) if src.any() else None,
                                                            round(float(L[~src].mean()), 4) if (~src).any() else None],
                     "choice right (gem, berry), probes where not every cave ties": [round(float(r.mean()), 4)
                                                                                     for r in right]}
    return rep, lls


def by_kind_and_gap(B, pr, src, mk_last, gap):
    """Log-likelihood of the ripe-bush condition at remembered ends, by what was last seen there and the gap."""
    out = {}
    for name in list(B) + ["upper bound"]:
        b = pr["ub"] if name == "upper bound" else B[name]
        L = loglik(b, pr["truth"])
        rows = {}
        for kname, k in (("gem", W6.GEM), ("dark floor", W6.DARK), ("ripe bush", W6.RIPE), ("bare bush", W6.BARE)):
            g = 0 if k in (W6.GEM, W6.DARK) else 1
            for lo, hi in ((1, 29), (30, 60), (61, 300)):
                m = src & (mk_last == k) & (gap >= lo) & (gap <= hi)
                if m.sum():
                    rows[f"{kname}, gap {lo}-{hi}"] = [round(float(L[..., g][m].mean()), 4), int(m.sum())]
        out[name] = rows
    return out


# ------------------------------------------------------------------------------------------------ paired histories
def script(layout, visits, final, wseed, T=None):
    """Scripted play on a fixed layout: each visit (cave, "look" | "use") walks to the cave's first tile (look) or into
    its end (use); then to the final place; then noops (to step T, and on to the episode's end)."""
    w = W6.World6(np.random.default_rng(wseed), layout)
    acts = []
    here = lambda: (int(w.agent[0]), int(w.agent[1]))

    def walk(target):
        for _ in range(200):
            if here() == target:
                return True
            a = w.path_step(target)
            if a == 0:
                return False
            acts.append(a)
            w.step(a)
        return False

    for j, mode in visits:
        ex, ey, _, side = w.caves[j]
        ok = walk((ex, ey - side * (W6.DEPTH6 - 1)) if mode == "look" else (ex, ey - side))
        if not ok:
            return None
        if mode == "use":
            a = next(a for a, dd in W6.MOVES.items() if dd == (0, side))
            acts.append(a)
            w.step(a)
    if not walk(final):
        return None
    return acts


def record(layout, acts, wseed):
    w = W6.World6(np.random.default_rng(wseed), layout)
    rec = {k: [] for k in ("obs", "act", "sem", "pos", "grid", "event")}
    for s in range(STEPS):
        a = acts[s] if s < len(acts) else 0
        rec["obs"].append(w.frame())
        rec["sem"].append(w.view())
        rec["pos"].append(w.agent.copy())
        rec["grid"].append(w.g[W6.PAD6:W6.PAD6 + GH, W6.PAD6:W6.PAD6 + GW].copy())
        rec["act"].append(a)
        ev = w.step(a)
        rec["event"].append((0 if ev[0] == "take" else 1, *ev[1]) if ev else (-1, -1, -1))
    return {k: np.array(v) for k, v in rec.items()}, np.array(w.caves)


def make_pairs(n_per_goal, seed=1093):
    """Pairs of histories on one layout with the same final place and step and different best caves. Gem pairs: dark
    caves D1, D2 hold gems and D3 none; history A takes D1's gem and looks into D3, B takes D2's and looks into D3.
    Berry pairs: sand caves S1, S2 hold ripe bushes; A looks at S2 and eats S1 last, B the reverse."""
    rng = np.random.default_rng(seed)
    eps, meta = [], []
    while len(meta) < 2 * n_per_goal:
        goal = 0 if len(meta) < n_per_goal else 1
        cols = {side: W6.World6.columns(type("R", (), {"rng": rng})()) for side in (-1, 1)}
        caves = [(side, c) for side in (-1, 1) for c in cols[side]]
        order = rng.permutation(6)
        kind, end = {}, {}
        for r, j in enumerate(order):
            sc = caves[j]
            dark = (r < 3) if goal == 0 else (r >= 2)
            kind[sc] = 0 if dark else 1
            if goal == 0:
                end[sc] = W6.GEM if r < 2 else (W6.DARK if dark else W6.RIPE)
            else:
                end[sc] = W6.RIPE if r < 2 else (W6.GEM if rng.random() < 0.5 else W6.DARK)
        layout = {"cols": cols, "kind": kind, "end": end,
                  "start": np.array([int(rng.integers(1, GW - 1)), int(rng.choice(W6.CORRIDOR6))])}
        idx = {sc: i for i, sc in enumerate(caves)}                # World6 lists caves side by side, columns sorted
        j1, j2 = idx[caves[order[0]]], idx[caves[order[1]]]
        if goal == 0:
            j3 = idx[caves[order[2]]]
            hA = [(j1, "use"), (j3, "look")]
            hB = [(j2, "use"), (j3, "look")]
            if rng.random() < 0.5:
                hA, hB = hA[::-1], hB[::-1]
        else:
            hA, hB = [(j2, "look"), (j1, "use")], [(j1, "look"), (j2, "use")]
        final = (int(rng.integers(1, GW - 1)), W6.CORRIDOR6[1])
        wseed = int(rng.integers(1 << 30))
        aA, aB = script(layout, hA, final, wseed), script(layout, hB, final, wseed)
        if aA is None or aB is None:
            continue
        T = max(len(aA), len(aB)) + 1
        if T >= STEPS:
            continue
        rA, cA = record(layout, aA, wseed)
        rB, cB = record(layout, aB, wseed)
        best = []
        for r, c in ((rA, cA), (rB, cB)):
            best.append(pair_best(r, c, T, goal))
        if None in best or best[0] == best[1]:
            continue
        for r, c, bst in ((rA, cA, best[0]), (rB, cB, best[1])):
            r["caves"] = c
            eps.append(r)
        meta.append((goal, T, best[0], best[1]))
    return eps, meta


def pair_best(r, caves, T, goal):
    """The single best cave at step T under the upper bound (None if it is not single)."""
    d = {"pos": r["pos"], "grid": r["grid"], "sem": r["sem"].reshape(STEPS, 7, 9), "caves": caves[None],
         "event": r["event"], "code": np.zeros((STEPS, 81), np.int64)}
    _, pr = replay(d, [0], probe_at={0: T})
    ub = pr["ub"][0][:, goal]
    top = ub.max()
    return int(ub.argmax()) if (ub >= top - 1e-9).sum() == 1 and top > 0.05 else None


def pairs_data(n_per_goal):
    eps, meta = make_pairs(n_per_goal)
    enc = EN.Enc().to(dev)
    enc.load_state_dict(torch.load(OUT / "world6_encoder.pt"))
    enc.eval()
    protos = np.load(OUT / "world6_codes.npz")["protos"]
    obs = np.concatenate([e["obs"] for e in eps])
    E = EN.embed(enc, EN.tiles(obs))
    d = {"V": torch.as_tensor(E.astype(np.float16)).to(dev),
         "code": TK.nearest(E.reshape(-1, 32), protos).reshape(len(E), 81).astype(np.int64),
         "pos": np.concatenate([e["pos"] for e in eps]), "grid": np.concatenate([e["grid"] for e in eps]),
         "sem": np.concatenate([e["sem"] for e in eps]), "event": np.concatenate([e["event"] for e in eps]),
         "caves": np.stack([e["caves"] for e in eps]), "act": np.concatenate([e["act"] for e in eps])}
    d["ep"] = np.repeat(np.arange(len(eps)), STEPS)
    return d, meta


def check():
    d = data("train")
    ev, _ = replay(d, range(500))
    K = codes_of(d)
    gap = ev["t"] - ev["win_t"][:, CENTRE]
    seen = ev["win_t"][:, CENTRE] >= 0
    last_kind = d["sem"].reshape(-1, 63)[ev["win_t"][:, CENTRE].clip(min=0), ev["win_i"][:, CENTRE].astype(int)]
    out = {"events (places coming into view)": int(len(ev["t"])), "of them cave ends": int(ev["end"].sum()),
           "kind codes": {W6.NAMES6[k]: c for k, c in K.items()}}
    rows = {}
    for name, k in (("gem", W6.GEM), ("dark floor", W6.DARK), ("ripe bush", W6.RIPE), ("bare bush", W6.BARE)):
        m = ev["end"] & seen & (last_kind == k)
        rows[f"last seen as {name}"] = {f"gap {lo}-{hi}": int((m & (gap >= lo) & (gap <= hi)).sum())
                                        for lo, hi in ((1, 9), (10, 29), (30, 60), (61, 120), (121, 300))}
    out["cave ends seen again, by what was last seen and the gap"] = rows
    out["bare bushes seen ripe on their next sighting"] = int((ev["end"] & seen & (last_kind == W6.BARE)
                                                               & (ev["kind"] == W6.RIPE)).sum())
    fs = ev["end"] & ~seen
    out["first sightings of cave ends (gem, dark floor, ripe, bare)"] = [int((fs & (ev["kind"] == k)).sum())
                                                                         for k in (W6.GEM, W6.DARK, W6.RIPE, W6.BARE)]
    (OUT / "data_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def run():
    global PROTOS
    t_start = time.time()
    clock = lambda: round(time.time() - t_start, 1)
    tr = data("train")
    PROTOS = torch.as_tensor(np.load(OUT / "world6_codes.npz")["protos"], device=dev, dtype=torch.float32)
    nc = len(PROTOS)
    K = codes_of(tr)
    goal = [K[W6.GEM], K[W6.RIPE]]
    ev, _ = replay(tr, range(500))
    ep_of = ev["t"] // STEPS
    i_tr = np.flatnonzero(ep_of < 400)
    rng = np.random.default_rng(0)
    i_fit = rng.choice(np.flatnonzero((ep_of >= 400) & (ep_of < 450)), 40000, replace=False)
    i_sel = np.flatnonzero(ep_of >= 450)
    print("replay", clock(), "events", len(ev["t"]), flush=True)
    R = Recall(tr, ev, i_tr, nc)
    fit = R.fit(tr, ev, i_fit)
    print("recall fitted", clock(), fit["s"], fit["h"], fit["sc"], flush=True)
    net, log_tau = train_net(tr, ev, i_tr, int(arg("--steps", "8000")), 1091)
    print("N trained", clock(), flush=True)

    # the arm: held-out log-likelihood on episodes 450-499, every place coming into view (and cave ends, reported)
    sel = {}
    with torch.no_grad():
        for name in ("R", "N"):
            P = []
            for s in range(0, len(i_sel), 20000):
                i = i_sel[s:s + 20000]
                if name == "R":
                    P.append(R.probs(*recall_keys(tr, ev["win_t"][i], ev["win_i"][i], ev["t"][i])))
                else:
                    P.append(net_probs(net, log_tau, tr, ev["win_t"][i], ev["win_i"][i], ev["t"][i]))
            P = torch.cat(P)
            L = torch.log(P.gather(1, torch.as_tensor(ev["code"][i_sel], device=dev)[:, None]).clamp(min=CLIP))[:, 0]
            L = L.cpu().numpy()
            sel[name] = {"every place": round(float(L.mean()), 4), "cave ends": round(float(L[ev["end"][i_sel]].mean()), 4)}
    arm = max(("R", "N"), key=lambda k: sel[k]["every place"])
    print("selection", sel, "arm", arm, clock(), flush=True)
    del tr["V"], ev
    torch.cuda.empty_cache()

    te = data("test")
    _, pr = replay(te, range(200), probe_every=10)
    B, src, cost = beliefs(te, pr, R, net, log_tau, goal)
    rep, lls = score(B, pr, src)
    ep = pr["ep"]
    crit1 = {}
    for base in ("view only", "static"):
        crit1[f"{arm} minus {base}"] = boot_p(lls[arm] - lls[base], ep)
    crit1[f"upper bound minus {arm}"] = boot_p(lls["upper bound"] - lls[arm], ep)
    other = "N" if arm == "R" else "R"
    crit1[f"{arm} minus {other}"] = boot_p(lls[arm] - lls[other], ep)
    wt_c = pr["win_t"][:, :, CENTRE]
    mk_last = np.where(wt_c >= 0, te["sem"].reshape(-1, 63)[wt_c.clip(min=0), pr["win_i"][:, :, CENTRE].astype(int)], -1)
    gap = pr["t"][:, None] - wt_c
    kinds = by_kind_and_gap(B, pr, src, mk_last, gap)
    print("test probes scored", clock(), flush=True)
    np.savez_compressed(OUT / "probes.npz", **{f"b_{k}": v for k, v in B.items()}, ub=pr["ub"], truth=pr["truth"],
                        ep=ep, t=pr["t"], src=src, dist=pr["dist"])
    del te["V"]
    torch.cuda.empty_cache()

    # criterion 3: paired histories
    pd, meta = pairs_data(int(arg("--pairs", "100")))
    probe_at = {2 * i + m: meta[i][1] for i in range(len(meta)) for m in (0, 1)}
    _, pp = replay(pd, range(len(pd["caves"])), probe_at=probe_at)
    Bp, srcp, _ = beliefs(pd, pp, R, net, log_tau, goal)
    paired = {}
    for name in list(Bp) + ["upper bound"]:
        b = pp["ub"] if name == "upper bound" else Bp[name]
        ch = choose(b, pp["dist"])
        follow = []
        for i, (g, T, bA, bB) in enumerate(meta):
            follow.append(ch[2 * i, g] == bA and ch[2 * i + 1, g] == bB)
        follow = np.array(follow)
        gs = np.array([m[0] for m in meta])
        paired[name] = {"pairs where the choice follows the history (gem, berry)":
                        [round(float(follow[gs == g].mean()), 4) for g in (0, 1)],
                        "pairs": [int((gs == g).sum()) for g in (0, 1)]}
    print("pairs scored", clock(), flush=True)

    # P19: criterion 1 against the number of training episodes (recall's stores; bandwidths kept)
    curve = {}
    tr = data("train")
    ev, _ = replay(tr, range(300))
    ep_of = ev["t"] // STEPS
    te = data("test")
    for n in (10, 30, 100, 300):
        Rn = Recall(tr, ev, np.flatnonzero(ep_of < n), nc)
        Rn.s, Rn.h, Rn.sc = R.s, R.h, R.sc
        Bn, srcn, _ = beliefs(te, pr, Rn, None, None, goal)
        curve[n] = round(float(loglik(Bn["R"], pr["truth"]).mean()), 4)
    print("curve", curve, clock(), flush=True)

    out = {"arm chosen": arm, "selection (held-out log-likelihood, episodes 450-499)": sel, "recall fit": fit,
           "cost per query, seconds (R, N)": {k: float(f"{v:.3g}") for k, v in cost.items()},
           "test probes": rep, "criterion 1 (paired bootstrap over episodes: mean, 95% interval, p not above 0)": crit1,
           "remembered ends by last sighting and gap (log-likelihood, count)": kinds,
           "criterion 3 (paired histories)": paired,
           "recall's log-likelihood by training episodes (P19)": curve, "seconds": clock()}
    (OUT / "run.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def report():
    """runs/109/{data_check,run}.json and the never-seen breakdown from probes.npz → the card's results.json."""
    out = {"data check (section 4)": json.loads((OUT / "data_check.json").read_text())}
    out.update(json.loads((OUT / "run.json").read_text()))
    z = np.load(OUT / "probes.npz")
    ub, src, dist = z["ub"], z["src"], z["dist"]
    ns = ~src
    groups = {"known dark cave, gem": (ns & np.isclose(ub[..., 0], 0.5), 0),
              "known sand cave, ripe bush": (ns & np.isclose(ub[..., 0], 0) & ~np.isclose(ub[..., 1], 0), 1),
              "cave never seen, gem": (ns & np.isclose(ub[..., 0], 0.25), 0),
              "cave never seen, ripe bush": (ns & np.isclose(ub[..., 0], 0.25), 1)}
    rows = {}
    for name, (m0, g) in groups.items():
        for lo, hi in ((0, 6), (7, 12), (13, 40)):
            m = m0 & (dist >= lo) & (dist <= hi)
            if not m.any():
                continue
            rows[f"{name}, agent {lo}-{hi} steps away"] = {
                "count": int(m.sum()), **{f: round(float(z[f"b_{f}"][..., g][m].mean()), 3) for f in ("N", "R")},
                "upper bound": round(float(ub[..., g][m].mean()), 3)}
    out["never-seen cave ends: mean belief by the agent's distance (test probes)"] = rows
    L = lambda b: np.where(z["truth"], np.log(np.clip(b, CLIP, 1)), np.log(np.clip(1 - b, CLIP, 1)))
    gap = L(ub) - L(z["b_N"])
    share = {name: round(float(gap[m0][..., g].sum() / gap.size), 4) for name, (m0, g) in groups.items()}
    share["remembered places"] = round(float(gap[src].sum() / gap.size), 4)
    out["upper bound minus N, nats per place and condition, by where it arises"] = share
    (ROOT / "experiments" / "109-conditions-from-memory" / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(share, indent=1))


if __name__ == "__main__":
    {"check": check, "run": run, "report": report}[CMD]()
