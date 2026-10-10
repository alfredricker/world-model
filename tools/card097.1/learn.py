"""Card 097.1: walking learned from the agent's own moves (System 1), judged against card 067's field (offline).

Input: the believed 13 x 13 view in the agent's frame (token vectors, card 070's encoder) and a target placement in
that frame (its place marked in four heading planes). Output: Q(view, target, move), the steps still to go after
the move, for left, right and forward. Learned by fitted Q-iteration on the recorded transitions (walkrec.py; the
agent's episodes on training seeds, 30% of actions random): Q(s, g, a) = 1 + min_a' Q'(s', g', a'), 0 to go once
the agent stands at g; g' is g seen from the placement after the move (a blocked move leaves it where it was).
Targets: later placements of the same episode within view (hindsight) and random placements in view.
Judged on the walking decisions of the test seeds' episodes (version 24, unperturbed): System 1's move (least Q over
the need's target placements in view; forward, left, right among equals) against the moves the field rates closest.

  bin/prun python tools/card097.1/learn.py      → runs/097.1/learn.json, runs/097.1/walk_net.pt
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "097.1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
R, W13 = 6, 13
NV = W13 * W13
CENTRE, FRONT = R * W13 + R, (R - 1) * W13 + R
LEFT, RIGHT, FWD = 0, 1, 2
ORDER = (FWD, LEFT, RIGHT)                             # card 023's order among equals
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))              # card 028: (column, row) steps in turning order
CAP = 40.0
log = lambda m: print(m, flush=True)


def rel(c, t):
    """Placement t = (px, py, pd) seen from placement c: (x right, y ahead, heading turned right)."""
    ahead, right = DIRS[c[2]], DIRS[(c[2] + 1) % 4]
    d = (t[0] - c[0], t[1] - c[1])
    return d[0] * right[0] + d[1] * right[1], d[0] * ahead[0] + d[1] * ahead[1], (t[2] - c[2]) % 4


def cell(x, y):
    """The view place of a where (x right, y ahead), or -1 outside the 13 x 13 view."""
    if abs(x) > R or abs(y) > R:
        return -1
    return (R - y) * W13 + (x + R)


def vectors(tier):
    arr = np.load(ROOT / "runs" / "095.1" / f"states_tier{tier}.npz")["arr"].astype(np.float32)
    return np.concatenate([arr, np.zeros((4096 - len(arr), arr.shape[1]), np.float32)])   # later handles: zero


def episodes(path):
    out = []
    for r in json.loads(Path(path).read_text())["per_episode"]:
        out.append(r["walkrec"])
    return out


K = int(CAP) + 1                                       # revision: steps to go as a distribution over 0..CAP


class Net(nn.Module):
    def __init__(self, d=32, h=96, bins=1):
        super().__init__()
        self.bins = bins
        c = d + 4
        self.c = nn.Sequential(nn.Conv2d(c, h, 3, padding=1), nn.ReLU(), nn.Conv2d(h, h, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(h, h, 3, padding=2, dilation=2), nn.ReLU(),
                               nn.Conv2d(h, h, 3, padding=4, dilation=4), nn.ReLU(),
                               nn.Conv2d(h, h, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(h * NV, 256), nn.ReLU(), nn.Linear(256, 3 * bins))

    def logits(self, V, G):
        x = torch.cat([V.transpose(1, 2), G], 1).view(len(V), -1, W13, W13)
        return self.head(self.c(x)).view(len(V), 3, self.bins)

    def forward(self, V, G):
        """V: (n, 169, 32) token vectors; G: (n, 4, 169) the target's heading planes. Steps to go per move."""
        z = self.logits(V, G)
        if self.bins == 1:
            return nn.functional.softplus(z[..., 0])
        return (z.softmax(-1) * torch.arange(self.bins, device=z.device, dtype=z.dtype)).sum(-1)


class Data:
    """Transitions (view before and after, move, placements before and after) of every recorded episode, and per
    episode its placements by step, for hindsight targets."""

    def __init__(self, files_by_tier):
        S, S2, A, P0, P1, E, K, T = [], [], [], [], [], [], [], []
        self.eps = []
        for tier, files in files_by_tier.items():
            for f in files:
                for steps in episodes(f):
                    for t in range(len(steps) - 1):
                        s, s2 = steps[t], steps[t + 1]
                        if s["a"] not in (LEFT, RIGHT, FWD) or s["pose"] is None or s2["pose"] is None:
                            continue
                        S.append(np.asarray(s["view"], np.int64))
                        S2.append(np.asarray(s2["view"], np.int64))
                        A.append(s["a"])
                        P0.append(s["pose"])
                        P1.append(s2["pose"])
                        E.append(len(self.eps))
                        K.append(t)
                        T.append(tier)
                    self.eps.append([s["pose"] for s in steps])
        self.view, self.view2 = torch.as_tensor(np.stack(S)), torch.as_tensor(np.stack(S2))
        self.A = torch.as_tensor(np.array(A))
        self.P0, self.P1 = np.array(P0), np.array(P1)
        self.E, self.K, self.tier = np.array(E), np.array(K), np.array(T)
        self.arr = {t: torch.as_tensor(vectors(t), device=dev) for t in files_by_tier}
        self.n = len(A)

    def vecs(self, i, after=False):
        src = self.view2 if after else self.view
        out = torch.zeros(len(i), NV, 32, device=dev)
        for t, arr in self.arr.items():
            m = self.tier[i] == t
            if m.any():
                out[torch.as_tensor(m, device=dev)] = arr[src[i[m]].to(dev)]
        return out


def goal_planes(gs):
    G = torch.zeros(len(gs), 4, NV, device=dev)
    for k, (x, y, h) in enumerate(gs):
        q = cell(x, y)
        if q >= 0:
            G[k, h, q] = 1.0
    return G


def sample_goals(d, i, rng):
    """Per transition: a placement its episode reached within the next 40 steps, within view (hindsight), or, half
    the time, a random placement in view."""
    gs = []
    for k in i.tolist():
        c = tuple(d.P0[k])
        if rng.random() < 0.5:
            poses = d.eps[d.E[k]]
            later = [rel(c, tuple(p)) for p in poses[d.K[k] + 1:d.K[k] + 41] if p is not None]
            later = [g for g in later if cell(g[0], g[1]) >= 0]
            if later:
                gs.append(later[rng.integers(len(later))])
                continue
        gs.append((int(rng.integers(-R, R + 1)), int(rng.integers(-R, R + 1)), int(rng.integers(4))))
    return gs


def train(d, iters=36, steps=500, bs=256, lr=3e-4, seed=97, bins=1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net, tgt = Net(bins=bins).to(dev), Net(bins=bins).to(dev)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.monotonic()
    for it in range(iters):
        for step in range(steps):
            i = rng.integers(d.n, size=bs)
            gs = sample_goals(d, i, rng)
            g2 = [rel(tuple(d.P1[k]), _abs(tuple(d.P0[k]), g)) for k, g in zip(i.tolist(), gs)]
            done = torch.as_tensor([g == (0, 0, 0) for g in g2], device=dev)
            V = d.vecs(i)
            with torch.no_grad():
                out = torch.as_tensor([cell(g[0], g[1]) < 0 for g in g2], device=dev)
                q2 = tgt(d.vecs(i, after=True), goal_planes(g2)).min(1).values
                y = 1.0 + torch.where(done, torch.zeros_like(q2), torch.where(out, torch.full_like(q2, CAP), q2))
                y = y.clamp(max=CAP)
            if bins == 1:
                q = net(V, goal_planes(gs)).gather(1, d.A[i].to(dev)[:, None])[:, 0]
                loss = nn.functional.smooth_l1_loss(q, y)
            else:                                      # the target: the next placement's best move's distribution, one step on
                with torch.no_grad():
                    G2 = goal_planes(g2)
                    V2 = d.vecs(i, after=True)
                    p2 = tgt.logits(V2, G2).softmax(-1)
                    e2 = (p2 * torch.arange(bins, device=dev)).sum(-1)
                    p2 = p2[torch.arange(bs, device=dev), e2.argmin(1)]
                    yd = torch.zeros(bs, bins, device=dev)
                    yd[:, 1:] = p2[:, :-1]
                    yd[:, -1] += p2[:, -1]
                    yd[done] = nn.functional.one_hot(torch.tensor(1, device=dev), bins).float()
                    yd[out & ~done] = nn.functional.one_hot(torch.tensor(bins - 1, device=dev), bins).float()
                z = net.logits(V, goal_planes(gs))[torch.arange(bs, device=dev), d.A[i].to(dev)]
                loss = -(yd * z.log_softmax(-1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        tgt.load_state_dict(net.state_dict())
        log(f"iteration {it + 1}: loss {float(loss.detach()):.4f} ({round(time.monotonic() - t0)} s)")
    return net


def _abs(c, g):
    """A target given relative to placement c, back in the first view's frame."""
    ahead, right = DIRS[c[2]], DIRS[(c[2] + 1) % 4]
    x, y, h = g
    return (c[0] + x * right[0] + y * ahead[0], c[1] + x * right[1] + y * ahead[1], (c[2] + h) % 4)


def open_space():
    """The trivial baseline: steps to each target in an empty room (offset and heading only), by search."""
    D, front = {(0, 0, 0): 0}, [(0, 0, 0)]
    while front:
        nxt = []
        for c in front:
            dx, dy = DIRS[c[2]]
            for p in ((c[0], c[1], (c[2] - 1) % 4), (c[0], c[1], (c[2] + 1) % 4), (c[0] + dx, c[1] + dy, c[2])):
                if p not in D and max(abs(p[0]), abs(p[1])) <= R + 2:
                    D[p] = D[c] + 1
                    nxt.append(p)
        front = nxt
    after = {LEFT: (0, 0, 3), RIGHT: (0, 0, 1), FWD: (0, -1, 0)}
    o = (0, 0, 0)

    def q(tg):
        return np.array([min(1 + D.get(_abs(o, rel(after[m], _abs(o, g))), 99) for g in tg)
                         for m in (LEFT, RIGHT, FWD)], np.float64)
    return q


def pick(q):
    best = q.min()
    return next(m for m in ORDER if q[m] <= best + 0.5)


def pick_int(q):
    """Steps to go are whole numbers: round, then forward, left, right among equals."""
    r = np.rint(q)
    return next(m for m in ORDER if r[m] == r.min())


@torch.no_grad()
def judge(net, tier, path, names):
    """Agreement with the field at the walking decisions of the recorded test episodes, by class."""
    arr = torch.as_tensor(vectors(tier), device=dev)
    nm = lambda h: names.get(int(h), "?")
    res = {"decisions": 0, "in view": 0, "agree": 0, "open space agrees": 0, "classes": {}}
    for ep in episodes(path):
        for s in ep:
            w = s["walk"]
            if w is None:
                continue
            res["decisions"] += 1
            tg = [tuple(g) for g in w["targets"] if cell(g[0], g[1]) >= 0]
            if not tg:
                continue
            res["in view"] += 1
            V = arr[torch.as_tensor(np.asarray(s["view"], np.int64), device=dev)][None].expand(len(tg), NV, 32)
            if isinstance(net, list):                  # revision: an ensemble; it answers only where its copies agree
                qs = [n(V, goal_planes(tg)).min(0).values.cpu().numpy() for n in net]
                picks = {pick_int(q) for q in qs}
                if len(picks) > 1:
                    res["copies disagree"] = res.get("copies disagree", 0) + 1
                    continue
                a = picks.pop()
            else:
                q = net(V, goal_planes(tg)).min(0).values.cpu().numpy()
                a = pick(q)
            res["answered"] = res.get("answered", 0) + 1
            ok = a in w["best"]
            ok0 = pick(OPEN(tg)) in w["best"]
            res["agree"] += ok
            res["open space agrees"] += ok0
            v = s["view"]
            nb = [v[CENTRE - 1], v[CENTRE + 1], v[FRONT], v[CENTRE + W13]]
            cls = []
            if nm(v[FRONT]).startswith("door") or nm(v[CENTRE]).startswith("door"):
                cls.append("doorway")
            if any(not (nm(h).startswith(("floor", "wall", "door", "unseen", "goal", "?"))) for h in nb):
                cls.append("beside a blocker")
            if all(g[1] < 0 for g in tg):
                cls.append("target behind")
            if not cls:
                cls.append("open")
            for c in cls:
                k = res["classes"].setdefault(c, [0, 0, 0])
                k[0] += ok
                k[1] += ok0
                k[2] += 1
    res["agree share"] = round(res["agree"] / max(res["answered"], 1), 4)
    res["open space share"] = round(res["open space agrees"] / max(res["answered"], 1), 4)
    res["answered share"] = round(res["answered"] / max(res["decisions"], 1), 4)
    res["in view share"] = round(res["in view"] / max(res["decisions"], 1), 4)
    res["classes"] = {c: {"agree": round(a / n, 4), "open space": round(b / n, 4), "n": n}
                      for c, (a, b, n) in res["classes"].items()}
    return res


OPEN = open_space()

if __name__ == "__main__":
    files = {t: [OUT / f"train_tier{t}.json"] for t in (1, 2, 3)}
    d = Data(files)
    log(f"transitions {d.n}, episodes {len(d.eps)}")
    tag = ""
    if "--revise" in sys.argv:                         # revision 1: distributional steps to go, three copies
        tag = "_r1"
        net = []
        for seed in (97, 98, 99):
            n = train(d, iters=48, seed=seed, bins=K)
            torch.save(n.state_dict(), OUT / f"walk_net_r1_s{seed}.pt")
            net.append(n.eval())
    else:
        net = train(d)
        torch.save(net.state_dict(), OUT / "walk_net.pt")
        net.eval()
    rep = {}
    for t in (1, 2, 3):
        names = {int(k): v for k, v in json.loads((ROOT / "runs" / "095.1" / f"names_tier{t}.json").read_text()).items()}
        rep[f"tier{t}"] = judge(net, t, OUT / f"eval_tier{t}.json", names)
        log(f"tier {t}: {json.dumps(rep[f'tier{t}'])}")
    (OUT / f"learn{tag}.json").write_text(json.dumps(rep, indent=1) + "\n")
