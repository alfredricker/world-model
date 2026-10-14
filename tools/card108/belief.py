"""Card 108: belief supplies "which one". Creatures seen now are matched to files (remembered individual tokens) by
each file's prediction of where it is now, at every gap k since it was last seen (1 in view, up to 50 out of view).

Forms (only the prediction differs; everything else is shared):
  model    card 108's transition model (tools/card108/train.py), asked about a 9 x 7 patch of remembered places
           around the file's last place, as last seen: "what is here k steps later?" (k = 1 with the agent's action,
           else the "k steps pass" token). The file's likelihood of a creature with code c at place q is the model's
           probability of code c at q (105.8's softmax of negative squared distances to the codes), less the same
           with the file's creature masked out: its own part, by intervention.
  modelq   diagnostic: model, with a link allowed only to a code the file's code shares a tracked object with
           (Q(c | the file's code) > 0, from the evaluator's tracks): the model's motion without its cross-kind links
  counted  card 102.3's counted motion: P(move | last move, code) propagated k steps over a +-20 window, times
           Q(c | the file's code) (102.3); counted on the evaluator's tracks of workers 0-7 (stronger than 102.3's).
  stays    the file stays where it was last seen (zero spread), times Q.
  oracle   the evaluator's place of the creature the file was last seen on (the gate's upper bound).

Shared, each step (an awake segment: episode, broken by sleep, as 102.3):
  candidates  view places whose code is a creature code, except the agent's own tile (the pose says where the body
              is). A creature code is one the model predicts leaves its place within 50 steps with probability at
              least 0.5, and not one of the agent's own codes (codes seen at the agent's tile in at least 90% of their
              occurrences).
  match       Hungarian assignment maximising the sum over links of log(r P_D L / lambda(c)), a link only where that
              is positive; every other candidate starts a new file. lambda(c): creature tokens of code c that no
              creature token at t - 1 explains (none at the same place or a neighbour), per place and step, counted
              on training play without labels.
  correct     a linked or new file takes the seen place, code and vector; k = 0; existence r = 1.
  unseen      r falls by the code's rate of vanishing (counted the same way, without labels), and on a miss (IPDA,
              P_D = 0.95): q is the file's predicted mass at places in view now that have not yet been looked at
              in this gap, r <- r (1 - P_D q) / (1 - r P_D q). A file is dropped after a gap of 51 (a return 50 steps
              away; the model is then asked at k = 50), at r < 0.001, or beyond the 8 most recently seen of the
              files not seen now.
  patch       for the model: the remembered places around the file's place (the map holds each place's last seen
              vector); places never seen and places last seen holding another creature code are masked.

Scored (labels only here): returns within 50 steps by day (cows, zombies, skeletons), taken by the file last seen
on the same creature; IDF1 over every day creature frame (detections on the evaluator's other objects, arrows and
plants, are distractors: MOT's rule), and 098.6's form with arrows as objects.

  bin/prun python tools/card108/belief.py <form> [--seed S] [--n N]       → runs/108/belief_<form>[_s<S>].json
  bin/prun python tools/card108/belief.py gate [--seed S]                 → runs/108/gate_s<S>.json
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
FORM, SEED, NMAX = sys.argv[1], int(arg("--seed", 1021)), int(arg("--n", 0))
TRACE = [int(x) for x in arg("--trace", "-1,-1,-1").split(",")]   # worker, first and last step to print (diagnosis)
sys.path.insert(0, str(ROOT / "tools" / "card105"))
sys.path.insert(0, str(ROOT / "tools" / "card102.3"))
sys.path.insert(0, str(ROOT / "tools" / "card098.5"))
from model import TransitionModel                        # noqa: E402
import belief as B3                                      # noqa: E402  (102.3: propagate, MOV)
from assign import linear_sum_assignment               # noqa: E402  (098.5's; scipy is not a dependency)

OUT = ROOT / "runs" / "108"
PLAY = ROOT / "runs" / "098.5"
dev = "cuda"
NT, KMAX, NA, K_FILES, PD, WIN, EPS = 81, 50, 66, 8, 0.95, 20, 0.01
AGENT = CENTRE = 31                                      # the view's centre; a patch's centre
COLS, ROWS = np.arange(NT) % 9, np.arange(NT) // 9
PROTOS = torch.as_tensor(np.load(ROOT / "runs" / "098.3" / "codes_joint_blocked.npz")["protos"], device=dev).float()
NC = len(PROTOS)
CREATURE_KINDS, ARROW, PLANT = (1, 2, 3), 4, 5           # the evaluator's cow, zombie, skeleton; arrow; plant
OFF, SIZE = 16, 96                                       # the map's offset and size in world places


# ------------------------------------------------------------------------------------------------ data
def load(w, vectors=True):
    z = np.load(PLAY / f"play_w{w}.npz")
    n = len(z["act"])
    sleep = z["sleep"]
    brk = np.r_[True, (z["ep"][1:] != z["ep"][:-1]) | sleep[1:] | sleep[:-1]]
    seg = np.cumsum(brk) + 1000000 * w
    seg[sleep] = -1
    d = {"w": w, "seg": seg, "act": z["act"].astype(np.int64), "pos": z["pos"].astype(np.int64), "light": z["light"],
         "code": np.load(PLAY / f"codes_w{w}.npy").astype(np.int64), "obj": z["obj"].astype(np.int64), "n": n}
    if vectors:
        d["V"] = torch.as_tensor(np.load(ROOT / "runs" / "105.2" / f"play_vec_w{w}.npy"))
    return d


def view_places(pos):
    """World places (x, y) of the 63 view tiles, for one pose."""
    return pos[0] - 4 + COLS[:63], pos[1] - 3 + ROWS[:63]


def truth(d):
    """Per step: the evaluator's objects in view, (id, kind, x, y)."""
    o = d["obj"]
    at = defaultdict(list)
    for t, i, k, dx, dy, *_ in o:
        at[int(t)].append((int(i), int(k), int(d["pos"][t, 0] + dx), int(d["pos"][t, 1] + dy)))
    return at


# ------------------------------------------------------------------------------------------------ the model
def load_model(seed):
    s = torch.load(OUT / f"model_s{seed}_net.pt")
    net = TransitionModel(32, NA, arm="Rb").to(dev)
    net.load_state_dict(s["net"])
    net.eval()
    return net, s["tau"]


FRAME = torch.tensor([0] * 63 + [1] * 18, device=dev)


@torch.no_grad()
def code_logp(net, tau, V, P, M, a):
    """Log-probabilities of each code at each of the 63 places (B, 63, NC)."""
    nxt, _ = net(V, P, FRAME.expand(len(V), NT), M, a)
    d2 = torch.cdist(nxt[:, :63].reshape(-1, 32), PROTOS) ** 2
    return torch.log_softmax(-d2 / tau, -1).view(len(V), 63, NC)


def own_part(net, tau, V, P, M, a):
    """The file's own part of the prediction, by intervention: p(code at place | the patch) minus the same with the
    file's creature masked out (the patch's centre), floored at 0. What remains is what this creature makes likely,
    not other creatures that may walk in (that is lambda's part)."""
    V0, M0 = V.clone(), M.clone()
    V0[:, CENTRE], M0[:, CENTRE] = 0, False
    lp = code_logp(net, tau, torch.cat([V, V0]), torch.cat([P, P]), torch.cat([M, M0]), torch.cat([a, a])).exp()
    return (lp[:len(V)] - lp[len(V):]).clamp(min=0)


def own_codes(train):
    """The agent's own codes: seen at its tile (the pose) in at least 90% of their occurrences in view."""
    at = np.zeros(NC)
    tot = np.zeros(NC)
    for d in train:
        c = d["code"][d["seg"] >= 0, :63]
        tot += np.bincount(c.ravel(), minlength=NC)
        at += np.bincount(c[:, AGENT], minlength=NC)
    return sorted(int(k) for k in np.flatnonzero((tot > 0) & (at >= 0.9 * tot)))


def creature_codes(net, tau, train, own):
    """Codes whose content the model predicts leaves its place within 50 steps with probability at least 0.5: per
    code, the mean over its view occurrences (every 10th awake step of training play) of 1 - p(same code at the same
    place, k steps later), at k = 50; k = 10 and 20 reported."""
    rows = {}
    for k in (10, 20, 50):
        s, n = np.zeros(NC), np.zeros(NC)
        for d in train:
            ts = np.flatnonzero(d["seg"] >= 0)[::10]
            for b in range(0, len(ts), 256):
                t = ts[b:b + 256]
                V = d["V"][t].to(dev).float()
                P = np.zeros((len(t), NT, 2), np.float32)
                P[:, :63, 0] = d["pos"][t, :1] - 4 + COLS[None, :63]
                P[:, :63, 1] = d["pos"][t, 1:] - 3 + ROWS[None, :63]
                P[:, 63:] = np.stack([COLS[63:], ROWS[63:]], 1)[None]
                M = torch.ones(len(t), NT, dtype=torch.bool, device=dev)
                lp = code_logp(net, tau, V, torch.as_tensor(P, device=dev), M,
                               torch.full((len(t),), 17 + k - 2, device=dev))
                c = torch.as_tensor(d["code"][t, :63], device=dev)
                stay = lp.gather(2, c[..., None])[..., 0].exp().cpu().numpy()
                c = c.cpu().numpy()
                np.add.at(s, c.ravel(), 1 - stay.ravel())
                np.add.at(n, c.ravel(), 1)
        rows[k] = (s / np.maximum(n, 1), n)
    leave, n = rows[50]
    movers = [int(c) for c in range(NC) if n[c] >= 5 and leave[c] >= 0.5 and c not in own]
    info = {str(c): {"occurrences": int(n[c]), "P(leaves) at k = 10, 20, 50": [round(float(rows[k][0][c]), 3)
                                                                             for k in (10, 20, 50)]}
            for c in range(NC) if n[c] > 0}
    return movers, info


# ------------------------------------------------------------------------------------------------ counts
def rates(train, movers):
    """Without labels, per creature code: lambda, the density of creature tokens that no creature token one step
    before explains (none at the same place or a 4-neighbour), per place and step; and the rate of vanishing, the
    share of creature tokens whose place and 4 neighbours are all in view a step later and hold no creature token."""
    mv = np.zeros(NC, bool)
    mv[movers] = True
    new, steps, gone, seen = np.zeros(NC), 0, np.zeros(NC), np.zeros(NC)
    nb = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
    for d in train:
        for t in range(d["n"] - 1):
            if d["seg"][t] < 0 or d["seg"][t + 1] != d["seg"][t]:
                continue
            steps += 1
            c0, c1 = d["code"][t, :63], d["code"][t + 1, :63]
            x0, y0 = view_places(d["pos"][t])
            x1, y1 = view_places(d["pos"][t + 1])
            p0 = {(x, y) for x, y, c, i in zip(x0, y0, c0, range(63)) if mv[c] and i != AGENT}
            p1 = {(x, y) for x, y, c, i in zip(x1, y1, c1, range(63)) if mv[c] and i != AGENT}
            v1 = set(zip(x1.tolist(), y1.tolist()))
            for x, y, c, i in zip(x1, y1, c1, range(63)):
                if mv[c] and i != AGENT and not any((x + a, y + b) in p0 for a, b in nb):
                    new[c] += 1
            for x, y, c, i in zip(x0, y0, c0, range(63)):
                if mv[c] and i != AGENT and all((x + a, y + b) in v1 for a, b in nb):
                    seen[c] += 1
                    gone[c] += not any((x + a, y + b) in p1 for a, b in nb)
    lam = (new + 0.5) / (steps * 63)
    vanish = (gone + 0.5) / (seen + 1)
    return lam, vanish


def counted(train_ids):
    """Card 102.3's counted motion on the evaluator's tracks (codes from perception at each object's place), workers
    0-7: per code, P(move | last move) and Q(b | a); a baseline, so labels are allowed."""
    trans, worn = {}, {}
    for w in train_ids:
        d = load(w, vectors=False)
        o = d["obj"]
        last = {}
        for t, i, k, dx, dy, *_ in o:
            t = int(t)
            if d["seg"][t] < 0:
                continue
            x, y = int(d["pos"][t, 0] + dx), int(d["pos"][t, 1] + dy)
            c = int(d["code"][t, (3 + dy) * 9 + 4 + dx])
            key = (int(d["seg"][t]), int(i))
            st = None
            if key in last and last[key][0] == t - 1:
                m = (x - last[key][1], y - last[key][2])
                if m in B3.MOV:
                    st = B3.MOV.index(m)
                    if last[key][3] is not None:
                        trans.setdefault(c, np.zeros((5, 5)))[last[key][3], st] += 1
            last[key] = (t, x, y, st)
            wn = worn.setdefault(key, {})
            wn[c] = wn.get(c, 0) + 1
    Q = {}
    for wn in worn.values():
        n = sum(wn.values())
        for a, na in wn.items():
            q = Q.setdefault(a, {})
            for b, nb in wn.items():
                q[b] = q.get(b, 0.0) + na * nb / n
    Q = {a: {b: v / sum(q.values()) for b, v in q.items()} for a, q in Q.items()}
    T = {c: (m + 0.5) / (m + 0.5).sum(1, keepdims=True) for c, m in trans.items()}
    pooled = sum(trans.values())
    T[-1] = (pooled + 0.5) / (pooled + 0.5).sum(1, keepdims=True)  # codes never counted: every code's motion pooled
    return T, Q


# ------------------------------------------------------------------------------------------------ belief
class Map:
    """Each world place's last seen token in this segment: (step, view index), or -1."""

    def __init__(self):
        self.t = np.full((SIZE, SIZE), -1, np.int64)
        self.i = np.full((SIZE, SIZE), -1, np.int64)
        self.code = np.full((SIZE, SIZE), -1, np.int64)

    def write(self, t, pos, codes):
        x, y = view_places(pos)
        self.t[y + OFF, x + OFF], self.i[y + OFF, x + OFF], self.code[y + OFF, x + OFF] = t, np.arange(63), codes[:63]


def patch(mp, d, x, y, t_carried, mover):
    """The 9 x 7 remembered places around (x, y) as the model's input: vectors, positions, present."""
    px, py = x - 4 + COLS[:63], y - 3 + ROWS[:63]
    ts, iv, cs = mp.t[py + OFF, px + OFF], mp.i[py + OFF, px + OFF], mp.code[py + OFF, px + OFF]
    present = (ts >= 0) & ~(mover[cs.clip(min=0)] & ((px != x) | (py != y)))
    V = torch.zeros(NT, 32, device=dev)
    k = np.flatnonzero(present)
    V[k] = d["V"][ts[k], iv[k]].to(dev).float()
    V[63:] = d["V"][t_carried, 63:].to(dev).float()
    P = np.zeros((NT, 2), np.float32)
    P[:63, 0], P[:63, 1] = px, py
    P[63:] = np.stack([COLS[63:], ROWS[63:]], 1)
    M = np.ones(NT, bool)
    M[:63] = present
    return V, torch.as_tensor(P, device=dev), torch.as_tensor(M, device=dev)


def run(d, form, net, tau, movers, lam, vanish, T, Q, at):
    """One pass over a worker's test play. Returns return events, detections and timing."""
    mover = np.zeros(NC + 1, bool)
    mover[movers] = True
    mover_t = torch.as_tensor(mover[:NC], device=dev)
    events, dets = [], []
    nf, files, mp, last_in = 0, {}, None, {}
    tb, nb = 0.0, 0
    n = d["n"] if not NMAX else min(NMAX, d["n"])
    for t in range(n):
        seg = d["seg"][t]
        if seg < 0:
            continue
        if t == 0 or d["seg"][t - 1] != seg:
            files, mp, last_in = {}, Map(), {}
        t0 = time.perf_counter()
        codes, pos = d["code"][t], d["pos"][t]
        vx, vy = view_places(pos)
        inview = set(zip(vx.tolist(), vy.tolist()))
        cand = [(int(vx[i]), int(vy[i]), int(codes[i])) for i in range(63) if i != AGENT and mover[codes[i]]]
        tid_at = {(x, y): i for i, k, x, y in at.get(t, [])}
        place_now = {i: (x, y) for i, k, x, y in at.get(t, [])}
        # 1. predict: each file's likelihood of each candidate, and its predicted mass at places in view now
        live = [f for f in files.values() if 1 <= t - f["t"] <= KMAX + 1]   # away 50 steps is a gap of 51
        L = np.zeros((len(live), len(cand)))
        mass = []
        if form in ("model", "modelq") and live:
            Vb = torch.stack([f["snap"][0] for f in live])
            Pb = torch.stack([f["snap"][1] for f in live])
            Mb = torch.stack([f["snap"][2] for f in live])
            k = np.array([min(t - f["t"], KMAX) for f in live])        # a gap of 51 is asked as 50
            a = np.where(k == 1, [d["act"][f["t"]] for f in live], 17 + k - 2)
            own = own_part(net, tau, Vb, Pb, Mb, torch.as_tensor(a, device=dev))
            pmov = (own * mover_t).sum(-1).cpu().numpy()                # the file's creature mass per patch place
            own = own.cpu().numpy()
        for a_, f in enumerate(live):
            k = t - f["t"]
            if form in ("model", "modelq"):
                px, py = f["x"] - 4 + COLS[:63], f["y"] - 3 + ROWS[:63]
                idx = {(int(x), int(y)): j for j, (x, y) in enumerate(zip(px, py))}
                qd = Q.get(f["code"], {f["code"]: 1.0})
                for j, (x, y, c) in enumerate(cand):
                    q = idx.get((x, y))
                    L[a_, j] = own[a_, q, c] if q is not None and (form == "model" or qd.get(c, 0) > 0) else 0.0
                mass.append({p: pmov[a_, q] for p, q in idx.items() if p in inview})
            elif form in ("counted", "stays"):
                Bm = f["B"].sum(0)
                x0, y0 = f["x"] - WIN, f["y"] - WIN
                qd = Q.get(f["code"], {f["code"]: 1.0})
                for j, (x, y, c) in enumerate(cand):
                    if 0 <= x - x0 < 2 * WIN + 1 and 0 <= y - y0 < 2 * WIN + 1:
                        L[a_, j] = Bm[y - y0, x - x0] * qd.get(c, 0.0)
                ys, xs = np.nonzero(Bm > 1e-6)
                mass.append({(int(x0 + xx), int(y0 + yy)): Bm[yy, xx] for yy, xx in zip(ys, xs)
                             if (int(x0 + xx), int(y0 + yy)) in inview})
            else:                                                       # oracle
                p = place_now.get(f["tid"])
                for j, (x, y, c) in enumerate(cand):
                    L[a_, j] = 1.0 if p == (x, y) else 0.0
                mass.append({p: 1.0} if p in inview else {})
        # 2. match
        match = {}
        if live and cand:
            r = np.array([f["r"] for f in live])
            lamc = np.array([lam[c] for _, _, c in cand])
            with np.errstate(divide="ignore"):
                W = np.log(r[:, None] * PD * L) - np.log(lamc[None])
            W = np.where(np.isfinite(W), np.maximum(W, 0), 0)
            ri, ci = linear_sum_assignment(-W)
            match = {int(j): live[i]["id"] for i, j in zip(ri, ci) if W[i, j] > 0}
        # score returns (before files take what they matched)
        light = d["light"][t] >= 0.3
        if light:
            cidx = {(x, y): j for j, (x, y, c) in enumerate(cand)}
            for i, k, x, y in at.get(t, []):
                if k not in CREATURE_KINDS or i not in last_in:
                    continue
                away = t - 1 - last_in[i]
                if 1 <= away <= KMAX:
                    j = cidx.get((x, y))
                    mine = [f for f in files.values() if f["tid"] == i]
                    fid = match.get(j) if j is not None else None
                    events.append({"seg": int(seg), "t": t, "id": i, "kind": k, "away": away,
                                   "candidate": j is not None, "has file": bool(mine),
                                   "linked": fid is not None,
                                   "kept": bool(fid is not None and files[fid]["tid"] == i)})
        if TRACE[1] <= t <= TRACE[2] and d["w"] == TRACE[0]:
            print(f"t {t} cand", [(x, y, c, tid_at.get((x, y))) for x, y, c in cand])
            for a_, f in enumerate(live):
                best = np.argsort(-L[a_])[:2]
                print(f"   file {f['id']} tid {f['tid']} at {(f['x'], f['y'])} k {t - f['t']} r {f['r']:.3f} L",
                      [(cand[j][:2], float(f"{L[a_, j]:.3g}")) for j in best if L[a_, j] > 0],
                      "mass in view", round(float(sum(mass[a_].values())), 3))
            print("   match", {cand[j][:2]: fid for j, fid in match.items()})
        # 3. correct
        for j, (x, y, c) in enumerate(cand):
            fid = match.get(j)
            if fid is None:
                fid, nf = nf, nf + 1
                files[fid] = {"id": fid, "st": 0}
                f = files[fid]
            else:
                f = files[fid]
                m = (x - f["x"], y - f["y"])
                f["st"] = B3.MOV.index(m) if t - f["t"] == 1 and m in B3.MOV else 0
            f.update(x=x, y=y, code=c, t=t, r=1.0, checked=set(), tid=tid_at.get((x, y)))
            dets.append((int(seg), t, fid, x, y))
        seen = set(match.values())
        for a_, f in enumerate(live):
            if f["id"] in seen:
                continue
            f["r"] *= 1 - vanish[f["code"]]
            q = min(1.0, sum(v for p, v in mass[a_].items() if p not in f["checked"]))
            f["checked"] |= set(mass[a_])
            f["r"] = f["r"] * (1 - PD * q) / max(1 - f["r"] * PD * q, 1e-12)
        for fid in [fid for fid, f in files.items() if f["t"] < t and (t + 1 - f["t"] > KMAX + 1 or f["r"] < 1e-3)]:
            del files[fid]
        old = sorted((f["t"], fid) for fid, f in files.items() if f["t"] < t)
        for _, fid in old[:max(0, len(old) - K_FILES)]:
            del files[fid]
        # 4. remember the view; predictions for the next step
        mp.write(t, pos, codes)
        for f in files.values():
            if form in ("model", "modelq") and f["t"] == t:
                f["snap"] = patch(mp, d, f["x"], f["y"], t, mover)
            elif form in ("counted", "stays"):
                if f["t"] == t:
                    f["B"] = np.zeros((5, 2 * WIN + 1, 2 * WIN + 1))
                    f["B"][f["st"], WIN, WIN] = 1.0
                if form == "counted":
                    f["B"] = B3.propagate(f["B"], T.get(f["code"], T[-1]))
        tb += time.perf_counter() - t0
        nb += 1
        for i, k, x, y in at.get(t, []):
            last_in[i] = t
    return events, dets, tb / max(nb, 1)


def idf1(d, dets, at, n):
    """IDF1 per segment over day frames: objects are the creatures in view (cow, zombie, skeleton); detections are
    every candidate's file. Main: detections on the evaluator's other objects (arrows, plants) are distractors and
    removed (MOT's rule). 098.6's form: arrows count as objects."""
    out = {}
    for name, kinds, drop in (("main", CREATURE_KINDS, (ARROW, PLANT)), ("arrows as objects (098.6)",
                                                                           CREATURE_KINDS + (ARROW,), (PLANT,))):
        gt_at = defaultdict(dict)
        dropped = set()
        ngt = 0
        for t in range(n):
            if d["seg"][t] < 0 or d["light"][t] < 0.3:
                continue
            for i, k, x, y in at.get(t, []):
                if k in kinds:
                    gt_at[t][(x, y)] = i
                    ngt += 1
                elif k in drop:
                    dropped.add((t, x, y))
        co = defaultdict(lambda: defaultdict(int))
        npred = 0
        for seg, t, fid, x, y in dets:
            if d["light"][t] < 0.3 or (t, x, y) in dropped:
                continue
            npred += 1
            g = gt_at[t].get((x, y))
            if g is not None:
                co[seg][(g, fid)] += 1
        idtp = 0
        for seg, pairs in co.items():
            G = sorted({g for g, p in pairs})
            Pd = sorted({p for g, p in pairs})
            gi, pi = {g: i for i, g in enumerate(G)}, {p: i for i, p in enumerate(Pd)}
            M = np.zeros((len(G), len(Pd)))
            for (g, p), c in pairs.items():
                M[gi[g], pi[p]] = c
            r, c = linear_sum_assignment(-M)
            idtp += M[r, c].sum()
        out[name] = {"idtp": float(idtp), "objects": ngt, "detections": npred}
    return out


def summarise(events, ids, secs):
    rep = {}
    ev = events
    f = lambda e: round(float(np.mean([x["kept"] for x in e])), 4) if e else None
    rep["1 returns within 50 steps, by day"] = len(ev)
    rep["1 identity kept on returns"] = f(ev)
    rep["returns: a candidate at the place / a file of the creature / linked to some file"] = [
        round(float(np.mean([e[k] for e in ev])), 4) for k in ("candidate", "has file", "linked")] if ev else None
    rep["by kind (cow 1, zombie 2, skeleton 3)"] = {k: {"returns": len(v), "kept": f(v)} for k in CREATURE_KINDS
                                                     for v in [[e for e in ev if e["kind"] == k]]}
    rep["by time away"] = {f"{lo}-{hi}": {"returns": len(v), "kept": f(v)} for lo, hi in ((1, 5), (6, 20), (21, 50))
                           for v in [[e for e in ev if lo <= e["away"] <= hi]]}
    for name in ids[0]:
        idtp = sum(x[name]["idtp"] for x in ids)
        g = sum(x[name]["objects"] for x in ids)
        p = sum(x[name]["detections"] for x in ids)
        key = "2 IDF1" if name == "main" else f"IDF1, {name}"
        rep[key] = round(2 * idtp / max(g + p, 1), 4)
        rep[f"{key}: ID recall, ID precision, objects, detections"] = [round(idtp / max(g, 1), 4),
                                                                      round(idtp / max(p, 1), 4), g, p]
    rep["belief seconds per step"] = round(secs, 5)
    return rep


# ------------------------------------------------------------------------------------------------ the gate's check
def gate(net, tau, movers, T, Q, test):
    """Held-out log-likelihood of the place and code a creature reaches k steps later (k = 5, 20), by day: the model
    asked about the patch around it as the belief builds it, against counted motion and "stays" (both with the
    evaluator's last move, so at their best). Every 3rd awake day step; the creature's place at t + k must be in view."""
    mover = np.zeros(NC + 1, bool)
    mover[movers] = True
    res = {k: {"model": [], "counted": [], "stays": [], "inside patch": [], "seg": []} for k in (5, 20)}
    spread = {}                                          # (code, last move, k) -> counted motion's map of places

    def counted_map(c0, st, k):
        if (c0, st, k) not in spread:
            B = np.zeros((5, 2 * WIN + 1, 2 * WIN + 1))
            B[st, WIN, WIN] = 1.0
            for _ in range(k):
                B = B3.propagate(B, T.get(c0, T[-1]))
            spread[(c0, st, k)] = B.sum(0)
        return spread[(c0, st, k)]

    for d in test:
        at = truth(d)
        where = {(t, i): (x, y, k) for t, v in at.items() for i, k, x, y in v}
        mp = None
        for t in range(d["n"]):
            if d["seg"][t] < 0:
                continue
            if t == 0 or d["seg"][t - 1] != d["seg"][t]:
                mp = Map()
            mp.write(t, d["pos"][t], d["code"][t])
            if t % 3 or d["light"][t] < 0.3:
                continue
            for k in (5, 20):
                if t + k >= d["n"] or d["seg"][t + k] != d["seg"][t]:
                    continue
                q = []
                for i, kind, x, y in at.get(t, []):
                    if kind not in CREATURE_KINDS or (t + k, i) not in where:
                        continue
                    x1, y1, _ = where[(t + k, i)]
                    vi = (y1 - d["pos"][t + k, 1] + 3) * 9 + (x1 - d["pos"][t + k, 0] + 4)
                    c1, c0 = int(d["code"][t + k, vi]), int(d["code"][t, (y - d["pos"][t, 1] + 3) * 9 + x - d["pos"][t, 0] + 4])
                    st = 0
                    if (t - 1, i) in where:
                        m = (x - where[(t - 1, i)][0], y - where[(t - 1, i)][1])
                        st = B3.MOV.index(m) if m in B3.MOV else 0
                    q.append((x, y, x1, y1, c0, c1, st))
                if not q:
                    continue
                snaps = [patch(mp, d, x, y, t, mover) for x, y, *_ in q]
                own = own_part(net, tau, torch.stack([s[0] for s in snaps]), torch.stack([s[1] for s in snaps]),
                               torch.stack([s[2] for s in snaps]),
                               torch.full((len(q),), 17 + k - 2, device=dev)).cpu().numpy()
                for b, (x, y, x1, y1, c0, c1, st) in enumerate(q):
                    dx, dy = x1 - x, y1 - y
                    inside = abs(dx) <= 4 and abs(dy) <= 3
                    Lm = own[b, (dy + 3) * 9 + dx + 4, c1] if inside else 0.0
                    qd = Q.get(c0, {c0: 1.0}).get(c1, 0.0)
                    Lc = counted_map(c0, st, k)[WIN + dy, WIN + dx] * qd if abs(dx) <= WIN and abs(dy) <= WIN else 0.0
                    Ls = float(dx == 0 and dy == 0) * qd
                    for name, Lv in (("model", Lm), ("counted", Lc), ("stays", Ls)):
                        res[k][name].append(float(np.log((1 - EPS) * Lv + EPS / 63)))
                    res[k]["inside patch"].append(inside)
                    res[k]["seg"].append(int(d["seg"][t]))
    out = {}
    for k, v in res.items():
        m, c = np.array(v["model"]), np.array(v["counted"])
        diff = m - c
        boot = np.random.default_rng(0).choice(diff, (2000, len(diff))).mean(1)
        segs, inv = np.unique(np.array(v["seg"]), return_inverse=True)   # the same, resampling whole segments
        sums, cnt = np.bincount(inv, diff), np.bincount(inv)
        idx = np.random.default_rng(0).integers(0, len(segs), (4000, len(segs)))
        boot_seg = sums[idx].sum(1) / cnt[idx].sum(1)
        out[f"k = {k}"] = {"creatures": len(m), "log-likelihood: model, counted, stays":
                           [round(float(m.mean()), 4), round(float(c.mean()), 4), round(float(np.mean(v["stays"])), 4)],
                           "model - counted (95% interval)": [round(float(diff.mean()), 4),
                                                              round(float(np.quantile(boot, 0.025)), 4),
                                                              round(float(np.quantile(boot, 0.975)), 4)],
                           "95% interval over segments": [round(float(np.quantile(boot_seg, q)), 4) for q in (0.025, 0.975)],
                           "segments": int(len(segs)),
                           "place at t + k inside the patch": round(float(np.mean(v["inside patch"])), 4)}
    return out


def main():
    t0 = time.monotonic()
    train = [load(w) for w in (0, 1)]
    net, tau = load_model(SEED if FORM in ("model", "modelq", "gate") else 1021)
    own = own_codes(train)
    movers, info = creature_codes(net, tau, train, own)
    print("own codes", own, "creature codes", movers, f"({time.monotonic() - t0:.0f} s)", flush=True)
    T, Q = counted(range(8))
    test = [load(w) for w in (8, 9)]
    if FORM == "gate":
        out = {"seed": SEED, "creature codes": movers, **gate(net, tau, movers, T, Q, test),
               "seconds": round(time.monotonic() - t0, 1)}
        (OUT / f"gate_s{SEED}.json").write_text(json.dumps(out, indent=1) + "\n")
        return print(json.dumps(out, indent=1))
    lam, vanish = rates(train, movers)
    print("rates", f"({time.monotonic() - t0:.0f} s)", flush=True)
    events, ids, secs = [], [], []
    for d in test:
        at = truth(d)
        ev, dets, s = run(d, FORM, net, tau, movers, lam, vanish, T, Q, at)
        events += ev
        ids.append(idf1(d, dets, at, d["n"] if not NMAX else min(NMAX, d["n"])))
        secs.append(s)
        print("worker done", f"({time.monotonic() - t0:.0f} s)", flush=True)
    rep = {"form": FORM, "seed": SEED if FORM in ("model", "modelq") else None, **summarise(events, ids, float(np.mean(secs))),
           "the agent's own codes": own, "creature codes": movers,
           "lambda (per place and step) and vanishing (per step), creature codes": {
               str(c): [float(f"{lam[c]:.3g}"), round(float(vanish[c]), 4)] for c in movers},
           "codes: P(leaves its place)": info, "seconds": round(time.monotonic() - t0, 1)}
    tag = f"{FORM}_s{SEED}" if FORM in ("model", "modelq") else FORM
    (OUT / f"belief_{tag}{arg('--tag', '')}.json").write_text(json.dumps(rep, indent=1) + "\n")
    np.save(OUT / f"events_{tag}{arg('--tag', '')}.npy", np.array(events, dtype=object))
    print(json.dumps({k: v for k, v in rep.items() if k != "codes: P(leaves its place)"}, indent=1))


if __name__ == "__main__":
    main()
