"""Card 102.5: the agent as a token.

Placing (the view's own motion), one rule for every world: each step the view's shift s among small rigid shifts
(-2..2 tiles each way) is the most probable given belief and the action,
    log p(s | view, belief, a) = log p(s | a) - sum over known places of |v - belief| / b + const,
with p(s | a) learned from the agent's own placements (counts, Dirichlet 1), and b the matching noise per number
(mean absolute residual at the chosen placements, learned the same way). Nothing says whether the view follows the
agent (Crafter, world 3) or stays fixed (worlds 2 and 4b): the prior learns it.

Controlled tokens: per code, the motion of its tiles in the world frame (the shift that best overlays its tiles at
one step on the next, among places seen at both), and the uncertainty coefficient U = I(action; motion) / H(motion) on training play. A code is
controlled when U >= 0.5 (declared); the evaluator's controlled codes are those whose tiles are mostly the player
(worlds 2, 3, Crafter) or blue (world 4b).

  bin/prun python tools/card102.5/self.py world2|world3|world4|crafter   → runs/102.5/self_<world>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("card102.2", "card102.3"):
    sys.path.insert(0, str(ROOT / "tools" / p))
import layers as LY                                    # noqa: E402
import belief as BL                                    # noqa: E402

RUNS = ROOT / "runs" / "102.5"
R, C, L = 7, 9, 63
SH = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)]
VY, VX = np.divmod(np.arange(L), C)


def load(world, split):
    if world == "world2":
        d = LY.load("world2", split)
        z = np.load(ROOT / "runs" / "card102" / f"world2_{split}.npz")
        d["ctrl_sem"], d["semc"] = 99, z["obj"].reshape(len(d["seg"]), L)             # the agent is object 99
        d["pos"] = None
    elif world == "world4":
        z = np.load(ROOT / "runs" / "card102" / f"world4_{split}.npz")
        n = len(z["act"])
        d = {"code": np.load(RUNS / f"world4_code_{split}.npy")[:, :L].reshape(n, R, C).astype(np.int64),
             "vec": np.load(RUNS / f"world4_vec_{split}.npy", mmap_mode="r"), "seg": z["ep"].astype(np.int64),
             "act": z["act"], "pos": None}
        d["ctrl_sem"], d["semc"] = 2, z["sem"].reshape(n, L)                           # blue
    else:
        d = BL.load(world, split)
        if world == "world3":
            d["act"] = np.load(ROOT / "runs" / "card102" / f"world3_{split}.npz")["act"]
            d["ctrl_sem"] = 4                                                          # world 3's agent kind
        else:
            ws = range(8) if split == "train" else (8, 9)
            d["act"] = np.concatenate([np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz")["act"] for w in ws])
            d["ctrl_sem"] = None                                                       # the view's centre (player)
        d["semc"] = d["sem"].reshape(len(d["seg"]), L)
    d["V"] = np.asarray(d["vec"][:, :L], np.float32) if world != "crafter" else d["vec"][:, :L]
    return d


def place(d, prior, b, learn, still=None, half=100, adapt="--adapt" in sys.argv):
    """Each step's view origin in the episode's world frame. prior: counts (actions, shifts); b: matching noise.
    Belief is the last vector seen at each world place of the episode (a canvas; NaN where never seen). still: per
    code, the probability its tiles stay put in the world frame; a place's mismatch counts by it (a place holds a still
    token, matched, or a moving one, which says nothing about the view's motion)."""
    seg, act, V = d["seg"], d["act"].astype(int), d["V"]
    n, dv = len(seg), V.shape[-1]
    org = np.zeros((n, 2), np.int64)
    shift = np.full(n, -1)
    res_sum, res_n = 0.0, 0
    canvas = np.full((2 * half + R, 2 * half + C, dv), np.nan, np.float32)
    touched = []
    code = d["code"].reshape(n, R, C)
    k0 = np.linspace(0, n - 1, min(n, 5000)).astype(int)
    S = np.asarray(V[k0], np.float32).reshape(-1, dv)
    mu, sp = S.mean(0), max(float(np.abs(S - S.mean(0)).mean()), 1e-3)        # the data's centre and spread
    for t in range(n):
        if seg[t] < 0:
            continue
        v = np.asarray(V[t], np.float32).reshape(R, C, dv)
        st_ = np.full((R, C), 0.99) if still is None else np.vectorize(lambda c: still.get(int(c), 0.5))(code[t])
        st_ = np.clip(st_, 1e-4, 1 - 1e-4)
        marg = np.abs(v - mu).sum(-1) / sp + dv * np.log(2 * sp)                # the place's cost as new content
        if t == 0 or seg[t - 1] != seg[t]:
            for (y, x) in touched:
                canvas[y:y + R, x:x + C] = np.nan
            touched = []
            org[t] = (0, 0)
        else:
            a = int(act[t - 1])
            lp = np.log(prior[a] / prior[a].sum())
            ll = np.empty(len(SH))
            bt = b                                        # this step's matching noise: the view's own median mismatch
            if adapt:                                     # under the action's likeliest shift, floored at b
                k0 = int(lp.argmax())
                o = org[t - 1] + SH[k0]
                y, x = int(o[1]) + half, int(o[0]) + half
                if 0 <= y <= 2 * half and 0 <= x <= 2 * half:
                    w = canvas[y:y + R, x:x + C]
                    known = ~np.isnan(w[..., 0])
                    if known.any():
                        bt = max(b, float(np.median(np.abs(v[known] - w[known]).mean(1))))
            rs = []
            for k, (dx, dy) in enumerate(SH):
                o = org[t - 1] + (dx, dy)
                y, x = int(o[1]) + half, int(o[0]) + half
                if not (0 <= y <= 2 * half and 0 <= x <= 2 * half):
                    ll[k] = -np.inf
                    rs.append(None)
                    continue
                w = canvas[y:y + R, x:x + C]
                known = ~np.isnan(w[..., 0])
                r = np.abs(v[known] - w[known]).sum(1)
                # every place in view: a known place holds a still token (its mismatch at the matching noise) or one
                # that moved (new content, by the data's spread), mixed by its code's stillness; a never-seen place is
                # new content (both Laplace, normalisers kept). Places never seen are left out, so each candidate is
                # judged by its known places' mean (card 098.8's form), the action's learned prior breaking ties
                kept = np.logaddexp(np.log(st_[known]) - r / bt - dv * np.log(2 * bt),
                                    np.log(1 - st_[known]) - marg[known])
                ll[k] = (lp[k] + kept.sum()) / max(int(known.sum()), 1) if known.any() else lp[k] - 1e6
                rs.append(r)
            k = int(ll.argmax())
            org[t] = org[t - 1] + SH[k]
            shift[t - 1] = k
            if learn:
                prior[a, k] += 1
            if rs[k] is not None and len(rs[k]):
                res_sum += float(rs[k].sum())
                res_n += rs[k].size * dv
        y, x = int(org[t, 1]) + half, int(org[t, 0]) + half
        if 0 <= y <= 2 * half and 0 <= x <= 2 * half:
            canvas[y:y + R, x:x + C] = v
            touched.append((y, x))
    return org, shift, (res_sum / res_n if res_n else b)


def motions(d, org):
    """Per step and code: the code's motion in the world frame, the shift that best overlays its tiles at t on its
    tiles at t+1 (ties to the smaller shift), among places seen at both steps."""
    seg, code = d["seg"], d["code"].reshape(len(d["seg"]), L)
    order = sorted(range(len(SH)), key=lambda k: abs(SH[k][0]) + abs(SH[k][1]))
    out = {}
    for t in np.flatnonzero((seg[:-1] >= 0) & (seg[:-1] == seg[1:])):
        o0, o1 = org[t], org[t + 1]
        lo, hi = np.maximum(o0, o1), np.minimum(o0, o1) + (C - 1, R - 1)              # seen at both steps
        x0, y0 = VX + o0[0], VY + o0[1]
        x1, y1 = VX + o1[0], VY + o1[1]
        in0 = (x0 >= lo[0]) & (x0 <= hi[0]) & (y0 >= lo[1]) & (y0 <= hi[1])
        in1 = (x1 >= lo[0]) & (x1 <= hi[0]) & (y1 >= lo[1]) & (y1 <= hi[1])
        for c in np.intersect1d(code[t][in0], code[t + 1][in1]):
            p0 = set(zip(x0[in0 & (code[t] == c)].tolist(), y0[in0 & (code[t] == c)].tolist()))
            p1 = set(zip(x1[in1 & (code[t + 1] == c)].tolist(), y1[in1 & (code[t + 1] == c)].tolist()))
            best, bk = -1, 0
            for k in order:
                dx, dy = SH[k]
                ov = len({(x + dx, y + dy) for x, y in p0} & p1)
                if ov > best:
                    best, bk = ov, k
            out.setdefault(int(c), []).append((int(d["act"][t]),) + SH[bk])
    return out


def entropy(x):
    _, n = np.unique(x, axis=0, return_counts=True)
    p = n / n.sum()
    return float(-(p * np.log(p)).sum())


def uncertainty(rows):
    r = np.array(rows)
    a, m = r[:, 0], r[:, 1:]
    hm = entropy(m)
    if hm < 1e-9:
        return 0.0, hm
    hc = sum((a == k).mean() * entropy(m[a == k]) for k in np.unique(a))
    return (hm - hc) / hm, hm


def truth_controlled(d, world):
    code, semc = d["code"].reshape(len(d["seg"]), L), d["semc"]
    ok = d["seg"] >= 0
    out = {}
    for c in np.unique(code[ok]):
        m = (code == c) & ok[:, None]
        if world == "crafter":                            # the player is the view's centre
            share = m[:, 31].sum() / m.sum()
        else:
            share = (semc[m] == d["ctrl_sem"]).mean()
        out[int(c)] = bool(share > 0.5)
    return out


def main():
    world = sys.argv[1]
    tr, te = load(world, "train"), load(world, "test")
    prior = np.ones((int(max(tr["act"].max(), te["act"].max())) + 1, len(SH)))
    b = 1.0
    still = None
    for it in range(2):                                   # b and each code's stillness, re-estimated once
        pr = prior.copy()
        org_tr, sh_tr, b_new = place(tr, pr, b, learn=True, still=still)
        b = max(b_new, 1e-3)
        M = motions(tr, org_tr)
        still = {c: (sum(1 for r in rows if r[1:] == (0, 0)) + 1) / (len(rows) + 2) for c, rows in M.items()}
        if "--world-only" in sys.argv:                    # revision: the controlled codes say nothing about the
            for c, rows in M.items():                     # world frame (the general form of "leave out the
                if len(rows) >= 50 and uncertainty(rows)[0] >= 0.5:      # agent's tile")
                    still[c] = 1e-4
    prior = pr
    org_te, sh_te, _ = place(te, prior.copy(), b, learn=False, still=still)
    if "--world-only" not in sys.argv and "--adapt" not in sys.argv:   # the placements, for card 102.4 revision 2
        np.save(RUNS / f"org_{world}_train.npy", org_tr)
        np.save(RUNS / f"org_{world}_test.npy", org_te)
    rep = {"world": world, "matching noise b": round(b, 4),
           "stillness of the least still codes": dict(sorted(((c, round(v, 3)) for c, v in still.items()),
                                                             key=lambda x: x[1])[:8]),
           "learned prior: most probable shift per action": {int(a): SH[int(prior[a].argmax())] for a in range(len(prior))
                                                             if prior[a].sum() > len(SH) + 10}}
    seg = te["seg"]
    st = np.flatnonzero((seg[:-1] >= 0) & (seg[:-1] == seg[1:]))
    got = np.array([SH[k] for k in sh_te[st]])
    if te["pos"] is not None:                             # the view follows the agent
        true = te["pos"][st + 1] - te["pos"][st]
    else:
        true = np.zeros_like(got)
    right = (got == true).all(1)
    rep["view's motion read right"] = round(float(right.mean()), 5)
    rep["steps"] = int(len(st))
    moved = (true != 0).any(1)
    rep["when the view moved"] = round(float(right[moved].mean()), 5) if moved.any() else None
    rep["when it did not"] = round(float(right[~moved].mean()), 5)
    pairs, cnt = np.unique(np.concatenate([true[~right], got[~right]], 1), axis=0, return_counts=True)
    rep["misreads, most common (true dx, dy, read dx, dy): count"] = {str(tuple(p)): int(c) for p, c in
                                                                     sorted(zip(pairs.tolist(), cnt), key=lambda x: -x[1])[:6]}
    if "light" in te:
        day = te["light"][st] >= 0.3
        rep["read right by day, by night"] = [round(float(right[day].mean()), 4),
                                              round(float(right[~day].mean()), 4) if (~day).any() else None]
    U = {c: uncertainty(r) for c, r in M.items() if len(r) >= 50}
    T = truth_controlled(tr, world)
    ctrl = {c for c, (u, _) in U.items() if u >= 0.5}
    truth = {c for c in U if T.get(c)}
    tp = len(ctrl & truth)
    rep["controlled codes"] = sorted(ctrl)
    rep["evaluator's controlled codes"] = sorted(truth)
    rep["precision"] = round(tp / len(ctrl), 4) if ctrl else None
    rep["recall"] = round(tp / len(truth), 4) if truth else None
    rep["U per code (U, motion entropy, sightings, controlled by the evaluator)"] = {
        c: (round(u, 3), round(h, 3), len(M[c]), bool(T.get(c))) for c, (u, h) in sorted(U.items(), key=lambda x: -x[1][0])}
    RUNS.mkdir(exist_ok=True)
    if "--world-only" in sys.argv:
        world += "_world_only"
    (RUNS / f"self_{world}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("U per")}, indent=1))
    print("top U:", list(rep["U per code (U, motion entropy, sightings, controlled by the evaluator)"].items())[:8])


if __name__ == "__main__":
    main()
