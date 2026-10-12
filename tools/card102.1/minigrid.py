"""Card 102.1, criterion 3: the same arms on card 095.1's MiniGrid tries.

Tokens: 095.1's state per interaction try (every token within 2 steps of the agent, the attended tokens beyond,
nearest first, at most 6, and the hand), each a what (the world's encoder vector) and a raw where: its place's
offset in card 062's believed 13 x 13 window (the agent at (0, 0), the front at (0, 1)), and for the hand the held
row, declared at (0, -7), outside the window. Divided by 6. Nothing else is declared.

Arm A is trained, as 095.1's, on the training tries of tiers 1, 2 and the decoy world, never tier 3. Each tier's
store holds its own training tries (as 095.1's arm B), with h and alpha chosen on the last tenth of that tier's
training episodes. Scored on the held-out tries (every tenth episode), weighted by memory's try weights, against
version 27's arm C (card 095.3's gate: front and hand only), and on tier 2's counterfactual boxes (095.3's gate:
300 held-out pick ups with an empty hand, the front set to each box).

  bin/prun python tools/card102.1/minigrid.py [--steps N]      → runs/102.1/minigrid.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card095.1"))
sys.path.insert(0, str(ROOT / "tools" / "card102.1"))
import tok                                             # noqa: E402
import arms as AR                                      # noqa: E402

BOXES = {"blue": 180, "red": 170, "grey": 190, "yellow": 185, "green": 175, "purple": 85}   # tier 2 (card 095.2)
FRONT, HAND = 71, 169
V27 = {"tier1": 1.0, "tier2": 1.0, "tier3": 0.998293}  # card 095.3's gate, front and hand only (runs/095.3/gate.json)


def data(world):
    z = tok.load(world)
    H, P, C, A = tok.build(z)
    tr, te = tok.split(z)
    wh = np.concatenate([z["wh"][:169], [[0, -7]]]).astype(np.float32) / 6.0
    valid = H >= 0
    d = {"V": (z["arr"][H.clip(0)] * valid[..., None]).astype(np.float16), "W": wh[P], "valid": valid,
         "code": np.where(valid, H, -1), "act": z["act"].astype(np.int64), "w": z["w"].astype(np.float64),
         "ep": z["ep"], "A": A, "C": C, "P": P,
         "locus": np.zeros((len(H), 2), np.float32)}       # the agent is the window's origin
    d["codea"] = np.where(C, A, d["code"])
    d["Va"] = (z["arr"][d["codea"].clip(0)] * valid[..., None]).astype(np.float16)
    return z, d, tr, te


def fit_world(net, d, tr, known):
    """The world's store and its h and alpha (last tenth of its training episodes), arm A fixed."""
    t = AR.sub(d, tr)
    eps = np.unique(t["ep"])
    val = np.isin(t["ep"], eps[int(0.9 * len(eps)):])
    ft, vd = AR.sub(t, ~val), AR.sub(t, val)
    _, H_f, _, S_f = AR.run_a(net, ft, "tokens")
    PA_v, H_v, AF_v, S_v = AR.run_a(net, vd, "tokens")
    store = AR.build_store(ft, H_f, S_f)
    rng = np.random.default_rng(0)
    a = H_v[vd["valid"]][rng.integers(0, int(vd["valid"].sum()), 4000)].astype(np.float32)
    scale = float(np.median(np.linalg.norm(a[:2000] - a[2000:], axis=1)))
    y = (vd["code"] != vd["codea"]).astype(np.float32)
    grid = []
    for hw in (0.02, 0.05, 0.1, 0.25):
        N, Nc, _, _ = AR.vote(store, vd["act"], vd["code"], H_v, hw * scale, results=False)
        for alpha in (0.3, 1.0, 3.0, 10.0, 30.0):
            L = AR.ll((Nc + alpha * PA_v) / (N + alpha), y, vd["valid"])
            grid.append((float((L * vd["w"]).sum() / vd["w"].sum()), hw, alpha))
    best = max(grid)
    alpha_r, acc = AR.choose_alpha_r(store, vd, H_v, S_v, AR.snap(AF_v, *known), best[1] * scale)
    _, H_t, _, S_t = AR.run_a(net, t, "tokens")
    store = AR.build_store(t, H_t, S_t)
    return AR.ArmC(net, store, best[1] * scale, best[2], "tokens", known, alpha_r), best + (alpha_r,)


def scores(P, res, d):
    C, A, valid, w = d["C"], d["A"], d["valid"], d["w"]
    ch = P >= 0.5
    right = ((ch == C) | ~valid).all(1) & ((~C) | (res == A) | ~valid).all(1)
    L = AR.ll(P, C.astype(np.float32), valid)
    return {"tries": int(len(C)), "exact next state": round(float((right * w).sum() / w.sum()), 6),
            "change log-likelihood per try": round(float((L * w).sum() / w.sum()), 6)}


def main():
    steps = int(AR.arg("--steps", "6000"))
    t0 = time.monotonic()
    worlds = {w: data(w) for w in ("tier1", "tier2", "tier3", "decoy")}
    train = [AR.sub(d, tr) for w, (z, d, tr, te) in worlds.items() if w != "tier3"]
    joint = {k: np.concatenate([t[k] for t in train]) for k in train[0]}
    print("training tries", len(joint["act"]), f"({time.monotonic() - t0:.0f} s)", flush=True)
    net = AR.train_a(joint, "tokens", steps, bs=256)
    rep = {"steps": steps}
    for w in ("tier1", "tier2", "tier3"):
        z, d, tr, te = worlds[w]
        known = np.unique(z["app"])
        arm, best = fit_world(net, d, tr, (z["arr"][known].astype(np.float32), known))
        q = AR.sub(d, te)
        P, PA, N, res, resA = arm(q)
        r = {"kernel width (x median distance)": best[1], "alpha": best[2], "alpha_r": best[3], "arm C": scores(P, res, q),
             "arm A alone": scores(PA, resA, q), "version 27's arm C (front and hand only)": V27[w]}
        r["difference from version 27 (points)"] = round(100 * (r["arm C"]["exact next state"] - V27[w]), 4)
        if w == "tier2":
            fs = int(np.flatnonzero(d["P"][0] == FRONT)[0])   # the front's slot (fixed: near places come first)
            cand = np.flatnonzero(te & (d["act"] == 3) & (d["code"][:, -1] == 0) & d["valid"][:, fs])
            Q = np.random.default_rng(0).choice(cand, min(300, len(cand)), replace=False)
            cf = {}
            for col, bx in BOXES.items():
                v = AR.sub(d, Q)
                v["code"] = v["code"].copy()
                v["V"] = v["V"].copy()
                v["code"][:, fs] = bx
                v["V"][:, fs] = z["arr"][bx]
                Pq, _, _, rq, _ = arm(v)
                cf[col] = round(float(((rq[:, -1] == bx) & (Pq[:, -1] >= 0.5)).mean()), 4)
            r["counterfactual boxes: hand holds that box"] = cf
        rep[w] = r
        print(w, json.dumps(r), f"({time.monotonic() - t0:.0f} s)", flush=True)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (AR.RUNS / f"minigrid{AR.arg('--tag', '')}.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
