"""Card 105.9: card 103.4's placing with its prior taken from the transition model. For each awake transition, 105.8's
token model gives the probability that the agent is at each tile of the 3 x 3 around the view's centre next (its own
read-out: the tile where an agent code is probable); that mass goes on the view shift that would centre the view there
(7 pixels per tile), mixed with 1% of the old prior p(shift | action). Evidence, tempering and candidates are 103.4's
as card 103.5 fitted them. The model reads positions only as differences, so its prediction needs no origin.

  bin/prun python tools/card105.9/place_model.py      → runs/105.9/prior_<src>_w<W>.npy, org_<src>_w<W>.npy, placing.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "105.9"
NET, TAU_M = ROOT / "runs" / "105.8" / "tokens_Rb_s1021_net.pt", None
MIX = 0.01                                              # the old prior's share, so that no shift is impossible

argv = sys.argv
sys.argv = ["run.py"]
sys.path.insert(0, str(ROOT / "tools" / "card105.2"))
import run as R                                          # noqa: E402
sys.argv = ["origins.py", "place", "0"]
spec = importlib.util.spec_from_file_location("origins1035", ROOT / "tools" / "card103.5" / "origins.py")
OR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OR)
sys.argv = argv
PL, TA = OR.PL, OR.TA
K, NS, UNIT = PL.K, PL.NS, 7
NEAR = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def source_data(kind, w):
    z = np.load(ROOT / "runs" / ("102.1" if kind == "curr" else "098.5") / (f"curr_w{w}.npz" if kind == "curr" else f"play_w{w}.npz"))
    d = {k: z[k] for k in ("act", "done", "pos", "sleep", "light", "ep")}
    d["obs"] = z["obs"][:, :PL.HW, :PL.WW]
    brk = np.r_[True, (d["ep"][1:] != d["ep"][:-1]) | d["sleep"][1:] | d["sleep"][:-1] | d["done"][:-1]]
    d["seg"] = np.cumsum(brk)
    d["seg"][d["sleep"]] = -1
    return d


@torch.no_grad()
def model_priors(srcs):
    """For each source, (n, NS * NS): the model's mixture where it predicts, else None (the old prior is used)."""
    tau = json.load(open(ROOT / "runs" / "105.8" / "tokens_Rb_s1021.json"))["tau"]
    tr = R.load(R.TRAIN, "tokens")
    net = R.TransitionModel(32, 17, arm="Rb").to(R.dev)
    net.load_state_dict(torch.load(NET))
    net.eval()
    Pt, plab = R.decoder(tr, "tokens")
    agent = torch.as_tensor(plab[:, 31] >= 1300, device=R.dev)
    near_idx = [31 + dx + 9 * dy for dx, dy in NEAR]
    near_k = [(K + UNIT * dy) * NS + (K + UNIT * dx) for dx, dy in NEAR]
    out = {}
    for kind, w in srcs:
        D = R.load([(kind, w)], "tokens")
        p = np.zeros((len(D["src"]), NS * NS), np.float32)
        has = np.zeros(len(D["src"]), bool)
        for s in range(0, len(D["t"]), 2048):
            i = np.arange(s, min(s + 2048, len(D["t"])))
            _, dist, _, _, _ = R.predict(net, D, i, Pt, plab)
            lg = -dist[:, near_idx] ** 2 / tau
            la = torch.logsumexp(lg[..., agent], -1) - torch.logsumexp(lg, -1)      # log p(agent there next), per tile
            q = torch.softmax(la, -1).cpu().numpy()                                 # over the 3 x 3: where it will be
            t = D["t"][i]
            p[np.ix_(t, near_k)] = q
            has[t] = True
        out[(kind, w)] = (p, has)
        print(f"model prior {kind} {w}: {int(has.sum())} transitions", flush=True)
    return out


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = [("curr", w) for w in range(10)] + [("play", w) for w in (0, 1, 8, 9)]
    pri = model_priors(srcs)
    z = np.load(ROOT / "runs" / "103.5" / "placing_tau300.npz")
    P = {"s2": float(z["s2"]), "eps": float(z["eps"]), "pi": z["pi"]}
    old = z["pr"] / z["pr"].sum(1, keepdims=True)                   # p(shift | action), as 103.5 fitted it
    cand = torch.as_tensor(z["cand"], device=PL.dev)
    rep = {}
    for kind, w in srcs:
        d = source_data(kind, w)
        n = len(d["act"])
        a = d["act"].astype(int)
        mp, has = pri[(kind, w)]
        prior = old[a].copy()
        prior[has] = (1 - MIX) * mp[has] + MIX * old[a[has]]
        np.save(OUT / f"prior_{kind}_w{w}.npy", prior.astype(np.float16))
        dd = dict(d, act=np.arange(n))                                # one prior and candidate set per step
        org, sh, _, _, _ = PL.place(dd, prior.astype(np.float64), P, learn=False, cand=cand[a])
        np.save(OUT / f"org_{kind}_w{w}.npy", org)
        # the evaluator's view of the result (scoring only)
        dp = np.r_[d["pos"][1:] - d["pos"][:-1], np.zeros((1, 2), np.int64)]
        ok = (sh != -99).all(1)
        mv = np.isin(a, R.MOVES) & ok
        right = (sh == UNIT * dp).all(1)
        blocked = mv & (dp == 0).all(1)
        rep[f"{kind} {w}"] = {"placements right": round(float(right[ok].mean()), 4),
                              "moves that succeeded": round(float(right[mv & ~blocked].mean()), 4),
                              "blocked moves": round(float(right[blocked].mean()), 4), "blocked moves (n)": int(blocked.sum()),
                              "still steps": round(float(right[ok & ~mv].mean()), 4)}
        print(kind, w, rep[f"{kind} {w}"], f"({time.monotonic() - t0:.0f} s)", flush=True)
        (OUT / "placing.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
