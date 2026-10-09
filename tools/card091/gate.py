"""Card 091's gate: the router against version 18's recall where the prior decides, a stored try whose whole
(front, held) combination is hidden from memory, in the agent's own World.

  version 18: its neighbour vote over situation groups (card 051's index, card 049's admitted conditions), with every
              group of the query's combination removed: s = (w_G·N_G + a/L) / (Σ w_G·N_G + a), the labels mapped to
              categories; scored on every stored group of pick up, drop and toggle (its counts as the truth). Forward:
              card 038's neighbour vote over its keys (its weights, KMIN and prior), the query's combination removed.
  router:     its kernel vote over the world's other stored tries of the action, the query's combination removed
              (runs/091/router_*.pt); scored on 4,000 stored tries per action.
Both are mean log-likelihood per try (memory's sampling weights).

  <version 18's flags> WM_STORED_FITS=1 bin/prun python tools/card091/gate.py tier2 runs/091/router_a.pt
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORLD = sys.argv[1]
ROUTER = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs" / "091" / "router_a.pt"
HIDE = sys.argv[3] if len(sys.argv) > 3 else "combination"         # or "front": every try with the same front tile
TIER = int(WORLD[-1]) if WORLD.startswith("tier") else 1
sys.argv = ["run.py", "tiers", "--tier", str(TIER), "--online", "0", "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402
import torch                                           # noqa: E402

sys.path.insert(0, str(ROOT / "tools" / "card091"))
import router as RT                                    # noqa: E402

T = RUN.T
VP = T.VP
log = lambda m: print(m, flush=True)
W, info = T.setup(TIER, log)
APP = T.Kd.APP


def v18_groups(a):
    kd = W.kinds[a]
    ix = kd.ix
    keys = [kd.keys[i] for i in ix["rep"]]
    combo = np.array([(int(k[0]), int(k[1])) for k in keys])
    Gc = ix["Gc"]
    L = kd.lc.shape[1]
    if Gc.shape[1] < L:
        Gc = np.hstack([Gc, np.zeros((len(Gc), L - Gc.shape[1]))])
    M = np.zeros((L, kd.ncat))
    M[np.arange(L), np.asarray(kd.lcat)[:L]] = 1.0
    ll, tot = 0.0, 0.0
    for i in range(0, len(keys), 256):
        wG = kd.group_weights(keys[i:i + 256])
        same = (combo[i:i + 256, None, 0] == combo[None, :, 0])
        if HIDE == "combination":
            same &= combo[i:i + 256, None, 1] == combo[None, :, 1]
        wG = np.where(same, 0.0, wG)
        Nn = wG @ Gc
        s = (Nn + kd.a / L) / (Nn.sum(1, keepdims=True) + kd.a)
        P = np.clip(s @ M, 1e-12, 1)
        Tq = Gc[i:i + 256] @ M
        ll += float((Tq * np.log(P)).sum())
        tot += float(Tq.sum())
    return ll / tot, len(keys)


def v18_forward():
    """Version 18's forward vote (card 038's neighbour_vote over its keys, share per key, KMIN, its prior), the
    query's (front, held) combination removed; scored on each key's stored counts."""
    kd = W.kinds[VP.FWD]
    V = VP if hasattr(VP, "neighbour_vote") else sys.modules["vector_planner"]
    share = kd.counts / kd.counts.sum(1, keepdims=True)
    combo = np.array([(int(k[0]), int(k[1]) if len(k) > 1 else 0) for k in kd.keys])
    k = np.exp(-V.wl1(kd.X, kd.X, kd.lam))
    k[k < V.KMIN] = 0.0
    same = combo[:, None, 0] == combo[None, :, 0]
    if HIDE == "combination":
        same &= combo[:, None, 1] == combo[None, :, 1]
    k[same] = 0.0
    ll = 0.0
    for i in range(len(kd.keys)):
        P, _ = V.neighbour_vote(k[i], share, None)
        ll += float((kd.counts[i] * np.log(np.clip(P, 1e-12, 1))).sum())
    return ll / float(kd.counts.sum()), len(kd.keys)


@torch.no_grad()
def router_scores(net, n_query=4000, seed=0):
    zc = RT.code_vectors()
    D = RT.Data([WORLD], zc)
    rng = np.random.default_rng(seed)
    out = {}
    hf = APP[D.d["front"].astype(np.int64)].astype(np.int64)
    hh = APP[D.d["held"].astype(np.int64)].astype(np.int64)
    combo = torch.as_tensor(hf * 100000 + (hh if HIDE == "combination" else 0), device="cuda")
    for a in RT.ACTS:
        r = D.rows[a]
        q = rng.choice(r, min(n_query, len(r)), replace=False)
        eq, ec = RT.embed(net, D, q), RT.embed(net, D, r)
        rt = torch.as_tensor(r, device="cuda")
        oc, wc, cc = D.t["out"][rt], D.w[rt], combo[rt]
        tau, al = net.log_tau.exp(), net.log_alpha.exp()
        lls = []
        for i in range(0, len(q), 256):
            qt = torch.as_tensor(q[i:i + 256], device="cuda")
            d = torch.cdist(eq[i:i + 256], ec, p=1)
            k = torch.exp(-d / tau) * wc[None]
            k = k.masked_fill(combo[qt][:, None] == cc[None], 0.0)
            N = k @ torch.nn.functional.one_hot(oc, RT.NOUT[a]).float()
            P = (N + al * D.prior[a][None]) / (N.sum(1, keepdim=True) + al)
            lls.append(torch.log(P[torch.arange(len(qt)), D.t["out"][qt]].clamp_min(1e-12)))
        ll = torch.cat(lls).cpu().numpy()
        wq = D.d["w"][q]
        f = D.prior[a].cpu().numpy()
        out[a] = {"router": round(float((ll * wq).sum() / wq.sum()), 4),
                  "action frequencies": round(float((np.log(f[D.d["out"][q]]) * wq).sum() / wq.sum()), 4)}
    return out


ck = torch.load(ROUTER)
net = RT.Router().cuda()
net.load_state_dict(ck["net"])
net.eval()
rep = {"note": "Card 091 gate, tools/card091/gate.py", "world": WORLD, "router": str(ROUTER), "hidden": HIDE,
       "router trained on": ck["worlds"]}
rs = router_scores(net)
for a, name in ((VP.FWD, "forward"), (VP.PICK, "pick up"), (VP.DROP, "drop"), (VP.TOG, "toggle")):
    v, n = v18_forward() if a == VP.FWD else v18_groups(a)
    rep[name] = {"version 18": round(v, 4), "stored keys or groups": n, **rs[a]}
    log(f"{name}: {rep[name]}")
out = ROOT / "runs" / "091" / f"gate_{WORLD}_{ROUTER.stem}{'' if HIDE == 'combination' else '_' + HIDE}.json"
out.write_text(json.dumps(rep, indent=1) + "\n")
