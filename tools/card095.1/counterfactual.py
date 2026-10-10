"""Card 095.1, criterion 3 (relations carry): tier 2's held-out pick ups with an empty hand, with the token in front
replaced by a box of each colour. Does the predicted hand after a pick up hold that box?

Version 20 (mode v20, needs the agent's setup): W.outcome for (box, empty hand, the try's believed view set with the
box added), from memory without the held-out episodes. Arm A: the network on the edited state. Arm B: pick up's
exemplars refitted (as arm_b.py) and asked about the edited state.

  <version 20's flags> bin/prun python tools/card095.1/counterfactual.py v20
  bin/prun python tools/card095.1/counterfactual.py arms
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card095.1"))
MODE = sys.argv[1]
BOXES = {"blue": 180, "red": 170, "grey": 190, "yellow": 185, "green": 175, "purple": 85}
EMPTY, NQ = 0, 300
if MODE == "v20":
    sys.argv = ["baseline.py", "tier2"]
    import baseline as BS                              # noqa: E402  (memory without the held-out episodes)
import tok                                             # noqa: E402

z = tok.load("tier2")
H, P, C, A = tok.build(z)
tr, te = tok.split(z)
fs = int(np.flatnonzero(P[0] == 71)[0])              # the front's slot (near places are fixed slots)
cand = np.flatnonzero(te & (z["act"] == 3) & (H[:, -1] == EMPTY) & (H[:, fs] >= 0))
Q = np.random.default_rng(0).choice(cand, min(NQ, len(cand)), replace=False)
out = {}

if MODE == "v20":
    T = BS.T
    W, info = T.setup(2, lambda m: None)
    APP = np.asarray(T.Kd.APP, np.int64)
    for col, b in BOXES.items():
        res = []
        for i in Q:
            hs = set(APP[np.flatnonzero(z["pres"][i])].tolist()) | {b}
            c, (fa, ha) = W.outcome(3, b, EMPTY, int(W.ctx_of(np.array(sorted(hs)))))
            res.append(ha)
        res = np.array([-1 if h is None else h for h in res])
        out[col] = {"hand holds that box": round(float((res == b).mean()), 4),
                    "other results": {str(k): int(v) for k, v in zip(*np.unique(res[res != b], return_counts=True))}}
        print("version 20", col, out[col], flush=True)
    (tok.RUNS / "cf_v20.json").write_text(json.dumps(out, indent=1) + "\n")
else:
    import torch
    import arm_a as AA
    import arm_b as AB
    net = AA.StateNet().to(AA.dev)
    net.load_state_dict(torch.load(tok.RUNS / "arm_a.pt")["net"])
    net.eval()
    arr = torch.as_tensor(z["arr"], device=AA.dev)
    known = np.unique(z["app"])
    K = arr[torch.as_tensor(known, device=AA.dev)]
    # arm B: pick up's exemplars on the training episodes (as arm_b.py)
    near_n = int((np.abs(z["wh"][:169]).sum(1) <= tok.DMAX).sum()) - 1
    F, r, c = AB.fields(H, P, near_n)
    ch, af, w = C[r, c], A[r, c], z["w"][r].astype(np.float64)
    s = np.flatnonzero(tr[r] & (z["act"][r] == 3))
    u, inv = np.unique(F[s], axis=0, return_inverse=True)
    inv = inv.ravel()
    bc = lambda x: np.bincount(inv, weights=x, minlength=len(u))
    arm = AB.Arm(z["arr"], u, bc(w[s] * ch[s]), bc(w[s]), bc(ch[s].astype(float)), bc((~ch[s]).astype(float)))
    arm.admit(lambda m: print(m, flush=True))
    sc = s[ch[s]]
    rep = {"arm A (network)": {}, "arm B (exemplars)": {}}
    for col, b in BOXES.items():
        Hq = H[Q].copy()
        Hq[:, fs] = b
        mask = Hq >= 0
        with torch.no_grad():
            Ht = torch.as_tensor(Hq, device=AA.dev)
            X = arr[Ht.clamp_min(0)] * torch.as_tensor(mask, device=AA.dev)[..., None]
            logit, after, ptr, g = net(X, torch.as_tensor(P[Q], device=AA.dev), torch.as_tensor(mask, device=AA.dev),
                                       torch.full((len(Q),), 3, device=AA.dev))
            snap = known[torch.cdist(after[:, -1], K, p=1).argmin(1).cpu().numpy()]
            pa = torch.sigmoid(logit[:, -1]).cpu().numpy()
        ok = (snap == b) & (pa > 0.5)
        rep["arm A (network)"][col] = {"hand holds that box": round(float(ok.mean()), 4),
                                       "P(hand changes) mean": round(float(pa.mean()), 4)}
        Fq, _, _ = AB.fields(Hq, P[Q], near_n)
        hq = Fq[Fq[:, 1] == tok.HAND]                  # the hand's instance per query
        pb, tb = arm.p_change(hq, totals=True)
        res, ways, src = AB.results(arm, hq, F[sc], w[sc], af[sc], known, z["arr"])
        okb = (res == b) & (pb > 0.5)
        rep["arm B (exemplars)"][col] = {"hand holds that box": round(float(okb.mean()), 4),
                                         "P(hand changes) mean": round(float(pb.mean()), 4)}
        al = json.loads((tok.RUNS / "c_alpha.json").read_text())["tier2"]["3"]
        pcc = (tb[:, 0] + al * pa) / (tb[:, 1] + al)
        resc = np.where(res >= 0, res, snap)
        okc = (resc == b) & (pcc > 0.5)
        rep.setdefault("arm C (B with A as prior)", {})[col] = {"hand holds that box": round(float(okc.mean()), 4),
                                                               "P(hand changes) mean": round(float(pcc.mean()), 4)}
        print(col, rep["arm A (network)"][col], rep["arm B (exemplars)"][col], rep["arm C (B with A as prior)"][col],
              flush=True)
    rep["queries"] = int(len(Q))
    (tok.RUNS / "cf_arms.json").write_text(json.dumps(rep, indent=1) + "\n")
