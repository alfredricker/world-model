"""Card 091's gate, "it reads memory": the router's vote with the candidates' outcomes shuffled among the candidates of
the same action (Kossen et al.'s control), against the true outcomes; combination hidden, as gate.py.

  bin/prun python tools/card091/shuffle.py tier3 runs/091/router_a.pt
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card091"))
import router as RT                                    # noqa: E402

world, path = sys.argv[1], Path(sys.argv[2])
ck = torch.load(path)
net = RT.Router().cuda()
net.load_state_dict(ck["net"])
net.eval()
D = RT.Data([world], RT.code_vectors())
rng = np.random.default_rng(0)
combo = torch.as_tensor(D.d["front"].astype(np.int64) * 1000 + D.d["held"].astype(np.int64), device="cuda")
rep = {"world": world, "router": str(path)}
with torch.no_grad():
    for a in RT.ACTS:
        r = D.rows[a]
        q = rng.choice(r, min(4000, len(r)), replace=False)
        eq, ec = RT.embed(net, D, q), RT.embed(net, D, r)
        rt = torch.as_tensor(r, device="cuda")
        oc_true = D.t["out"][rt]
        res = {}
        for name, oc in (("true outcomes", oc_true), ("shuffled outcomes", oc_true[torch.randperm(len(r), device="cuda")])):
            lls = []
            for i in range(0, len(q), 256):
                qt = torch.as_tensor(q[i:i + 256], device="cuda")
                k = torch.exp(-torch.cdist(eq[i:i + 256], ec, p=1) / net.log_tau.exp()) * D.w[rt][None]
                k = k.masked_fill(combo[qt][:, None] == combo[rt][None], 0.0)
                N = k @ torch.nn.functional.one_hot(oc, RT.NOUT[a]).float()
                al = net.log_alpha.exp()
                P = (N + al * D.prior[a][None]) / (N.sum(1, keepdim=True) + al)
                lls.append(torch.log(P[torch.arange(len(qt)), D.t["out"][qt]].clamp_min(1e-12)))
            ll = torch.cat(lls).cpu().numpy()
            wq = D.d["w"][q]
            res[name] = round(float((ll * wq).sum() / wq.sum()), 4)
        rep[str(a)] = res
        print(a, res, flush=True)
(ROOT / "runs" / "091" / f"shuffle_{world}_{path.stem}.json").write_text(json.dumps(rep, indent=1) + "\n")
