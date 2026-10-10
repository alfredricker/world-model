"""Card 096.1, the curve on events that change something and on events that change nothing, separately (version 24's
arm A and alpha).   bin/prun python tools/card096.1/split.py      → runs/096.1/split.json"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card096.2"))
import sleep as SL                                     # noqa: E402

net = SL.AA.StateNet()
net.load_state_dict(torch.load(ROOT / "runs" / "095.1" / "arm_a.pt", map_location="cpu")["net"])
net.eval()
out = {}
for tier in (2, 3):
    evs, known, arr, alpha0 = SL.base(tier)
    for a, an in SL.FS.ACTS.items():
        E = [e for e in evs if e["key"][0] == a]
        for grp, G in (("change", [e for e in E if (e["q"] > 0.5).any()]), ("no change", [e for e in E if not (e["q"] > 0.5).any()])):
            ps = [SL.FS.arm_a(net, arr, known, a, *e["key"][1:]) for e in G]
            r = {"events": len(G)}
            for n in SL.NS:
                r[f"B n={n}"] = round(float(np.mean([SL.judge(e, np.zeros(2), [-1, -1], None, n)[0] for e in G])), 3) if G else None
                r[f"C n={n}"] = round(float(np.mean([SL.judge(e, p, s, alpha0[str(a)], n)[0] for e, (p, s) in zip(G, ps)])), 3) if G else None
            r["A"] = round(float(np.mean([SL.judge(e, p, s, "A", 0)[0] for e, (p, s) in zip(G, ps)])), 3) if G else None
            out[f"tier{tier} {an} {grp}"] = r
            print(f"tier{tier} {an} {grp}: {r}", flush=True)
(ROOT / "runs" / "096.1" / os.environ.get("WM_SPLIT_OUT", "split.json")).write_text(json.dumps(out, indent=1) + "\n")
