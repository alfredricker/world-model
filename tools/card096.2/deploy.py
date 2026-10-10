"""Card 096.2 for the agent: arm A by event over tiers 1-3 and the decoy with nothing held out
(runs/096.2/arm_a_deploy.pt), and alpha per tier and action fitted on all held-out events of sleep.py's run with its
event3 arm (runs/096.2/alpha.json; tier 1, with 8 held-out events, takes tier 2's).

  bin/prun python tools/card096.2/deploy.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card096.2"))
import sleep as SL                                     # noqa: E402

rng = np.random.default_rng(96)
worlds = ("tier1", "tier2", "decoy", "tier3")
held = {w: SL.held_out(w, rng) for w in worlds}         # the same draw as sleep.py
net3 = SL.AA.StateNet()
net3.load_state_dict(torch.load(SL.OUT / "arm_a_event3.pt", map_location="cpu")["net"])
net3.eval()
alpha = {}
for tier in (2, 3):
    evs, known, arr, _ = SL.base(tier)
    alpha[f"tier{tier}"] = {}
    for a in (3, 4, 5):
        Eh = [e for e in evs if e["key"][0] == a and e["key"] in held[f"tier{tier}"]]
        pas = [SL.FS.arm_a(net3, arr, known, a, *e["key"][1:]) for e in Eh]
        alpha[f"tier{tier}"][str(a)] = SL.fit_alpha(Eh, pas)
alpha["tier1"] = dict(alpha["tier2"])
(SL.OUT / "alpha.json").write_text(json.dumps(alpha, indent=1) + "\n")
print("alpha", alpha, flush=True)
SL.train("deploy", worlds, {w: set() for w in worlds}, True)
