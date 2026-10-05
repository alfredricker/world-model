"""Card 052, step T: does a saved transition model know the colour cases? On tries collected as the probe is (no
tint, play starts at 0.5, stratified by kind), the share of tries whose predicted after-pieces fall in the real
after-tile's code tuple (front and held), per kind of try; the locked-door tries first.

  bin/prun python tools/card052/diag/transition_colour.py runs/052/sT_T0_399.pt [gates 0|1]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import effect as EF                                    # noqa: E402
import predflip as PF                                  # noqa: E402


def main():
    ck = torch.load(sys.argv[1])
    EF.DR.GN.TINT = 0.0
    EF.START_SHARE = 0.5
    PF.COLLECT_STEPS = 150000
    _, probe = PF.collect_strat(399 + 555)
    EF.INTERACTION = "transition"
    EF.GATES = len(sys.argv) > 2 and sys.argv[2] == "1"
    EF.DR.Encoder.EMA = 0.99
    enc = EF.Encoder(399)
    enc.enc.load_state_dict(ck["enc"])
    enc.trans.load_state_dict(ck["trans"])
    with torch.no_grad():
        enc.books.copy_(ck["books"].to(enc.dev))
        if ck.get("gates") is not None:
            enc.gates.copy_(ck["gates"].to(enc.dev))
    res = defaultdict(list)
    with torch.no_grad():
        for ai, a in enumerate(EF.ACTS):
            p = probe[a]
            zf, zh = enc.pieces(enc.x([t[0] for t in p])), enc.pieces(enc.x([t[1] for t in p]))
            pf, ph = enc.predict_after(ai, zf, zh)
            _, cf = enc.quant(pf)
            _, ch = enc.quant(ph)
            rf, _ = enc.codes(np.stack([t[4] for t in p]))
            rh, _ = enc.codes(np.stack([t[5] for t in p]))
            ok = (cf.cpu().numpy() == rf).all(1) & (ch.cpu().numpy() == rh).all(1)
            okf = (cf.cpu().numpy() == rf).all(1)
            for t, o, of in zip(p, ok, okf):
                res[PF.kind_of(a, t[3])].append((o, of, t[2]))
    for name, q in PF.COLOUR.items():
        v = res.get(q, [])
        print(f"{name}: {len(v)} tries; effect known (front and held) {np.mean([x[0] for x in v]):.2f}, "
              f"front only {np.mean([x[1] for x in v]):.2f}; outcomes {sorted(set(x[2] for x in v))}")
    allv = [x for v in res.values() for x in v]
    print(f"all {len(allv)} tries: {np.mean([x[0] for x in allv]):.3f}; worst kinds:",
          sorted(((round(float(np.mean([x[0] for x in v])), 2), str(q)) for q, v in res.items() if len(v) >= 5))[:6])


if __name__ == "__main__":
    main()
