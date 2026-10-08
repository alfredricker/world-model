"""Card 087's gate, parts (ii) and (iii), and version 18's per-tile baseline on part (i); in the agent's own World.

  <version 18's flags> WM_STORED_FITS=1 bin/prun python tools/card087/gate2.py none    decoy memory, nothing removed:
        (iii) and the baseline on (i)
  ... gate2.py red        a fold (the hue's openings removed, card 086): (ii), the tile toggling the fold's door
                          with its key becomes, as card 038's ways imagine it
  ... gate2.py tier2      tier 2's memory: (iii), and (ii) for its locked door (card 068's blue door)
Writes runs/087/gate2_<mode>.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODE = sys.argv[1]
TIER = 2 if MODE == "tier2" else 1
if TIER == 2:
    sys.argv = ["run.py", "tiers", "--tier", "2", "--online", "0", "--memory", "066"]
else:
    sys.argv = ["run.py", "decoy", "--fold", MODE if MODE != "none" else "red", "--hold", "opening", "--online", "0",
                "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402
sys.path.insert(0, str(ROOT / "tools" / "card087"))
import props as PR                                     # noqa: E402
import torch                                           # noqa: E402

T, R = RUN.T, RUN.R
VP = T.VP
CR = sys.modules["code_recall"]
t00 = time.monotonic()
log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)

if TIER == 1:                                          # as card 086's gate
    T.ENVS[1] = R.register()
    T.OUT = ROOT / "runs" / "070" / "decoy_memory066"
    R.DECOY.update(door=list(T.HUES), decoy=list(T.HUES))
    _memory = T.memory

    def memory(tier, pool=None):
        D, stats = _memory(tier, pool)
        if MODE == "none":
            return D, stats
        f = D["ego0"][:, T.FRONT].astype(np.int64) // 5 * 5
        h = D["ego0"][:, T.HELD].astype(np.int64) // 5 * 5
        drop = (D["act"] == VP.TOG) & (f == T.KR.code(("door", MODE, 0))) & (h == T.KR.code(("key", MODE)))
        keep_t = ~drop[np.isin(D["act"], T.INTER)]
        D = {k: (v[keep_t] if k in ("pres", "stale") else v[~drop]) for k, v in D.items()}
        return D, {**stats, "opening_tries_removed": int(drop.sum())}

    T.memory = memory

W, info = T.setup(TIER, log)
S = W.S
nets = []
for sd in torch.load(PR.OUT / "ensemble.pt"):
    n = PR.Net().cuda()
    n.load_state_dict(sd)
    nets.append(n.eval())
app = lambda o: int(T.Kd.APP[T.KR.code(o)])
_pl = T.S7.Plan047(W)                                  # a start situation's view context and empty hand (card 086)
_env = T.make(TIER)
_env.reset(seed=T.SEED_TEST + (999 if TIER == 1 else 1000 * TIER))
_codes = T.now_codes(_env)
_, _st, _, _ = _pl.observe(None, T.PV.crop(T.Kd.APP[_codes], _codes))
CTX, EMPTY = int(_pl.ctx_id(_st[0], _st[1])), int(_pl.facts[_st[0]][VP.HELD])
WALK = {VP.CHANGED: 0, VP.UNCHANGED: 1, VP.ENDED: 2}
NAMES = ["moved", "blocked", "ended"]
res = {"note": "Card 087 gate, parts (ii) and (iii); tools/card087/gate2.py", "mode": MODE}


def v18_walk(h):
    return NAMES[WALK[W.move_cat(VP.FWD, int(h))]]


def net_props(z):
    Wp, P, Tp = PR.predict(nets, np.asarray(z)[None])
    return {"walk": NAMES[int(Wp.mean(0)[0].argmax())], "p_moved": round(float(Wp.mean(0)[0][0]), 4),
            "pick": round(float(P.mean(0)[0]), 4), "tog": round(float(Tp.mean(0)[0]), 4),
            "spread_walk": round(float(Wp.std(0)[0].max()), 4)}


# -- (ii): the tile a toggled door becomes, as card 038's ways imagine it
def imagined(door, key):
    kd = W.kinds[VP.TOG]
    q = (app(door), app(key), CTX)
    real = app((door[0], door[1], 2))                   # the real open door of that colour
    out = {}
    for c in (1, 3):                                    # front changed; front and held changed
        try:
            fa = kd.result(q, c)[0]
        except Exception as e:                          # noqa: BLE001
            out[c] = {"error": repr(e)[:200]}
            continue
        if fa is None:
            continue
        out[c] = {"L1 to the real open door": round(float(np.abs(S.arr[fa] - S.arr[real]).sum()), 4),
                  "version 18 walk": v18_walk(fa), "network": net_props(S.arr[fa]),
                  "same identity code as the open door": bool(CR.code_id(S, fa) == CR.code_id(S, real))}
    out["real open door"] = {"version 18 walk": v18_walk(real), "network": net_props(S.arr[real])}
    return out


if MODE not in ("none", "tier2"):
    res["(ii)"] = imagined(("door", MODE, 0), ("key", MODE))
    log(json.dumps(res["(ii)"]))
if MODE == "tier2":
    env = T.make(2)
    env.reset(seed=T.SEED_TEST + 2000)
    g = env.unwrapped.grid
    door = next(g.get(i, j) for i in range(g.width) for j in range(g.height)
                if g.get(i, j) is not None and g.get(i, j).type == "door")
    res["(ii)"] = {door.color: imagined(("door", door.color, 0), ("key", door.color))}
    for c in T.HUES:                                    # card 068: every locked door colour in tier 2's memory
        res["(ii)"][c] = imagined(("door", c, 0), ("key", c))
    log(json.dumps(res["(ii)"]))

# -- version 18's per-tile reading of part (i)'s tiles (vectors as the agent encodes them, added as new handles)
if MODE == "none":
    zn = np.load(PR.OUT / "named_vectors.npz")
    rows = []
    for z, k, c in zip(zn["z"], zn["kind"], zn["colour"]):
        h = S.add(np.asarray(z, np.float64))
        truth = PR.labels(str(k))
        r = {"kind": str(k), "colour": str(c), "truth walk": NAMES[truth[0]], "version 18 walk": v18_walk(h),
             "network walk": net_props(z)["walk"]}
        o = {"floor": None, "wall": "wall", "goal": "goal", "open": ("door", str(c), 2), "closed": ("door", str(c), 1),
             "locked": ("door", str(c), 0), "key": ("key", str(c)), "ball": ("ball", str(c)), "box": ("box", str(c))}[str(k)]
        try:                                            # the agent's own catalogue tile, where it has one
            hc = app(o)
            r.update({"L1 to the catalogue tile": round(float(np.abs(S.arr[hc] - z).sum()), 4),
                      "version 18 walk, catalogue tile": v18_walk(hc), "network walk, catalogue tile": net_props(S.arr[hc])["walk"]})
        except Exception:                               # noqa: BLE001  (no catalogue tile: the three held-out hues)
            pass
        rows.append(r)
    res["(i) version 18 per tile"] = {
        "walk right": f"{sum(r['version 18 walk'] == r['truth walk'] for r in rows)} of {len(rows)}",
        "network walk right": f"{sum(r['network walk'] == r['truth walk'] for r in rows)} of {len(rows)}",
        "version 18 wrong": [(r["kind"], r["colour"], r["version 18 walk"]) for r in rows
                             if r["version 18 walk"] != r["truth walk"]],
        "catalogue tiles": f"{sum('L1 to the catalogue tile' in r for r in rows)}",
        "largest L1 to the catalogue tile": max([r.get("L1 to the catalogue tile", 0) for r in rows]),
        "version 18 right on catalogue tiles": f"{sum(r.get('version 18 walk, catalogue tile') == r['truth walk'] for r in rows)}",
        "network right on catalogue tiles": f"{sum(r.get('network walk, catalogue tile') == r['truth walk'] for r in rows)}",
        "rows": rows}
    log(json.dumps({k: v for k, v in res["(i) version 18 per tile"].items() if k != "rows"}))

# -- (iii): network against recall's neighbours, each identity code left out, on memory's own labels per identity
if MODE in ("none", "tier2"):
    out = {}
    for a, prop in ((VP.FWD, "walk"), (VP.PICK, "pick"), (VP.TOG, "tog")):
        kd = W.kinds[a]
        fronts = np.array([k[0] for k in kd.keys])
        if prop == "pick":                              # empty-handed pick-ups only
            sel = np.array([len(k) > 1 and k[1] == EMPTY for k in kd.keys])
        else:
            sel = np.ones(len(kd.keys), bool)
        fronts, counts = fronts[sel], kd.counts[sel]
        ids = np.array([CR.code_id(S, h) for h in fronts])
        lab = {}                                        # per identity: memory's label
        for i in np.unique(ids):
            c = counts[ids == i].sum(0)
            if prop == "walk":
                lab[i] = WALK.get(int(np.argmax(c)), 1)
            else:
                lab[i] = int(c[1:].sum() > 0)
        uid = np.array(sorted(lab))
        rep = {}
        for i in uid:
            rep[i] = fronts[ids == i][np.argmax(counts[ids == i].sum(1))]
        Z = S.arr[[rep[i] for i in uid]]
        y = np.array([lab[i] for i in uid])
        lam = np.asarray(kd.lam, np.float64)[:kd.D]
        Dm = np.abs(Z[:, None] - Z[None]) @ lam
        Kw = np.exp(-Dm)
        np.fill_diagonal(Kw, 0.0)                      # each identity left out
        if prop == "walk":
            Y = np.eye(3)[y]
            Pn = (Kw @ Y + 1 / 3) / (Kw.sum(1, keepdims=True) + 1)
            Wp, _, _ = PR.predict(nets, Z)
            Pf = Wp.mean(0)
        else:
            pn = (Kw @ y + 0.5) / (Kw.sum(1) + 1)
            Pn = np.stack([1 - pn, pn], 1)
            _, P, Tp = PR.predict(nets, Z)
            pf = (P if prop == "pick" else Tp).mean(0)
            Pf = np.stack([1 - pf, pf], 1)
        ll = lambda Pm: float(np.log(np.clip(Pm[np.arange(len(y)), y], 1e-9, 1)).mean())
        out[prop] = {"identities": len(uid), "neighbours right": f"{int((Pn.argmax(1) == y).sum())} of {len(y)}",
                     "network right": f"{int((Pf.argmax(1) == y).sum())} of {len(y)}",
                     "neighbours log-lik": round(ll(Pn), 4), "network log-lik": round(ll(Pf), 4),
                     "network better": ll(Pf) > ll(Pn)}
    res["(iii)"] = out
    log(json.dumps(out))

res["setup"] = {k: v for k, v in info.items() if k in ("model_seconds",)}
(PR.OUT / f"gate2_{MODE}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
