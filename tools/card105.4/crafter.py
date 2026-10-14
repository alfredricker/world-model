"""Card 105.4 in Crafter: conditions read from card 105.2's transition model by intervention.

Events on the held-out worlds: makes that made a tool (the curriculum's worlds 8-9), and moves that succeeded or were
blocked (all four test sources). For a make, the outcome asked is whether the tool's carried slot changes; for a move,
where the agent will be. Every view tile j is replaced, one at a time, by each other kind's vector seen in the same
view (one tile of each kind; the agent's tile is left alone); for a make, every carried slot is also replaced by an
empty slot's vector from the same state (the item used up). j is a condition if some replacement changes the
model's outcome. Scoring only: the truth is Crafter's own rule under the same replacement (data.yaml): a tool is made
when the table (and for iron tools the furnace) is within the 3 x 3 around the agent and the materials are carried;
a move succeeds into grass, sand, path or lava with no creature on it. Only events whose unaltered outcome the model
predicts right are asked. Baseline: "the 3 x 3 around the agent, and every carried slot".

  bin/prun python tools/card105.4/crafter.py --net runs/105.2/tokens_Rb_s1021_net.pt [--per 300]
      → runs/105.4/crafter_<net name>.json
  [--origins pose]: card 105.10's models, the view placed by the agent's own pose → crafter_105.10_<net name>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
NET = Path(arg("--net", "runs/105.2/tokens_Rb_s1021_net.pt"))
PER = int(arg("--per", 300))
TAU = float(arg("--tau", "nan")) if "--tau" in sys.argv else None
DIAG_ON = "--diag" in sys.argv   # card 105.8's models: read places by probability
ORIGINS = arg("--origins", None)  # card 105.10: "pose", the agent's own pose places the view
sys.argv = ["run.py"] + (["--origins", ORIGINS] if ORIGINS else [])
sys.path.insert(0, str(ROOT / "tools" / "card105.2"))
import run as R                                           # noqa: E402

dev = "cuda"
OUT = ROOT / "runs" / "105.4"
GRASS, SAND, PATH, LAVA, TABLE, FURNACE = 2, 5, 4, 7, 11, 12
WALK = (GRASS, SAND, PATH, LAVA)
NEAR9 = [r * 9 + c for r in (2, 3, 4) for c in (3, 4, 5)]          # the 3 x 3 around the agent (tile 31)
SLOT = {"wood": 68, "stone": 69, "coal": 70, "iron": 71}
NEEDS = {11: ({"wood": 1}, (TABLE,)), 12: ({"wood": 1, "stone": 1}, (TABLE,)),
         13: ({"wood": 1, "coal": 1, "iron": 1}, (TABLE, FURNACE)), 14: ({"wood": 1}, (TABLE,)),
         15: ({"wood": 1, "stone": 1}, (TABLE,)), 16: ({"wood": 1, "coal": 1, "iron": 1}, (TABLE, FURNACE))}
STEP = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def made_by_rule(view, counts, a):
    uses, near = NEEDS[a]
    return all(k in view[NEAR9] for k in near) and all(counts[SLOT[m]] >= n for m, n in uses.items())


from collections import Counter, defaultdict
DIAG = defaultdict(Counter)


def main():
    t0 = time.monotonic()
    rng = np.random.default_rng(1021)
    tr, te = R.load(R.TRAIN, "tokens"), R.load(R.TEST, "tokens")
    net = R.TransitionModel(32, 17, arm=NET.name.split("_")[1]).to(dev)
    net.load_state_dict(torch.load(ROOT / NET))
    net.eval()
    Pt, plab = R.decoder(tr, "tokens")
    agent_protos = torch.as_tensor(plab[:, 31] >= 1300, device=dev)
    T, act = te["t"], te["act"][te["t"]]
    lab0 = te["lab"][T]
    inv = (lab0[:, 63:79] - 100) % 10                                   # carried counts (evaluator, scoring only)
    counts = np.zeros((len(T), R.NT), np.int64)
    counts[:, 63:79] = inv

    def outcome(kind, i, V):
        n = len(V)
        _, dist, valid, code, code0 = R.predict(net, te, np.full(n, i), Pt, plab, V0=V)
        if kind == "make":
            sl = R.MAKE_SLOT[int(act[i])]
            return code[:, sl] != code0[:, sl]
        if TAU is not None:                                              # the tile where an agent code is most probable
            lg = -dist[:, :63] ** 2 / TAU
            lp = torch.logsumexp(lg[..., agent_protos], -1) - torch.logsumexp(lg, -1)
            lp[~torch.as_tensor(valid[:, :63], device=dev)] = -float("inf")
            return lp.argmax(1).cpu().numpy()
        da = dist[..., agent_protos].min(-1).values[:, :63]
        da[~torch.as_tensor(valid[:, :63], device=dev)] = float("inf")
        return da.argmin(1).cpu().numpy()

    # events
    _, _, _, code_all, code0_all = (None,) * 5
    made = np.zeros(len(T), bool)
    for s in range(0, len(T), 2048):
        i = np.arange(s, min(s + 2048, len(T)))
        _, _, _, code, code0 = R.predict(net, te, i, Pt, plab)
        sl = np.array([R.MAKE_SLOT.get(int(a), 73) for a in act[i]])
        lab1 = te["lab"][T[i] + 1]
        made[i] = np.isin(act[i], list(R.MAKE_SLOT)) & (lab1[np.arange(len(i)), sl] != lab0[i, sl])
    d = te["pos_true"][T + 1] - te["pos_true"][T]
    mv = np.isin(act, R.MOVES)
    blocked = mv & (d == 0).all(1)
    pick = lambda m: sorted(rng.choice(np.flatnonzero(m), min(PER, int(m.sum())), replace=False))
    evs = {"make (a tool made)": pick(made & (te["src"][T] < 2)), "move, blocked": pick(blocked),
           "move, succeeded": pick(mv & ~blocked)}
    rep = {"net": str(NET)}
    for name, lst in evs.items():
        kind = "make" if name.startswith("make") else "move"
        tp = fp = fn = btp = bfp = bfn = exact = asked = wrong = 0
        sizes = []
        for i in lst:
            a = int(act[i])
            view = lab0[i, :63]
            V0 = te["V"][T[i]].to(dev).float()
            base = outcome(kind, i, V0[None])[0]
            if kind == "make":
                truth0 = made_by_rule(view, counts[i], a)
            else:
                dx, dy = STEP[a]
                tgt = 31 + dx + 9 * dy
                truth0 = tgt if view[tgt] in WALK else 31
            if (kind == "make" and base != truth0) or (kind == "move" and base != truth0):
                wrong += 1
                continue
            asked += 1
            reps = {}
            for q in range(63):
                if q != 31 and view[q] >= 0 and view[q] not in reps:
                    reps[int(view[q])] = q
            empty = [q for q in range(67, 79) if inv[i, q - 63] == 0]
            var, js, tc = [], [], np.zeros(R.NT, bool)
            for j in range(63):
                if j == 31 or view[j] < 0:
                    continue
                for k, q in reps.items():
                    if k == view[j]:
                        continue
                    v2 = view.copy()
                    v2[j] = k
                    if kind == "make":
                        flips = made_by_rule(v2, counts[i], a) != truth0
                    else:
                        flips = (tgt if v2[tgt] in WALK else 31) != truth0
                    tc[j] |= flips
                    vj = V0.clone()
                    vj[j] = V0[q]
                    var.append(vj)
                    js.append(j)
            if kind == "make" and empty:
                sl = R.MAKE_SLOT[a]
                for j in range(67, 79):
                    if j == sl or inv[i, j - 63] == 0:
                        continue
                    c2 = counts[i].copy()
                    c2[j] = 0
                    tc[j] |= made_by_rule(view, c2, a) != truth0
                    vj = V0.clone()
                    vj[j] = V0[empty[0]]
                    var.append(vj)
                    js.append(j)
            mc = np.zeros(R.NT, bool)
            ks_of = []
            for j in range(63):
                if j == 31 or view[j] < 0:
                    continue
                ks_of += [(j, k) for k in reps if k != view[j]]
            if kind == "make" and empty:
                ks_of += [(j, "empty") for j in range(67, 79) if j != R.MAKE_SLOT[a] and inv[i, j - 63] > 0]
            for s in range(0, len(var), 256):
                o = outcome(kind, i, torch.stack(var[s:s + 256]))
                for j, x, jk in zip(js[s:s + 256], o, ks_of[s:s + 256]):
                    mc[j] |= x != base
                    if x != base:
                        role = ("carried" if j >= 63 else "3x3" if j in NEAR9 else
                                f"ring {max(abs(j % 9 - 4), abs(j // 9 - 3))}")
                        DIAG[name][("flip", role, str(jk[1]))] += 1
            near = np.zeros(R.NT, bool)
            near[NEAR9] = True
            near[31] = False
            near[67:79] = kind == "make"
            for j in np.flatnonzero(mc & ~tc):
                DIAG[name][("extra", "3x3" if j in NEAR9 else "carried" if j >= 63 else
                            f"ring {max(abs(j % 9 - 4), abs(j // 9 - 3))}", int(view[j]) if j < 63 else j)] += 1
            for j in np.flatnonzero(~mc & tc):
                DIAG[name][("missed", int(view[j]) if j < 63 else j)] += 1
                if j < 63:
                    DIAG[name][("missed with replacements available", tuple(sorted(k for k in reps if k != view[j])))] += 1
            tp += int((mc & tc).sum()); fp += int((mc & ~tc).sum()); fn += int((~mc & tc).sum())
            btp += int((near & tc).sum()); bfp += int((near & ~tc).sum()); bfn += int((~near & tc).sum())
            exact += int((mc == tc).all())
            sizes.append(int(tc.sum()))
        f = lambda x, y: round(x / max(x + y, 1), 4)
        rep[name] = {"events asked (model right on the unaltered outcome)": asked,
                     "events the model got wrong (not asked)": wrong,
                     "precision": f(tp, fp), "recall": f(tp, fn),
                     "events with exactly the true conditions": round(exact / max(asked, 1), 4),
                     "baseline '3 x 3 and carried slots': precision": f(btp, bfp), "baseline: recall": f(btp, bfn),
                     "true conditions per event (mean)": round(float(np.mean(sizes)), 2) if sizes else None}
        print(name, rep[name], f"({time.monotonic() - t0:.0f} s)", flush=True)
        if DIAG_ON:
            print("  diag:", [(k, v) for k, v in DIAG[name].most_common(30) if k[0] in ("extra", "flip", "missed", "missed with replacements available")][:20], flush=True)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"crafter_{NET.parent.name + '_' if ORIGINS else ''}{NET.stem}.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
