"""Card 106 in Crafter: card 105.4's conditions by intervention on card 105.10's models (the view placed by the agent's
own pose), with two filters on which replaced states count (one pass gives every arm).

  H (the world could be in it): under the model's own prediction for doing nothing (action 0), the event's token (the
    tool's carried slot for a make; the move's target tile) and the replaced token keep their codes.
  K (the model knows it): the copy (another seed) predicts the same outcome (the tool made or not; the agent's next
    tile, read by probability with the copy's own tau).
  Arms: none (105.4), H, K, HK. The truth is 105.4's: Crafter's rule (data.yaml) on the replaced view.

  bin/prun python tools/card106/crafter.py --net runs/105.10/tokens_Rb_s1021_net.pt --copy runs/105.10/tokens_Rb_s7_net.pt
      → runs/106/crafter_<net name>.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
NET, COPY = Path(arg("--net", "runs/105.10/tokens_Rb_s1021_net.pt")), Path(arg("--copy", "runs/105.10/tokens_Rb_s7_net.pt"))
PER = int(arg("--per", 300))
sys.argv = ["run.py", "--origins", "pose"]
sys.path.insert(0, str(ROOT / "tools" / "card105.2"))
import run as R                                           # noqa: E402

dev = "cuda"
OUT = ROOT / "runs" / "106"
ARMS = ("none", "H", "K", "HK")
GRASS, SAND, PATH, LAVA, TABLE, FURNACE = 2, 5, 4, 7, 11, 12
WALK = (GRASS, SAND, PATH, LAVA)
NEAR9 = [r * 9 + c for r in (2, 3, 4) for c in (3, 4, 5)]
SLOT = {"wood": 68, "stone": 69, "coal": 70, "iron": 71}
NEEDS = {11: ({"wood": 1}, (TABLE,)), 12: ({"wood": 1, "stone": 1}, (TABLE,)),
         13: ({"wood": 1, "coal": 1, "iron": 1}, (TABLE, FURNACE)), 14: ({"wood": 1}, (TABLE,)),
         15: ({"wood": 1, "stone": 1}, (TABLE,)), 16: ({"wood": 1, "coal": 1, "iron": 1}, (TABLE, FURNACE))}
STEP = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def made_by_rule(view, counts, a):
    uses, near = NEEDS[a]
    return all(k in view[NEAR9] for k in near) and all(counts[SLOT[m]] >= n for m, n in uses.items())


def load(path):
    net = R.TransitionModel(32, 17, arm=path.name.split("_")[1]).to(dev)
    net.load_state_dict(torch.load(ROOT / path))
    tau = json.load(open(ROOT / path.parent / path.name.replace("_net.pt", ".json")))["tau"]
    return net.eval(), tau


@torch.no_grad()
def predict(net, D, i, Pt, V0, a=None):
    """Codes of the next state and of the input, as R.predict; a replaces the action (no-op = 0)."""
    V0, _, valid, P, F, M, act = R.batch(D, i, V0)
    if a is not None:
        act = torch.full_like(act, a)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        nxt, _ = net(V0, P, F, M, act)
    dist = torch.cdist(nxt.float().reshape(-1, Pt.shape[1]), Pt).reshape(len(i), R.NT, -1)
    code0 = torch.cdist(V0.reshape(-1, Pt.shape[1]), Pt).argmin(1).reshape(len(i), R.NT).cpu().numpy()
    return dist, valid, dist.argmin(-1).cpu().numpy(), code0


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1021)
    tr, te = R.load(R.TRAIN, "tokens"), R.load(R.TEST, "tokens")
    (net, tau), (net2, tau2) = load(NET), load(COPY)
    Pt, plab = R.decoder(tr, "tokens")
    agent_protos = torch.as_tensor(plab[:, 31] >= 1300, device=dev)
    T, act = te["t"], te["act"][te["t"]]
    lab0 = te["lab"][T]
    inv = (lab0[:, 63:79] - 100) % 10                                   # carried counts (evaluator, scoring only)
    counts = np.zeros((len(T), R.NT), np.int64)
    counts[:, 63:79] = inv

    def outcome(m, tu, kind, i, V):
        dist, valid, code, code0 = predict(m, te, np.full(len(V), i), Pt, V)
        if kind == "make":
            sl = R.MAKE_SLOT[int(act[i])]
            return code[:, sl] != code0[:, sl]
        lg = -dist[:, :63] ** 2 / tu                                    # the tile where an agent code is most probable
        lp = torch.logsumexp(lg[..., agent_protos], -1) - torch.logsumexp(lg, -1)
        lp[~torch.as_tensor(valid[:, :63], device=dev)] = -float("inf")
        return lp.argmax(1).cpu().numpy()

    made = np.zeros(len(T), bool)
    for s in range(0, len(T), 2048):
        i = np.arange(s, min(s + 2048, len(T)))
        sl = np.array([R.MAKE_SLOT.get(int(a), 73) for a in act[i]])
        lab1 = te["lab"][T[i] + 1]
        made[i] = np.isin(act[i], list(R.MAKE_SLOT)) & (lab1[np.arange(len(i)), sl] != lab0[i, sl])
    d = te["pos_true"][T + 1] - te["pos_true"][T]
    mv = np.isin(act, R.MOVES)
    blocked = mv & (d == 0).all(1)
    pick = lambda m: sorted(rng.choice(np.flatnonzero(m), min(PER, int(m.sum())), replace=False))
    evs = {"make (a tool made)": pick(made & (te["src"][T] < 2)), "move, blocked": pick(blocked),
           "move, succeeded": pick(mv & ~blocked)}
    rep = {"net": str(NET), "copy": str(COPY)}
    for name, lst in evs.items():
        kind = "make" if name.startswith("make") else "move"
        st = {a: Counter() for a in ARMS}
        set_aside = {a: Counter() for a in ("H", "K")}
        wrong = disagree = 0
        for i in lst:
            a = int(act[i])
            view = lab0[i, :63]
            V0 = te["V"][T[i]].to(dev).float()
            base = outcome(net, tau, kind, i, V0[None])[0]
            if kind == "make":
                truth0 = made_by_rule(view, counts[i], a)
                tok = R.MAKE_SLOT[a]
            else:
                dx, dy = STEP[a]
                tok = 31 + dx + 9 * dy
                truth0 = tok if view[tok] in WALK else 31
            if base != truth0:
                wrong += 1
                continue
            base_known = outcome(net2, tau2, kind, i, V0[None])[0] == base
            disagree += int(not base_known)
            reps = {}
            for q in range(63):
                if q != 31 and view[q] >= 0 and view[q] not in reps:
                    reps[int(view[q])] = q
            empty = [q for q in range(67, 79) if inv[i, q - 63] == 0]
            var, js, kinds, tc = [], [], [], np.zeros(R.NT, bool)
            for j in range(63):
                if j == 31 or view[j] < 0:
                    continue
                for k, q in reps.items():
                    if k == view[j]:
                        continue
                    v2 = view.copy()
                    v2[j] = k
                    tc[j] |= (made_by_rule(v2, counts[i], a) != truth0) if kind == "make" else \
                        ((tok if v2[tok] in WALK else 31) != truth0)
                    vj = V0.clone()
                    vj[j] = V0[q]
                    var.append(vj)
                    js.append(j)
                    kinds.append(k)
            if kind == "make" and empty:
                for j in range(67, 79):
                    if j == tok or inv[i, j - 63] == 0:
                        continue
                    c2 = counts[i].copy()
                    c2[j] = 0
                    tc[j] |= made_by_rule(view, c2, a) != truth0
                    vj = V0.clone()
                    vj[j] = V0[empty[0]]
                    var.append(vj)
                    js.append(j)
                    kinds.append("empty slot")
            js = np.array(js, dtype=np.int64)
            o, o2, keepH = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)], [np.zeros(0, bool)]   # an event may have no replacement
            for s in range(0, len(var), 256):
                Vb = torch.stack(var[s:s + 256])
                o.append(outcome(net, tau, kind, i, Vb))
                o2.append(outcome(net2, tau2, kind, i, Vb))
                _, _, code, code0 = predict(net, te, np.full(len(Vb), i), Pt, Vb, a=0)
                jj = js[s:s + 256]
                keepH.append((code[:, tok] == code0[:, tok]) & (code[np.arange(len(jj)), jj] == code0[np.arange(len(jj)), jj]))
            o, o2, keepH = np.concatenate(o), np.concatenate(o2), np.concatenate(keepH)
            keepK = (o2 == o) & base_known
            flips = o != base
            for arm, keep in (("none", np.ones(len(js), bool)), ("H", keepH), ("K", keepK), ("HK", keepH & keepK)):
                if arm in ("K", "HK") and not base_known:
                    continue
                mc = np.zeros(R.NT, bool)
                np.logical_or.at(mc, js[keep & flips], True)
                s_ = st[arm]
                s_["asked"] += 1
                s_["tp"] += int((mc & tc).sum()); s_["fp"] += int((mc & ~tc).sum()); s_["fn"] += int((~mc & tc).sum())
                s_["exact"] += int((mc == tc).all())
            for arm, keep in (("H", keepH), ("K", keepK)):         # which replacements each filter set aside
                for k in np.array(kinds, dtype=object)[~keep]:
                    set_aside[arm][str(k)] += 1
        f = lambda x, y: round(x / max(x + y, 1), 4)
        rep[name] = {"events the model got wrong (not asked)": wrong, "events the copies disagree on": disagree}
        for arm in ARMS:
            s_ = st[arm]
            rep[name][arm] = {"asked": s_["asked"], "precision": f(s_["tp"], s_["fp"]), "recall": f(s_["tp"], s_["fn"]),
                              "events with exactly the true conditions": round(s_["exact"] / max(s_["asked"], 1), 4)}
        for arm in ("H", "K"):
            rep[name][f"replacements {arm} set aside, by kind (codes' labels, scoring only)"] = dict(set_aside[arm].most_common(8))
        print(name, json.dumps({a: (rep[name][a]["precision"], rep[name][a]["recall"]) for a in ARMS}),
              f"({time.monotonic() - t0:.0f} s)", flush=True)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"crafter_{NET.stem}.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
