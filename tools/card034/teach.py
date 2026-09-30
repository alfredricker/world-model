"""Card 034: recall teaches the encoder.

Card 031's encoder objective gains card 033's recall term on the things' own numbers (card 033's arm 5,
no relation part), with 16 codes per codebook: each remembered event (action, tile in front, held tile,
what changed) is predicted by a similarity-weighted vote of the other events of its world and action,
k = exp(-sum_j lambda_j |x_j - x'_j|), and the leave-one-out log-likelihood trains the encoder. Forward
moves (moved, blocked, reached the goal) join card 033's pick-ups, drops and toggles. The test: keys and
open doors of a colour never seen (yellow, purple), predicted and acted on by card 029's model over the
codes (card 031's check), in card 031's world (a) made colour-general.

Steps:
  bin/prun python tools/card034/teach.py --collect                       data, memory, reference steps
  bin/prun python tools/card034/teach.py --sweep MU [--M 16]             gate: 10 encoders at weight MU
  bin/prun python tools/card034/teach.py --main --arm 2 --mu MU --seeds 200-204 --out runs/034_arm2_a.json
  bin/prun python tools/card034/teach.py --labels --out runs/034_arm1.json
  bin/prun python tools/card034/teach.py --diag --mu MU --out runs/034_diag.json
"""
import copy
import dataclasses
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card033"))
import recall as R  # noqa: E402  (appends yellow, then imports card 031, which appends purple)
import kinds_probe as KP  # noqa: E402

C, KR, F, dl, ld, G = R.C, R.KR, R.F, R.dl, R.ld, R.G
APP0, NEW, FRONT, HELD, NV = R.APP0, R.NEW, R.FRONT, R.HELD, R.NV
FWD, PICK, DROP, TOG = C.FWD, C.PICK, C.DROP, C.TOG
ACTIONS = (FWD, PICK, DROP, TOG)
ANAME = {FWD: "forward", PICK: "pick up", DROP: "drop", TOG: "toggle"}
FKINDS = ("moved", "blocked", "reached the goal")            # forward's outcomes, in columns 0-2 of 4
MOVED, BLOCKED, GOAL = range(3)
NOTHING, SWAPPED, IN_PLACE = R.NOTHING, R.SWAPPED, R.IN_PLACE
WORLDS = C.WORLDS
COLOURS = ("yellow", "purple")
ROOT = Path(__file__).resolve().parents[2]
SMALL = "--small" in sys.argv                  # smoke test: little data, one seed, separate files
SUF = "_small" if SMALL else ""
DATA = ROOT / "runs" / f"034_data{SUF}.pkl"
MEMORY = ROOT / "runs" / f"034_memory{SUF}.pkl"
REF = ROOT / "runs" / f"034_ref{SUF}.json"
SEEDS = (200,) if SMALL else tuple(range(200, 210))
FLOOR, code = R.FLOOR, KR.code

# card 031's worlds (a) and (b), one per new colour
TESTS = {f"{tk}_{col}": (world, door_open) for col in COLOURS for tk, (world, door_open) in
         (("a", ("key", 1)), ("b", ("switch", 0)))}
C.TESTS = TESTS


# ---------------------------------------------------------------- data

def cases(layouts, tr, world, col):
    """Card 031's section 4 events, named for the colour (evaluator)."""
    s0, s1 = tr["s0"].astype(np.int64), tr["s1"].astype(np.int64)
    act = tr["act"].astype(np.int64)
    dv = np.array(ld.DIR_VEC)
    fx, fy = s0[:, 0] + dv[s0[:, 2], 0], s0[:, 1] + dv[s0[:, 2], 1]
    door = np.array([layouts[e].door for e in tr["ep"]])
    at_door = (fx == door[:, 0]) & (fy == door[:, 1])
    in_door = (s0[:, 0] == door[:, 0]) & (s0[:, 1] == door[:, 1])
    at_key = (fx == s0[:, 4]) & (fy == s0[:, 5])
    opened, closed = s0[:, 8] == 1, s0[:, 8] == 0
    moved = (s1[:, :2] != s0[:, :2]).any(1)
    c = {f"pick up the {col} key": (act == PICK) & (s0[:, 3] == 0) & (s1[:, 3] == 1),
         f"drop the {col} key": (act == DROP) & (s0[:, 3] == 1) & (s1[:, 3] == 0),
         f"forward into the {col} key (blocked)": (act == FWD) & at_key,
         f"forward into the closed {col} door (blocked)": (act == FWD) & at_door & closed,
         f"forward onto the open {col} door": (act == FWD) & at_door & opened,
         f"toggle the open {col} door (closes)": (act == TOG) & at_door & opened}
    tc = (act == TOG) & at_door & closed
    if world == "switch":
        c[f"toggle the closed {col} door, switch on (opens)"] = tc & (s0[:, 9] == 1)
        c[f"toggle the closed {col} door, switch off (stays)"] = tc & (s0[:, 9] == 0)
    else:
        c[f"toggle the closed {col} door without the {col} key (stays)"] = tc & (s0[:, 3] != 1)
        c[f"toggle the closed {col} door holding the {col} key"] = tc & (s0[:, 3] == 1)
    extra = {f"forward out of the {col} doorway (reported only)": (act == FWD) & in_door & moved,
             f"turn in the {col} doorway (reported only)": np.isin(act, (C.LEFT, C.RIGHT)) & in_door}
    return c, extra


def criterion_cases(col):
    return [f"pick up the {col} key", f"drop the {col} key", f"forward onto the open {col} door"]


def setup(episodes, heldout, n_test, n_tep, res, save, log):
    """Card 031's familiar data (card 033's setup), and card 031's worlds (a) and (b) in each new colour."""
    data, tiles, pairs = R.setup(episodes, heldout, n_test, res, save, log)
    tests = {}
    for tk, (world, door_open) in TESTS.items():
        col = tk.split("_")[1]
        t0 = time.monotonic()
        ld.make_layout = lambda size, rule, rng, col=col: dataclasses.replace(C._MAKE(size, rule, rng), door_colour=col)
        ld.start_state = C.open_start if door_open else C._START
        pool = F.pool20()
        layouts, tr, _, _ = dl.collect(pool, world, n_tep, 31, 0.0, p_uniform=1.0, seq_episodes=4)
        rng = np.random.default_rng(777)
        test = [ld.make_layout(8, world, rng) for _ in range(n_test)]
        sh, _ = G.shortest(pool, test, world)
        pool.close()
        C.patch()
        crit, extra = cases(layouts, tr, world, col)
        allc = {**crit, **extra}
        other = np.flatnonzero(~np.any(np.stack(list(allc.values())), 0))
        other = np.sort(np.random.default_rng(31).choice(other, min(20000, len(other)), replace=False))
        allc["other transitions (a sample of 20,000; reported only)"] = np.isin(np.arange(len(tr["act"])), other)
        sel = np.flatnonzero(np.any(np.stack(list(allc.values())), 0))
        pos = np.full(len(tr["act"]), -1)
        pos[sel] = np.arange(len(sel))
        tests[tk] = {"act": tr["act"][sel].astype(np.int64), "term1": tr["term1"][sel].astype(bool),
                     "ego0": C.ego_codes(tr["c0"][sel], tr["s0"][sel]).astype(np.uint8),
                     "ego1": C.ego_codes(tr["c1"][sel], tr["s1"][sel]).astype(np.uint8),
                     "cases": {k: pos[np.flatnonzero(m)] for k, m in allc.items()},
                     "criterion_cases": criterion_cases(col) if tk.startswith("a") else [],
                     "test": test, "shortest": sh["mean_steps"]}
        res.setdefault("data", {})[f"test_{tk}"] = {"world": world, "door_open_at_start": bool(door_open),
                                                     "transitions": int(len(tr["act"])),
                                                     "cases": {k: int(m.sum()) for k, m in allc.items()},
                                                     "shortest_route": sh, "seconds": round(time.monotonic() - t0, 1)}
        log(f"test {tk} ({world}, door open {bool(door_open)}): {res['data'][f'test_{tk}']['cases']}; shortest {sh}")
        save()
    return data, tests, tiles, pairs


def forward_memory(D):
    """Forward moves: each distinct (front, held) appearance pair, weighted counts of moved, blocked and
    reached the goal (read from the egocentric view, as card 027)."""
    m = np.flatnonzero(D["act"] == FWD)
    e0, e1, w = D["ego0"][m], D["ego1"][m], D["w"][m]
    changed = (APP0[e0][:, :NV] != APP0[e1][:, :NV]).any(1)
    out = np.where(D["term1"][m], GOAL, np.where(changed, MOVED, BLOCKED))
    fc, hc = e0[:, FRONT].astype(np.int64), e0[:, HELD].astype(np.int64)
    key = APP0[fc].astype(np.int64) * 256 + APP0[hc]
    u, first, inv = np.unique(key, return_index=True, return_inverse=True)
    counts = np.zeros((len(u), len(R.KINDS)))
    np.add.at(counts, (inv.ravel(), out), w)
    return {"front": fc[first], "held": hc[first], "counts": counts}


def memory_all(data):
    mem = {}
    for w in WORLDS:
        mem[w], _ = R.build_memory(data[w])
        mem[w][FWD] = forward_memory(data[w])
    return mem


def memory_report(mem):
    out = {}
    for w in WORLDS:
        out[w] = {ANAME[a]: len(mem[w][a]["counts"]) for a in ACTIONS}
        m = mem[w][FWD]
        per = {}
        for f, c in zip(m["front"], m["counts"]):
            nm = C.name_of_code(f)
            per.setdefault(nm, np.zeros(3))
            per[nm] += c[:3]
        out[w]["forward by tile in front (weighted: moved, blocked, goal)"] = {k: [round(float(x)) for x in v] for k, v in per.items()}
    return out


# ---------------------------------------------------------------- recall with forward moves

def prior_of(a):
    p = np.zeros(len(R.KINDS))
    n = len(FKINDS) if a == FWD else len(R.KINDS)
    p[:n] = 1.0 / n
    return p


class Groups:
    """The memories as tensors, one group per (world, action), with each group's prior."""

    def __init__(self, mem, dev):
        import torch
        self.names, self.f, self.h, self.c = [], [], [], []
        for w in WORLDS:
            for a in ACTIONS:
                m = mem[w][a]
                self.names.append((w, a))
                self.f.append(torch.as_tensor(m["front"], device=dev))
                self.h.append(torch.as_tensor(m["held"], device=dev))
                self.c.append(torch.as_tensor(m["counts"], dtype=torch.float32, device=dev))
        self.prior = torch.as_tensor(np.stack([prior_of(a) for _, a in self.names]), dtype=torch.float32, device=dev)


PRIOR = None


def loo_loglik(X, Rr, Mk, lam):
    """Card 033's leave-one-out log-likelihood, with a prior per group (forward has three outcomes)."""
    import torch
    d = (torch.abs(X[:, :, None] - X[:, None]) * lam[:, None, None]).sum(-1)
    k = torch.exp(-d) * Mk[:, :, None] * Mk[:, None] * (1 - torch.eye(X.shape[1], device=X.device))
    P = (k @ Rr + PRIOR[:, None, :]) / (k.sum(-1, keepdim=True) + 1.0)
    return ((Rr * torch.log(P.clamp_min(1e-30))).sum(-1) * Mk).sum(1) / Mk.sum(1)


R.loo_loglik = loo_loglik                    # train_encoder and fit_theta read it at call time


def use_groups(mem, dev):
    global PRIOR
    g = Groups(mem, dev)
    PRIOR = g.prior
    return g


class Recall(R.Recall):
    """Card 033's recall on own numbers, with the action's prior."""

    def __init__(self, z, m, theta, a):
        super().__init__(z, m, theta, "own")
        self.prior = prior_of(a)

    def predict(self, f, h, loo=False, vote=True):
        q = self.find(f, h)
        own = np.zeros(len(R.KINDS)) if (q is None or loo) else self.c[q]
        num, den = own + self.prior, own.sum() + 1.0
        kw = 0.0
        if vote:
            xq = self.key(f, h)
            for i in range(len(self.f)):
                if i == q:
                    continue
                k = float(np.exp(-(self.lam * np.abs(xq - self.key(self.f[i], self.h[i]))).sum()))
                num, den, kw = num + k * self.c[i] / self.c[i].sum(), den + k, kw + k
        return num / den, float(own.sum()), kw


def recall_reports(z, theta, mem):
    """Recall's own predictions of the criterion cases (no codes), and the tries curve at the closed
    door of each new colour, in the key world."""
    names = [(w, a) for w in WORLDS for a in ACTIONS]
    recs = {(w, a): Recall(z, mem[w][a], theta[j], a) for j, (w, a) in enumerate(names) if w == "key"}
    out = {}
    for col in COLOURS:
        key, opn, cld = code(("key", col)), code(("door", col, 2)), code(("door", col, 0))
        q = [("pick up the key", PICK, key, FLOOR, SWAPPED), ("drop the key", DROP, FLOOR, key, SWAPPED),
             ("forward onto the open door", FWD, opn, FLOOR, MOVED),
             ("toggle the open door (closes)", TOG, opn, FLOOR, IN_PLACE)]
        r = out[col] = {"predicted": {}, "closed_door_tries": {}}
        for nmk, a, f, h, true in q:
            P, own, kw = recs[("key", a)].predict(f, h)
            r["predicted"][nmk] = {"right": R.right(P, true), "p_true": round(float(P[true]), 3),
                                   "vote_weight": round(kw, 3)}
        tq = [("forward into it (blocked)", FWD, cld, FLOOR, BLOCKED),
              ("pick it up (nothing)", PICK, cld, FLOOR, NOTHING),
              ("toggle it, hand empty (nothing)", TOG, cld, FLOOR, NOTHING),
              ("toggle it holding a red key (nothing)", TOG, cld, code(("key", "red")), NOTHING),
              (f"toggle it holding the {col} key (opens)", TOG, cld, key, IN_PLACE)]
        for nmk, a, f, h, true in tq:
            curve = {}
            for k in (0, 1, 2, 4, 8):
                rec = copy.deepcopy(recs[("key", a)])
                for _ in range(k):
                    rec.add(f, h, true)
                P, own, kw = rec.predict(f, h)
                curve[k] = {"right": R.right(P, true), "p_true": round(float(P[true]), 3), "vote_weight": round(kw, 3)}
            first = next((k for k in (0, 1, 2, 4, 8) if curve[k]["right"]), None)
            r["closed_door_tries"][nmk] = {"first_right_after": first, "curve": curve}
    return out


# ---------------------------------------------------------------- codes

def new_tiles_report(ca, tiles):
    """The new tiles' codes, and the familiar tiles sharing each code."""
    names = {c: C.name_of_code(c) for c in tiles}
    by = {}
    for c in tiles:
        by.setdefault(tuple(int(v) for v in ca[c]), []).append(names[c])
    out = {}
    for nm, c in KP.new_tiles().items():
        row = [int(v) for v in ca[c]]
        out[nm] = {"codes": ["new" if v == NEW else v for v in row],
                   "shared_per_codebook": [[names[t] for t in tiles if ca[t, k] == row[k]] if row[k] != NEW else []
                                           for k in range(ca.shape[1])],
                   "same_tuple_as": by.get(tuple(row), [])}
    return out


def kinds_check(z, tiles):
    """Evaluator: does a new tile fall in with its kind, in any piece or in all 32 numbers?"""
    names = {c: C.name_of_code(c) for c in tiles}
    kinds = {}
    for c in tiles:
        kinds.setdefault(KP.kind_of(names[c]), []).append(c)
    kinds = {k: v for k, v in kinds.items() if len(v) > 1}
    dist = lambda X: np.linalg.norm(X[:, None] - X[None], axis=-1)
    parts = {f"piece {k}": z[:, 8 * k:8 * k + 8] for k in range(4)}
    parts["all 32"] = z
    return {p: KP.check(dist(X), tiles, kinds, KP.new_tiles()) for p, X in parts.items()}


def verdicts(o):
    c1 = all(o[w]["criterion_1"]["pass"] for w in WORLDS)
    c2 = all(o["key"][f"test_a_{col}"]["criterion_2"]["pass"] for col in COLOURS)
    c3 = all(o["key"][f"test_a_{col}"]["criterion_3"]["pass"] for col in COLOURS)
    return {"criterion_1": c1, "criterion_2": c2, "criterion_3": c3,
            "per_colour": {col: {"criterion_2": o["key"][f"test_a_{col}"]["criterion_2"]["pass"],
                                 "criterion_3": o["key"][f"test_a_{col}"]["criterion_3"]["pass"]} for col in COLOURS}}


# ---------------------------------------------------------------- main

def configure():
    R.configure()


def main():
    args = sys.argv[1:]
    flag = lambda f: f in args
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    configure()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    assert KR.N_CODES <= 255

    if flag("--collect"):
        res = {"note": "Card 034: data, memory and reference steps."}
        out = ROOT / "runs" / f"034_collect{SUF}.json"
        save = lambda: out.write_text(json.dumps(res, indent=1, default=str) + "\n")
        data, tests, tiles, pairs = setup(*((300, 60, 20, 60) if SMALL else (5000, 1000, 500, 1000)), res, save, log)
        mem = memory_all(data)
        MEMORY.write_bytes(pickle.dumps({"mem": mem, "tiles": tiles, "pairs": pairs}))
        res["tiles"] = [C.name_of_code(c) for c in tiles]
        res["memory"] = memory_report(mem)
        save()
        DATA.write_bytes(pickle.dumps({"data": data, "tests": tests, "tiles": tiles, "pairs": pairs}, protocol=5))
        log(f"memory: {json.dumps(res['memory'], default=str)[:1500]}")
        # reference: card 029's exact tile names (card 031's arm 3), for criterion 1's steps and as the
        # trivial comparison on the new colours
        o = C.run_codes("exact names", APP0[:, None].astype(np.int64), False, data, tests, dev, log)
        res["exact_names"] = o
        REF.write_text(json.dumps({w: o[w]["acting"]["mean_steps_when_successful"] for w in WORLDS}) + "\n")
        res["seconds"] = round(time.monotonic() - t00, 1)
        save()
        log("collect done")
        return

    cache = pickle.loads(MEMORY.read_bytes())
    mem, tiles, pairs = cache["mem"], cache["tiles"], cache["pairs"]
    groups = use_groups(mem, dev)

    if flag("--sweep"):
        mu, M = float(get("--sweep", "0")), int(get("--M", "16"))
        res = {"mu": mu, "M": M, "seeds": []}
        out = ROOT / "runs" / f"034_sweep_M{M}_{mu}{SUF}.json"
        for seed in SEEDS:
            ca, rep, z, th = R.train_encoder(tiles, pairs, groups, mu, mode="own", seed=seed, dev=dev, M=M)
            rep["new_tiles"] = [f"{C.name_of_code(c)} (codebook {k})" for c in tiles for k in range(ca.shape[1]) if ca[c, k] == NEW]
            ok = R.tiles_ok(ca, tiles)
            res["seeds"].append({"seed": seed, "encoder": rep, **ok})
            log(f"mu {mu} M {M} seed {seed}: {rep}; {ok}")
            out.write_text(json.dumps(res, indent=1) + "\n")
        res["qualifies"] = all(s["distinct"] and s["none_new"] for s in res["seeds"])
        res["seeds_ok"] = sum(s["distinct"] and s["none_new"] for s in res["seeds"])
        out.write_text(json.dumps(res, indent=1) + "\n")
        log(f"mu {mu} M {M}: qualifies {res['qualifies']} ({res['seeds_ok']} of 10)")
        return

    if flag("--diag"):
        mu = float(get("--mu", "0"))
        out = Path(get("--out", "runs/034_diag.json"))
        res = {"note": "Closed-door diagnostic: arm 2 with 4 times the updates.", "mu": mu, "seeds": []}
        for seed in SEEDS:
            ca, rep, z, th = R.train_encoder(tiles, pairs, groups, mu, mode="own", seed=seed, dev=dev, M=16,
                                             updates=500 if SMALL else 20000)
            res["seeds"].append({"seed": seed, "encoder": rep, "tiles": R.tiles_ok(ca, tiles),
                                 "new_tiles": new_tiles_report(ca, tiles), "kinds_check": kinds_check(z, tiles),
                                 "recall": recall_reports(z, th, mem)})
            out.write_text(json.dumps(res, indent=1, default=str) + "\n")
            log(f"diag seed {seed}: {rep}")
        return

    d = pickle.loads(DATA.read_bytes())
    data, tests = d["data"], d["tests"]
    ref = json.loads(REF.read_text())
    out = Path(get("--out", "runs/034_main.json"))
    res = {"note": "Card 034, tools/card034/teach.py", "arms": {}}
    save = lambda: out.write_text(json.dumps(res, indent=1, default=str) + "\n")

    if flag("--labels"):
        ca = C.oracle_codes()
        o = C.run_codes("arm 1", ca, True, data, tests, dev, log, ref)
        o["new_tiles"] = new_tiles_report(ca, tiles)
        o["verdicts"] = verdicts(o)
        res["arms"]["1_labels"] = o
        log(f"arm 1 verdicts {o['verdicts']}")
        save()
        return

    arm = get("--arm", "2")
    mu = {"2": float(get("--mu", "0")), "3": 0.0, "4": float(get("--mu", "0"))}[arm]
    M = {"2": 16, "3": 16, "4": 8}[arm]
    a, b = get("--seeds", "200-209").split("-")
    runs = res["arms"][{"2": "2_main", "3": "3_no_recall_term", "4": "4_eight_codes"}[arm]] = {"mu": mu, "M": M, "seeds": []}
    for seed in range(int(a), int(b) + 1):
        ca, rep, z, th = R.train_encoder(tiles, pairs, groups, mu, mode="own", seed=seed, dev=dev, M=M,
                                         **({"updates": 500} if SMALL else {}))
        o = {"seed": seed, "encoder": rep, "tiles": R.tiles_ok(ca, tiles), "codes": C.codes_report(ca, tiles),
             "new_tiles": new_tiles_report(ca, tiles), "kinds_check": kinds_check(z, tiles)}
        if mu > 0:
            o["recall"] = recall_reports(z, th, mem)
        oc = C.run_codes(f"arm {arm} seed {seed}", ca, True, data, tests, dev, log, ref)
        o["worlds"] = oc
        o["verdicts"] = verdicts(oc)
        runs["seeds"].append(o)
        log(f"arm {arm} seed {seed}: tiles {o['tiles']}; verdicts {o['verdicts']}")
        save()
    runs["seeds_passing"] = {c: sum(s["verdicts"][c] for s in runs["seeds"]) for c in ("criterion_1", "criterion_2", "criterion_3")}
    res["seconds"] = round(time.monotonic() - t00, 1)
    save()
    log(f"done: {runs['seeds_passing']} -> {out}")


if __name__ == "__main__":
    main()
