"""Card 098.7's Crafter harness: card 098.6's (tools/card098.6/crafter.py) with card 098.7's files level (the width of
"like this" chosen by held-out prediction). Play, vectors and codes are card 098.5's (runs/098.5/).

  bin/prun python tools/card098.7/crafter.py track --mode tokens|truth|nofiles --workers 8,9 [--tag T]
       --save F   after the run, keep the files level's long-term memory (stores, kinds) in runs/098.7/F
       --load F   start from it, and run only --workers (otherwise workers 0-7 are run first, then --workers)
       --only     from nothing, run only --workers (the first of runs in parts)
       --check W  the gate's prediction check on worker W (pair events, recall against the prior)
       --steps N  smoke tests only
"""
import pickle
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card098.5"))   # assign.py
sys.path.insert(0, str(ROOT / "tools" / "card098.7"))
import files as FL                                     # noqa: E402
from assign import linear_sum_assignment               # noqa: E402

PLAY = ROOT / "runs" / "098.5"
OUT = ROOT / "runs" / "098.7"
CODES = ROOT / "runs" / "098.3" / "codes_joint_blocked.npz"
SHIFTS = ROOT / "runs" / "098.3" / "results_joint_blocked.json"
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
MODE = arg("--mode", "tokens")
CENTRE = 31
REL = np.array([(k % 9 - 4, k // 9 - 3) for k in range(63)], np.float64)    # view place -> offset (x right, y down)
VIEW = np.array([k for k in range(63) if k != CENTRE])
MOVERS = {1: "cow", 2: "zombie", 3: "skeleton", 4: "arrow"}
NCLASS = 19


def euclid(A, B):
    A, B = np.asarray(A, np.float32), np.asarray(B, np.float32)
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T
    return np.sqrt(np.maximum(d, 0))


def track(workers):
    radius = float(np.load(CODES)["radius"]) if MODE != "truth" else 0.5
    learned = json.loads(SHIFTS.read_text())["learned shifts (action: dx, dy)"]
    shifts = {int(a): np.array(s, np.float64) for a, s in learned.items()}
    step_len = float(np.median([np.linalg.norm(s) for s in shifts.values()]))   # the agent's own step, learned
    FS = FL.Files(euclid, radius, step_len, own=[-d for d in shifts.values()])   # the agent's own displacements
    out = {}
    order = sorted(set(range(8)) | set(workers))
    if arg("--load", None):
        FS = pickle.loads((OUT / arg("--load", None)).read_bytes())
        FS.reset()
        order = workers
    if "--only" in sys.argv:                            # from nothing, only these (the first of runs in parts)
        order = workers
    if MODE == "nofiles":                               # identity by place; the kinds the tokens run confirmed
        FS.kinds = np.load(OUT / "kinds_tokens.npy")
        order = workers
    for w in order:
        t0 = time.time()
        st0 = dict(FS.stats)
        z = np.load(PLAY / f"play_w{w}.npz")
        act, ep, done, pos = z["act"], z["ep"], z["done"], z["pos"].astype(np.float64)
        FS.log = w == int(arg("--check", -1))
        calls0 = FS.stats["recall_calls"]
        if MODE == "truth":
            V = np.eye(NCLASS, dtype=np.float32)[z["sem"].reshape(len(act), 63).clip(0)]
        else:
            V = np.load(PLAY / f"vec_w{w}.npy").astype(np.float32)
        if arg("--steps", None):                        # smoke tests only: scores are then meaningless
            act = act[:int(arg("--steps", None))]
        rec = {"agent": np.zeros((len(act), 2)), "files": [], "level1_right": [], "stats": None}
        layout, agent = {}, np.zeros(2)
        for t in range(len(act)):
            first = t == 0 or ep[t] != ep[t - 1] or done[t - 1]
            if first:
                layout, agent = {}, np.zeros(2)
                FS.reset()
            else:
                cands = [np.zeros(2)]
                if int(act[t - 1]) in shifts:
                    cands.append(-shifts[int(act[t - 1])])     # the world shifts one way, the agent the other
                if MODE == "truth":
                    best = pos[t] - pos[t - 1]
                else:
                    near = [f["pos"] for f in FS.files if f["confirmed"]] if MODE == "tokens" else []
                    score = []
                    for c in cands:
                        P = agent + c + REL[VIEW]
                        keys = [tuple(p) for p in P.round().astype(int)]
                        m = np.array([k in layout and not any(np.abs(p - q).max() <= step_len for q in near)
                                      for k, p in zip(keys, P)])
                        if not m.any():
                            score.append(np.inf)
                            continue
                        Bv = np.stack([layout[k] for k, mm in zip(keys, m) if mm])
                        score.append(float(np.linalg.norm(V[t][VIEW][m] - Bv, axis=1).mean()))
                    best = cands[int(np.argmin(score))] if np.isfinite(min(score)) else cands[0]
                rec["level1_right"].append(bool(np.allclose(best, pos[t] - pos[t - 1])))
                agent = agent + best
            rec["agent"][t] = agent
            P = agent + REL[VIEW]
            keys = [tuple(p) for p in P.round().astype(int)]
            known = np.array([k in layout for k in keys])
            B = np.stack([layout.get(k, np.zeros(V.shape[-1], np.float32)) for k in keys])
            if MODE == "nofiles":
                explained = np.zeros(len(VIEW), bool)
                seen = [((int(ep[t]),) + keys[i], P[i].copy()) for i in np.flatnonzero(FS.is_kind(V[t][VIEW]))]
            else:
                explained, seen = FS.step(t, P, V[t][VIEW], B, known, (agent + REL[0], agent + REL[62]))
            for i in np.flatnonzero(~explained):
                layout[keys[i]] = V[t][VIEW][i]
            rec["files"].append(seen)
        rec["stats"] = {k: FS.stats[k] - st0.get(k, 0) for k in FS.stats}
        rec["recall_calls_per_step"] = round((FS.stats["recall_calls"] - calls0) / len(act), 4)
        if FS.log:
            rec["check"] = check(z, rec, FS)
        out[w] = rec
        print(f"track {MODE} worker {w}: {time.time() - t0:.0f} s, kinds {0 if FS.kinds is None else len(FS.kinds)}",
              flush=True)
    return out, FS


def check(z, rec, FS):
    """The gate's prediction check on one worker: pair events by the true class of the token that vanished, and the
    log-likelihood of confirmed files' moves under recall and under the prior, for movers."""
    sem = z["sem"].reshape(len(z["act"]), 63)
    light = z["light"]
    at = {tuple(r): i for i, r in enumerate(REL.astype(int))}

    def cls(t, p):
        i = at.get(tuple(np.round(np.asarray(p) - rec["agent"][t]).astype(int)))
        return int(sem[t, i]) if i is not None else -1
    MOVE = {14, 15, 16, 17}
    out = {}
    for tag in ("day", "night"):
        st = [r0 for t, u, k, r0, pm in FS.log_pairs if (light[t] < 0.3) == (tag == "night")
              and 0 <= cls(t - 1, u) <= 12]
        mv = [r0 for t, u, k, r0, pm in FS.log_pairs if (light[t] < 0.3) == (tag == "night") and cls(t - 1, u) in MOVE]
        out[tag] = {"static pair events": len(st),
                    "share that would start a file": round(float(np.mean(np.array(st) >= FL.DELETE)), 4)
                    if st else None,
                    "mover pair events": len(mv),
                    "share of movers' that would": round(float(np.mean(np.array(mv) >= FL.DELETE)), 4) if mv else None}
        for where, kn in (("first seen", False), ("changed", True)):
            r = [(r0, cls(t, p)) for t, p, r0, k in FS.log_births if k == kn and (light[t] < 0.3) == (tag == "night")]
            st = np.array([r0 for r0, c in r if 0 <= c <= 12])
            mv = np.array([r0 for r0, c in r if c in MOVE])
            out[tag][f"tokens {where}, no file or pair: static, share that start / confirm a file"] = \
                [len(st), round(float((st >= FL.DELETE).mean()), 4), round(float((st >= FL.CONFIRM).mean()), 4)] \
                if len(st) else None
            out[tag][f"tokens {where}, no file or pair: movers, share that start / confirm a file"] = \
                [len(mv), round(float((mv >= FL.DELETE).mean()), 4), round(float((mv >= FL.CONFIRM).mean()), 4)] \
                if len(mv) else None
    lr = [(a, b) for t, p, d, a, b in FS.log_moves if cls(t - 1, p) in MOVE]
    out["confirmed files' moves on movers"] = len(lr)
    if lr:
        a, b = np.array(lr).T
        out["mean log-likelihood: recall, prior"] = [round(float(a.mean()), 4), round(float(b.mean()), 4)]
    FS.log_pairs, FS.log_moves, FS.log_births = [], [], []
    return out


def idf1(z, rec, w, light):
    """IDF1 over the movers (cow, zombie, skeleton, arrow) of one worker; per kind, the share of each kind's frames
    whose object is followed by its assigned file (ID recall); motion right for files followed across two frames."""
    obj = z["obj"]
    ep, pos = z["ep"], z["pos"].astype(np.float64)
    gt = defaultdict(list)                              # t -> [(gid, kind, rel)]
    for t, i, k, dx, dy, fx, fy in obj:
        if k in MOVERS:
            gt[int(t)].append(((int(ep[t]), int(i)), int(k), np.array([dx, dy], np.float64)))
    co = defaultdict(lambda: defaultdict(int))          # episode -> (gid, pid) -> frames together
    ngt, npred, gkind = defaultdict(int), defaultdict(int), {}
    at = {}                                             # (t, pid) -> gid it sits on
    on = {}                                             # (t, gid) -> the file on it
    for t, seen in enumerate(rec["files"]):
        e = int(ep[t])
        for g, k, r in gt.get(t, []):
            ngt[e] += 1
            gkind[g] = k
        npred[e] += len(seen)
        for pid, p in seen:
            rel = p - rec["agent"][t]
            for g, k, r in gt.get(t, []):
                if np.abs(rel - r).max() < 0.5:
                    co[e][(g, pid)] += 1
                    at[(t, pid)] = g
                    on[(t, g)] = pid
    idtp, idtp_kind, n_kind = 0, defaultdict(int), defaultdict(int)
    for t in gt:
        for g, k, r in gt[t]:
            n_kind[k] += 1
    assigned = {}
    for e, pairs in co.items():
        G = sorted({g for g, p in pairs})
        Pd = sorted({p for g, p in pairs})
        gi, pi = {g: i for i, g in enumerate(G)}, {p: i for i, p in enumerate(Pd)}
        M = np.zeros((len(G), len(Pd)))
        for (g, p), c in pairs.items():
            M[gi[g], pi[p]] = c
        r, c = linear_sum_assignment(-M)
        for i, j in zip(r, c):
            if M[i, j] > 0:
                idtp += M[i, j]
                idtp_kind[gkind[G[i]]] += M[i, j]
                assigned[Pd[j]] = G[i]
    tot_gt, tot_pred = sum(ngt.values()), sum(npred.values())
    # motion: a file seen at t-1 and t on the same object both times
    right, n, right_n, n_n = 0, 0, 0, 0
    last = {}
    for t, seen in enumerate(rec["files"]):
        cur = {pid: p for pid, p in seen}
        for pid, p in cur.items():
            if (t - 1, pid) in at and (t, pid) in at and at[(t - 1, pid)] == at[(t, pid)] and pid in last \
                    and ep[t] == ep[t - 1]:
                g = at[(t, pid)]
                r0 = next(r for gg, k, r in gt[t - 1] if gg == g)
                r1 = next(r for gg, k, r in gt[t] if gg == g)
                true_d = (pos[t] + r1) - (pos[t - 1] + r0)
                ok = bool(np.allclose(p - last[pid], true_d))
                right += ok
                n += 1
                if light[t] < 0.3:
                    right_n += ok
                    n_n += 1
        last = cur
    # where identity is lost: each object frame followed by its own file, another file or none, by day and night
    own = {g: p for p, g in assigned.items()}
    lost = defaultdict(int)
    for t in gt:
        tag = "night" if light[t] < 0.3 else "day"
        for g, k, r in gt[t]:
            f = on.get((t, g))
            lost[(tag, "none" if f is None else "own" if own.get(g) == f else "another")] += 1
    # file detections on tokens that never move (the stop rule)
    sem = z["sem"].reshape(len(z["act"]), 63)
    cell = {tuple(r): i for i, r in enumerate(REL.astype(int))}
    on_static = 0
    for t, seen in enumerate(rec["files"]):
        for pid, p in seen:
            i = cell.get(tuple(np.round(p - rec["agent"][t]).astype(int)))
            if i is not None and (0 <= sem[t, i] <= 12 or sem[t, i] == 18):
                on_static += 1
    return {"idtp": idtp, "gt": tot_gt, "pred": tot_pred, "idtp_kind": dict(idtp_kind), "n_kind": dict(n_kind),
            "motion_right": right, "motion_n": n, "motion_right_night": right_n, "motion_n_night": n_n,
            "on_static": on_static, "lost": dict(lost)}


def main():
    workers = [int(x) for x in arg("--workers", "8,9").split(",")]
    out, FS = track(workers)
    if MODE == "tokens" and FS.kinds is not None:
        np.save(OUT / "kinds_tokens.npy", FS.kinds)
    if arg("--save", None):
        FS.log_pairs, FS.log_moves, FS.log_births = [], [], []
        (OUT / arg("--save", None)).write_bytes(pickle.dumps(FS))
    S = defaultdict(float)
    kinds = defaultdict(lambda: [0, 0])
    l1 = []
    lost = defaultdict(int)
    for w in workers:
        z = np.load(PLAY / f"play_w{w}.npz")
        r = idf1(z, out[w], w, z["light"])
        for k in ("idtp", "gt", "pred", "motion_right", "motion_n", "motion_right_night", "motion_n_night",
                  "on_static"):
            S[k] += r[k]
        for k in MOVERS:
            kinds[MOVERS[k]][0] += r["idtp_kind"].get(k, 0)
            kinds[MOVERS[k]][1] += r["n_kind"].get(k, 0)
        l1 += out[w]["level1_right"]
        for k, v in r["lost"].items():
            lost[k] += v
    rep = {"mode": MODE, "workers": workers,
           "IDF1 (cow, zombie, skeleton, arrow pooled)": round(2 * S["idtp"] / (S["gt"] + S["pred"]), 4),
           "ID recall / ID precision": [round(S["idtp"] / max(S["gt"], 1), 4), round(S["idtp"] / max(S["pred"], 1), 4)],
           "object frames / file detections": [int(S["gt"]), int(S["pred"])],
           "share of file detections on static tokens": round(S["on_static"] / max(S["pred"], 1), 4),
           "ID recall by kind": {k: round(a / max(b, 1), 4) for k, (a, b) in kinds.items()},
           "motion right (files on the same object two frames running)": [
               round(S["motion_right"] / max(S["motion_n"], 1), 4), int(S["motion_n"])],
           "motion right at night (light < 0.3)": [
               round(S["motion_right_night"] / max(S["motion_n_night"], 1), 4), int(S["motion_n_night"])],
           "level 1: the agent's own motion right": round(float(np.mean(l1)), 4),
           "kinds that have had a file": 0 if FS.kinds is None else int(len(FS.kinds)),
           "movers' object frames followed by their own file / another file / none": {
               tag: [round(lost[(tag, c)] / max(sum(lost[(tag, x)] for x in ("own", "another", "none")), 1), 4)
                     for c in ("own", "another", "none")] + [sum(lost[(tag, x)] for x in ("own", "another", "none"))]
               for tag in ("day", "night")},
           "width of like this (step, width, held-out log-likelihood per candidate)": getattr(FS, "widths", []),
           "file events (judged workers)": {w: out[w]["stats"] for w in workers},
           "recall calls per step": {w: out[w].get("recall_calls_per_step") for w in workers},
           "check": {w: out[w]["check"] for w in out if "check" in out[w]}}
    (OUT / f"{MODE}{arg('--tag', '')}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
