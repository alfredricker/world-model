"""Card 110 in Crafter: confidence for System 1's predictions, and conditions read through it, on 105.10's models.

106's events and replacements (the same draw); arms as tools/card110/world1.py: none, P, K, R, RK, oracle.
- Outcomes as 106: a make's tool slot changed or not; a move's next place of the agent (the tile where an agent code
  is most probable, with the model's own tau). Their probability (P) uses the temperature fitted on held-out tokens:
  the slot's probability of keeping or leaving its code; for a move, the softmax over tiles of the agent's log-odds.
- Familiarity (R): the vector of the outcome's token (the slot; the move's target tile).
- Oracle (gate, scoring only): the 5 x 5 kinds around the agent, the carried wood, stone, coal and iron (to 9), and
  the action occur together in training.
The truth is 106's: Crafter's rule (data.yaml) on the replaced view.

  bin/prun python tools/card110/crafter.py --net runs/105.10/tokens_Rb_s1021_net.pt --copy runs/105.10/tokens_Rb_s7_net.pt
      [--per 300] [--smoke] → runs/110/crafter_<net name>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
NET, COPY = Path(arg("--net", "runs/105.10/tokens_Rb_s1021_net.pt")), Path(arg("--copy", "runs/105.10/tokens_Rb_s7_net.pt"))
PER = int(arg("--per", 300))
SMOKE = "--smoke" in sys.argv
sys.argv = ["run.py", "--origins", "pose"]
sys.path.insert(0, str(ROOT / "tools" / "card105.2"))
sys.path.insert(0, str(ROOT / "tools" / "card110"))
import run as R                                           # noqa: E402
import common as CM                                       # noqa: E402

dev = "cuda"
OUT = ROOT / "runs" / "110"
ARMS = ("none", "P", "K", "R", "RK", "oracle")
GRASS, SAND, PATH, LAVA, TABLE, FURNACE = 2, 5, 4, 7, 11, 12
WALK = (GRASS, SAND, PATH, LAVA)
NEAR9 = [r * 9 + c for r in (2, 3, 4) for c in (3, 4, 5)]
SQ25 = [r * 9 + c for r in range(1, 6) for c in range(2, 7)]
SLOT = {"wood": 68, "stone": 69, "coal": 70, "iron": 71}
NEEDS = {11: ({"wood": 1}, (TABLE,)), 12: ({"wood": 1, "stone": 1}, (TABLE,)),
         13: ({"wood": 1, "coal": 1, "iron": 1}, (TABLE, FURNACE)), 14: ({"wood": 1}, (TABLE,)),
         15: ({"wood": 1, "stone": 1}, (TABLE,)), 16: ({"wood": 1, "coal": 1, "iron": 1}, (TABLE, FURNACE))}
STEP = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
N_MEM_STATES = 8000 if SMOKE else 40000
P19 = (10, 30, 100, 300)


def made_by_rule(view, counts, a):
    uses, near = NEEDS[a]
    return all(k in view[NEAR9] for k in near) and all(counts[SLOT[m]] >= n for m, n in uses.items())


def load(path):
    net = R.TransitionModel(32, 17, arm=path.name.split("_")[1]).to(dev)
    net.load_state_dict(torch.load(ROOT / path))
    tau = json.load(open(ROOT / path.parent / path.name.replace("_net.pt", ".json")))["tau"]
    return net.eval(), tau


def okey(view, inv4, a):
    """Oracle key (scoring only): the 5 x 5 kinds around the agent (labels + 2, base 32), carried counts, action."""
    key = np.asarray(a, np.uint64).copy()
    for q in SQ25:
        key = key * np.uint64(1000003) + (view[:, q] + 2).astype(np.uint64)
    for m in range(4):
        key = key * np.uint64(1000003) + inv4[:, m].clip(0, 9).astype(np.uint64)
    return key


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1021)                                     # 106's draw of events
    rng2 = np.random.default_rng(110)                                     # everything else
    tr, te = R.load(R.TRAIN, "tokens"), R.load(R.TEST, "tokens")
    (net, tau), (net2, tau2) = load(NET), load(COPY)
    vec = CM.Vectors(net)
    Pt, plab = R.decoder(tr, "tokens")
    agent_protos = torch.as_tensor(plab[:, 31] >= 1300, device=dev)

    @torch.no_grad()
    def run(m, D, i, V0=None, a=None):
        """dist (B, NT, C), valid, next code, input code, vectors (net only), the true next code (no V0)."""
        V0_, Vn, valid, P, F, M, act = R.batch(D, i, V0)
        if a is not None:
            act = torch.full_like(act, a)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt, _ = m(V0_, P, F, M, act)
        dist = torch.cdist(nxt.float().reshape(-1, Pt.shape[1]), Pt).reshape(len(i), R.NT, -1)
        code0 = torch.cdist(V0_.reshape(-1, Pt.shape[1]), Pt).argmin(1).reshape(len(i), R.NT).cpu().numpy()
        code1 = torch.cdist(Vn.reshape(-1, Pt.shape[1]), Pt).argmin(1).reshape(len(i), R.NT).cpu().numpy()
        return dist, valid, dist.argmin(-1).cpu().numpy(), code0, (vec.v if m is net else None), code1

    # ---- memory over training: all tokens of N states, and every token whose code changed
    n_tr = len(tr["t"])
    idx_tr = np.arange(n_tr) if not SMOKE else np.sort(rng2.choice(n_tr, 30000, replace=False))
    whole = np.zeros(n_tr, bool)
    whole[rng2.choice(idx_tr, min(N_MEM_STATES, len(idx_tr)), replace=False)] = True
    mv, ma, ms = [], [], []
    for s in range(0, len(idx_tr), 1024):
        i = idx_tr[s:s + 1024]
        _, valid, _, code0, v, code1 = run(net, tr, i)
        keep = (valid.cpu().numpy() & (code1 != code0)) | whole[i][:, None]
        r, c = np.nonzero(keep)
        mv.append(v[torch.as_tensor(r, device=dev), torch.as_tensor(c, device=dev)].cpu().numpy())
        ma.append(tr["act"][tr["t"][i]][r])
        ms.append(i[r])
    mem = CM.Memory(np.concatenate(mv), np.concatenate(ma), np.concatenate(ms))
    print("memory", len(mem.vec), f"({time.monotonic() - t0:.0f} s)", flush=True)
    rep = {"net": str(NET), "copy": str(COPY), "memory vectors": int(len(mem.vec)), "smoke": SMOKE}

    # ---- oracle keys over training (scoring only)
    labtr = tr["lab"][tr["t"]]
    okeys = np.unique(okey(labtr[:, :63], (labtr[:, 68:72] - 100) % 10, tr["act"][tr["t"]]))

    def known(keys):
        pos = np.searchsorted(okeys, keys).clip(max=len(okeys) - 1)
        return okeys[pos] == keys

    # ---- events: 106's draw
    T, act = te["t"], te["act"][te["t"]]
    lab0 = te["lab"][T]
    inv = (lab0[:, 63:79] - 100) % 10
    counts = np.zeros((len(T), R.NT), np.int64)
    counts[:, 63:79] = inv
    made = np.zeros(len(T), bool)
    for s in range(0, len(T), 2048):
        i = np.arange(s, min(s + 2048, len(T)))
        sl = np.array([R.MAKE_SLOT.get(int(a), 73) for a in act[i]])
        lab1 = te["lab"][T[i] + 1]
        made[i] = np.isin(act[i], list(R.MAKE_SLOT)) & (lab1[np.arange(len(i)), sl] != lab0[i, sl])
    d = te["pos_true"][T + 1] - te["pos_true"][T]
    mvm = np.isin(act, R.MOVES)
    blocked = mvm & (d == 0).all(1)
    pick = lambda m: sorted(rng.choice(np.flatnonzero(m), min(PER, int(m.sum())), replace=False))
    evs = {"make (a tool made)": pick(made & (te["src"][T] < 2)), "move, blocked": pick(blocked),
           "move, succeeded": pick(mvm & ~blocked)}
    ev_i = {i for lst in evs.values() for i in lst}

    # ---- fits on held-out real tokens (test steps not among the events), criterion 1 on others
    pool = np.array([i for i in rng2.permutation(len(T))[:12000] if i not in ev_i])
    n_fit = 1000 if SMOKE else 4000
    fit_i, eval_i = pool[:n_fit], pool[n_fit:n_fit + n_fit // 2]

    def tokens(ii):
        out = {k: [] for k in ("nd2", "v", "a", "y", "y0", "lab")}
        for s in range(0, len(ii), 512):
            i = ii[s:s + 512]
            dist, valid, _, code0, v, code1 = run(net, te, i)
            m = valid.cpu().numpy()
            r, c = np.nonzero(m)
            out["nd2"].append((-dist[torch.as_tensor(r, device=dev), torch.as_tensor(c, device=dev)] ** 2).cpu().numpy())
            out["v"].append(v[torch.as_tensor(r, device=dev), torch.as_tensor(c, device=dev)].cpu().numpy())
            out["a"].append(act[i][r]); out["y"].append(code1[r, c]); out["y0"].append(code0[r, c])
            out["lab"].append(lab0[i][r, c])
        return {k: np.concatenate(v) for k, v in out.items()}

    F = tokens(fit_i)
    tauT = CM.fit_temperature(F["nd2"], F["y"])
    pf_all = torch.softmax(torch.as_tensor(F["nd2"] / tauT), -1).numpy()
    pf, rightf = pf_all.max(1), (pf_all.argmax(1) == F["y"]).astype(float)
    Df = mem.knn(F["v"], F["a"])
    kj, ab, ktab = CM.choose_k(pf, Df, rightf)
    rep["fit"] = {"model tau": tau, "fitted tau": round(tauT, 5), "tokens": int(len(pf)), "wrong tokens": int((rightf == 0).sum()),
                  "k chosen": CM.KS[kj], "alpha, beta": [round(ab[0], 4), round(ab[1], 4)], "by k": ktab,
                  "raw distance AUROC for a wrong token (diagnostic), by k":
                      {str(k): CM.auroc(Df[:, j], rightf == 0) for j, k in enumerate(CM.KS)}}
    print("fit", rep["fit"], f"({time.monotonic() - t0:.0f} s)", flush=True)

    E = tokens(eval_i)
    pe_all = torch.softmax(torch.as_tensor(E["nd2"] / tauT), -1).numpy()
    pe, righte = pe_all.max(1), (pe_all.argmax(1) == E["y"]).astype(float)
    ce = pe * CM.applic(np.log(mem.knn(E["v"], E["a"])[:, kj] + 1e-6), ab)
    c1 = {}
    for g, acts in R.GROUPS.items():
        for part, m in (("all tokens", np.ones(len(pe), bool)), ("tokens whose code changed", E["y"] != E["y0"]),
                        ("creatures", np.isin(E["lab"], R.CREATURES))):
            m = m & np.isin(E["a"], acts)
            if m.sum() >= 100:
                c1[f"{g}, {part}"] = {"n": int(m.sum()), "ECE P": CM.ece(pe[m], righte[m]),
                                      "ECE R": CM.ece(ce[m], righte[m]), "accuracy": round(float(righte[m].mean()), 4)}
    rep["1 calibration"] = c1
    rep["1 reliability R (confidence, accuracy, n)"] = CM.reliability(ce, righte)
    print("criterion 1", json.dumps({k: (v["ECE P"], v["ECE R"]) for k, v in c1.items()}), flush=True)

    def outcome(m, tu, kind, i, V):
        dist, valid, code, code0, v, _ = run(m, te, np.full(len(V), i), V)
        a = int(act[i])
        if kind == "make":
            sl = R.MAKE_SLOT[a]
            o = code[:, sl] != code0[:, sl]
            p0 = torch.softmax(-dist[:, sl] ** 2 / tauT, -1)[torch.arange(len(V)), torch.as_tensor(code0[:, sl])].cpu().numpy()
            p = np.where(o, 1 - p0, p0)
            tokq = sl
        else:
            lg = -dist[:, :63] ** 2 / tu
            lp = torch.logsumexp(lg[..., agent_protos], -1) - torch.logsumexp(lg, -1)
            lp[~valid[:, :63]] = -float("inf")
            o = lp.argmax(1).cpu().numpy()
            lgT = -dist[:, :63] ** 2 / tauT
            lpT = torch.logsumexp(lgT[..., agent_protos], -1) - torch.logsumexp(lgT, -1)
            lpT[~valid[:, :63]] = -float("inf")
            p = torch.softmax(lpT, 1)[torch.arange(len(V)), torch.as_tensor(o)].cpu().numpy()
            dx, dy = STEP[a]
            tokq = 31 + dx + 9 * dy
        return o, p, (v[:, tokq].cpu().numpy() if v is not None else None)

    rows, vecs, base = [], [], []
    for name, lst in evs.items():
        kind = "make" if name.startswith("make") else "move"
        for i in lst:
            a = int(act[i])
            view = lab0[i, :63]
            V0 = te["V"][T[i]].to(dev).float()
            o, p, v = outcome(net, tau, kind, i, V0[None])
            if kind == "make":
                truth0 = made_by_rule(view, counts[i], a)
                tok = R.MAKE_SLOT[a]
            else:
                dx, dy = STEP[a]
                tok = 31 + dx + 9 * dy
                truth0 = tok if view[tok] in WALK else 31
            if o[0] != truth0:
                base.append(dict(kind=name, asked=False))
                continue
            o2, _, _ = outcome(net2, tau2, kind, i, V0[None])
            vecs.append(v[0])
            e = dict(kind=name, asked=True, bk=o[0], p=float(p[0]), known=bool(o2[0] == o[0]), vi=len(vecs) - 1, a=a,
                     oracle=bool(known(okey(view[None], inv[i:i + 1, 5:9], [a]))[0]), tc=np.zeros(R.NT, bool), r0=len(rows))
            reps = {}
            for q in range(63):
                if q != 31 and view[q] >= 0 and view[q] not in reps:
                    reps[int(view[q])] = q
            empty = [q for q in range(67, 79) if inv[i, q - 63] == 0]
            var, meta, okk = [], [], []
            for j in range(63):
                if j == 31 or view[j] < 0:
                    continue
                for k, q in reps.items():
                    if k == view[j]:
                        continue
                    v2 = view.copy()
                    v2[j] = k
                    tk = made_by_rule(v2, counts[i], a) if kind == "make" else (tok if v2[tok] in WALK else 31)
                    e["tc"][j] |= tk != truth0
                    vj = V0.clone()
                    vj[j] = V0[q]
                    var.append(vj)
                    meta.append((j, tk))
                    okk.append(okey(v2[None], inv[i:i + 1, 5:9], [a])[0])
            if kind == "make" and empty:
                for j in range(67, 79):
                    if j == tok or inv[i, j - 63] == 0:
                        continue
                    c2 = counts[i].copy()
                    c2[j] = 0
                    tk = made_by_rule(view, c2, a)
                    e["tc"][j] |= tk != truth0
                    vj = V0.clone()
                    vj[j] = V0[empty[0]]
                    var.append(vj)
                    meta.append((j, tk))
                    okk.append(okey(view[None], c2[None, 68:72], [a])[0])
            orcs = known(np.array(okk, np.uint64)) if okk else []
            for s in range(0, len(var), 256):
                Vb = torch.stack(var[s:s + 256])
                o, p, v = outcome(net, tau, kind, i, Vb)
                o2, _, _ = outcome(net2, tau2, kind, i, Vb)
                for q in range(len(Vb)):
                    j, tk = meta[s + q]
                    vecs.append(v[q])
                    rows.append((name, j, int(o[q]), int(o2[q]), float(p[q]), int(tk), bool(orcs[s + q]), len(vecs) - 1, a))
            e["r1"] = len(rows)
            base.append(e)
        print(name, len(lst), "events", len(rows), "replacements", f"({time.monotonic() - t0:.0f} s)", flush=True)

    vecs = np.stack(vecs).astype(np.float16)
    acts = np.zeros(len(vecs), np.int64)
    for e in base:
        if e["asked"]:
            acts[e["vi"]] = e["a"]
    for r in rows:
        acts[r[7]] = r[8]

    def score(D_all):
        out, au = {}, {}
        appl = CM.applic(np.log(D_all[:, kj] + 1e-6), ab)
        allk, allp, allc, allr, allo, allw, allD = [], [], [], [], [], [], []
        for name in evs:
            es = [e for e in base if e["kind"] == name and e["asked"]]
            bP = np.array([e["p"] for e in es])
            bR = bP * appl[[e["vi"] for e in es]]
            thrP, thrR = (np.percentile(bP, 5), np.percentile(bR, 5)) if len(es) else (0, 0)
            st = {a: [0, 0, 0, 0] for a in ARMS}
            for e in es:
                rr = rows[e["r0"]:e["r1"]]
                if not rr:
                    continue
                js = np.array([r[1] for r in rr], np.int64)
                pred = np.array([r[2] for r in rr])
                copy = np.array([r[3] for r in rr])
                pk = np.array([r[4] for r in rr])
                truth = np.array([r[5] for r in rr])
                orc = np.array([r[6] for r in rr], bool)
                cR = pk * appl[[r[7] for r in rr]]
                flips = pred != e["bk"]
                keep = {"none": np.ones(len(rr), bool), "P": pk >= thrP, "K": copy == pred, "R": cR >= thrR, "oracle": orc}
                keep["RK"] = keep["R"] & keep["K"]
                ask = {"none": True, "P": e["p"] >= thrP, "K": e["known"], "R": e["p"] * appl[e["vi"]] >= thrR, "oracle": True}
                ask["RK"] = ask["R"] and ask["K"]
                for arm in ARMS:
                    if not ask[arm]:
                        continue
                    mc = np.zeros(R.NT, bool)
                    np.logical_or.at(mc, js[keep[arm] & flips], True)
                    s_ = st[arm]
                    s_[0] += int((mc & e["tc"]).sum()); s_[1] += int((mc & ~e["tc"]).sum())
                    s_[2] += int((~mc & e["tc"]).sum()); s_[3] += 1
                allk += [name] * len(rr); allp += list(pk); allc += list(copy == pred); allr += list(cR)
                allo += list(orc); allw += list(pred != truth); allD += [r[7] for r in rr]
            out[name] = {arm: {"asked": st[arm][3], "precision, recall": list(CM.prf(*st[arm][:3]))} for arm in ARMS}
            out[name]["thresholds P, R"] = [round(float(thrP), 4), round(float(thrR), 4)]
        wrong, kinds = np.array(allw, bool), np.array(allk)
        conf = {"P": np.array(allp), "K": np.array(allc, float), "R": np.array(allr), "oracle": np.array(allo, float)}
        conf["RK"] = conf["R"] * conf["K"]
        for arm, c in conf.items():
            au[arm] = {"all": CM.auroc(-c, wrong)}
            for name in evs:
                m = kinds == name
                au[arm][name] = CM.auroc(-c[m], wrong[m])
        rowD = D_all[allD]
        au["raw distance (diagnostic), by k: AUROC for wrong"] = {str(k): CM.auroc(rowD[:, j], wrong) for j, k in enumerate(CM.KS)}
        au["replacements, wrong"] = [int(len(wrong)), int(wrong.sum())]
        return out, au

    D_all = mem.knn(vecs, acts)
    out, au = score(D_all)
    rep["3 conditions"] = out
    rep["2 AUROC for a wrong prediction"] = au
    for name in evs:
        print(name, json.dumps({a: out[name][a]["precision, recall"] for a in ARMS}), flush=True)
    print("criterion 2", json.dumps({a: au[a]["all"] for a in ("P", "K", "R", "RK", "oracle")}),
          au["raw distance (diagnostic), by k: AUROC for wrong"], flush=True)

    # ---- P19: the memory holding about 10, 30, 100, 300 training occurrences of each kind of event
    Ttr, atr = tr["t"], tr["act"][tr["t"]]
    l0, l1 = tr["lab"][Ttr], tr["lab"][Ttr + 1]
    sl = np.array([R.MAKE_SLOT.get(int(a), 73) for a in atr])
    dtr = tr["pos_true"][Ttr + 1] - tr["pos_true"][Ttr]
    mtr = np.isin(atr, R.MOVES)
    occ = {"make (a tool made)": int((np.isin(atr, list(R.MAKE_SLOT)) & (l1[np.arange(len(Ttr)), sl] != l0[np.arange(len(Ttr)), sl])).sum()),
           "move, blocked": int((mtr & (dtr == 0).all(1)).sum()), "move, succeeded": int((mtr & ~(dtr == 0).all(1)).sum())}
    p19 = {}
    for name in evs:
        p19[name] = {"training occurrences": occ[name]}
        for n in P19:
            if n >= occ[name]:
                continue
            keep_states = np.random.default_rng(n).choice(n_tr, int(n_tr * n / occ[name]), replace=False)
            o, _ = score(mem.subset(keep_states).knn(vecs, acts))
            p19[name][str(n)] = o[name]["R"]["precision, recall"]
    rep["P19: R's precision and recall against the memory's occurrences of the kind"] = p19
    print("P19", json.dumps(p19), flush=True)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"crafter_{NET.stem}{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print("done", rep["seconds"], flush=True)


if __name__ == "__main__":
    main()
