"""Card 110 in breaking world 1: confidence for System 1's predictions, and conditions read through it.

105.4's events and replacements (one tile replaced at a time by another kind's vector seen in the same state), on
105.6's models. Arms, all from one pass:
  none: every replacement counts (105.4);
  P:    the model's own probability of the predicted outcome, temperature-scaled on held-out real tokens;
  K:    an independently trained copy predicts the same outcome (106);
  R:    P x familiarity by recall (tools/card110/common.py);
  RK:   R and K both;
  oracle (gate, scoring only): the tile's 5 x 5 surroundings of kinds, with the action, occur in training.
P and R keep a replaced state if its confidence is at least the 5th percentile of the confidences of the real events of
that kind (held-out states); an event whose own real state falls below is not asked, as K does with disagreements.

Scoring only: kinds from the evaluator's grid; the truth is the world's own rule on the replaced grid (105.4).

  bin/prun python tools/card110/world1.py --net runs/105.6/tokens_Rb_s1021_net.pt --copy runs/105.6/tokens_Rb_s7_net.pt
      [--per 300] [--smoke] → runs/110/world1_<net name>.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card105"))
sys.path.insert(0, str(ROOT / "tools" / "card110"))
from model import TransitionModel                          # noqa: E402
import common as CM                                        # noqa: E402

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
NET, COPY = Path(arg("--net", "runs/105.6/tokens_Rb_s1021_net.pt")), Path(arg("--copy", "runs/105.6/tokens_Rb_s7_net.pt"))
PER = int(arg("--per", 300))
SMOKE = "--smoke" in sys.argv
spec = importlib.util.spec_from_file_location("c1054w1", ROOT / "tools" / "card105.4" / "world1.py")
W1 = importlib.util.module_from_spec(spec)
sys.argv = ["world1.py"]
spec.loader.exec_module(W1)
OUT = ROOT / "runs" / "110"
dev = "cuda"
NT, W, H = W1.NT, W1.W, W1.H
FLOOR, WALL, AGENT, BOX, LAMP, LIT = range(6)
ARMS = ("none", "P", "K", "R", "RK", "oracle")
N_MEM_STATES = 8000 if SMOKE else 40000
P19 = (10, 30, 100, 300)


def load(path, dim):
    net = TransitionModel(dim, 5, arm=path.name.split("_")[1]).to(dev)
    net.load_state_dict(torch.load(ROOT / path))
    return net.eval()


def patch_keys(sem63, cells, act):
    """Hash of the 5 x 5 kinds around each cell (padding 7) with the action: the oracle's familiarity (scoring only)."""
    g = np.full((len(sem63), H + 4, W + 4), 7, np.int64)
    g[:, 2:-2, 2:-2] = sem63.reshape(-1, H, W)
    y, x = cells // W, cells % W
    key = np.asarray(act, np.uint64).copy()
    for dy in range(5):
        for dx in range(5):
            key = key * np.uint64(1000003) + g[np.arange(len(cells)), y + dy, x + dx].astype(np.uint64)
    return key


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1021)
    z = np.load(ROOT / "runs" / "card102" / "world1_test.npz")
    sem = z["sem"].reshape(-1, 63).astype(np.int64)
    act_te = z["act"].astype(np.int64)
    V = np.load(ROOT / "runs" / "102.1" / "world1_vec_test.npy")
    C = np.load(ROOT / "runs" / "102.1" / "world1_code_test.npy").astype(np.int64)
    ztr = np.load(ROOT / "runs" / "card102" / "world1_train.npz")
    sem_tr = ztr["sem"].reshape(-1, 63).astype(np.int64)
    act_tr = ztr["act"].astype(np.int64)
    Ctr = np.load(ROOT / "runs" / "102.1" / "world1_code_train.npy").astype(np.int64)
    Vtr = np.load(ROOT / "runs" / "102.1" / "world1_vec_train.npy", mmap_mode="r")
    kind_of = np.full(64, -1)                                                          # scoring only, as 105.4
    for k in np.unique(Ctr[:, :63]):
        kind_of[k] = np.bincount(sem_tr.reshape(-1)[Ctr[:, :63].reshape(-1) == k]).argmax()
    Pr = torch.as_tensor(np.load(ROOT / "runs" / "102.1" / "world1_codes.npz")["protos"], device=dev).float()
    kind_mask = torch.as_tensor(np.stack([kind_of[:len(Pr)] == k for k in range(6)]), device=dev).float()  # (6, C)
    net, net2 = load(NET, V.shape[-1]), load(COPY, V.shape[-1])
    vec = CM.Vectors(net)
    Pt = W1.P_TILE.to(dev)
    Ft = W1.F_TILE.to(dev)

    @torch.no_grad()
    def run(m, Vb, a):
        """Next-code squared distances (B, 63, C) and, for net, the last-layer vectors (B, 63, d)."""
        Vb = torch.as_tensor(Vb, device=dev).float()
        B = len(Vb)
        a = torch.as_tensor(np.broadcast_to(np.asarray(a), (B,)).copy(), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt, _ = m(Vb, Pt.expand(B, NT, 2), Ft.expand(B, NT), torch.ones(B, NT, dtype=torch.bool, device=dev), a)
        d2 = torch.cdist(nxt[:, :63].float().reshape(-1, Pr.shape[1]), Pr).view(B, 63, -1) ** 2
        return d2, (vec.v[:, :63] if m is net else None)

    # ---- memory: the model's vectors over training (all tiles of N states, and every tile whose code changed)
    valid_tr = np.flatnonzero(np.r_[ztr["ep"][1:] == ztr["ep"][:-1], False] & ~ztr["done"])
    if SMOKE:
        valid_tr = np.sort(rng.choice(valid_tr, 60000, replace=False))
    whole = set(rng.choice(valid_tr, N_MEM_STATES, replace=False).tolist())
    mv, ma, ms = [], [], []
    for s in range(0, len(valid_tr), 2048):
        t = valid_tr[s:s + 2048]
        _, v = run(net, np.asarray(Vtr[t]), act_tr[t])
        ch = Ctr[t + 1, :63] != Ctr[t, :63]
        keep = ch | np.isin(t, list(whole))[:, None]
        r, c = np.nonzero(keep)
        mv.append(v[torch.as_tensor(r, device=dev), torch.as_tensor(c, device=dev)].cpu().numpy())
        ma.append(act_tr[t][r])
        ms.append(t[r])
    mem = CM.Memory(np.concatenate(mv), np.concatenate(ma), np.concatenate(ms))
    print("memory", len(mem.vec), f"({time.monotonic() - t0:.0f} s)", flush=True)
    rep = {"net": str(NET), "copy": str(COPY), "memory vectors": int(len(mem.vec)), "smoke": SMOKE}

    # ---- oracle keys over training (scoring only)
    okeys = np.unique(np.concatenate([patch_keys(sem_tr[valid_tr], np.full(len(valid_tr), c), act_tr[valid_tr])
                                      for c in range(63)]))

    def known(keys):
        pos = np.searchsorted(okeys, keys).clip(max=len(okeys) - 1)
        return okeys[pos] == keys

    # ---- events (105.4's draw, same seed) and held-out real tiles for the fits, from other test steps
    W1.PER = PER
    ev = W1.events(z, rng)
    ev_t = {t for lst in ev.values() for t, _ in lst}
    valid_te = np.flatnonzero(np.r_[z["ep"][1:] == z["ep"][:-1], False] & ~z["done"])
    pool = np.array([t for t in rng.permutation(valid_te)[:12000] if t not in ev_t])
    n_fit = 1000 if SMOKE else 4000
    fit_t, eval_t = pool[:n_fit], pool[n_fit:n_fit + n_fit // 2]

    def tiles(ts):
        d2, v = [], []
        for s in range(0, len(ts), 1024):
            t = ts[s:s + 1024]
            a, b = run(net, V[t], act_te[t])
            d2.append(a.cpu().numpy())
            v.append(b.cpu().numpy())
        d2, v = np.concatenate(d2), np.concatenate(v)
        return d2.reshape(-1, d2.shape[-1]), v.reshape(-1, v.shape[-1]), np.repeat(act_te[ts], 63), \
            C[ts + 1, :63].reshape(-1), C[ts, :63].reshape(-1), sem[ts].reshape(-1)

    d2f, vf, af, yf, _, _ = tiles(fit_t)
    tau = CM.fit_temperature(-d2f, yf)
    pf_all = torch.softmax(torch.as_tensor(-d2f / tau), -1).numpy()
    pf, rightf = pf_all.max(1), (pf_all.argmax(1) == yf).astype(float)
    Df = mem.knn(vf, af)
    kj, ab, ktab = CM.choose_k(pf, Df, rightf)
    rep["fit"] = {"tau": round(tau, 5), "tiles": int(len(pf)), "wrong tiles": int((rightf == 0).sum()),
                  "k chosen": CM.KS[kj], "alpha, beta": [round(ab[0], 4), round(ab[1], 4)], "by k": ktab}
    print("fit", rep["fit"], f"({time.monotonic() - t0:.0f} s)", flush=True)

    # ---- criterion 1: calibration on held-out real tiles, by action and changed or not, and lamps
    d2e, ve, ae, ye, y0e, seme = tiles(eval_t)
    pe_all = torch.softmax(torch.as_tensor(-d2e / tau), -1).numpy()
    pe, righte = pe_all.max(1), (pe_all.argmax(1) == ye).astype(float)
    De = mem.knn(ve, ae)[:, kj]
    ce = pe * CM.applic(np.log(De + 1e-6), ab)
    c1 = {}
    for a in range(5):
        for part, m in (("all tiles", np.ones(len(pe), bool)), ("tiles whose code changed", ye != y0e),
                        ("lamps", np.isin(seme, (LAMP, LIT)))):
            m = m & (ae == a)
            if m.sum() >= 100:
                c1[f"action {a}, {part}"] = {"n": int(m.sum()), "ECE P": CM.ece(pe[m], righte[m]),
                                             "ECE R": CM.ece(ce[m], righte[m]), "accuracy": round(float(righte[m].mean()), 4)}
    rep["1 calibration"] = c1
    rep["1 reliability R (confidence, accuracy, n)"] = CM.reliability(ce, righte)
    print("criterion 1", json.dumps({k: (v["ECE P"], v["ECE R"]) for k, v in c1.items()}), flush=True)

    # ---- events: replacements, outcomes, confidences
    rows = []                 # per replacement: kind, event no, j, pred kind, copy kind, p, truth kind, possible, oracle
    vecs = []
    base = []                 # per event: kind, asked, p, copy agrees, vector index, oracle, truth conds
    for kind, lst in ev.items():
        for t, i in lst:
            a = int(act_te[t])
            g = sem[t].reshape(H, W)
            base_true = W1.true_next(g, a).reshape(-1)[i]
            d2, v = run(net, V[t][None], a)
            p_k = (torch.softmax(-d2[0, i] / tau, -1) @ kind_mask.T).cpu().numpy()
            b_kind = int(kind_of[int(d2[0, i].argmin())])
            if b_kind != base_true:
                base.append(dict(kind=kind, asked=False))
                continue  # only events whose unaltered outcome is predicted right are asked (105.4)
            d2c, _ = run(net2, V[t][None], a)
            b_known = int(kind_of[int(d2c[0, i].argmin())]) == b_kind
            vecs.append(v[0, i].cpu().numpy())
            e = dict(kind=kind, asked=True, bk=b_kind, p=float(p_k[b_kind]), known=b_known, vi=len(vecs) - 1, a=a, i=i,
                     oracle=known(patch_keys(sem[t][None], np.array([i]), [a]))[0],
                     tc=np.zeros(63, bool), r0=len(rows))
            present = {k: int(np.flatnonzero(sem[t] == k)[0]) for k in (FLOOR, WALL, BOX, LAMP, LIT) if (sem[t] == k).any()}
            var, meta, g2s = [], [], []
            v0 = V[t].astype(np.float32)
            for j in range(63):
                if j == i:
                    continue
                for k in ([FLOOR] if sem[t, j] == AGENT else [k for k in present if k != sem[t, j]]):
                    g2 = g.copy().reshape(-1)
                    g2[j] = k
                    tk = W1.true_next(g2.reshape(H, W), a).reshape(-1)[i]
                    g3 = g2.reshape(H, W).copy()
                    W1.light(g3)
                    e["tc"][j] |= tk != base_true
                    vj = v0.copy()
                    vj[j] = v0[present[k]]
                    var.append(vj)
                    meta.append((j, int(tk), bool((g3.reshape(-1) == g2).all())))
                    g2s.append(g2)
            orcs = known(patch_keys(np.stack(g2s), np.full(len(g2s), i), np.full(len(g2s), a))) if g2s else []
            for s in range(0, len(var), 512):
                Vb = np.stack(var[s:s + 512])
                d2, v = run(net, Vb, a)
                d2c, _ = run(net2, Vb, a)
                pk = (torch.softmax(-d2[:, i] / tau, -1) @ kind_mask.T).cpu().numpy()
                kk = kind_of[d2[:, i].argmin(1).cpu().numpy()]
                kc = kind_of[d2c[:, i].argmin(1).cpu().numpy()]
                vv = v[:, i].cpu().numpy()
                for q in range(len(Vb)):
                    (j, tk, poss), orc = meta[s + q], bool(orcs[s + q])
                    vecs.append(vv[q])
                    rows.append((kind, len(base), j, int(kk[q]), int(kc[q]), float(pk[q, kk[q]]), tk, poss, orc,
                                 len(vecs) - 1, a))
            e["r1"] = len(rows)
            base.append(e)
        print(kind, len(lst), "events", len(rows), "replacements", f"({time.monotonic() - t0:.0f} s)", flush=True)

    vecs = np.stack(vecs).astype(np.float16)
    acts = np.zeros(len(vecs), np.int64)
    for e in base:
        if e["asked"]:
            acts[e["vi"]] = e["a"]
    for r in rows:
        acts[r[9]] = r[10]
    D_all = mem.knn(vecs, acts)

    def score(D_all, tag=""):
        """Precision and recall of the conditions per kind and arm; criterion 2's AUROC per arm."""
        out, au = {}, {}
        appl = CM.applic(np.log(D_all[:, kj] + 1e-6), ab)
        R_kind, R_pk, R_kk, R_tk, R_poss, R_orc, R_vi, R_known = [], [], [], [], [], [], [], []
        for kind in ev:
            es = [e for e in base if e["kind"] == kind and e["asked"]]
            bP = np.array([e["p"] for e in es])
            bR = bP * appl[[e["vi"] for e in es]]
            thrP, thrR = (np.percentile(bP, 5), np.percentile(bR, 5)) if len(es) else (0, 0)
            st = {a: [0, 0, 0, 0] for a in ARMS}          # tp, fp, fn, asked
            for e in es:
                rr = rows[e["r0"]:e["r1"]]
                js = np.array([r[2] for r in rr], np.int64)
                pred = np.array([r[3] for r in rr])
                copy = np.array([r[4] for r in rr])
                pk = np.array([r[5] for r in rr])
                truth = np.array([r[6] for r in rr])
                orc = np.array([r[8] for r in rr], bool)
                cR = pk * appl[[r[9] for r in rr]]
                flips = pred != e["bk"]
                keep = {"none": np.ones(len(rr), bool), "P": pk >= thrP, "K": copy == pred, "R": cR >= thrR}
                keep["RK"] = keep["R"] & keep["K"]
                keep["oracle"] = orc
                ask = {"none": True, "P": e["p"] >= thrP, "K": e["known"], "R": e["p"] * appl[e["vi"]] >= thrR,
                       "oracle": True}
                ask["RK"] = ask["R"] and ask["K"]
                for arm in ARMS:
                    if not ask[arm]:
                        continue
                    mc = np.zeros(63, bool)
                    np.logical_or.at(mc, js[keep[arm] & flips], True)
                    s_ = st[arm]
                    s_[0] += int((mc & e["tc"]).sum()); s_[1] += int((mc & ~e["tc"]).sum())
                    s_[2] += int((~mc & e["tc"]).sum()); s_[3] += 1
                for r, c_r in zip(rr, cR):
                    R_kind.append(kind); R_pk.append(r[5]); R_kk.append(r[3]); R_tk.append(r[6]); R_poss.append(r[7])
                    R_orc.append(r[8]); R_vi.append(c_r); R_known.append(r[4] == r[3])
            out[kind] = {arm: {"asked": st[arm][3], "precision, recall": list(CM.prf(*st[arm][:3]))} for arm in ARMS}
            out[kind]["thresholds P, R"] = [round(float(thrP), 4), round(float(thrR), 4)]
        wrong = np.array(R_kk) != np.array(R_tk)
        kinds = np.array(R_kind)
        conf = {"P": np.array(R_pk), "K": np.array(R_known, float), "R": np.array(R_vi),
                "oracle": np.array(R_orc, float)}
        conf["RK"] = conf["R"] * conf["K"]
        for arm, c in conf.items():
            au[arm] = {"all": CM.auroc(-c, wrong)}
            for kind in ev:
                m = kinds == kind
                au[arm][kind] = CM.auroc(-c[m], wrong[m])
        imp = ~np.array(R_poss)
        au["R: AUROC for impossible states (diagnostic)"] = CM.auroc(-conf["R"], imp)
        au["K: AUROC for impossible states (diagnostic)"] = CM.auroc(-conf["K"], imp)
        au["P: AUROC for impossible states (diagnostic)"] = CM.auroc(-conf["P"], imp)
        rowD = D_all[[r[9] for r in rows]]
        au["raw distance (diagnostic), by k: AUROC for impossible, for wrong"] = {
            str(k): [CM.auroc(rowD[:, j], imp), CM.auroc(rowD[:, j], wrong)] for j, k in enumerate(CM.KS)}
        lo = kinds == "lamp goes out"
        au["raw distance (diagnostic), lamp goes out, by k: AUROC for impossible, for wrong"] = {
            str(k): [CM.auroc(rowD[lo, j], imp[lo]), CM.auroc(rowD[lo, j], wrong[lo])] for j, k in enumerate(CM.KS)}
        au["replacements, wrong, impossible"] = [int(len(wrong)), int(wrong.sum()), int(imp.sum())]
        au["mean R confidence: possible and right, possible and wrong, impossible"] = [
            round(float(conf["R"][~imp & ~wrong].mean()), 4) if (~imp & ~wrong).any() else None,
            round(float(conf["R"][~imp & wrong].mean()), 4) if (~imp & wrong).any() else None,
            round(float(conf["R"][imp].mean()), 4) if imp.any() else None]
        return out, au

    out, au = score(D_all)
    rep["3 conditions"] = out
    rep["2 AUROC for a wrong prediction"] = au
    for kind in ev:
        print(kind, json.dumps({a: out[kind][a]["precision, recall"] for a in ARMS}), flush=True)
    print("criterion 2", json.dumps({a: au[a]["all"] for a in ("P", "K", "R", "RK", "oracle")}),
          au["R: AUROC for impossible states (diagnostic)"], flush=True)

    # ---- P19: the memory holding about 10, 30, 100, 300 training occurrences of each kind of event
    W1.PER = 10 ** 9
    ev_tr = W1.events(ztr, np.random.default_rng(0))
    p19 = {}
    for kind in ev:
        n_kind = len(ev_tr[kind])
        p19[kind] = {"training occurrences": n_kind}
        for n in P19:
            if n >= n_kind:
                continue
            keep_states = np.random.default_rng(n).choice(valid_tr, int(len(valid_tr) * n / n_kind), replace=False)
            sub = mem.subset(keep_states)
            o, _ = score(sub.knn(vecs, acts))
            p19[kind][str(n)] = o[kind]["R"]["precision, recall"]
    rep["P19: R's precision and recall against the memory's occurrences of the kind"] = p19
    print("P19", json.dumps(p19), flush=True)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"world1_{NET.stem}{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print("done", rep["seconds"], flush=True)


if __name__ == "__main__":
    main()
