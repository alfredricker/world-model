"""Card 174: scoring. Truth is read here and nowhere else.

  score174.py [--set val|fresh] <name> ...      runs/174/<name>.pt (or runs/173/<name>.pt) -> runs/174/score_<set>_<name>.json
  score174.py [--set val|fresh] random8         the encoder with random weights, K = 8 (trivial baseline)

Factors and criterion 2(a), 2(b): card 173's (tools/card173/score173.py), over 5-step windows.
Criterion 1: each variable read through its own factor (the 'own' AUROC of card 172's scoring); whole state reported.
Criterion 2(c): per variable, its factor's mean change over windows in which that variable does not differ, as a share of
  its mean change over windows in which only that variable differs. Windows with a lock flip within 3 steps of either end
  are left out (the hand is on a button; section 4) and reported apart.
"""
import json, sys
import numpy as np
import torch
import factors as FA
import score as SC
import score173 as S3

OUT = FA.ROOT / "runs/174"
import os
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", 0.5)))      # at most half the card
DEV, NAMES, GAP, T = FA.DEV, SC.NAMES, S3.GAP, FA.T
SET = FA.arg("--set", "val")


def load_set():
    if SET == "val":
        return FA.load_meta("val"), np.load(FA.OUT / "feat_val.npy", mmap_mode="r").reshape(-1, 1024, 64)
    m = np.load(OUT / "meta_fresh.npz")
    return {k: m[k] for k in m.files}, np.load(OUT / "feat_fresh.npy", mmap_mode="r").reshape(-1, 1024, 64)


def pair_sets(mv):
    rng = np.random.default_rng(0)
    e, i, j = SC.val_pairs(mv, rng)
    cell, same = SC.cells(mv["cube"][e, i], mv["cube"][e, j], mv["drawer"][e, i] - mv["drawer"][e, j], mv["window"][e, i] - mv["window"][e, j],
                          mv["locks"][e, i, 0] - mv["locks"][e, j, 0], mv["locks"][e, i, 1] - mv["locks"][e, j, 1], SC.on_table(mv["cube"][e, i]) & SC.on_table(mv["cube"][e, j]))
    return {SET: (cell, same, e * T + i, e * T + j)}


def near_flip(mv, gap=GAP, w=3):
    lk = mv["locks"]; E = lk.shape[0]
    flip = np.zeros((E, T), bool); flip[:, 1:] = (lk[:, 1:] != lk[:, :-1]).any(-1)
    near = np.zeros((E, T), bool)
    for d in range(-w, w + 1):
        sh = np.zeros((E, T), bool)
        if d >= 0:
            sh[:, d:] = flip[:, :T - d]
        else:
            sh[:, :d] = flip[:, -d:]
        near |= sh
    return near[:, gap:] | near[:, :-gap]


def crit2c(Sv, used, groups, best, mv):
    D = S3.differ(mv); n = D.sum(-1); nf = near_flip(mv)
    c = np.abs(Sv[:, GAP:] - Sv[:, :-GAP]) * used[None, None]
    tot = c.sum(-1)
    r = dict(per_variable_pct={}, per_variable_all_windows_pct={}, per_variable_hand_on_button_pct={})
    for i, v in enumerate(NAMES):
        f = c[..., groups[best[i]]].sum(-1)
        own = f[D[..., i] & (n == 1)].mean()
        r["per_variable_pct"][v] = round(100 * float(f[~D[..., i] & ~nf].mean() / own), 1)
        r["per_variable_all_windows_pct"][v] = round(100 * float(f[~D[..., i]].mean() / own), 1)
        r["per_variable_hand_on_button_pct"][v] = round(100 * float(f[~D[..., i] & nf].mean() / own), 1)
    one = tot[n == 1].mean()
    r["whole_state_nothing_differs_pct"] = dict(all_windows=round(100 * float(tot[n == 0].mean() / one), 1), hand_off_buttons=round(100 * float(tot[(n == 0) & ~nf].mean() / one), 1))
    r["windows_hand_on_button_pct"] = round(100 * float(nf.mean()), 1)
    r["worst_pct"] = max(r["per_variable_pct"].values())
    return r


def main():
    mv, Xv = load_set()
    sets = pair_sets(mv)
    Xt = np.load(FA.OUT / "feat_train.npy", mmap_mode="r")
    E = mv["drawer"].shape[0]
    for name in [a for a in sys.argv[1:] if not a.startswith("--") and a not in ("val", "fresh")]:
        if name == "slots168":                                         # card 168's rule-built slots, each variable on its own slot
            import model171 as M1
            M = M1.M
            ck8 = torch.load(FA.ROOT / "runs/168/scene_wnet_0.pt", map_location=DEV, weights_only=False)
            n8 = M.SceneWNet(n_slots=ck8["n_slots"], mov_si=ck8["mov_si"]).to(DEV); n8.load_state_dict(ck8["state_dict"]); n8.eval()
            vm = ck8["var_map"]
            parts = [M.encode_all(n8, np.asarray(Xv[i:i + 2048], np.float32)) for i in range(0, len(Xv), 2048)]
            Val, A = np.concatenate([q[1] for q in parts], 0), np.concatenate([q[3] for q in parts], 0)
            cell, same, ia, ib = sets[SET]
            own = dict(cube=np.linalg.norm(A[ia] - A[ib], axis=1), **{v: np.abs(Val[ia, vm[v]] - Val[ib, vm[v]]) for v in ("drawer", "window", "lock0", "lock1")})
            r = dict(set=SET, reading_own={v: SC.auroc(own[v][cell[v]], own[v][same]) for v in NAMES}, pairs={v: int(cell[v].sum()) for v in NAMES})
            json.dump(r, open(OUT / f"score_{SET}_{name}.json", "w"), indent=1)
            print(f"[{SET} {name}] own {r['reading_own']}", flush=True)
            continue
        if name.startswith("random"):
            K = int(name[6:]); torch.manual_seed(123)
            net = FA.Factors(K=K).to(DEV).eval(); ck = dict(K=K, seed=None, steps=0)
        else:
            p = OUT / f"{name}.pt"
            ck = torch.load(p if p.exists() else S3.OUT / f"{name}.pt", map_location=DEV, weights_only=False)
            K = ck["K"]
            net = FA.Factors(K=K).to(DEV); net.load_state_dict(ck["state_dict"]); net.eval()
        S_tr = SC.encode(net, Xt[:100].reshape(-1, 1024, 64)).reshape(100, T, K)
        S = SC.encode(net, Xv); Sv = S.reshape(E, T, K)
        used, groups, _ = S3.factor_rule(S_tr)
        if not groups:
            used, groups = np.ones(K, bool), [[k] for k in range(K)]
        c2, best = S3.crit2(Sv, used, groups, mv)

        def dist(nm, ia, ib):
            d = np.abs(S[ia] - S[ib]) * used[None]
            return dict(whole=d.sum(-1), own={v: d[:, groups[best[i]]].sum(-1) for i, v in enumerate(NAMES)})
        c1 = SC.crit1(dist, sets)[SET]
        r = dict(set=SET, K=K, seed=ck["seed"], steps=ck["steps"], reading_own={v: c1[v]["own"] for v in NAMES}, reading_whole={v: c1[v]["whole"] for v in NAMES},
                 pairs={v: c1[v]["n"] for v in NAMES}, criterion2=c2, criterion2c=crit2c(Sv, used, groups, best, mv),
                 std_per_coordinate=[round(float(x), 3) for x in S_tr.reshape(-1, K).std(0)])
        if not name.startswith("random") and not ck.get("bound") and SET == "val":
            ckg = torch.load(FA.OUT / "g.pt", map_location=DEV, weights_only=False)
            g = FA.PoseTerm().to(DEV); g.load_state_dict(ckg["state_dict"]); g.eval()
            r["held_out"] = S3.val_nce(Sv, torch.from_numpy(((mv["pose"] - ckg["mu"]) / ckg["sd"]).astype(np.float32)).to(DEV), g)
        json.dump(r, open(OUT / f"score_{SET}_{name}.json", "w"), indent=1)
        print(f"[{SET} {name}] K {K} own {r['reading_own']} | whole {r['reading_whole']} | factors {c2['n_factors']} {c2['factors']} distinct {c2['distinct']} share {c2['share_in_best']}"
              f" | 2c {r['criterion2c']} | held-out {r.get('held_out')}", flush=True)


if __name__ == "__main__":
    main()
    import resource
    print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB", flush=True)
