"""Card 172: scoring. Truth is read here and nowhere else.

  score.py <name> [<name> ...]     runs/172/<name>.pt -> runs/172/score_<name>.json
  score.py baselines               the pose alone, 32 principal components of the mean patch vector, a random encoder,
                                   and card 168's SceneWNet slots (the reference built with rules)

Criterion 1  for pairs of frames of one episode: the state's distance separates pairs in which exactly one variable
             differs (drawer or window by 4-10 cm, a lock, the cube on the table by over 4 cm) from pairs in which no
             variable differs (cube and slides within 1 cm, locks equal). AUROC, on the 100 validation episodes (pairs up
             to 40 steps apart) and on card 141's 300 test episodes (frames before touches, every pair of an episode).
Criterion 2  factors read by the card's rule from 100 training episodes, no labels; then on validation steps:
             (a) each variable's best factor is a different one; (b) the share of the state's change that lies in that
             factor over the steps in which only that variable changes (ratio of sums); (c) the state's mean change in
             steps where no variable changes over its mean change where exactly one does.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import factors as FA

ROOT, OUT, DEV = FA.ROOT, FA.OUT, FA.DEV
NAMES = ["cube", "drawer", "window", "lock0", "lock1"]
USED_STD, MOVE_X, MOVE_MIN, TIE = 0.02, 5.0, 0.02, 0.5        # the factor rule's four constants (card, section 2)


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    x = np.concatenate([pos, neg]); r = np.empty(len(x)); o = np.argsort(x, kind="stable"); r[o] = np.arange(1, len(x) + 1)
    return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


@torch.no_grad()
def encode(net, X):
    out = []
    for i in range(0, len(X), 256):
        out.append(net(torch.from_numpy(np.asarray(X[i:i + 256], np.float32)).to(DEV)).cpu().numpy())
    return np.concatenate(out, 0)


def factor_rule(S):
    """S [E, T, K] on training frames, no labels -> used [K] bool, factors (list of coordinate lists)."""
    K = S.shape[-1]
    used = S.reshape(-1, K).std(0) >= USED_STD
    c = np.abs(np.diff(S, axis=1)).reshape(-1, K)
    moves = c > np.maximum(MOVE_X * np.median(c, 0), MOVE_MIN)[None]
    inter = (moves[:, :, None] & moves[:, None, :]).sum(0).astype(np.float64)
    n = moves.sum(0).astype(np.float64)
    union = n[:, None] + n[None, :] - inter
    tied = (inter / np.maximum(union, 1.0) >= TIE) & (union > 0)
    lab = -np.ones(K, int); groups = []
    for k in range(K):
        if not used[k] or lab[k] >= 0:
            continue
        stack, g = [k], []
        lab[k] = len(groups)
        while stack:
            a = stack.pop(); g.append(a)
            for b in np.where(tied[a] & used & (lab < 0))[0]:
                lab[b] = len(groups); stack.append(int(b))
        groups.append(sorted(g))
    return used, groups, dict(moves_share_per_coordinate=[round(float(x), 3) for x in moves.mean(0)])


def step_changes(m):
    c = m["cube"]
    return np.stack([np.linalg.norm(np.diff(c, axis=1), axis=-1) > 0.003, np.abs(np.diff(m["drawer"], axis=1)) > 0.003,
                     np.abs(np.diff(m["window"], axis=1)) > 0.003, np.diff(m["locks"][..., 0], axis=1) != 0, np.diff(m["locks"][..., 1], axis=1) != 0], -1)


def cells(cube_a, cube_b, dr, wi, l0, l1, table):
    """Per pair: which cell it is a positive of, and whether nothing differs. dr, wi, l0, l1: differences; table: both on the table."""
    cd = np.linalg.norm(cube_a - cube_b, axis=-1)
    c_same, d_same, w_same, l0_same, l1_same = cd <= 0.01, np.abs(dr) <= 0.01, np.abs(wi) <= 0.01, l0 == 0, l1 == 0
    return dict(
        cube=(cd > 0.04) & table & d_same & w_same & l0_same & l1_same,
        drawer=(np.abs(dr) > 0.04) & (np.abs(dr) <= 0.10) & c_same & w_same & l0_same & l1_same,
        window=(np.abs(wi) > 0.04) & (np.abs(wi) <= 0.10) & c_same & d_same & l0_same & l1_same,
        lock0=(l0 != 0) & c_same & d_same & w_same & l1_same,
        lock1=(l1 != 0) & c_same & d_same & w_same & l0_same,
    ), c_same & d_same & w_same & l0_same & l1_same


def on_table(c):
    ins = (np.abs(c[..., 0] - 0.33) < 0.12) & (c[..., 2] > 0.04) & (c[..., 1] < -0.1)
    return (~ins) & (c[..., 2] < 0.03)


def val_pairs(m, rng, max_gap=40, per_ep=6000):
    E, T = m["drawer"].shape
    e = np.repeat(np.arange(E), per_ep)
    i = rng.integers(0, T - 1, len(e)); j = np.minimum(i + rng.integers(1, max_gap + 1, len(e)), T - 1)
    return e, i, j


def crit1(dist_fn, sets):
    """dist_fn(set_name, a_index, b_index) -> dict(whole=[n], own={var: [n]} or None)."""
    res = {}
    for name, (cell, same, ia, ib) in sets.items():
        d = dist_fn(name, ia, ib)
        r = {}
        for v in NAMES:
            r[v] = dict(n=int(cell[v].sum()), whole=auroc(d["whole"][cell[v]], d["whole"][same]) if d.get("whole") is not None else None,
                        own=auroc(d["own"][v][cell[v]], d["own"][v][same]) if d.get("own") else None)
        r["n_same"] = int(same.sum())
        res[name] = r
    return res


def build_sets():
    rng = np.random.default_rng(0)
    mv = FA.load_meta("val")
    e, i, j = val_pairs(mv, rng)
    cell, same = cells(mv["cube"][e, i], mv["cube"][e, j], mv["drawer"][e, i] - mv["drawer"][e, j], mv["window"][e, i] - mv["window"][e, j],
                       mv["locks"][e, i, 0] - mv["locks"][e, j, 0], mv["locks"][e, i, 1] - mv["locks"][e, j, 1], on_table(mv["cube"][e, i]) & on_table(mv["cube"][e, j]))
    sets = {"validation": (cell, same, e * FA.T + i, e * FA.T + j)}
    import common as C141
    t = C141.truth(C141.TEST)
    ia, ib = [], []
    for ep in np.unique(t["ep"]):
        idx = np.where(t["ep"] == ep)[0]
        a, b = np.triu_indices(len(idx), 1)
        ia.append(idx[a]); ib.append(idx[b])
    ia, ib = np.concatenate(ia), np.concatenate(ib)
    tab = t["where"] == "table"
    cell, same = cells(t["cube"][ia], t["cube"][ib], t["drawer"][ia] - t["drawer"][ib], t["window"][ia] - t["window"][ib],
                       t["locks"][ia, 0] - t["locks"][ib, 0], t["locks"][ia, 1] - t["locks"][ib, 1], tab[ia] & tab[ib])
    sets["test frames (card 141)"] = (cell, same, ia, ib)
    return sets, mv


def test_frames():
    return np.concatenate([np.load(ROOT / f"runs/132/z_{n}.npy", mmap_mode="r") for n in ("t500", "t700", "val")], 0)


def crit2(S_val, used, groups, mv):
    ch = step_changes(mv); n = ch.sum(-1)
    c = np.abs(np.diff(S_val, axis=1)) * used[None, None]
    tot = c.sum(-1)
    share = np.zeros((5, len(groups)))
    for v in range(5):
        m = ch[..., v] & (n == 1)
        den = tot[m].sum()
        for f, g in enumerate(groups):
            share[v, f] = c[m][:, g].sum() / max(den, 1e-9)
    best = share.argmax(1) if len(groups) else np.zeros(5, int)
    arm_only, one = tot[n == 0].mean(), tot[n == 1].mean()
    return dict(
        n_used=int(used.sum()), n_factors=len(groups), factors=[[int(k) for k in g] for g in groups],
        best_factor={v: int(best[i]) for i, v in enumerate(NAMES)}, distinct=bool(len(set(best.tolist())) == 5 and len(groups) >= 5),
        share_in_best={v: round(100 * float(share[i, best[i]]), 1) if len(groups) else 0.0 for i, v in enumerate(NAMES)},
        steps_only={v: int((ch[..., i] & (n == 1)).sum()) for i, v in enumerate(NAMES)},
        arm_only_over_one_variable_pct=round(100 * float(arm_only / max(one, 1e-9)), 1),
        per_variable_change_over_arm_only={v: round(float(tot[ch[..., i] & (n == 1)].mean() / max(arm_only, 1e-9)), 1) for i, v in enumerate(NAMES)},
    ), best


def score_state(states, mv, sets, rule_S=None):
    """states: dict set name -> S [frames, K] (validation in episode order); rule_S [E, T, K] training frames or None."""
    Sv = states["validation"].reshape(mv["drawer"].shape[0], FA.T, -1)
    K = Sv.shape[-1]
    if rule_S is not None:
        used, groups, extra = factor_rule(rule_S)
    else:
        used, groups, extra = np.ones(K, bool), [[k] for k in range(K)], {}
    c2, best = crit2(Sv, used, groups, mv)

    def dist(name, ia, ib):
        S = states[name]
        d = np.abs(S[ia] - S[ib]) * used[None]
        return dict(whole=d.sum(-1), own={v: d[:, groups[best[i]]].sum(-1) for i, v in enumerate(NAMES)} if len(groups) else None)
    return dict(criterion1=crit1(dist, {k: v for k, v in sets.items() if k in states}), criterion2=c2, **extra)


def brief(name, r):
    c1 = r["criterion1"]
    line = f"[{name}]"
    for s in c1:
        line += f" | {s}: " + " ".join(f"{v} {c1[s][v]['whole']}/{c1[s][v]['own']}({c1[s][v]['n']})" for v in NAMES) + f" same {c1[s]['n_same']}"
    if "criterion2" in r:
        c2 = r["criterion2"]
        line += f" | used {c2['n_used']} factors {c2['n_factors']} distinct {c2['distinct']} share {c2['share_in_best']} arm-only/one {c2['arm_only_over_one_variable_pct']}%"
    print(line, flush=True)


def main():
    sets, mv = build_sets()
    Xv = np.load(OUT / "feat_val.npy", mmap_mode="r").reshape(-1, 1024, 64)
    if sys.argv[1] == "baselines":
        res = {}
        ck = torch.load(OUT / "g.pt", map_location="cpu", weights_only=False)
        pose = ((mv["pose"] - ck["mu"]) / ck["sd"]).reshape(-1, 12)
        res["pose alone"] = score_state({"validation": pose}, mv, sets)
        Xt = np.load(OUT / "feat_train.npy", mmap_mode="r")
        mean_tr = np.concatenate([np.asarray(Xt[e], np.float32).mean(1) for e in range(0, 100, 2)], 0)       # [frames, 64]
        mu = mean_tr.mean(0); _, sv, Vt = np.linalg.svd(mean_tr - mu, full_matrices=False)
        proj = lambda X: np.concatenate([((np.asarray(X[i:i + 512], np.float32).mean(1) - mu) @ Vt[:32].T) / (sv[:32] / np.sqrt(len(mean_tr))) for i in range(0, len(X), 512)], 0)
        Zt = test_frames()
        res["32 principal components of the mean patch vector"] = score_state({"validation": proj(Xv), "test frames (card 141)": proj(Zt)}, mv, sets)
        torch.manual_seed(123)
        net = FA.Factors(K=32).to(DEV).eval()
        rule_S = encode(net, Xt[:100].reshape(-1, 1024, 64)).reshape(100, FA.T, -1)
        res["random encoder"] = score_state({"validation": encode(net, Xv), "test frames (card 141)": encode(net, Zt)}, mv, sets, rule_S)
        # card 168's slots, scored on each variable's own slot (truth names the slot: bookkeeping, as in card 168)
        import model171 as M1
        M = M1.M
        ck8 = torch.load(ROOT / "runs/168/scene_wnet_0.pt", map_location=DEV, weights_only=False)
        n8 = M.SceneWNet(n_slots=ck8["n_slots"], mov_si=ck8["mov_si"]).to(DEV); n8.load_state_dict(ck8["state_dict"]); n8.eval()
        vm = ck8["var_map"]
        r8 = {}
        for name, X in (("validation", Xv), ("test frames (card 141)", Zt)):
            parts = [M.encode_all(n8, np.asarray(X[i:i + 2048], np.float32)) for i in range(0, len(X), 2048)]
            Val, A = np.concatenate([q[1] for q in parts], 0), np.concatenate([q[3] for q in parts], 0)
            cell, same, ia, ib = sets[name]
            own = dict(cube=np.linalg.norm(A[ia] - A[ib], axis=1), drawer=np.abs(Val[ia, vm["drawer"]] - Val[ib, vm["drawer"]]),
                       window=np.abs(Val[ia, vm["window"]] - Val[ib, vm["window"]]), lock0=np.abs(Val[ia, vm["lock0"]] - Val[ib, vm["lock0"]]),
                       lock1=np.abs(Val[ia, vm["lock1"]] - Val[ib, vm["lock1"]]))
            r8[name] = {v: dict(n=int(cell[v].sum()), whole=None, own=auroc(own[v][cell[v]], own[v][same])) for v in NAMES}
            r8[name]["n_same"] = int(same.sum())
        res["card 168's slots (own slot)"] = dict(criterion1=r8)
        json.dump(res, open(OUT / "score_baselines.json", "w"), indent=1)
        for k, v in res.items():
            brief(k, v)
        return
    Xt = np.load(OUT / "feat_train.npy", mmap_mode="r")
    Zt = test_frames()
    for name in sys.argv[1:]:
        ck = torch.load(OUT / f"{name}.pt", map_location=DEV, weights_only=False)
        net = FA.Factors(K=ck["K"]).to(DEV); net.load_state_dict(ck["state_dict"]); net.eval()
        rule_S = encode(net, Xt[:100].reshape(-1, 1024, 64)).reshape(100, FA.T, -1)
        r = score_state({"validation": encode(net, Xv), "test frames (card 141)": encode(net, Zt)}, mv, sets, rule_S)
        r.update(K=ck["K"], seed=ck["seed"], steps=ck["steps"], sec=ck["sec"], bound=ck["bound"], picks_true_pct=ck["picks_true_pct"],
                 tau=round(float(net.log_tau.exp()), 3), std_per_coordinate=[round(float(x), 3) for x in rule_S.reshape(-1, ck["K"]).std(0)])
        json.dump(r, open(OUT / f"score_{name}.json", "w"), indent=1)
        brief(name, r)


if __name__ == "__main__":
    main()
