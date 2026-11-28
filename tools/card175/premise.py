"""Card 175, data check (no training): do the premises of a decorrelation penalty hold on the fits that exist?

  premise.py <name> ...   -> runs/175/premise.json     (validation episodes; truth read for scoring only)

Per fit: correlation between coordinates and the penalty it would pay (sum over i != j of corr^2); which variable moves each
coordinate (mean 5-step change where only that variable differs); linear R^2 of the cube's two table axes and of the drawer
from each coordinate and from sets of coordinates (fitted on half the episodes, tested on the other half).
Also: the correlation between the scene's true variables in the training episodes, which a decorrelated state cannot copy.
"""
import json, os, sys
import numpy as np
import torch
import factors as FA
import score as SC
import score173 as S3

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", 0.5)))
OUT = FA.ROOT / "runs/175"
NAMES, GAP, T = SC.NAMES, S3.GAP, FA.T
mv = FA.load_meta("val")
Xv = np.load(FA.OUT / "feat_val.npy", mmap_mode="r").reshape(-1, 1024, 64)
Xt = np.load(FA.OUT / "feat_train.npy", mmap_mode="r")
E = mv["drawer"].shape[0]
ep = np.repeat(np.arange(E), T); tr, te = ep < E // 2, ep >= E // 2
tab = SC.on_table(mv["cube"]).reshape(-1)


def r2(X, y, a=tr, b=te):
    X1 = np.c_[X, np.ones(len(X))]
    w, *_ = np.linalg.lstsq(X1[a], y[a], rcond=None)
    p = X1[b] @ w
    return round(float(1 - ((y[b] - p) ** 2).sum() / ((y[b] - y[b].mean()) ** 2).sum()), 3)


def pen(C):
    return round(float((C ** 2).sum() - (np.diag(C) ** 2).sum()), 3)


def truth_corr():
    mt = FA.load_meta("train")
    tb = SC.on_table(mt["cube"]).reshape(-1)
    V = np.stack([mt["cube"][..., 0].reshape(-1), mt["cube"][..., 1].reshape(-1), mt["drawer"].reshape(-1), mt["window"].reshape(-1),
                  mt["locks"][..., 0].reshape(-1).astype(float), mt["locks"][..., 1].reshape(-1).astype(float)], 1)
    names = ["cube_x", "cube_y", "drawer", "window", "lock0", "lock1"]
    r = {}
    for tag, m in (("all_frames", np.ones(len(V), bool)), ("cube_on_table", tb)):
        C = np.corrcoef(V[m].T)
        r[tag] = dict(frames=int(m.sum()), corr=[[round(float(x), 2) for x in row] for row in C], penalty=pen(C),
                      largest={f"{names[i]}~{names[j]}": round(float(C[i, j]), 2) for i in range(6) for j in range(i) if abs(C[i, j]) >= 0.15})
    r["cube_on_table_share_pct"] = round(100 * float(tb.mean()), 1)
    return names, r


def main():
    names, tc = truth_corr()
    out = dict(truth=dict(names=names, **tc), fits={})
    print("truth", json.dumps(tc), flush=True)
    D = S3.differ(mv); n = D.sum(-1)
    cx, cy, dr = mv["cube"][..., 0].reshape(-1), mv["cube"][..., 1].reshape(-1), mv["drawer"].reshape(-1)
    for name in sys.argv[1:]:
        p = FA.ROOT / "runs/174" / f"{name}.pt"
        ck = torch.load(p if p.exists() else S3.OUT / f"{name}.pt", map_location=FA.DEV, weights_only=False)
        K = ck["K"]
        net = FA.Factors(K=K).to(FA.DEV); net.load_state_dict(ck["state_dict"]); net.eval()
        S_tr = SC.encode(net, Xt[:100].reshape(-1, 1024, 64)).reshape(100, T, K)
        S = SC.encode(net, Xv); Sv = S.reshape(E, T, K)
        used, groups, _ = S3.factor_rule(S_tr)
        u = np.where(used)[0]
        C = np.corrcoef(S[:, u].T)
        c = np.abs(Sv[:, GAP:] - Sv[:, :-GAP])
        mov = np.stack([c[D[..., i] & (n == 1)].mean(0) for i in range(5)], 0)             # [variable, coordinate]
        dom = mov.argmax(0)
        cube_k = [int(k) for k in u if dom[k] == 0]
        drawer_k = [int(k) for k in u if dom[k] == 1]
        r = dict(K=K, used=[int(k) for k in u], factors=[[int(k) for k in g] for g in groups], penalty=pen(C),
                 pairs_over_half={f"{int(u[i])}~{int(u[j])}": round(float(C[i, j]), 2) for i in range(len(u)) for j in range(i) if abs(C[i, j]) >= 0.5},
                 variable_moving_each_coordinate={int(k): NAMES[dom[k]] for k in u},
                 change_per_coordinate={v: [round(float(x), 3) for x in mov[i]] for i, v in enumerate(NAMES)},
                 nothing_differs_per_coordinate=[round(float(x), 3) for x in c[n == 0].mean(0)],
                 std=[round(float(x), 3) for x in S.std(0)],
                 cube_coordinates=cube_k, drawer_coordinates=drawer_k,
                 cube_x=dict(whole=r2(S[:, u], cx, tr & tab, te & tab), cube_coords=r2(S[:, cube_k], cx, tr & tab, te & tab) if cube_k else None,
                             each={int(k): r2(S[:, [k]], cx, tr & tab, te & tab) for k in u}),
                 cube_y=dict(whole=r2(S[:, u], cy, tr & tab, te & tab), cube_coords=r2(S[:, cube_k], cy, tr & tab, te & tab) if cube_k else None,
                             each={int(k): r2(S[:, [k]], cy, tr & tab, te & tab) for k in u}),
                 drawer=dict(whole=r2(S[:, u], dr), each={int(k): r2(S[:, [k]], dr) for k in drawer_k}))
        if len(cube_k) > 1:
            Cc = np.corrcoef(S[:, cube_k].T)
            r["cube_coordinates_corr"] = [[round(float(x), 2) for x in row] for row in Cc]
        if len(drawer_k) > 1:
            r["drawer_coordinates_corr"] = [[round(float(x), 2) for x in row] for row in np.corrcoef(S[:, drawer_k].T)]
        out["fits"][name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(out, open(OUT / "premise.json", "w"), indent=1)
    import resource
    print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
