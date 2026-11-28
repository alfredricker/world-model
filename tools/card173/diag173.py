"""Card 173, diagnostic after the gate (no training): where the state's change goes when nothing in the scene differs.
  diag173.py <name> ...  -> runs/173/diag_<name>.json"""
import json, sys
import numpy as np
import torch
import factors as FA
import score as SC
import score173 as S3

mv = FA.load_meta("val")
Xv = np.load(FA.OUT / "feat_val.npy", mmap_mode="r").reshape(-1, 1024, 64)
Xt = np.load(FA.OUT / "feat_train.npy", mmap_mode="r")
ckg = torch.load(FA.OUT / "g.pt", map_location="cpu", weights_only=False)
pose = ((mv["pose"] - ckg["mu"]) / ckg["sd"])
E, T, G = mv["drawer"].shape[0], FA.T, S3.GAP
ep = np.repeat(np.arange(E), T); tr, te = ep < E // 2, ep >= E // 2


def r2(X, y, a=tr, b=te):
    X1 = np.c_[X, np.ones(len(X))]
    w, *_ = np.linalg.lstsq(X1[a], y[a], rcond=None)
    p = X1[b] @ w
    return round(float(1 - ((y[b] - p) ** 2).sum() / ((y[b] - y[b].mean()) ** 2).sum()), 3)


for name in sys.argv[1:]:
    ck = torch.load(S3.OUT / f"{name}.pt", map_location=FA.DEV, weights_only=False)
    K = ck["K"]
    net = FA.Factors(K=K).to(FA.DEV); net.load_state_dict(ck["state_dict"]); net.eval()
    S_tr = SC.encode(net, Xt[:100].reshape(-1, 1024, 64)).reshape(100, T, K)
    S = SC.encode(net, Xv); Sv = S.reshape(E, T, K)
    used, groups, _ = S3.factor_rule(S_tr)
    D = S3.differ(mv); n = D.sum(-1)
    c = np.abs(Sv[:, G:] - Sv[:, :-G]) * used[None, None]
    fac = np.stack([c[..., g].sum(-1) for g in groups], -1)                       # [E, T-G, F]
    cube_d = np.linalg.norm(mv["cube"][:, G:] - mv["cube"][:, :-G], axis=-1)
    slide_d = np.maximum(np.abs(mv["drawer"][:, G:] - mv["drawer"][:, :-G]), np.abs(mv["window"][:, G:] - mv["window"][:, :-G]))
    still = (n == 0) & (cube_d < 0.001) & (slide_d < 0.001)
    creep = (n == 0) & ~still
    pose_d = np.abs(pose[:, G:] - pose[:, :-G]).sum(-1)
    one = fac.sum(-1)[n == 1].mean()
    tab = SC.on_table(mv["cube"]).reshape(-1)
    out = dict(
        factors=[[int(k) for k in g] for g in groups],
        mean_change_per_factor=dict(nothing_differs=[round(float(x), 4) for x in fac[n == 0].mean(0)],
                                    **{v: [round(float(x), 4) for x in fac[D[..., i] & (n == 1)].mean(0)] for i, v in enumerate(SC.NAMES)}),
        windows=dict(nothing_differs=int((n == 0).sum()), of_which_truly_still=int(still.sum()), of_which_under_1cm_creep=int(creep.sum())),
        change_over_one_variable_pct=dict(truly_still=round(100 * float(fac.sum(-1)[still].mean() / one), 1), under_1cm_creep=round(100 * float(fac.sum(-1)[creep].mean() / one), 1)),
        median_change_over_one_variable_median_pct=round(100 * float(np.median(fac.sum(-1)[n == 0]) / np.median(fac.sum(-1)[n == 1])), 1),
        still_windows_change_vs_pose_change_corr=round(float(np.corrcoef(fac.sum(-1)[still], pose_d[still])[0, 1]), 3),
        still_windows_top_decile_share_of_change_pct=round(100 * float(np.sort(fac.sum(-1)[still])[-max(1, int(still.sum()) // 10):].sum() / fac.sum(-1)[still].sum()), 1),
        truth_from_state=dict(drawer=r2(S, mv["drawer"].reshape(-1)), window=r2(S, mv["window"].reshape(-1)),
                              lock0=r2(S, mv["locks"][..., 0].reshape(-1).astype(float)), lock1=r2(S, mv["locks"][..., 1].reshape(-1).astype(float)),
                              cube_x_on_table=r2(S, mv["cube"][..., 0].reshape(-1), tr & tab, te & tab), cube_y_on_table=r2(S, mv["cube"][..., 1].reshape(-1), tr & tab, te & tab)),
        truth_from_each_factor={v: [r2(S[:, g], y) for g in groups] for v, y in (("drawer", mv["drawer"].reshape(-1)), ("window", mv["window"].reshape(-1)),
                                ("lock0", mv["locks"][..., 0].reshape(-1).astype(float)), ("lock1", mv["locks"][..., 1].reshape(-1).astype(float)))},
        pose_from_state_max_r2=max(r2(S, pose.reshape(-1, 12)[:, i]) for i in range(12)),
        state_from_pose_r2_per_factor=[round(float(np.mean([r2(pose.reshape(-1, 12), S[:, k]) for k in g])), 3) for g in groups],
    )
    json.dump(out, open(S3.OUT / f"diag_{name}.json", "w"), indent=1)
    print(name, json.dumps(out), flush=True)
