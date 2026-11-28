"""Card 172, diagnostic: what the state holds. Linear read-outs fitted on half the validation episodes, scored on the
other half (truth for scoring only).   diag.py <name> ...  -> runs/172/diag_<name>.json"""
import json, sys
import numpy as np
import torch
import factors as FA
import score as SC

mv = FA.load_meta("val")
Xv = np.load(FA.OUT / "feat_val.npy", mmap_mode="r").reshape(-1, 1024, 64)
ck_g = torch.load(FA.OUT / "g.pt", map_location="cpu", weights_only=False)
pose = ((mv["pose"] - ck_g["mu"]) / ck_g["sd"]).reshape(-1, 12)
E = mv["drawer"].shape[0]
ep = np.repeat(np.arange(E), FA.T)
tr, te = ep < E // 2, ep >= E // 2
tab = SC.on_table(mv["cube"]).reshape(-1)


def r2(X, y, m_tr, m_te):
    X1 = np.c_[X, np.ones(len(X))]
    w, *_ = np.linalg.lstsq(X1[m_tr], y[m_tr], rcond=None)
    p = X1[m_te] @ w
    return round(float(1 - ((y[m_te] - p) ** 2).sum() / ((y[m_te] - y[m_te].mean()) ** 2).sum()), 3)


for name in sys.argv[1:]:
    ck = torch.load(FA.OUT / f"{name}.pt", map_location=FA.DEV, weights_only=False)
    net = FA.Factors(K=ck["K"]).to(FA.DEV); net.load_state_dict(ck["state_dict"]); net.eval()
    S = SC.encode(net, Xv)
    out = dict(
        truth_from_state=dict(
            drawer=r2(S, mv["drawer"].reshape(-1), tr, te), window=r2(S, mv["window"].reshape(-1), tr, te),
            lock0=r2(S, mv["locks"][..., 0].reshape(-1).astype(float), tr, te), lock1=r2(S, mv["locks"][..., 1].reshape(-1).astype(float), tr, te),
            cube_x_on_table=r2(S, mv["cube"][..., 0].reshape(-1), tr & tab, te & tab), cube_y_on_table=r2(S, mv["cube"][..., 1].reshape(-1), tr & tab, te & tab)),
        truth_from_pose=dict(drawer=r2(pose, mv["drawer"].reshape(-1), tr, te), lock0=r2(pose, mv["locks"][..., 0].reshape(-1).astype(float), tr, te),
                             cube_x_on_table=r2(pose, mv["cube"][..., 0].reshape(-1), tr & tab, te & tab)),
        pose_from_state={f"pose {i}": r2(S, pose[:, i], tr, te) for i in range(12)},
        state_from_pose_mean_r2=round(float(np.mean([r2(pose, S[:, k], tr, te) for k in range(S.shape[1])])), 3),
    )
    # effective number of dimensions the state uses (participation ratio of its covariance spectrum)
    ev = np.linalg.eigvalsh(np.cov(S.T)); ev = ev[ev > 0]
    out["effective_dimensions"] = round(float(ev.sum() ** 2 / (ev ** 2).sum()), 1)
    json.dump(out, open(FA.OUT / f"diag_{name}.json", "w"), indent=1)
    print(name, json.dumps(out), flush=True)
