"""Card 174, section 4 (no training): what moves card 173's lock factors when no lock flips.
  locks.py  -> runs/174/locks.json     Uses runs/173/k8_s0.pt on the 100 validation episodes; truth for the analysis only."""
import json
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
import factors as FA
import score as SC
import score173 as S3

mv = FA.load_meta("val")
Xv = np.load(FA.OUT / "feat_val.npy", mmap_mode="r").reshape(-1, 1024, 64)
ck = torch.load(S3.OUT / "k8_s0.pt", map_location=FA.DEV, weights_only=False)
net = FA.Factors(K=ck["K"]).to(FA.DEV); net.load_state_dict(ck["state_dict"]); net.eval()
E, T = mv["drawer"].shape[0], FA.T
S = SC.encode(net, Xv).reshape(E, T, -1)
pose, eff = mv["pose"], mv["pose"][..., 6:9]                     # effector position (observation units)
locks = mv["locks"]
ep = np.repeat(np.arange(E), T).reshape(E, T); te = ep >= E // 2
out = {}
for li, k in ((0, 4), (1, 1)):                                   # lock -> its coordinate in k8_s0
    x, st = S[..., k], locks[..., li]
    flip = np.zeros((E, T), bool); flip[:, 1:] = st[:, 1:] != st[:, :-1]
    btn = np.median(eff[flip], 0)                                # where the hand is when this lock flips
    dist = np.linalg.norm(eff - btn, axis=-1)
    r = dict(coordinate=k, mean_by_state=[round(float(x[st == s].mean()), 3) for s in (0, 1)], std_within_state=[round(float(x[st == s].std()), 3) for s in (0, 1)],
             hand_to_button_at_flip_median=round(float(np.median(dist[flip])), 3), hand_to_button_overall_median=round(float(np.median(dist)), 3))
    # residual from the state's mean, by how near the hand is to the button
    res = np.abs(x - np.where(st == 1, x[st == 1].mean(), x[st == 0].mean()))
    qs = np.quantile(dist, [0, 0.05, 0.1, 0.25, 0.5, 1.0])
    r["abs_residual_by_hand_distance"] = {f"{qs[i]:.2f}-{qs[i+1]:.2f}": round(float(res[(dist >= qs[i]) & (dist <= qs[i + 1])].mean()), 3) for i in range(5)}
    r["share_of_residual_in_nearest_10pct_of_frames"] = round(100 * float(res[dist <= qs[2]].sum() / res.sum()), 1)
    # change over a gap with the lock's state constant, by gap
    r["mean_abs_change_state_constant_by_gap"] = {}
    for g in (1, 2, 5, 10, 20, 40):
        same = np.ones((E, T - g), bool)
        for j in range(1, g + 1):
            same &= st[:, j:T - g + j] == st[:, :T - g]
        r["mean_abs_change_state_constant_by_gap"][g] = round(float(np.abs(x[:, g:] - x[:, :-g])[same].mean()), 4)
    # near a flip in time?
    near = np.zeros((E, T), bool)
    for d in range(-3, 4):
        near |= np.roll(flip, d, axis=1)
    c5 = np.abs(x[:, 5:] - x[:, :-5]); const = (st[:, 5:] == st[:, :-5])
    nf = near[:, 5:] | near[:, :-5]
    r["change_5_steps_state_same"] = dict(within_3_steps_of_a_flip=round(float(c5[const & nf].mean()), 4), share_of_windows_pct=round(100 * float((const & nf).sum() / const.sum()), 1),
                                          elsewhere=round(float(c5[const & ~nf].mean()), 4))
    # what predicts the residual: pose; pose and the rest of the scene
    y = (x - np.where(st == 1, x[st == 1].mean(), x[st == 0].mean()))
    feats = {"pose": pose.reshape(-1, 12), "rest of the scene": np.c_[mv["cube"].reshape(-1, 3), mv["drawer"].reshape(-1), mv["window"].reshape(-1), locks[..., 1 - li].reshape(-1)],
             "pose and rest of the scene": np.c_[pose.reshape(-1, 12), mv["cube"].reshape(-1, 3), mv["drawer"].reshape(-1), mv["window"].reshape(-1), locks[..., 1 - li].reshape(-1)],
             "episode step": np.tile(np.arange(T), E)[:, None].astype(float)}
    r["residual_r2_from"] = {}
    for nm, Xf in feats.items():
        m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Xf[~te.reshape(-1)], y.reshape(-1)[~te.reshape(-1)])
        p = m.predict(Xf[te.reshape(-1)]); yt = y.reshape(-1)[te.reshape(-1)]
        r["residual_r2_from"][nm] = round(float(1 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()), 3)
    # between-episode share of the within-state variance
    em = np.array([[x[e][st[e] == s].mean() if (st[e] == s).any() else np.nan for s in (0, 1)] for e in range(E)])
    r["std_of_episode_means_by_state"] = [round(float(np.nanstd(em[:, s])), 3) for s in (0, 1)]
    out[f"lock{li}"] = r
    print(f"lock{li}", json.dumps(r), flush=True)
json.dump(out, open(FA.ROOT / "runs/174/locks.json", "w"), indent=1)
