"""Card 171, section 4 (data check). Training episodes only, frozen card 168 model (seed 0); no test episode read.

  bash tools/card170/run.sh 240 python tools/card171/datacheck.py

Counts chains of consecutive touches, how far the end of a pull varies (readings, and simulator truth for
scoring only), and how often a put-in is followed by a drawer touch that hides the cube.
"""
import numpy as np, torch, json
import model as M
LEARN = ["m400", "m600", "m800", "tail"]
Z, tc, ep = M.load_split(LEARN)
ck = torch.load(M.ROOT / "runs/168/scene_wnet_0.pt", map_location=M.DEV, weights_only=False)
net = M.SceneWNet(n_slots=ck["n_slots"], mov_si=ck["mov_si"]).to(M.DEV); net.load_state_dict(ck["state_dict"]); net.eval()
meta = ck["meta"]; kinds = meta["slot_kind"]; cuts = {int(k): float(v) for k, v in meta["slot_cuts"].items()}
H, Val, V, A = M.encode_all(net, Z)
N = len(ep); print("frames", N, "episodes", len(np.unique(ep)))
same = ep[:-1] == ep[1:]
for L in (1, 2, 3, 4, 6, 8):
    ok = np.ones(N - L, bool)
    for j in range(L): ok &= same[j:N - L + j]
    print(f"chains of {L} touches: {int(ok.sum())}")
grp = meta["group"][np.linalg.norm(tc[:, None, :3] - meta["cen"][None], axis=-1).argmin(1)]
truth = {}
for nm in LEARN:
    ev = np.load(M.ROOT / f"runs/132/ev_{nm}.npz")
    for k in ev.files:
        truth.setdefault(k, []).append(ev[k])
print("ev keys", list(truth.keys()))
tr = {k: np.concatenate(v, 0) for k, v in truth.items() if k in ("drawer", "window", "cube", "buttons", "locks")}
for k, v in tr.items(): print(k, v.shape, np.round(np.quantile(v.reshape(len(v), -1)[:, 0], [0, .5, 1]), 3))
s = np.where(same)[0]
slides = [k for k, kd in enumerate(kinds) if kd == "slide"]
print("kinds", kinds, "cuts", cuts, "cause_of", meta["cause_of"], "cont", meta.get("cont_si"))
for k in slides:
    t = s[grp[s] == meta["slots"][k]]
    dv = Val[t + 1, k] - Val[t, k]
    lo, hi = np.quantile(Val[:, k], [0.03, 0.97]); mid = 0.5 * (lo + hi)
    print(f"slide slot {k}: range {lo:.2f}..{hi:.2f}, touches {len(t)}, moved {(np.abs(dv) > cuts[k]).sum()}")
    for d, m in (("lo", (dv < -1.2 * cuts[k])), ("hi", (dv > 1.2 * cuts[k]))):
        i = t[m]; e = Val[i + 1, k]; st = Val[i, k]
        q = np.quantile(e, [.1, .5, .9])
        X = np.c_[st, tc[i, :3], tc[i, 6:7], np.ones(len(i))]
        w, *_ = np.linalg.lstsq(X, e, rcond=None); r2 = 1 - ((e - X @ w) ** 2).sum() / ((e - e.mean()) ** 2).sum()
        within = (np.abs(e - np.median(e)) <= cuts[k]).mean()
        line = f"  pull {d}: n={len(i)} end 10/50/90% = {q.round(2)}, within one threshold of the median end {100*within:.0f}%, R2 of end from start reading+contact {r2:.2f}"
        for nm in ("drawer", "window"):
            if nm in tr:
                tv = tr[nm].reshape(len(Val), -1)[:, 0]
                c = np.corrcoef(e, tv[i + 1])[0, 1]
                line += f", corr(end reading, true {nm} after)={c:.2f} true-end 10/50/90%={np.quantile(tv[i+1],[.1,.5,.9]).round(3)}"
        print(line)
# compound the planner needs: a put-in followed by a drawer touch; a pull followed later in the chain by a press
mov = meta["mov_si"]; zc = meta["z_cut"]
vis = V >= 0.5
cont = meta.get("cont_si")
put = same & (grp[:-1] == meta["slots"][mov]) & vis[1:] & (A[1:, 2] > zc) & ~(A[:-1, 2] > zc)
i = np.where(put)[0]; i = i[i + 2 < N]; i = i[ep[i + 2] == ep[i]]
print("put-ins read", int(put.sum()), "followed in the episode by one more touch", len(i), "of which next touch on the drawer", int((grp[i + 1] == meta["slots"][cont]).sum()) if cont is not None else None,
      "and the cube hidden after it", int(((grp[i + 1] == meta["slots"][cont]) & ~vis[i + 2]).sum()) if cont is not None else None)
hid = same & vis[:-1] & ~vis[1:]
print("transitions that hide the cube", int(hid.sum()), "; that show it again", int((same & ~vis[:-1] & vis[1:]).sum()))
