"""Card 175, data check: can card 172's encoder read both table axes of the cube when it is told to (labels)?

  cube_bound.py [--steps 1500]   -> runs/175/cube_bound.json
A supervised fit of Factors(K=2) on the cube's x and y (frames with the cube on the table, 96 training episodes), scored by
linear R^2 on the validation episodes. Also: how the cube moves in the data, and what the extra cube coordinates of the
12- and 16-coordinate fits of card 174 follow.
"""
import json, os, resource, time
import numpy as np
import torch
import factors as FA
import score as SC
import score173 as S3

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", 0.5)))
OUT, DEV, T, GAP = FA.ROOT / "runs/175", FA.DEV, FA.T, S3.GAP
STEPS = int(FA.arg("--steps", 1500))


def r2(p, y):
    X1 = np.c_[p, np.ones(len(p))]
    h = len(p) // 2
    w, *_ = np.linalg.lstsq(X1[:h], y[:h], rcond=None)
    q = X1[h:] @ w
    return round(float(1 - ((y[h:] - q) ** 2).sum() / ((y[h:] - y[h:].mean()) ** 2).sum()), 3)


def main():
    out = {}
    mt, mv = FA.load_meta("train"), FA.load_meta("val")
    tb_t, tb_v = SC.on_table(mt["cube"]), SC.on_table(mv["cube"])
    xy = mt["cube"][..., :2][tb_t]
    lo, hi = xy.min(0), xy.max(0)
    D = S3.differ(mv); n = D.sum(-1)
    only = D[..., 0] & (n == 1) & tb_v[:, GAP:] & tb_v[:, :-GAP]
    d = np.abs(mv["cube"][:, GAP:] - mv["cube"][:, :-GAP])[only]
    out["cube_on_table"] = dict(x_range=[round(float(lo[0]), 3), round(float(hi[0]), 3)], y_range=[round(float(lo[1]), 3), round(float(hi[1]), 3)],
                                x_std=round(float(xy[:, 0].std()), 3), y_std=round(float(xy[:, 1].std()), 3))
    out["cube_only_windows_on_table"] = dict(n=int(only.sum()), mean_abs_dx=round(float(d[:, 0].mean()), 4), mean_abs_dy=round(float(d[:, 1].mean()), 4),
                                            mostly_x_pct=round(100 * float((d[:, 0] > 2 * d[:, 1]).mean()), 1), mostly_y_pct=round(100 * float((d[:, 1] > 2 * d[:, 0]).mean()), 1),
                                            dy_under_1cm_pct=round(100 * float((d[:, 1] < 0.01).mean()), 1), dx_under_1cm_pct=round(100 * float((d[:, 0] < 0.01).mean()), 1))
    print(json.dumps(out), flush=True)

    # supervised fit
    torch.manual_seed(0); rng = np.random.default_rng(0)
    X = np.load(FA.OUT / "feat_train.npy", mmap_mode="r")
    ids = rng.choice(X.shape[0], 96, replace=False)
    feat = torch.from_numpy(np.stack([np.asarray(X[i]) for i in ids])).to(DEV).reshape(-1, 1024, 64)
    tgt = ((mt["cube"][ids][..., :2] - lo) / (hi - lo) * 1.6 - 0.8).reshape(-1, 2).astype(np.float32)
    ok = torch.from_numpy(np.where(tb_t[ids].reshape(-1))[0]).to(DEV)
    tgt = torch.from_numpy(tgt).to(DEV)
    net = FA.Factors(K=2).to(DEV)
    net.log_tau.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    t0 = time.time()
    for st in range(1, STEPS + 1):
        i = ok[torch.randint(0, len(ok), (256,), device=DEV)]
        loss = ((net(feat[i].float()) - tgt[i]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if st % 250 == 0:
            print(f"step {st} {time.time()-t0:.0f}s mse {loss.item():.4f}", flush=True)
    net.eval()
    del feat
    Xv = np.load(FA.OUT / "feat_val.npy", mmap_mode="r").reshape(-1, 1024, 64)
    with torch.no_grad():
        S = SC.encode(net, Xv)
    m = tb_v.reshape(-1); cv = mv["cube"].reshape(-1, 3)
    out["supervised_K2"] = dict(steps=STEPS, sec=round(time.time() - t0), x_r2=r2(S[m][:, [0]], cv[m, 0]), y_r2=r2(S[m][:, [1]], cv[m, 1]),
                                x_abs_err_cm=round(100 * float(np.abs((S[m, 0] + 0.8) / 1.6 * (hi[0] - lo[0]) + lo[0] - cv[m, 0]).mean()), 2),
                                y_abs_err_cm=round(100 * float(np.abs((S[m, 1] + 0.8) / 1.6 * (hi[1] - lo[1]) + lo[1] - cv[m, 1]).mean()), 2))
    print(json.dumps(out["supervised_K2"]), flush=True)

    # what the extra cube coordinates of the large fits follow (all frames; linear R^2 of the truth from one coordinate)
    ins = (~tb_v).reshape(-1).astype(float)
    for name, ks in (("k12_s0h", [7, 10]), ("k16_s0h", [1, 7, 8]), ("k6_s0h", [1])):
        ck = torch.load(FA.ROOT / "runs/174" / f"{name}.pt", map_location=DEV, weights_only=False)
        f = FA.Factors(K=ck["K"]).to(DEV); f.load_state_dict(ck["state_dict"]); f.eval()
        with torch.no_grad():
            Sf = SC.encode(f, Xv)
        out[name] = {int(k): dict(x_all=r2(Sf[:, [k]], cv[:, 0]), y_all=r2(Sf[:, [k]], cv[:, 1]), z_all=r2(Sf[:, [k]], cv[:, 2]), off_table=r2(Sf[:, [k]], ins),
                                  x_table=r2(Sf[m][:, [k]], cv[m, 0]), y_table=r2(Sf[m][:, [k]], cv[m, 1]),
                                  x_table_cubic=r2(np.stack([Sf[m, k], Sf[m, k] ** 2, Sf[m, k] ** 3], 1), cv[m, 0])) for k in ks}
        out[name]["all_coords_x_table_quadratic"] = r2(np.c_[Sf[m], Sf[m] ** 2], cv[m, 0])
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(OUT / "cube_bound.json", "w"), indent=1)
    print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
