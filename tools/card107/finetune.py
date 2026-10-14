"""Card 107: DA3METRIC-LARGE's depth head tuned by view synthesis with the known pose (Monodepth2's loss, pose given).

  tools/card107/dprun tools/card107/finetune.py gate                        → runs/107/gate.json
  tools/card107/dprun tools/card107/finetune.py train [--steps 3000 --bs 4 --lr 1e-4]
                                                                            → runs/107/head.pt, runs/107/train.json
  tools/card107/dprun tools/card107/finetune.py export <clip> released|tuned
                                                                            → runs/107/clips/<clip>_depth_<arm>.npz

Frames are resized to 504 x 280 and normalised as DA3's own input processor does; the backbone (frozen) runs in
bfloat16 without gradients, the DPT head in float32. Metric depth = f * output / 300 (DA3's README), f the mean focal
length at 504 x 280. Frame i is drawn from pose i - 1 (Craftium); the camera sits 1.5 above the feet.
Sources for a target frame t: the nearest earlier and later frames (within 12 steps) whose camera moved at least
0.25 and at most 4 blocks from t's. Loss: per pixel the smallest of the warped sources' photometric errors
(0.85 (1 - SSIM) / 2 + 0.15 L1) and the unwarped sources' (Monodepth2's automatic mask), plus 1e-3 edge-aware
smoothness of mean-normalised inverse depth.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "107"
CLIPS = RUNS / "clips"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from truth import EYE, FOV_Y  # noqa: E402

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
dev = torch.device("cuda")
MW, MH = 504, 280
TRAIN = ["train_a", "train_b", "train_c", "train_d", "train_e", "train_f"]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def intrinsics(W, H):
    f = (H / 2) / np.tan(np.radians(FOV_Y) / 2)          # at the clip's own size
    fx, fy = f * MW / W, f * MH / H
    return fx, fy, MW / 2, MH / 2


def cameras(P):
    """Per frame: centre C [T, 3] and world-to-camera rotation R [T, 3, 3] (rows: right, down, forward)."""
    T = len(P)
    C, R = np.zeros((T, 3)), np.zeros((T, 3, 3))
    for t in range(T):
        q = P[max(t - 1, 0)]
        yaw, p = np.radians(q[3]), np.radians(q[4])
        f = np.array([-np.sin(yaw) * np.cos(p), -np.sin(p), np.cos(yaw) * np.cos(p)])
        r = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        u = np.array([-np.sin(yaw) * np.sin(p), np.cos(p), np.cos(yaw) * np.sin(p)])
        C[t], R[t] = [q[0], q[1] + EYE, q[2]], np.stack([r, -u, f])
    return C, R


def sources(C, t, lo=0.25, hi=4.0, reach=12):
    out = []
    for direction in (-1, 1):
        s = None
        for k in range(1, reach + 1):
            j = t + direction * k
            if j < 0 or j >= len(C):
                break
            dist = np.linalg.norm(C[j] - C[t])
            if dist > hi:
                break
            if dist >= lo:
                s = j
                break
        out.append(s)
    return out


def load_clip(stem):
    z = np.load(CLIPS / f"{stem}.npz")
    Fr = torch.as_tensor(z["frames"]).permute(0, 3, 1, 2)
    H, W = Fr.shape[2:]
    small = torch.cat([F.interpolate(Fr[i:i + 256].float(), size=(MH, MW), mode="bilinear", antialias=True,
                                     align_corners=False).round().clamp(0, 255).byte() for i in range(0, len(Fr), 256)])
    C, R = cameras(z["poses"])
    return small, C, R, (W, H)


def load_model():
    from depth_anything_3.api import DepthAnything3
    net = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(dev).eval()
    return net.model


def depth(m, imgs, K, train=False):
    """imgs: [B, 3, MH, MW] uint8 on the GPU → metric depth [B, MH, MW] (float32)."""
    x = ((imgs.float() / 255) - MEAN) / STD
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        feats, _ = m.backbone(x[:, None], cam_token=None, export_feat_layers=[], ref_view_strategy="first")
    feats = [[t.float() if torch.is_tensor(t) else t for t in f] if isinstance(f, (list, tuple)) else f.float()
             for f in feats]
    with torch.autocast("cuda", enabled=False), torch.set_grad_enabled(train):
        out = m.head(feats, MH, MW, patch_start_idx=0)
    d = out["depth"]
    d = d.reshape(d.shape[0], MH, MW) if d.dim() == 4 else d
    return (K[0] + K[1]) / 2 * d / 300.0


def ssim(x, y):
    p = lambda a: F.avg_pool2d(F.pad(a, (1, 1, 1, 1), mode="reflect"), 3, 1)
    mx, my = p(x), p(y)
    sx, sy, sxy = p(x * x) - mx ** 2, p(y * y) - my ** 2, p(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    n = (2 * mx * my + c1) * (2 * sxy + c2)
    dd = (mx ** 2 + my ** 2 + c1) * (sx + sy + c2)
    return torch.clamp((1 - n / dd) / 2, 0, 1)


def photometric(a, b):
    return 0.85 * ssim(a, b).mean(1) + 0.15 * (a - b).abs().mean(1)


def warp(src, D, Ct, Rt, Cs, Rs, K):
    """Sample the source image where each target pixel lands, given target depth D [B, H, W] (z along forward)."""
    B = D.shape[0]
    fx, fy, cx, cy = K
    v, u = torch.meshgrid(torch.arange(MH, device=dev, dtype=torch.float32) + 0.5,
                          torch.arange(MW, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
    Xc = torch.stack([(u - cx) / fx * D, (v - cy) / fy * D, D], -1)              # [B, H, W, 3] target camera
    Xw = torch.einsum("bji,bhwj->bhwi", Rt, Xc) + Ct[:, None, None]               # R^T x + C
    Xs = torch.einsum("bij,bhwj->bhwi", Rs, Xw - Cs[:, None, None])
    z = Xs[..., 2].clamp(min=1e-3)
    us, vs = fx * Xs[..., 0] / z + cx, fy * Xs[..., 1] / z + cy
    grid = torch.stack([us / MW * 2 - 1, vs / MH * 2 - 1], -1)
    return F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)


def smooth(D, img):
    disp = 1 / D.clamp(min=1e-3)
    disp = disp / disp.mean((1, 2), keepdim=True)
    gx, gy = (disp[:, :, :-1] - disp[:, :, 1:]).abs(), (disp[:, :-1] - disp[:, 1:]).abs()
    ix = (img[:, :, :, :-1] - img[:, :, :, 1:]).abs().mean(1)
    iy = (img[:, :, :-1] - img[:, :, 1:]).abs().mean(1)
    return (gx * torch.exp(-ix)).mean() + (gy * torch.exp(-iy)).mean()


def batch_loss(m, clip, idx, K):
    small, C, R = clip
    tgt, srcs = [], []
    for t in idx:
        s = sources(C, t)
        srcs.append([j if j is not None else t for j in s])
        tgt.append(t)
    T_ = torch.as_tensor(np.array(tgt))
    img_t = small[T_].to(dev)
    D = depth(m, img_t, K, train=True)
    a = img_t.float() / 255
    errs = []
    for side in (0, 1):
        S_ = np.array([s[side] for s in srcs])
        img_s = small[torch.as_tensor(S_)].to(dev).float() / 255
        Ct, Rt = (torch.as_tensor(C[T_], device=dev, dtype=torch.float32), torch.as_tensor(R[T_], device=dev, dtype=torch.float32))
        Cs, Rs = (torch.as_tensor(C[S_], device=dev, dtype=torch.float32), torch.as_tensor(R[S_], device=dev, dtype=torch.float32))
        w = warp(img_s, D, Ct, Rt, Cs, Rs, K)
        e = photometric(w, a)
        same = torch.as_tensor(S_ == T_.numpy(), device=dev)[:, None, None]
        errs.append(torch.where(same, torch.full_like(e, 10.0), e))               # no source on this side
        errs.append(photometric(img_s, a) + 1e-5 * torch.randn_like(e) + same * 10.0)   # unwarped (automatic mask)
    E = torch.stack(errs, 0)
    m_ = E.min(0)
    warped_wins = (m_.indices % 2 == 0).float().mean()
    return m_.values.mean() + 1e-3 * smooth(D, a), float(warped_wins)


def gate():
    """The loss must prefer the truth: point-sampled photometric error at true depth vs true x0.8, x1.25, released."""
    m = load_model()
    rep = {}
    for stem in ("test_pasture", "test_roam"):
        small, C, R, (W, H) = load_clip(stem)
        K = intrinsics(W, H)
        g = np.load(CLIPS / f"{stem}_gt.npz")
        Dg, ts = g["D"], g["ts"]
        GH, GW = Dg.shape[1:]
        vs, us = np.meshgrid((np.arange(GH) + 0.5) * MH / GH, (np.arange(GW) + 0.5) * MW / GW, indexing="ij")
        keep = ~((vs > 0.80 * MH) | ((us > 0.70 * MW) & (vs > 0.45 * MH)))     # the item bar and the hand
        tot = {k: [] for k in ("true", "true x0.8", "true x1.25", "released")}
        used = 0
        for n, t in enumerate(ts):
            s_ = [j for j in sources(C, t) if j is not None]
            ok = np.isfinite(Dg[n]) & keep
            if not s_ or ok.sum() < 50:
                continue
            used += 1
            img_t = small[t:t + 1].to(dev)
            with torch.no_grad():
                Dr = depth(m, img_t, K)[0].cpu().numpy()
            dr = Dr[np.clip(vs.astype(int), 0, MH - 1), np.clip(us.astype(int), 0, MW - 1)]
            a = small[t].float().numpy() / 255
            per = {}
            for name, dep in (("true", Dg[n]), ("true x0.8", Dg[n] * 0.8), ("true x1.25", Dg[n] * 1.25), ("released", dr)):
                errs = []
                dep = np.where(np.isfinite(dep), dep, 1.0)        # pixels without true depth are masked out below
                for s in s_:
                    b = small[s].float().numpy() / 255
                    X = np.stack([(us - K[2]) / K[0] * dep, (vs - K[3]) / K[1] * dep, dep], -1)
                    Xw = X @ R[t] + C[t]
                    Xs = (Xw - C[s]) @ R[s].T
                    z = np.maximum(Xs[..., 2], 1e-3)
                    uu, vv = K[0] * Xs[..., 0] / z + K[2], K[1] * Xs[..., 1] / z + K[3]
                    inside = (uu >= 0) & (uu < MW - 1) & (vv >= 0) & (vv < MH - 1) & (Xs[..., 2] > 0.1)
                    x0, y0 = np.floor(np.clip(uu, 0, MW - 2)).astype(int), np.floor(np.clip(vv, 0, MH - 2)).astype(int)
                    fx_, fy_ = np.clip(uu, 0, MW - 2) - x0, np.clip(vv, 0, MH - 2) - y0
                    samp = (b[:, y0, x0] * (1 - fx_) * (1 - fy_) + b[:, y0, x0 + 1] * fx_ * (1 - fy_)
                            + b[:, y0 + 1, x0] * (1 - fx_) * fy_ + b[:, y0 + 1, x0 + 1] * fx_ * fy_)
                    tv = a[:, vs.astype(int), us.astype(int)]
                    e = np.abs(samp - tv).mean(0)
                    errs.append(np.where(inside, e, np.inf))
                per[name] = np.min(errs, 0)
            common = ok & np.all([np.isfinite(e) for e in per.values()], 0)   # points every candidate can be scored on
            if common.sum() < 50:
                used -= 1
                continue
            for name, e in per.items():
                tot[name].append(float(e[common].mean()))
        rep[stem] = {"frames used (true depth and a source)": used,
                     **{f"photometric error at {k}": round(float(np.mean(v)), 5) for k, v in tot.items()}}
        rep[stem]["truth lowest"] = bool(all(np.mean(tot["true"]) < np.mean(v) for k, v in tot.items() if k != "true"))
        print(stem, json.dumps(rep[stem]), flush=True)
    rep["gate passes"] = all(rep[s]["truth lowest"] for s in ("test_pasture", "test_roam"))
    json.dump(rep, open(RUNS / "gate.json", "w"), indent=1)
    print("gate passes:", rep["gate passes"])


def train():
    steps, bs, lr = int(arg("--steps", 3000)), int(arg("--bs", 4)), float(arg("--lr", 1e-4))
    m = load_model()
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in m.head.parameters()], lr=lr, weight_decay=1e-4)
    clips, pool = [], []
    for c, stem in enumerate(TRAIN):
        small, C, R, (W, H) = load_clip(stem)
        clips.append((small, C, R))
        for t in range(len(C)):
            if any(s is not None for s in sources(C, t)):
                pool.append((c, t))
    K = intrinsics(W, H)
    print(f"training targets with a source: {len(pool)} in {len(TRAIN)} clips", flush=True)
    rng = np.random.default_rng(1021)
    m.head.train()
    t0, log = time.monotonic(), []
    for it in range(steps):
        c = int(rng.integers(len(clips)))
        cand = [t for cc, t in pool if cc == c]
        idx = rng.choice(cand, bs, replace=False)
        loss, wins = batch_loss(m, clips[c], idx, K)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == steps - 1:
            log.append({"step": it, "loss": round(float(loss), 5), "warped wins": round(wins, 3),
                        "seconds": round(time.monotonic() - t0)})
            print(json.dumps(log[-1]), flush=True)
    torch.save(m.head.state_dict(), RUNS / "head.pt")
    json.dump({"steps": steps, "batch": bs, "lr": lr, "targets": len(pool), "log": log}, open(RUNS / "train.json", "w"),
              indent=1)


def export(stem, arm):
    m = load_model()
    if arm == "tuned":
        m.head.load_state_dict(torch.load(RUNS / "head.pt"))
    m.eval()
    small, C, R, (W, H) = load_clip(stem)
    K = intrinsics(W, H)
    out = []
    with torch.no_grad():
        for i in range(0, len(small), 16):
            D = depth(m, small[i:i + 16].to(dev), K)
            out.append(F.interpolate(D[:, None], size=(180, 320), mode="bilinear", align_corners=False)[:, 0]
                       .half().cpu().numpy())
    np.savez_compressed(CLIPS / f"{stem}_depth_{arm}.npz", D=np.concatenate(out), no_base=np.zeros(len(small), bool))
    print(f"{stem} {arm}: {len(small)} frames", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gate":
        gate()
    elif mode == "train":
        train()
    else:
        export(sys.argv[2], sys.argv[3])
