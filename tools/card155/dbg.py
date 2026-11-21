"""why never-met objects of one behaviour get the wrong class: dbg.py <seed> <by>"""
import sys, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "tools/card155")
import pipeline as P
W, M, R, Fb = P.W, P.M, P.R, P.Fb
seed, by = int(sys.argv[1]), sys.argv[2]
world = W.make_world(seed, by); log = {}; L = P.learn(world, 100, seed * 100 + 10, log)
print("kind_of", world["kind_of"], "classes", log["class_sizes"], log["classes_by_true_behaviour"], "true_of_kind", L["true_of_kind"])
sh = L["shape"]; names = W._w147.SHAPES
true_sil = {(n, s): F.adaptive_avg_pool2d(torch.tensor(W.shape_mask(n, s), dtype=torch.float32)[None], 8)[0].reshape(-1) for n in names for s in (6, 8, 10, 12, 14, 16)}
for k, p in enumerate(sh.P):
    d = sorted(((float(0.5 * ((p - v) ** 2).sum()), key) for key, v in true_sil.items()))[:2]
    print("shape code", k, "nearest true silhouettes", [(round(a, 2), b) for a, b in d])
used = sorted(L["idx_of"], key=L["idx_of"].get)
sil = torch.stack([L["app"][p]["sil"] for p in used]); code = sh.energy(sil).argmin(1).numpy(); en = sh.energy(sil).min(1).values.numpy()
tk = np.array([next(o for o in [None]) if False else 0 for _ in used])
keys = [L["mem"].info[p]["key"] for p in used]; iou = np.array([L["mem"].info[p]["iou"] for p in used])
print("objects per (shape code, class):"); 
for c in range(len(sh.P)):
    sel = code == c; print("   code", c, "n", int(sel.sum()), "classes", np.bincount(L["kind"][sel], minlength=L["dyn"].K).tolist(), "mean pattern iou", iou[sel].mean().round(2) if sel.any() else None, "mean energy", en[sel].mean().round(3) if sel.any() else None)
oh = lambda i, n: torch.eye(n)[i][None]
print("appearance prior by shape code (colour uniform):", [torch.softmax(L["looks"].field(oh(c, len(sh.P)), torch.ones(1, len(L["colour"].P)) / len(L["colour"].P)), -1)[0].numpy().round(2).tolist() for c in range(len(sh.P))])
print("shape scale", sh.scale)
for n in names:
    for s in (6, 8, 10, 12, 14, 16):
        v = true_sil[(n, s)][None]; e = sh.energy(v)[0]; b = sh.belief(v)[0]
        print("  true", n, s, "-> code", int(e.argmin()), "energy", round(float(e.min()), 3), "belief", b.numpy().round(2).tolist()) if n == names[int(np.nonzero(world["kind_of"] == 2)[0][0])] or s == 10 else None
