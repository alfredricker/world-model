"""Card 146 gate, on the 320 test images of Tetrominoes (seconds):
what the 8 x 8 patch grid allows, what colour alone gives, and whether settling terms 1 and 2
recovers the objects when the couplings are perfect, for both forms of a patch's state.
usage: tools/card146/run.sh 300 python tools/card146/gate.py
"""
import json, sys, numpy as np, torch
sys.path.insert(0, "tools/card146")
import energy as E, score as S
from scipy.cluster.hierarchy import fcluster, linkage

d = np.load("data/tetrominoes/tetrominoes_test.npz")
img, gt = d["images"], d["labels"].astype(int)                  # [320,3,32,32] in 0..1, [320,32,32]
dev, ps, G = "cuda", 4, 8
out = {}
maj = S.patch_fg_majority(gt, ps)                               # [320,8,8]
m = torch.tensor(maj.reshape(320, -1), device=dev)

def merged(lab):
    """Share of images where two of {the background, the objects} mostly carry the same label."""
    r = []
    for l, mm in zip(lab, m):
        tops = [torch.bincount(l[mm == o], minlength=64).argmax().item() for o in torch.unique(mm)]
        r.append(len(set(tops)) < len(tops))
    return float(np.mean(r))

def rep(name, lab):
    lab = torch.as_tensor(np.asarray(lab) if not torch.is_tensor(lab) else lab).reshape(320, -1).to(dev)
    p = S.upsample(lab.view(320, G, G).cpu().numpy(), 32)
    out[name] = {"fgari": float(S.fgari(gt, p).mean()), "mbo": float(S.mbo(gt, p).mean()), "merged": merged(lab)}
    print(f"{name:58s} FG-ARI {out[name]['fgari']*100:5.1f}  MBO {out[name]['mbo']*100:5.1f}  merged {out[name]['merged']*100:3.0f}%", flush=True)

print("patches per image: background %.1f, per object %.1f" % ((maj == 0).sum((1, 2)).mean(), (maj > 0).sum((1, 2)).mean() / 3))
rep("ceiling: each patch its main object", maj)
rep("trivial: one group", np.ones_like(maj))
col = img.reshape(320, 3, G, ps, G, ps).mean((3, 5)).transpose(0, 2, 3, 1).reshape(320, G * G, 3)
rep("trivial: colour, average-linkage clustering, 4 clusters",
    np.stack([fcluster(linkage(c, "average"), t=4, criterion="maxclust") for c in col]))

pos = E.grid(G, dev)
eye = torch.eye(G * G, device=dev)
same = (m[:, :, None] == m[:, None, :]).float()
near = (torch.cdist(pos, pos) <= 2.01).float() * (1 - eye)
far = (1 - near) * (1 - eye)
J_all = (2 * same - 1) * (1 - eye) / 16
for K in (4, 8):
    rep(f"perfect couplings, all pairs, beliefs, K={K}", E.settle_labels(J_all, K=K)[0])
for K in (4, 8):
    rep(f"perfect couplings, all pairs, unit vectors, K={K}", E.settle_vectors(J_all / 4, K=K)[0])
for eps in (0.0, 0.3):
    J = ((2 * same - 1) * near - eps * far) / 16
    rep(f"perfect within 2 patches, far pairs repel {eps}, beliefs, K=8", E.settle_labels(J, K=8)[0])
json.dump(out, open("runs/146/gate.json", "w"), indent=1)
