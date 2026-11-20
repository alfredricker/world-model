"""Card 146, arm A: objects by settling the energy on couplings taken from a trained AKOrN's features,
against the published read-out (average-linkage clustering of the same features).
Run inside the AKOrN clone:  AKORN=1 tools/card146/run.sh 600 python ../../../tools/card146/arm_a.py <name> <checkpoint>
Parameters are chosen on the first 320 validation images and scored once on the 320 test images.
"""
import itertools, json, sys, numpy as np, torch, torch.nn.functional as F
ROOT = "/home/fred/Projects/world-model"
sys.path.insert(0, ROOT + "/tools/card146"); sys.path.insert(0, ".")
import energy as E, score as S
import fastcluster
from scipy.cluster.hierarchy import fcluster
from ema_pytorch import EMA
from source.models.objs.knet import AKOrN

name, ckpt = sys.argv[1], sys.argv[2]
V2 = len(sys.argv) > 3 and sys.argv[3] == "v2"       # card 146.1: thresholds around 0.8 and a colder, longer descent
dev, G = "cuda", 8
torch.manual_seed(0)
net = AKOrN(4, ch=128, L=1, T=8, gamma=1.0, J="attn", c_norm="none", psize=4, imsize=32, gta=True).to(dev)
model = EMA(net, beta=0.998, update_every=10, update_after_step=200)
model.load_state_dict(torch.load(ckpt, weights_only=True)["model_state_dict"])
model = model.ema_model.eval()

def load(split, n=320):
    d = np.load(f"{ROOT}/data/tetrominoes/tetrominoes_{split}.npz")
    return torch.tensor(d["images"][:n]).to(dev), d["labels"][:n].astype(int)

@torch.no_grad()
def feats(img):                                  # what AKOrN's evaluation clusters: [B, 64, 128], unit rows
    c = model.feature(img)[0]
    return F.normalize(c, dim=1).flatten(2).transpose(1, 2)

def scores(gt, lab):
    lab = np.asarray(lab).reshape(len(gt), G, G)
    p = S.upsample(lab, 32)
    maj = torch.tensor(S.patch_fg_majority(gt, 4).reshape(len(gt), -1))
    l = torch.tensor(lab.reshape(len(gt), -1))
    merged = np.mean([len({torch.bincount(a[b == o], minlength=64).argmax().item() for o in torch.unique(b)}) < len(torch.unique(b)) for a, b in zip(l, maj)])
    return {"fgari": float(S.fgari(gt, p).mean()) * 100, "mbo": float(S.mbo(gt, p).mean()) * 100, "merged": float(merged) * 100,
            "labels_used": float(np.mean([len(np.unique(a)) for a in lab]))}

def cluster(f, **kw):
    return np.stack([fcluster(fastcluster.average(x), **kw) for x in f.cpu().numpy()])

pos = E.grid(G, dev)
(iv, gv), (it, gtest) = load("val"), load("test")
fv, ft = feats(iv), feats(it)
out = {"name": name, "checkpoint": ckpt}

out["clustering, told 4"] = scores(gtest, cluster(ft, t=4, criterion="maxclust"))
out["clustering, told 8"] = scores(gtest, cluster(ft, t=8, criterion="maxclust"))
ths = np.linspace(0.1, 1.4, 27)
best_th = max(ths, key=lambda th: scores(gv, cluster(fv, t=th, criterion="distance"))["fgari"])
out["clustering, distance threshold from validation"] = dict(scores(gtest, cluster(ft, t=best_th, criterion="distance")), threshold=float(best_th))

if V2:
    grid = [dict(tau=t, sharp=1.0, radius=r, far=fa, scale=0.25) for t, (r, fa) in
            itertools.product((0.7, 0.75, 0.8, 0.85, 0.9), ((None, 0.0), (2, 0.1), (2, 0.3)))]
    descent = dict(T=400, eta=0.2, b1=300.0)
else:
    grid = [dict(tau=t, sharp=s, radius=r, far=fa) for t, s, (r, fa) in
            itertools.product((0.3, 0.5, 0.7, 0.9, 0.95), (1.0, 4.0, 16.0), ((None, 0.0), (2, 0.1), (2, 0.3)))]
    descent = {}
def run(f, kw, K=8):
    return E.settle_labels(E.couplings(f, pos, **kw), K=K, **descent)[0].cpu().numpy()
val = [(scores(gv, run(fv, kw))["fgari"], i) for i, kw in enumerate(grid)]
kw = grid[max(val)[1]]
out["settling, K=8"] = dict(scores(gtest, run(ft, kw)), params=kw, validation_fgari=max(val)[0])
out["settling, K=4, same parameters"] = scores(gtest, run(ft, kw, K=4))
out["settling, K=16, same parameters"] = scores(gtest, run(ft, kw, K=16))
out["validation grid"] = [dict(grid[i], fgari=v) for v, i in val]
for k, v in out.items():
    if isinstance(v, dict):
        print(f"{k:48s} FG-ARI {v['fgari']:5.1f}  MBO {v['mbo']:5.1f}  merged {v['merged']:3.0f}%  labels used {v['labels_used']:.1f}", {a: b for a, b in v.items() if a in ('params', 'threshold')} or "")
json.dump(out, open(f"{ROOT}/runs/146/arm_a_{'v2_' if V2 else ''}{name}.json", "w"), indent=1)
