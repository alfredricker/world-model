"""Card 172, diagnostic after the gate: are the factors in the state but not on its axes?

  rotate.py <name> [n ...]   -> runs/172/rotate_<name>.json
The state's changes between consecutive steps of 100 training episodes (no labels) are unmixed by independent component
analysis into n directions; the state is re-expressed in those directions and scored as before, each direction one factor.
"""
import json, sys
import numpy as np
import torch
from sklearn.decomposition import FastICA
import factors as FA
import score as SC

name = sys.argv[1]
ns = [int(x) for x in sys.argv[2:]] or [8, 12, 16]
sets, mv = SC.build_sets()
ck = torch.load(FA.OUT / f"{name}.pt", map_location=FA.DEV, weights_only=False)
net = FA.Factors(K=ck["K"]).to(FA.DEV); net.load_state_dict(ck["state_dict"]); net.eval()
Xt = np.load(FA.OUT / "feat_train.npy", mmap_mode="r")
S_tr = SC.encode(net, Xt[:100].reshape(-1, 1024, 64)).reshape(100, FA.T, -1)
S_val = SC.encode(net, np.load(FA.OUT / "feat_val.npy", mmap_mode="r").reshape(-1, 1024, 64))
S_te = SC.encode(net, SC.test_frames())
dS = np.diff(S_tr, axis=1).reshape(-1, S_tr.shape[-1])
mu = S_tr.reshape(-1, S_tr.shape[-1]).mean(0)
res = {}
for n in ns:
    ica = FastICA(n_components=n, whiten="unit-variance", random_state=0, max_iter=1000).fit(dS)
    W = ica.components_                                                    # [n, K]
    rot = lambda S: (S - mu) @ W.T
    r = SC.score_state({"validation": rot(S_val), "test frames (card 141)": rot(S_te)}, mv, sets)
    res[n] = r
    SC.brief(f"{name}, {n} independent directions of change", r)
json.dump(res, open(FA.OUT / f"rotate_{name}.json", "w"), indent=1)
