"""Card 171, diagnostic: does a prediction change when the slots a touch does not act on are replaced by their
recalled twins (the mean of the 16 remembered vectors nearest in value, as the search builds imagined states)?

  bash tools/card171/run.sh 240 python tools/card171/twin.py   -> runs/171/twin.json
Training episodes only; no simulator truth read.
"""
import json
import numpy as np
import torch
import model171 as M1
import effects171 as E

M = M1.M
Z, tc, ep = M.load_split(E.LEARN)
res = {}
for name, path in (("card168", M.ROOT / "runs/168/scene_wnet_0.pt"), ("card171 both", E.OUT / "scene_wnet_0.pt"), ("card171 range only", E.OUT / "scene_wnet_rng_0.pt")):
    net, ck = E.load(path)
    meta = ck["meta"]; mov = meta["mov_si"]; S = net.n_slots
    H, Val, V, A = M.encode_all(net, Z)
    bank_h, bank_val = torch.from_numpy(H).to(M.DEV), torch.from_numpy(Val).to(M.DEV)
    u = torch.from_numpy(M.make_u_act(tc, meta)).to(M.DEV)
    Zt = torch.from_numpy(Z).to(M.DEV)
    s = np.where(ep[:-1] == ep[1:])[0]
    grp = meta["group"][np.linalg.norm(tc[:, None, :3] - meta["cen"][None], axis=-1).argmin(1)]
    own = np.array([meta["slots"].index(g) if g in meta["slots"] else -1 for g in grp[s]])
    p_real, p_twin, v_real, v_twin = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(s), 256):
            b = torch.as_tensor(s[i:i + 256], dtype=torch.long, device=M.DEV)
            eo = net.encode(Zt[b])
            o = torch.as_tensor(own[i:i + 256], device=M.DEV)
            h2 = eo["h"].clone()
            for k in range(S):
                if k == mov:
                    continue
                rows = torch.where(o != k)[0]                 # every slot the touch does not act on
                h2[rows, k] = M1.recall_h(bank_h, bank_val, k, eo["val"][rows, k])
            t1 = net.forward_transition(eo["h"], eo["vis_logit"], eo["aim"], u[b])
            t2 = net.forward_transition(h2, eo["vis_logit"], eo["aim"], u[b])
            p_real.append(t1["p_chg"].cpu().numpy()); p_twin.append(t2["p_chg"].cpu().numpy())
            v_real.append(t1["vis_next_logit"].cpu().numpy()); v_twin.append(t2["vis_next_logit"].cpu().numpy())
    p_real, p_twin = np.concatenate(p_real), np.concatenate(p_twin)
    v_real, v_twin = np.concatenate(v_real), np.concatenate(v_twin)
    ok = own >= 0
    r = np.arange(len(s))[ok]
    flip = (p_real[r, own[ok]] >= 0.5) != (p_twin[r, own[ok]] >= 0.5)
    out = dict(touches=int(ok.sum()), own_slot_prediction_flips_pct=round(100.0 * float(flip.mean()), 1))
    for k in range(S):
        m = own[ok] == k
        out[f"slot {k} ({meta['slot_kind'][k]}) touches"] = dict(n=int(m.sum()), flips_pct=round(100.0 * float(flip[m].mean()), 1))
    dr = own == meta["cont_si"]
    out["cube in view after a drawer touch: flips_pct"] = round(100.0 * float(((v_real[dr] >= 0) != (v_twin[dr] >= 0)).mean()), 1)
    res[name] = out
    print(name, json.dumps(out), flush=True)
json.dump(res, open(E.OUT / "twin.json", "w"), indent=1)
