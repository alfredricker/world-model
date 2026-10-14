"""Card 106 in breaking world 1: card 105.4's conditions by intervention, with two filters on which replaced states
count (one pass gives every arm).

  H (the world could be in it): under the model's own prediction for doing nothing (action 0), the event's token and
    the replaced token keep their codes.
  K (the model knows it): an independently trained copy predicts the same next code for the event's token.
  Arms: none (105.4), H, K, HK.

Scoring only:
- the truth is 105.4's: the world's own rule on the replaced grid;
- the gate counts H's verdicts against the world's rule, where a state is possible when its lamps agree with its boxes
  (light() leaves it unchanged);
- precision and recall are also given against the truth over possible states only.

  bin/prun python tools/card106/world1.py --net runs/105.6/tokens_Rb_s1021_net.pt --copy runs/105.6/tokens_Rb_s7_net.pt
      → runs/106/world1_<net name>.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card105"))
from model import TransitionModel                          # noqa: E402

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
NET, COPY = Path(arg("--net", "runs/105.6/tokens_Rb_s1021_net.pt")), Path(arg("--copy", "runs/105.6/tokens_Rb_s7_net.pt"))
PER = int(arg("--per", 300))
spec = importlib.util.spec_from_file_location("c1054w1", ROOT / "tools" / "card105.4" / "world1.py")
W1 = importlib.util.module_from_spec(spec)
sys.argv = ["world1.py"]
spec.loader.exec_module(W1)
W1.PER = PER
OUT = ROOT / "runs" / "106"
dev = "cuda"
NT, W, H = W1.NT, W1.W, W1.H
FLOOR, WALL, AGENT, BOX, LAMP, LIT = range(6)
ARMS = ("none", "H", "K", "HK")


def load(path, dim):
    net = TransitionModel(dim, 5, arm=path.name.split("_")[1]).to(dev)
    net.load_state_dict(torch.load(ROOT / path))
    return net.eval()


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1021)
    z = np.load(ROOT / "runs" / "card102" / "world1_test.npz")
    sem = z["sem"].reshape(-1, 63).astype(np.int64)
    V = np.load(ROOT / "runs" / "102.1" / "world1_vec_test.npy")
    C = np.load(ROOT / "runs" / "102.1" / "world1_code_test.npy").astype(np.int64)     # perception's codes
    ztr = np.load(ROOT / "runs" / "card102" / "world1_train.npz")
    c = np.load(ROOT / "runs" / "102.1" / "world1_code_train.npy")[:, :63].reshape(-1).astype(np.int64)
    l = ztr["sem"].reshape(-1)
    kind_of = np.full(64, -1)                                                          # scoring only, as 105.4
    for k in np.unique(c):
        kind_of[k] = np.bincount(l[c == k]).argmax()
    Pr = torch.as_tensor(np.load(ROOT / "runs" / "102.1" / "world1_codes.npz")["protos"], device=dev).float()
    net, net2 = load(NET, V.shape[-1]), load(COPY, V.shape[-1])

    @torch.no_grad()
    def codes(m, Vb, a):
        B = len(Vb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt, _ = m(Vb, W1.P_TILE.to(dev).expand(B, NT, 2), W1.F_TILE.to(dev).expand(B, NT),
                       torch.ones(B, NT, dtype=torch.bool, device=dev), torch.full((B,), a, device=dev))
        return torch.cdist(nxt[:, :63].float().reshape(-1, Pr.shape[1]), Pr).argmin(1).view(B, 63).cpu().numpy()

    rep = {"net": str(NET), "copy": str(COPY)}
    gate = np.zeros((2, 2), np.int64)              # [H keeps, sets aside] x [the rule calls it possible, impossible]
    ev = W1.events(z, rng)
    for kind, lst in ev.items():
        st = {a: {"tp": 0, "fp": 0, "fn": 0, "tp_p": 0, "fp_p": 0, "fn_p": 0, "exact": 0, "asked": 0} for a in ARMS}
        wrong = disagree = 0
        for t, i in lst:
            a = int(z["act"][t])
            g = sem[t].reshape(H, W)
            base_true = W1.true_next(g, a).reshape(-1)[i]
            v0 = torch.as_tensor(V[t], device=dev).float()
            base = codes(net, v0[None], a)[0, i]
            if kind_of[base] != base_true:
                wrong += 1
                continue
            base_known = codes(net2, v0[None], a)[0, i] == base
            disagree += int(not base_known)
            present = {k: int(np.flatnonzero(sem[t] == k)[0]) for k in (FLOOR, WALL, BOX, LAMP, LIT) if (sem[t] == k).any()}
            var, js, newc, tc, tcp, poss = [], [], [], np.zeros(63, bool), np.zeros(63, bool), []
            for j in range(63):
                if j == i:
                    continue
                for k in ([FLOOR] if sem[t, j] == AGENT else [k for k in present if k != sem[t, j]]):
                    g2 = g.copy().reshape(-1)
                    g2[j] = k
                    flip = W1.true_next(g2.reshape(H, W), a).reshape(-1)[i] != base_true
                    g3 = g2.reshape(H, W).copy()
                    W1.light(g3)
                    p = bool((g3.reshape(-1) == g2).all())
                    tc[j] |= flip
                    tcp[j] |= flip and p
                    vj = v0.clone()
                    vj[j] = v0[present[k]]
                    var.append(vj)
                    js.append(j)
                    newc.append(C[t, present[k]])
                    poss.append(p)
            js, newc, poss = np.array(js), np.array(newc), np.array(poss)
            pk, pk2, nop = [], [], []
            for s in range(0, len(var), 512):
                Vb = torch.stack(var[s:s + 512])
                pk.append(codes(net, Vb, a)[:, i])
                pk2.append(codes(net2, Vb, a)[:, i])
                nop.append(codes(net, Vb, 0))
            pk, pk2, nop = np.concatenate(pk), np.concatenate(pk2), np.concatenate(nop)
            keepH = (nop[:, i] == C[t, i]) & (nop[np.arange(len(js)), js] == newc)
            keepK = (pk2 == pk) & base_known
            gate += np.array([[(keepH & poss).sum(), (keepH & ~poss).sum()], [(~keepH & poss).sum(), (~keepH & ~poss).sum()]])
            flips = kind_of[pk] != kind_of[base]
            for arm, keep in (("none", np.ones(len(js), bool)), ("H", keepH), ("K", keepK), ("HK", keepH & keepK)):
                if arm in ("K", "HK") and not base_known:
                    continue                                   # the copies disagree on the event itself: not asked
                mc = np.zeros(63, bool)
                np.logical_or.at(mc, js[keep & flips], True)
                s_ = st[arm]
                s_["asked"] += 1
                s_["tp"] += int((mc & tc).sum()); s_["fp"] += int((mc & ~tc).sum()); s_["fn"] += int((~mc & tc).sum())
                s_["tp_p"] += int((mc & tcp).sum()); s_["fp_p"] += int((mc & ~tcp).sum()); s_["fn_p"] += int((~mc & tcp).sum())
                s_["exact"] += int((mc == tc).all())
        f = lambda x, y: round(x / max(x + y, 1), 4)
        rep[kind] = {"events the model got wrong (not asked)": wrong, "events the copies disagree on": disagree}
        for arm in ARMS:
            s_ = st[arm]
            rep[kind][arm] = {"asked": s_["asked"], "precision": f(s_["tp"], s_["fp"]), "recall": f(s_["tp"], s_["fn"]),
                              "events with exactly the true conditions": round(s_["exact"] / max(s_["asked"], 1), 4),
                              "against the truth over possible states: precision, recall":
                                  [f(s_["tp_p"], s_["fp_p"]), f(s_["tp_p"], s_["fn_p"])]}
        print(kind, json.dumps({a: (rep[kind][a]["precision"], rep[kind][a]["recall"]) for a in ARMS}),
              f"({time.monotonic() - t0:.0f} s)", flush=True)
    rep["gate: H keeps possible, keeps impossible, sets aside possible, sets aside impossible"] = gate.reshape(-1).tolist()
    rep["gate: share of possible kept, share of impossible set aside"] = [
        round(gate[0, 0] / max(gate[:, 0].sum(), 1), 4), round(gate[1, 1] / max(gate[:, 1].sum(), 1), 4)]
    print("gate", rep["gate: share of possible kept, share of impossible set aside"], gate.reshape(-1).tolist(), flush=True)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"world1_{NET.stem}.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
