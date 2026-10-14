"""Card 105.4 in breaking world 1: conditions read from the transition model by intervention.

For an event at step t (a lamp lights or goes out, a box is pushed, the agent moves, a move is blocked), token i is
the one whose outcome is asked about (the lamp; the box's destination; the move's target; the agent's own tile when
blocked). Every other world tile j is replaced, one at a time, by each other kind's vector seen in the same state
(the agent's tile only by floor: the agent removed). j is a condition of i if some replacement changes i's predicted
next kind. Scoring only: the truth is the same replacement on the evaluator's grid, run through the world's own rule
(tools/card102/worlds.py), and j is a true condition if some replacement changes i's true next kind. Only events whose
unaltered outcome the model predicts right are asked. Baseline: "every tile within one step of i or of the agent".

  bin/prun python tools/card105.4/world1.py --net runs/105.6/tokens_Rb_s1021_net.pt [--per 300]
      → runs/105.4/world1_<net name>.json
"""
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
NET = Path(arg("--net", "runs/105.6/tokens_Rb_s1021_net.pt"))
PER = int(arg("--per", 300))
ARM = NET.name.split("_")[1]
OUT = ROOT / "runs" / "105.4"
dev = "cuda"
NT, W, H = 81, 9, 7
FLOOR, WALL, AGENT, BOX, LAMP, LIT = range(6)
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
P_TILE = torch.tensor([(k % 9, k // 9) for k in range(NT)], dtype=torch.float32)
F_TILE = torch.tensor([0 if k < 63 else 1 for k in range(NT)])


def light(g):
    for y, x in zip(*np.nonzero((g == LAMP) | (g == LIT))):
        near = any(0 <= y + dy < H and 0 <= x + dx < W and g[y + dy, x + dx] == BOX for dx, dy in MOVES.values())
        g[y, x] = LIT if near else LAMP


def true_next(grid, a):
    """World 1's step (card 102) on a 7 x 9 grid with the agent drawn in it; without an agent nothing moves."""
    g = grid.copy()
    ag = np.argwhere(g == AGENT)
    if len(ag) != 1 or a not in MOVES:
        light(g)
        return g
    y, x = ag[0]
    g[y, x] = FLOOR
    dx, dy = MOVES[a]
    ny, nx = y + dy, x + dx
    if 0 <= ny < H and 0 <= nx < W and g[ny, nx] == FLOOR:
        y, x = ny, nx
    elif 0 <= ny < H and 0 <= nx < W and g[ny, nx] == BOX:
        by, bx = ny + dy, nx + dx
        if 0 <= by < H and 0 <= bx < W and g[by, bx] == FLOOR:
            g[by, bx], g[ny, nx] = BOX, FLOOR
            y, x = ny, nx
    light(g)
    g[y, x] = AGENT
    return g


def events(z, rng):
    sem, act, ep, done, push = z["sem"].reshape(-1, 63).astype(np.int64), z["act"].astype(np.int64), z["ep"], \
        z["done"], z["push"]
    T = np.flatnonzero(np.r_[ep[1:] == ep[:-1], False] & ~done)
    ev = {k: [] for k in ("lamp lights", "lamp goes out", "push", "move", "blocked")}
    for t in T:
        s0, s1, a = sem[t], sem[t + 1], int(act[t])
        for i in np.flatnonzero((s0 == LAMP) & (s1 == LIT)):
            ev["lamp lights"].append((t, int(i)))
        for i in np.flatnonzero((s0 == LIT) & (s1 == LAMP)):
            ev["lamp goes out"].append((t, int(i)))
        if a in MOVES:
            ag = int(np.argmax(s0 == AGENT))
            dx, dy = MOVES[a]
            tgt = ag + dx + 9 * dy
            if push[t]:
                ev["push"].append((t, tgt + dx + 9 * dy))
            elif s1[ag] == AGENT:
                ev["blocked"].append((t, ag))
            else:
                ev["move"].append((t, tgt))
    return {k: [v[j] for j in sorted(rng.choice(len(v), min(PER, len(v)), replace=False))] for k, v in ev.items()}


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1021)
    z = np.load(ROOT / "runs" / "card102" / "world1_test.npz")
    sem = z["sem"].reshape(-1, 63).astype(np.int64)
    V = np.load(ROOT / "runs" / "102.1" / "world1_vec_test.npy")
    # each code's kind, by majority in training (scoring only), as card 105.1
    ztr = np.load(ROOT / "runs" / "card102" / "world1_train.npz")
    c = np.load(ROOT / "runs" / "102.1" / "world1_code_train.npy")[:, :63].reshape(-1).astype(np.int64)
    l = ztr["sem"].reshape(-1)
    kind_of = np.full(64, -1)
    for k in np.unique(c):
        kind_of[k] = np.bincount(l[c == k]).argmax()
    Pr = torch.as_tensor(np.load(ROOT / "runs" / "102.1" / "world1_codes.npz")["protos"], device=dev).float()
    net = TransitionModel(V.shape[-1], 5, arm=ARM).to(dev)
    net.load_state_dict(torch.load(ROOT / NET))
    net.eval()

    @torch.no_grad()
    def pred_kind(Vb, a, i):
        B = len(Vb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt, _ = net(Vb, P_TILE.to(dev).expand(B, NT, 2), F_TILE.to(dev).expand(B, NT),
                         torch.ones(B, NT, dtype=torch.bool, device=dev), torch.full((B,), a, device=dev))
        return kind_of[torch.cdist(nxt[:, i].float(), Pr).argmin(1).cpu().numpy()]

    rep = {"net": str(NET), "arm": ARM}
    ev = events(z, rng)
    for kind, lst in ev.items():
        tp = fp = fn = btp = bfp = bfn = exact = asked = wrong = 0
        sizes = []
        for t, i in lst:
            a = int(z["act"][t])
            g = sem[t].reshape(H, W)
            base_true = true_next(g, a).reshape(-1)[i]
            v0 = torch.as_tensor(V[t], device=dev).float()
            base_pred = pred_kind(v0[None], a, i)[0]
            if base_pred != base_true:
                wrong += 1
                continue
            asked += 1
            present = {k: int(np.flatnonzero(sem[t] == k)[0]) for k in (FLOOR, WALL, BOX, LAMP, LIT) if (sem[t] == k).any()}
            var, js, tc = [], [], np.zeros(63, bool)
            for j in range(63):
                if j == i:
                    continue
                ks = [FLOOR] if sem[t, j] == AGENT else [k for k in present if k != sem[t, j]]
                for k in ks:
                    g2 = g.copy().reshape(-1)
                    g2[j] = k
                    if true_next(g2.reshape(H, W), a).reshape(-1)[i] != base_true:
                        tc[j] = True
                    vj = v0.clone()
                    vj[j] = v0[present[k]]
                    var.append(vj)
                    js.append(j)
            mc = np.zeros(63, bool)
            for s in range(0, len(var), 512):
                pk = pred_kind(torch.stack(var[s:s + 512]), a, i)
                for j, p in zip(js[s:s + 512], pk):
                    mc[j] |= p != base_pred
            ag = int(np.argmax(sem[t] == AGENT))
            near = np.zeros(63, bool)
            for j in range(63):
                if j != i and (max(abs(j % 9 - i % 9), abs(j // 9 - i // 9)) <= 1
                               or max(abs(j % 9 - ag % 9), abs(j // 9 - ag // 9)) <= 1):
                    near[j] = True
            tp += int((mc & tc).sum()); fp += int((mc & ~tc).sum()); fn += int((~mc & tc).sum())
            btp += int((near & tc).sum()); bfp += int((near & ~tc).sum()); bfn += int((~near & tc).sum())
            exact += int((mc == tc).all())
            sizes.append(int(tc.sum()))
        f = lambda x, y: round(x / max(x + y, 1), 4)
        rep[kind] = {"events asked (model right on the unaltered outcome)": asked,
                     "events the model got wrong (not asked)": wrong,
                     "precision": f(tp, fp), "recall": f(tp, fn),
                     "events with exactly the true conditions": round(exact / max(asked, 1), 4),
                     "baseline 'within one step of i or the agent': precision": f(btp, bfp),
                     "baseline: recall": f(btp, bfn),
                     "true conditions per event (mean)": round(float(np.mean(sizes)), 2) if sizes else None}
        print(kind, rep[kind], f"({time.monotonic() - t0:.0f} s)", flush=True)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"world1_{NET.stem}.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
