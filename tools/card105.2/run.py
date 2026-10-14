"""Card 105.2: series 105's one transition model on Crafter, with every awake step (moves included) and the view's
tiles at world places.

Tokens: the 81 tiles of the 9 x 9 grid (card 098.3's encoder vectors, 32-d). The 63 view tiles sit in the world frame
at card 103.4's learned origin (rounded to tiles) plus their place in the view; the 18 carried tiles sit in the
screen's frame. A view tile at t is the same token as the tile at t + 1 with the same world place; tiles leaving the
view have no target, tiles entering it are not predicted. The action is one more token. Forms: tokens, or truth (the
evaluator's kinds, heading and counts as one-hot vectors, card 102.1's: the gate). Arms as in card 105.1 (R, Rb, A).

Data: the crafting curriculum (card 102.1; worlds 0-7 train, 8-9 test; placed by tools/card105.2/origins_curr.py)
and random play (card 098.5; workers 0-1 train, 8-9 test; placed by card 103.5).

Scored by the evaluator (scoring only): each predicted vector is decoded to its nearest code, each code read as its
most common label in training (102.1's labels: kind; carried slot and count; the agent with its heading).

  bin/prun python tools/card105.2/run.py prep                       → runs/105.2/play_vec_w{0,1,8,9}.npy
  bin/prun python tools/card105.2/run.py tokens|truth R|Rb|A [--seed 1021] [--steps 30000]
      → runs/105.2/<form>_<arm>_s<seed>.json, _net.pt
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card105"))
from model import TransitionModel, code_loss, weighted_loss   # noqa: E402

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
OUT = ROOT / "runs" / "105.2"
R102 = ROOT / "runs" / "102.1"
dev = "cuda"
NT, UNIT = 81, 7
MOVES = (1, 2, 3, 4)
CREATURES = (14, 15, 16, 17)
GRASS, TABLE = 2, 11
MAKE_SLOT = {11: 73, 12: 74, 13: 75, 14: 76, 15: 77, 16: 78}   # make action → its tool's carried slot (scoring)
GROUPS = {"move": (1, 2, 3, 4), "do": (5,), "place": (7, 8, 9, 10), "make": tuple(MAKE_SLOT), "noop": (0,)}
NEAR8 = [r * 9 + c for r in (2, 3, 4) for c in (3, 4, 5) if (r, c) != (3, 4)]
FAR16 = [r * 9 + c for r in range(1, 6) for c in range(2, 7) if max(abs(r - 3), abs(c - 4)) == 2]
SQ5 = NEAR8 + FAR16
COLS, ROWS = np.arange(NT) % 9, np.arange(NT) // 9
LOSS = arg("--loss", "plain" if "--plain" in sys.argv else "weighted")   # weighted (105.2), plain (105.7), code (105.8)
PLAIN = LOSS == "plain"
TRUE_ORIGINS = arg("--origins", "learned") in ("true", "pose")   # the agent's position: a diagnostic (105.8), then
POSE_ORIGINS = arg("--origins", "learned") == "pose"             # proprioception under GOAL C1 as amended (105.10)
MODEL_ORIGINS = arg("--origins", "learned") == "model"  # card 105.9: the placing's prior from 105.8's model
TRAIN = [("curr", w) for w in range(8)] + [("play", 0), ("play", 1)]
TEST = [("curr", 8), ("curr", 9), ("play", 8), ("play", 9)]


def prep():
    os.environ.update(WM98_NEAR="0.0005", WM98_EXACT="1", WM98_CHANGED="1", WM98_PAIR="1", WM98_RADIUS="exact")
    sys.path.insert(0, str(ROOT / "tools" / "card098"))
    import encoder as EN
    enc = EN.Enc().to(EN.dev)
    enc.load_state_dict(torch.load(ROOT / "runs" / "098.3" / "encoder_joint_blocked.pt"))
    enc.eval()
    for w in (0, 1, 8, 9):
        z = np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz")
        E = EN.embed(enc, EN.tiles(z["obs"]))
        np.save(OUT / f"play_vec_w{w}.npy", E.astype(np.float16))
        print("play worker", w, E.shape, flush=True)


def vec_truth(sem, face, inv):
    """Card 102.1's one-hot tokens: kind (19), the agent's heading (4), carried slot (16) and count (10)."""
    n = len(sem)
    V = np.zeros((n, NT, 49), np.float16)
    r, c = np.nonzero(sem >= 0)
    V[r, c, sem[r, c]] = 1
    hd = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    V[np.arange(n), 31, 19 + np.array([hd.get((int(a), int(b)), 0) for a, b in face])] = 1
    for k in range(16):
        V[:, 63 + k, 23 + k] = 1
        V[np.arange(n), 63 + k, 39 + inv[:, k].clip(0, 9)] = 1
    return V


def load(sources, form):
    """Every awake transition of the sources, with each token's place, its partner at t + 1 and the labels."""
    D = {k: [] for k in ("V", "lab", "pos", "act", "pos_true", "src")}
    T, KM, off = [], [], 0
    for kind, w in sources:
        if kind == "curr":
            z = np.load(R102 / f"curr_w{w}.npz")
            org = np.load(ROOT / "runs" / "105.2" / f"org_curr_w{w}.npy")
        else:
            z = np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz")
            org = np.load(ROOT / "runs" / "103.5" / f"org_tau300_w{w}.npy")
        n = len(z["act"])
        act, ep, done, sleep = z["act"].astype(np.int64), z["ep"], z["done"], z["sleep"]
        sem = z["sem"].reshape(n, 63).astype(np.int64)
        inv, face = z["inv"].astype(np.int64), z["face"].astype(np.int64)
        lab = np.full((n, NT), -1, np.int64)
        lab[:, :63] = sem
        lab[:, 63:79] = 100 + 10 * np.arange(16)[None] + inv.clip(0, 9)
        lab[:, 31] = 1300 + (face[:, 0] + 1) * 3 + (face[:, 1] + 1)
        if form == "truth":
            V = vec_truth(sem, face, inv)
        else:
            V = np.load(R102 / f"curr_vec_w{w}.npy") if kind == "curr" else np.load(ROOT / "runs" / "105.2" / f"play_vec_w{w}.npy")
        if MODEL_ORIGINS:
            org = np.load(ROOT / "runs" / "105.9" / f"org_{kind}_w{w}.npy")
        o = np.rint(org[:n] / UNIT).astype(np.int64)                    # the view's origin in world tiles (x, y)
        if TRUE_ORIGINS:                                                # diagnostic: the evaluator's origins
            o = z["pos"].astype(np.int64) - (4, 3)
        t = np.flatnonzero(np.r_[ep[1:] == ep[:-1], False] & ~done & ~sleep & ~np.r_[sleep[1:], True])
        sh = o[t + 1] - o[t]
        c1, r1 = COLS[None, :63] - sh[:, :1], ROWS[None, :63] - sh[:, 1:]
        km = np.full((len(t), NT), -1, np.int64)
        km[:, :63] = np.where((c1 >= 0) & (c1 < 9) & (r1 >= 0) & (r1 < 7), r1 * 9 + c1, -1)
        km[:, 63:] = np.arange(63, NT)
        pos = np.zeros((n, NT, 2), np.float32)
        pos[:, :63] = o[:, None, :] + np.stack([COLS[:63], ROWS[:63]], 1)[None]
        pos[:, 63:] = np.stack([COLS[63:], ROWS[63:]], 1)[None]
        for k, v in (("V", V), ("lab", lab), ("pos", pos), ("act", act), ("pos_true", z["pos"].astype(np.int64)),
                     ("src", np.full(n, len(D["src"])))):
            D[k].append(v)
        T.append(t + off)
        KM.append(km)
        off += n
    D = {k: np.concatenate(v) for k, v in D.items()}
    D["t"], D["km"] = np.concatenate(T), np.concatenate(KM)
    D["V"] = torch.as_tensor(D["V"])
    return D


FRAME = torch.tensor([0] * 63 + [1] * 18)


def batch(D, i, V0=None):
    """Transitions i (indices into D["t"]): inputs, targets, validity. V0 replaces the inputs (the role test)."""
    t, km = D["t"][i], D["km"][i]
    if V0 is None:
        V0 = D["V"][t].to(dev).float()
    V1 = D["V"][t + 1].to(dev).float()
    kmt = torch.as_tensor(km, device=dev)
    Vn = torch.gather(V1, 1, kmt.clamp(min=0)[..., None].expand(-1, -1, V1.shape[-1]))
    valid = kmt >= 0
    P = torch.as_tensor(D["pos"][t], device=dev)
    F = FRAME.to(dev).expand(len(i), NT)
    M = torch.ones(len(i), NT, dtype=torch.bool, device=dev)
    a = torch.as_tensor(D["act"][t], device=dev)
    return V0, Vn, valid, P, F, M, a


def decoder(tr, form):
    """Prototypes and each one's label at each place (scoring only), as card 102.1: the most common label of the pair
    (code, place) in training, else of the code. Prototypes: 098.3's codes, or the distinct one-hot tokens."""
    idx = np.random.default_rng(0).choice(tr["t"], min(30000, len(tr["t"])), replace=False)
    Vs = tr["V"][idx]
    if form == "truth":
        U = torch.as_tensor(np.unique(Vs.reshape(-1, Vs.shape[-1]).numpy(), axis=0), device=dev).float()
    else:
        U = torch.as_tensor(np.load(ROOT / "runs" / "098.3" / "codes_joint_blocked.npz")["protos"], device=dev).float()
    c = np.concatenate([torch.cdist(Vs[s:s + 2000].to(dev).float().reshape(-1, U.shape[1]), U).argmin(1).cpu().numpy()
                        for s in range(0, len(idx), 2000)]).reshape(len(idx), NT)
    L = tr["lab"][idx]
    by_code = np.full(len(U), -2, np.int64)
    pair = np.full((len(U), NT), -2, np.int64)
    for k in np.unique(c):
        m = c == k
        by_code[k] = np.bincount(L[m] + 2).argmax() - 2
        for q in np.unique(np.nonzero(m)[1]):
            pair[k, q] = np.bincount(L[:, q][m[:, q]] + 2).argmax() - 2
    pair = np.where(pair == -2, by_code[:, None], pair)
    return U, pair


@torch.no_grad()
def predict(net, D, i, Pt, plab, V0=None):
    V0, Vn, valid, P, F, M, a = batch(D, i, V0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        nxt, _ = net(V0, P, F, M, a)
    dist = torch.cdist(nxt.float().reshape(-1, Pt.shape[1]), Pt).reshape(len(i), NT, -1)
    code = dist.argmin(-1).cpu().numpy()
    code0 = torch.cdist(V0.reshape(-1, Pt.shape[1]), Pt).argmin(1).reshape(len(i), NT).cpu().numpy()
    return plab[code, np.arange(NT)[None]], dist, valid.cpu().numpy(), code, code0


def evaluate(net, te, Pt, plab, tau=None):
    """tau (card 105.8): the model's probabilities, p(c) proportional to exp(-d^2 / tau); the agent's next place is then
    the tile where an agent code is most probable. Without tau, the tile nearest an agent code (squared-error models)."""
    agent_protos = torch.as_tensor(plab[:, 31] >= 1300, device=dev)    # codes read as the agent at the view's centre
    pred, apl = np.zeros((len(te["t"]), NT), np.int64), np.zeros(len(te["t"]), np.int64)
    apl_d = np.zeros(len(te["t"]), np.int64)
    same = np.zeros((len(te["t"]), NT), bool)                         # the predicted code is the current one
    for s in range(0, len(te["t"]), 1024):
        i = np.arange(s, min(s + 1024, len(te["t"])))
        p, dist, valid, code, code0 = predict(net, te, i, Pt, plab)
        pred[i], same[i] = p, code == code0
        da = dist[..., agent_protos].min(-1).values[:, :63]                # nearest agent code, per view tile
        da[~torch.as_tensor(valid[:, :63], device=dev)] = float("inf")
        apl_d[i] = da.argmin(1).cpu().numpy()
        if tau is not None:
            lg = -dist[:, :63] ** 2 / tau
            lp = torch.logsumexp(lg[..., agent_protos], -1) - torch.logsumexp(lg, -1)
            lp[~torch.as_tensor(valid[:, :63], device=dev)] = -float("inf")
            apl[i] = lp.argmax(1).cpu().numpy()
        else:
            apl[i] = apl_d[i]
    t, km = te["t"], te["km"]
    lab0 = te["lab"][t]
    lab1 = np.where(km >= 0, np.take_along_axis(te["lab"][t + 1], km.clip(min=0), 1), -1)
    act = te["act"][t]
    ch = lab1 != lab0
    # as card 102.1: a token that did not change is right if no change is predicted (its code kept); one that changed
    # is right if its predicted code reads as the evaluator's next label
    right = np.where(ch, pred == lab1, same)
    # scored (102.1's): non-creature view tiles with a target, at neither of the agent's places; carried items
    d = te["pos_true"][t + 1] - te["pos_true"][t]
    a_next = (3 + d[:, 1]) * 9 + (4 + d[:, 0])                          # the agent's place at t + 1, as a tile at t
    scored = np.zeros_like(right)
    scored[:, :63] = (km[:, :63] >= 0) & (lab0[:, :63] >= 0) & ~np.isin(lab0[:, :63], CREATURES) \
        & ~np.isin(lab1[:, :63], CREATURES)
    scored[:, 31] = False
    scored[np.arange(len(t)), np.clip(a_next, 0, 62)] &= ~((a_next >= 0) & (a_next < 63))
    scored[:, 67:79] = True
    whole = (right | ~scored).all(1)
    still = (~ch | ~scored).all(1)
    rep = {"test transitions": int(len(t))}
    sapling = np.zeros_like(ch)                                         # saplings gained (probability 0.1 per try)
    sapling[:, 67] = (lab1[:, 67] > lab0[:, 67]) & (act == 5)[:, None][:, 0]
    for g, acts in GROUPS.items():
        m = np.isin(act, acts)
        rep[g] = {"steps": int(m.sum()), "whole scored next state right": round(float(whole[m].mean()), 4),
                  "nothing changes: whole scored next state right": round(float(still[m].mean()), 4),
                  "scored tokens that changed": int((ch & scored)[m].sum()),
                  "changed scored tokens right": round(float(right[m][(ch & scored)[m]].mean()), 4)
                  if (ch & scored)[m].any() else None,
                  "changed scored tokens right, saplings left out (105.8)":
                      round(float(right[m][(ch & scored & ~sapling)[m]].mean()), 4)
                      if (ch & scored & ~sapling)[m].any() else None}
    slot = np.array([MAKE_SLOT.get(int(a), 73) for a in act])
    made = np.isin(act, list(MAKE_SLOT)) & ch[np.arange(len(t)), slot]
    rep["1 makes that made a tool: whole scored next state right"] = round(float(whole[made].mean()), 4)
    rep["makes that made a tool"] = int(made.sum())
    mv = np.isin(act, MOVES)
    blocked = mv & (d == 0).all(1)
    ok = apl == a_next
    rep["2 move steps: the agent's next place right, moved"] = round(float(ok[mv & ~blocked].mean()), 4)
    rep["2 move steps: the agent's next place right, blocked"] = round(float(ok[blocked].mean()), 4)
    rep["2 move steps: the agent's next place right, all"] = round(float(ok[mv].mean()), 4)
    rep["move steps (moved, blocked)"] = [int((mv & ~blocked).sum()), int(blocked.sum())]
    rep["2 read-out"] = "most probable agent code (105.8)" if tau is not None else "nearest agent code"
    if tau is not None:
        okd = apl_d == a_next
        rep["2 by the nearest agent code instead: moved, blocked, all"] = [
            round(float(okd[mv & ~blocked].mean()), 4), round(float(okd[blocked].mean()), 4), round(float(okd[mv].mean()), 4)]
    rep["3 role test"] = role_test(net, te, made, Pt, plab, lab0)
    return rep


def role_test(net, te, made, Pt, plab, lab0):
    """102.1's role test: held-out makes that made a tool with exactly one table in the 5 x 5 square; the table's
    vector moved to each of the 8 near and 16 far places (its old place given a grass vector from outside the
    square). Passing: the tool's carried slot is predicted to change at all 8 near places and at none of the far."""
    rows, skipped = [], 0
    for s in np.flatnonzero(made & (te["src"][te["t"]] < 2)):          # the curriculum's test worlds
        tab = [p for p in SQ5 if lab0[s, p] == TABLE]
        grass = [p for p in range(63) if p not in SQ5 and lab0[s, p] == GRASS]
        if len(tab) != 1 or not grass:
            skipped += 1
            continue
        p0, g = tab[0], grass[0]
        V = te["V"][te["t"][s]].to(dev).float()[None].repeat(24, 1, 1)
        for j, q in enumerate(SQ5):
            V[j, p0], V[j, q] = V[j, g].clone(), te["V"][te["t"][s], p0].to(dev).float()
        _, _, _, code, code0 = predict(net, te, np.full(24, s), Pt, plab, V0=V)
        sl = MAKE_SLOT[int(te["act"][te["t"][s]])]
        rows.append(code[:, sl] != code0[:, sl])                        # the tool's slot predicted to change
    R = np.array(rows).reshape(-1, 24)
    ok = R[:, :8].all(1) & ~R[:, 8:].any(1)
    f = lambda x: round(float(x), 4) if len(R) else None
    return {"tests": len(R), "skipped (not one table in the 5 x 5 square, or no grass outside)": skipped,
            "share passing (made at all 8 near, at none of the 16 far)": f(ok.mean()),
            "near places made": f(R[:, :8].mean()), "far places made": f(R[:, 8:].mean())}


def main():
    if sys.argv[1] == "prep":
        return prep()
    FORM, ARM = sys.argv[1], sys.argv[2]
    if "--eval" in sys.argv:                                            # re-score a saved net: --eval <net> --tau <tau>
        tr, te = load(TRAIN, FORM), load(TEST, FORM)
        net = TransitionModel(tr["V"].shape[-1], 17, arm=ARM).to(dev)
        net.load_state_dict(torch.load(arg("--eval", None)))
        net.eval()
        Pt, plab = decoder(tr, FORM)
        tau = float(arg("--tau", "nan")) if "--tau" in sys.argv else None
        rep = {"net": arg("--eval", None), "tau": tau, **evaluate(net, te, Pt, plab, tau)}
        Path(arg("--eval", None)).with_suffix(".rescored.json").write_text(json.dumps(rep, indent=1) + "\n")
        return print(json.dumps({k: v for k, v in rep.items() if k[0] in "23"}, indent=1))
    SEED, STEPS, BS = int(arg("--seed", 1021)), int(arg("--steps", 30000)), int(arg("--bs", 128))
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.monotonic()
    tr, te = load(TRAIN, FORM), load(TEST, FORM)
    print(f"loaded: {len(tr['t'])} training and {len(te['t'])} test transitions ({time.monotonic() - t0:.0f} s)",
          flush=True)
    net = TransitionModel(tr["V"].shape[-1], 17, arm=ARM).to(dev)
    log_tau = torch.nn.Parameter(torch.tensor(-2.3, device=dev))      # card 105.8's temperature, tau = 0.1 at the start
    Pt, plab = decoder(tr, FORM)
    opt = torch.optim.AdamW([*net.parameters(), log_tau], lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
    for step in range(STEPS):
        i = rng.choice(len(tr["t"]), BS, replace=False)
        V0, Vn, valid, P, F, M, a = batch(tr, i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt, pres = net(V0, P, F, M, a)
        if LOSS == "code":
            loss = code_loss(nxt.float(), pres.float(), Vn, M, valid, Pt, log_tau)
        else:
            loss = weighted_loss(nxt.float(), pres.float(), Vn, M, V0, valid, plain=PLAIN)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"step {step} loss {float(loss.detach()):.5f} ({time.monotonic() - t0:.0f} s)", flush=True)
    net.eval()
    tau = float(log_tau.exp()) if LOSS == "code" else None
    rep = {"form": FORM, "arm": ARM, "seed": SEED, "steps": STEPS, "loss": LOSS, "tau": round(float(log_tau.exp()), 5),
           "training transitions": int(len(tr["t"])),
           **evaluate(net, te, Pt, plab, tau), "seconds": round(time.monotonic() - t0, 1)}
    out = {"weighted": OUT, "plain": ROOT / "runs" / "105.7", "code": ROOT / "runs" / "105.8"}[LOSS]
    if TRUE_ORIGINS:
        out = out / "true_origins"
        rep["origins"] = "evaluator's (diagnostic)"
    if POSE_ORIGINS:
        out = ROOT / "runs" / "105.10"
        rep["origins"] = "the agent's own pose (proprioception, GOAL C1 as amended 2026-10-13; card 105.10)"
    if MODEL_ORIGINS:
        out = ROOT / "runs" / "105.9"
        rep["origins"] = "103.4's placing with 105.8's model as its prior (105.9)"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{FORM}_{ARM}_s{SEED}.json").write_text(json.dumps(rep, indent=1) + "\n")
    torch.save(net.state_dict(), out / f"{FORM}_{ARM}_s{SEED}_net.pt")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
