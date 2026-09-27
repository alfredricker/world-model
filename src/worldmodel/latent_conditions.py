"""Card 006: conditions in the learned state.

Card 005's final network is frozen. Its internal state z (256 numbers per
frame) is split into parts by a sparse autoencoder (a dictionary of 1024
directions, a few active per frame); raw units of z are the baseline. Card
003's condition finder then runs with parts as atoms ("part i on" / "part i
off"), labelled by the supplied success signals. Found parts are tested by
switching them off inside z and reading the heads again. The simulator's
variables are used only to say what a found part means.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import collect, rules_from_matrices
from .envs.keydoor import FORWARD, PICKUP, TOGGLE, variables
from .learn_keydoor import CELLS, GOALS, TARGETS, Frames, Net, auc, build

UNLOCK, KEY = GOALS.index("door_unlocked"), GOALS.index("holding_matching_key")
WALK_GOAL = TARGETS.index("goal")
MAX_RULES = 6          # rules reported per goal; the rest are counted as uncovered
PROBE_VARS = [("front", "door"), ("carrying_matches_door", True), ("door", "open"),
              ("front_matches_door", True), ("carrying", "none")]


# ---------------------------------------------------------------- latent state

@torch.no_grad()
def encode(net, frames: Frames, idx: np.ndarray, device, chunk=16384) -> torch.Tensor:
    out = []
    for s in range(0, len(idx), chunk):
        i = torch.from_numpy(idx[s:s + chunk]).to(device)
        out.append(net.enc(frames(i)))
    return torch.cat(out)


@torch.no_grad()
def heads(net, z):
    logits = net.achieve(z).view(-1, len(GOALS), 5)
    walk = torch.sigmoid(net.walk(z).view(-1, len(TARGETS), 3)).max(2).values
    return torch.sigmoid(logits), walk


class SAE(nn.Module):
    """Sparse autoencoder: f = relu(E(z - b) + c), z' = D f + b, unit-norm columns of D."""

    def __init__(self, dim: int, size: int):
        super().__init__()
        self.enc = nn.Linear(dim, size)
        self.dec = nn.Parameter(torch.randn(size, dim) / dim ** 0.5)
        self.b = nn.Parameter(torch.zeros(dim))

    def directions(self):
        return F.normalize(self.dec, dim=1)

    def forward(self, z):
        f = F.relu(self.enc(z - self.b))
        return f, f @ self.directions() + self.b


def train_sae(z: torch.Tensor, size: int, l1: float, steps: int, seed: int = 0):
    torch.manual_seed(seed)
    sae = SAE(z.shape[1], size).to(z.device)
    with torch.no_grad():
        sae.b.copy_(z.mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    scale = z.pow(2).sum(1).mean()
    for _ in range(steps):
        batch = z[torch.randint(len(z), (4096,), device=z.device)]
        f, zr = sae(batch)
        loss = (batch - zr).pow(2).sum(1).mean() / scale + l1 * f.sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        f, zr = sae(z[:200_000])
        stats = {"l1": l1, "active_per_frame": round(float((f > 0).sum(1).float().mean()), 1),
                 "unexplained_variance": round(float((z[:200_000] - zr).pow(2).sum(1).mean() / scale), 4),
                 "dead_parts": int(((f > 0).sum(0) == 0).sum())}
    return sae, stats


# ---------------------------------------------------------------- rows

def rows_for(d: dict, action: int, goal: int, limit_states: int | None = None):
    """Transitions with this action where the goal did not hold: state index, achieved."""
    i, a = d["t_idx"], d["t_act"]
    m = (a == action) & ~d["hold"][i, goal]
    if limit_states is not None:
        m &= i < limit_states
    return i[m], d["achieved"][m, goal]


def atoms_of(active: np.ndarray, prefix: str):
    """Boolean parts -> atoms "part on" and "part off"."""
    n = active.shape[1]
    atoms = [(f"{prefix}{j}", "on") for j in range(n)] + [(f"{prefix}{j}", "off") for j in range(n)]
    return atoms, np.concatenate([active, ~active], 1)


def rule_stats(members, X, y):
    m = np.ones(len(y), bool)
    for j in members:
        m &= X[:, j]
    out = {"attempts": int(m.sum()), "achieves": round(float(y[m].mean()), 4) if m.any() else None,
           "share_of_successes": round(float((m & y).sum() / max(y.sum(), 1)), 4), "removal": {}}
    for j in members:
        mm = np.ones(len(y), bool)
        for k in members:
            if k != j:
                mm &= X[:, k]
        without = mm & ~X[:, j]
        out["removal"][j] = round(float(y[m].mean() - y[without].mean()), 4) if m.any() and without.any() else None
    return out


_LABELS = {}


def meaning(score: np.ndarray, vars_: list[dict], top=2):
    """Which simulator variable value a part tracks: best areas under the curve."""
    if id(vars_) not in _LABELS:
        keys = sorted({(k, v) for d in vars_ for k, v in d.items() if k not in ("x", "y")}, key=str)
        _LABELS[id(vars_)] = [(f"{k}={v}", np.array([d[k] == v for d in vars_])) for k, v in keys]
    res = []
    for name, label in _LABELS[id(vars_)]:
        if 0 < label.sum() < len(label):
            res.append((round(auc(score, label), 4), name))
    return sorted(res, reverse=True)[:top]


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, default=Path("runs/005_data10k_long/net.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5000, help="training episodes used for rows")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--sae-steps", type=int, default=8000)
    parser.add_argument("--l1", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    started = time.monotonic()
    log_f = open(args.out / "log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    net = Net().to(device)
    net.load_state_dict(torch.load(args.net))
    net.eval()
    train_data = collect(8, 10000, 640, 3)[:args.episodes]
    test_data = collect(8, 500, 640, 99)
    d, dt = build(train_data), build(test_data)
    fr, frt = Frames(d["codes"], device), Frames(dt["codes"], device)
    tr_states = [(ep["layout"], s) for ep in train_data for s in ep["states"]]
    te_states = [(ep["layout"], s) for ep in test_data for s in ep["states"]]
    te_vars = [variables(l, s) for l, s in te_states]
    zt = encode(net, frt, np.arange(len(dt["codes"])), device)
    log(f"train states {len(d['codes'])}, test states {len(dt['codes'])}")
    result = {}

    # Gate: linear read-out of z for the needed variables.
    rng = np.random.default_rng(0)
    sample = rng.choice(len(d["codes"]), 200_000, replace=False)
    zs = encode(net, fr, sample, device)
    s_vars = [variables(*tr_states[i]) for i in sample]
    gate = {}
    for k, v in PROBE_VARS:
        y = torch.tensor([x[k] == v for x in s_vars], dtype=torch.float32, device=device)
        lin = nn.Linear(zs.shape[1], 1).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
        mu, sd = zs.mean(0), zs.std(0) + 1e-6
        for _ in range(1500):
            loss = F.binary_cross_entropy_with_logits(lin((zs - mu) / sd).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            score = lin((zt - mu) / sd).squeeze(1).cpu().numpy()
        gate[f"{k}={v}"] = round(auc(score, np.array([x[k] == v for x in te_vars])), 4)
    result["gate_linear_readout_auc"] = gate
    log("gate " + json.dumps(gate))

    # Criterion-1 cells of card 005, on z and on a rebuilt z.
    def cells(z):
        A, _ = heads(net, z)
        A = A.cpu().numpy()
        out = {}
        for name, goal, action, cond, truth in CELLS:
            g = GOALS.index(goal)
            rows = [i for i, a in zip(dt["t_idx"], dt["t_act"]) if a == action and not dt["hold"][i, g] and cond(te_vars[i])]
            out[name] = round(float(((A[rows, g, action] > 0.5) == bool(truth)).mean()), 4)
        return out

    base_cells = cells(zt)
    # Dictionary: sparsest one whose rebuilt state keeps every cell within 2 points.
    zfit = encode(net, fr, rng.choice(len(d["codes"]), 1_000_000, replace=False), device)
    sweep, sae = [], None
    for l1 in sorted(args.l1):
        cand, stats = train_sae(zfit, args.size, l1, args.sae_steps)
        with torch.no_grad():
            rebuilt = torch.cat([cand(zt[s:s + 65536])[1] for s in range(0, len(zt), 65536)])
        c = cells(rebuilt)
        stats["cells"] = c
        stats["fidelity_ok"] = all(abs(c[k] - base_cells[k]) <= 0.02 for k in c)
        sweep.append(stats)
        log("sae " + json.dumps(stats))
        if stats["fidelity_ok"]:
            sae, chosen = cand, stats
    result["cells_on_z"] = base_cells
    result["sae_sweep"] = sweep
    if sae is None:
        result["stopped"] = "no dictionary kept the heads' answers within 2 points"
        (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        return
    result["sae_chosen"] = chosen
    torch.save(sae.state_dict(), args.out / "sae.pt")
    dirs = sae.directions().detach()

    @torch.no_grad()
    def parts(z):
        return torch.cat([sae(z[s:s + 65536])[0] for s in range(0, len(z), 65536)])

    ft = parts(zt)                                   # test parts, (N_test, size)
    ft_on = (ft > 0).cpu().numpy()

    def run_basis(name, part_fn, on_test, switch_off, n_parts):
        """Criteria 1-2 for one way of splitting z into parts."""
        res = {}
        for goal, action, label in ((UNLOCK, TOGGLE, "unlock"), (KEY, PICKUP, "holding_key")):
            idx, y = rows_for(d, action, goal)
            on = np.concatenate([part_fn(encode(net, fr, idx[s:s + 200_000], device))
                                 for s in range(0, len(idx), 200_000)])
            atoms, X = atoms_of(on, name)
            rules = rules_from_matrices({action: (atoms, X, y)}, max_rules=MAX_RULES,
                                        action_names=("left", "right", "forward", "pickup", "toggle"))
            del X
            ti, ty = rows_for(dt, action, goal)
            _, Xt = atoms_of(on_test[ti], name)
            found = []
            for r in rules:
                st = rule_stats(r["members"], Xt, ty)
                st["parts"] = [f"{atoms[j][0]}={atoms[j][1]}" for j in r["members"]]
                st["train"] = {k: r[k] for k in ("achievement_probability", "share_of_successes")}
                st["meaning"] = {}
                for j in r["members"]:
                    p = j % n_parts
                    score = on_test[:, p].astype(float) * (1 if j < n_parts else -1)
                    st["meaning"][f"{atoms[j][0]}={atoms[j][1]}"] = meaning(score, te_vars)
                found.append(st)
            res[label] = {"rules": found}
            # Criterion 2 on test frames where the action succeeded.
            succ = ti[ty]
            if found and label == "unlock":
                zs_ = zt[succ]
                p0 = heads(net, zs_)[0][:, goal, action].cpu().numpy()
                inter = {"successes": len(succ), "predicted_before": round(float(p0.mean()), 4)}
                for r, st in zip(rules, found):
                    covered = np.ones(len(succ), bool)
                    for j in r["members"]:
                        covered &= _atom(on_test[succ], j, n_parts)
                    for j in r["members"]:
                        key = f"{atoms[j][0]}={atoms[j][1]}"
                        if j >= n_parts:
                            inter[key] = "off-part: not switched"
                            continue
                        zc = switch_off(zs_[covered], j)
                        p = heads(net, zc)[0][:, goal, action].cpu().numpy()
                        inter[key] = {"frames": int(covered.sum()), "below_0.1": round(float((p < 0.1).mean()), 4),
                                      "mean_after": round(float(p.mean()), 4)}
                    # a random active part not in the rule
                    rule_parts = {j for j in r["members"] if j < n_parts}
                    keep_high, n = 0, 0
                    g = np.random.default_rng(1)
                    for k in np.flatnonzero(covered):
                        cand = [p for p in np.flatnonzero(on_test[succ[k]]) if p not in rule_parts]
                        if not cand:
                            continue
                        zc = switch_off(zs_[k:k + 1], int(g.choice(cand)))
                        keep_high += float(heads(net, zc)[0][0, goal, action]) >= 0.9
                        n += 1
                    inter[f"rule {found.index(st)}: other active part off, still >= 0.9"] = round(keep_high / max(n, 1), 4)
                res["interventions"] = inter
        return res

    def _atom(on, j, n):
        return on[:, j] if j < n else ~on[:, j - n]

    def sae_parts(z):
        return np.concatenate([(sae(z[s:s + 65536])[0] > 0).cpu().numpy() for s in range(0, len(z), 65536)])

    def sae_off(z, j):
        f = sae(z)[0][:, j:j + 1]
        return z - f * dirs[j]

    def save():
        (args.out / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")

    result["dictionary"] = run_basis("part", sae_parts, ft_on, sae_off, args.size)
    save()
    log("dictionary done")

    def raw_parts(z):
        return (z > 0).cpu().numpy()

    def raw_off(z, j):
        z = z.clone()
        z[:, j] = 0
        return z

    result["raw_units"] = run_basis("unit", raw_parts, (zt > 0).cpu().numpy(), raw_off, zt.shape[1])
    save()
    log("raw units done")

    # Criterion 3: the part that means holding a key, as a subgoal.
    unlock_rules = result["dictionary"]["unlock"]["rules"]
    carry_parts = []
    for st in unlock_rules:
        for part, m in st["meaning"].items():
            if part.endswith("=on") and m and m[0][1].startswith(("carrying_matches_door=True", "carrying_colour=")):
                carry_parts.append((int(part.split("=")[0][4:]), m[0][1]))
    sub = {}
    i_t, i_a = dt["t_idx"], dt["t_act"]
    for p, var in carry_parts[:2]:
        k, v = var.split("=")
        before = np.array([str(te_vars[i][k]) == v for i in i_t])
        after = np.array([str(te_vars[i + 1][k]) == v for i in i_t])
        event = ~before & after
        switch = ~ft_on[i_t, p] & ft_on[i_t + 1, p]
        entry = {"means": var, "events": int(event.sum()), "switch_ons": int(switch.sum()),
                 "recall": round(float(switch[event].mean()), 4) if event.any() else None,
                 "precision": round(float(event[switch].mean()), 4) if switch.any() else None}
        # Finder with "part p switches on" as the goal: rows from 1500 training episodes.
        n_states = sum(len(ep["states"]) for ep in train_data[:1500])
        ti = d["t_idx"][d["t_idx"] < n_states]
        ta = d["t_act"][d["t_idx"] < n_states]
        on_b = np.concatenate([sae_parts(encode(net, fr, ti[s:s + 200_000], device)) for s in range(0, len(ti), 200_000)])
        on_a = np.concatenate([sae_parts(encode(net, fr, ti[s:s + 200_000] + 1, device)) for s in range(0, len(ti), 200_000)])
        y = ~on_b[:, p] & on_a[:, p]
        mats = {}
        for a in range(5):
            m = (ta == a) & ~on_b[:, p]
            atoms, X = atoms_of(on_b[m], "part")
            mats[a] = (atoms, X, y[m])
        rules = rules_from_matrices(mats, max_rules=2, action_names=("left", "right", "forward", "pickup", "toggle"))
        entry["subgoal_rules"] = []
        for r in rules[:2]:
            parts_m = {}
            for j in r["members"]:
                q = j % args.size
                score = ft_on[:, q].astype(float) * (1 if j < args.size else -1)
                parts_m[f"{atoms[j][0]}={atoms[j][1]}"] = meaning(score, te_vars)
            entry["subgoal_rules"].append({"action": r["action"], "achieves": r["achievement_probability"],
                                           "share": r["share_of_successes"], "parts": parts_m})
        sub[f"part{p}"] = entry
        del mats
    result["subgoal"] = sub
    save()
    log("subgoal done")

    # Also reported: walk to goal square, labelled by the network's own W > 0.1.
    n_states = sum(len(ep["states"]) for ep in train_data)
    idx = np.arange(0, n_states, 8)
    idx = idx[~d["at"][idx, WALK_GOAL] & ~d["term"][idx]]
    z = encode(net, fr, idx, device)
    y = (heads(net, z)[1][:, WALK_GOAL] > 0.1).cpu().numpy()
    atoms, X = atoms_of(sae_parts(z), "part")
    rules = rules_from_matrices({0: (atoms, X, y)}, max_rules=2, action_names=("walk to goal",))
    result["walk_to_goal"] = [{"parts": {f"{atoms[j][0]}={atoms[j][1]}": meaning(
        ft_on[:, j % args.size].astype(float) * (1 if j < args.size else -1), te_vars) for j in r["members"]},
        "achieves": r["achievement_probability"], "share": r["share_of_successes"]} for r in rules[:2]]
    result["seconds"] = round(time.monotonic() - started, 1)
    (args.out / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    log("walk " + json.dumps(result["walk_to_goal"]))
    log(f"done in {result['seconds']} s")


if __name__ == "__main__":
    main()
