"""Card 173: scoring. Truth is read here and nowhere else (and in the gate's upper-bound fit).

  score173.py <name> [<name> ...]      runs/173/<name>.pt -> runs/173/score_<name>.json

Criterion 1  as card 172 (tools/card172/score.py): the distance over the whole state separates pairs of frames that
             differ in exactly one variable from pairs that differ in none. AUROC.
Factors      card 172's rule with its four constants, read from changes over 5 steps (not 1) of 100 training episodes,
             no labels: unused if std < 0.02; moves if the 5-step change is over 5 times its median and over 0.02;
             tied if the windows in which two coordinates move overlap by half or more; factor = connected group.
Criterion 2  over 5-step windows of the validation episodes ("differs" = cube or slide over 1 cm, a lock flipped, as in
             card 172's data check): (a) each variable's best factor is a different one; (b) share of the state's
             change in that factor over windows where only that variable differs; (c) the state's mean change in windows
             where nothing differs over its mean change where exactly one variable does.
Also         the held-out InfoNCE loss the ladder chooses K by (40,000 validation moments, the same for every fit), and
             card 172's single-step version of (c) as a diagnostic.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
import factors as FA
import score as SC
import factors173 as F3

OUT, DEV, NAMES, GAP = F3.OUT, FA.DEV, SC.NAMES, 5


def factor_rule(S, gap=GAP):
    K = S.shape[-1]
    used = S.reshape(-1, K).std(0) >= SC.USED_STD
    c = np.abs(S[:, gap:] - S[:, :-gap]).reshape(-1, K)
    moves = c > np.maximum(SC.MOVE_X * np.median(c, 0), SC.MOVE_MIN)[None]
    inter = (moves[:, :, None] & moves[:, None, :]).sum(0).astype(np.float64)
    n = moves.sum(0).astype(np.float64)
    union = n[:, None] + n[None, :] - inter
    tied = (inter / np.maximum(union, 1.0) >= SC.TIE) & (union > 0)
    lab = -np.ones(K, int); groups = []
    for k in range(K):
        if not used[k] or lab[k] >= 0:
            continue
        stack, g = [k], []
        lab[k] = len(groups)
        while stack:
            a = stack.pop(); g.append(a)
            for b in np.where(tied[a] & used & (lab < 0))[0]:
                lab[b] = len(groups); stack.append(int(b))
        groups.append(sorted(g))
    return used, groups, dict(moves_share_per_coordinate=[round(float(x), 3) for x in moves.mean(0)],
                              overlap=[[round(float(x), 2) for x in r] for r in inter / np.maximum(union, 1.0)])


def differ(m, gap=GAP):
    c, d, w, l = m["cube"], m["drawer"], m["window"], m["locks"]
    return np.stack([np.linalg.norm(c[:, gap:] - c[:, :-gap], axis=-1) > 0.01, np.abs(d[:, gap:] - d[:, :-gap]) > 0.01,
                     np.abs(w[:, gap:] - w[:, :-gap]) > 0.01, l[:, gap:, 0] != l[:, :-gap, 0], l[:, gap:, 1] != l[:, :-gap, 1]], -1)


def crit2(S, used, groups, mv, gap=GAP):
    D = differ(mv, gap); n = D.sum(-1)
    c = np.abs(S[:, gap:] - S[:, :-gap]) * used[None, None]
    tot = c.sum(-1)
    share = np.zeros((5, max(len(groups), 1)))
    for v in range(5):
        m = D[..., v] & (n == 1)
        den = tot[m].sum()
        for f, g in enumerate(groups):
            share[v, f] = c[m][:, g].sum() / max(den, 1e-9)
    best = share.argmax(1)
    none, one = float(tot[n == 0].mean()), float(tot[n == 1].mean())
    sh = {v: round(100 * float(share[i, best[i]]), 1) for i, v in enumerate(NAMES)}
    return dict(
        n_used=int(used.sum()), n_factors=len(groups), factors=[[int(k) for k in g] for g in groups],
        best_factor={v: int(best[i]) for i, v in enumerate(NAMES)}, distinct=bool(len(set(best.tolist())) == 5 and len(groups) >= 5),
        share_in_best=sh, n_at_80=int(sum(x >= 80.0 for x in sh.values())),
        windows_only={v: int((D[..., i] & (n == 1)).sum()) for i, v in enumerate(NAMES)}, windows_none=int((n == 0).sum()),
        nothing_differs_over_one_variable_pct=round(100 * none / max(one, 1e-9), 1),
        per_variable_change_over_nothing={v: round(float(tot[D[..., i] & (n == 1)].mean() / max(none, 1e-9)), 1) for i, v in enumerate(NAMES)},
        share_matrix=[[round(100 * float(x), 1) for x in r] for r in share],
    ), best


@torch.no_grad()
def val_nce(S_val, pose, g):
    """Held-out InfoNCE with the frozen pose term, 40,000 moments drawn the same way for every fit."""
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    S = torch.from_numpy(S_val).to(DEV)
    e, t, ce, ct = FA.sample(pose, 40000, gen)
    gl = g(pose[e, t][:, None].expand(-1, ce.shape[1], -1), pose[ce, ct])
    logit = -(S[e, t][:, None] - S[ce, ct]).abs().sum(-1) / F3.TAU + gl
    zero = torch.zeros(len(e), dtype=torch.long, device=DEV)
    return dict(loss=round(float(F.cross_entropy(logit, zero)), 4), picks_true_pct=round(100 * float((logit.argmax(1) == 0).float().mean()), 1),
                pose_term_alone_loss=round(float(F.cross_entropy(gl, zero)), 4), chance_loss=round(float(np.log(ce.shape[1])), 4))


def main():
    sets, mv = SC.build_sets()
    Xv = np.load(FA.OUT / "feat_val.npy", mmap_mode="r").reshape(-1, 1024, 64)
    Xt = np.load(FA.OUT / "feat_train.npy", mmap_mode="r")
    Zt = SC.test_frames()
    ckg = torch.load(FA.OUT / "g.pt", map_location=DEV, weights_only=False)
    g = FA.PoseTerm().to(DEV); g.load_state_dict(ckg["state_dict"]); g.eval()
    pose = torch.from_numpy(((mv["pose"] - ckg["mu"]) / ckg["sd"]).astype(np.float32)).to(DEV)
    E = mv["drawer"].shape[0]
    for name in sys.argv[1:]:
        ck = torch.load(OUT / f"{name}.pt", map_location=DEV, weights_only=False)
        K = ck["K"]
        net = FA.Factors(K=K).to(DEV); net.load_state_dict(ck["state_dict"]); net.eval()
        S_tr = SC.encode(net, Xt[:100].reshape(-1, 1024, 64)).reshape(100, FA.T, K)
        states = {"validation": SC.encode(net, Xv), "test frames (card 141)": SC.encode(net, Zt)}
        Sv = states["validation"].reshape(E, FA.T, K)
        used, groups, extra = factor_rule(S_tr)
        c2, best = crit2(Sv, used, groups, mv)
        c2_single, _ = SC.crit2(Sv, used, groups, mv) if len(groups) else ({}, None)

        def dist(nm, ia, ib):
            d = np.abs(states[nm][ia] - states[nm][ib]) * used[None]
            return dict(whole=d.sum(-1), own={v: d[:, groups[best[i]]].sum(-1) for i, v in enumerate(NAMES)} if len(groups) else None)
        r = dict(K=K, seed=ck["seed"], steps=ck["steps"], sec=ck["sec"], bound=ck["bound"], train_picks_true_pct=ck["picks_true_pct"],
                 criterion1=SC.crit1(dist, sets), criterion2=c2,
                 single_step_arm_only_over_one_variable_pct=c2_single.get("arm_only_over_one_variable_pct"),
                 held_out=None if ck["bound"] else val_nce(Sv, pose, g),
                 std_per_coordinate=[round(float(x), 3) for x in S_tr.reshape(-1, K).std(0)],
                 at_box_edge_pct=round(100 * float((np.abs(S_tr) > 0.95).mean()), 1), **extra)
        json.dump(r, open(OUT / f"score_{name}.json", "w"), indent=1)
        c1 = r["criterion1"]
        print(f"[{name}] K {K} " + " | ".join(f"{s}: " + " ".join(f"{v} {c1[s][v]['whole']}" for v in NAMES) for s in c1)
              + f" | factors {c2['n_factors']} of {c2['n_used']} used: {c2['factors']} distinct {c2['distinct']} share {c2['share_in_best']}"
              + f" nothing/one {c2['nothing_differs_over_one_variable_pct']}% (single step {r['single_step_arm_only_over_one_variable_pct']}%)"
              + f" | held-out {r['held_out']}", flush=True)


if __name__ == "__main__":
    main()
