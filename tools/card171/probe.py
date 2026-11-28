"""Card 171, diagnostic: where does an imagined put-away chain break?

  probe.py --arm card170|rng|both [--n 60]   -> runs/171/probe_<arm>.json

For self-made locked put-away problems (problem seed 101; the agent does not act, so these are not the same
problems as the development run), the chain every solution needs is imagined step by step through the scene model,
as the search would: unlock the drawer, pull it open, put the cube in, pull it shut, and press the lock again if the
goal shows it locked. The first link at which the imagined chain leaves the route to the goal is recorded. Nothing is
executed in the scene and no simulator truth is read.
"""
import json, os
from collections import Counter
import gymnasium as gym
import numpy as np
import torch
import plan as P
import agent171 as A1

OUT = A1.OUT


@torch.no_grad()
def step(ag, st, op):
    """One imagined touch, as agent171.RangeAgent.bfs_plan / card 170's bfs_plan build it. Returns (next state, p_chg)."""
    tr = ag.net.forward_transition(
        torch.from_numpy(st["h"][None]).to(ag.dev), torch.tensor([st["vis_logit"]], dtype=torch.float32, device=ag.dev),
        torch.from_numpy(st["aim"][None].astype(np.float32)).to(ag.dev), torch.from_numpy(op["u_act"][None]).to(ag.dev))
    p_chg, val_next = tr["p_chg"][0].cpu().numpy(), tr["val_next"][0].cpu().numpy()
    use_rng = getattr(ag, "use_rng", False)
    q_lo = tr["q_lo"][0].cpu().numpy() if use_rng else val_next
    q_hi = tr["q_hi"][0].cpu().numpy() if use_rng else val_next
    chg = p_chg >= 0.5
    nh, nval, nrng = st["h"].copy(), st["val"].copy(), dict(st.get("rng", {}))
    for k in np.where(chg)[0]:
        if k != ag.mov_si:
            nval[k] = val_next[k]; nh[k] = ag.snap_slide_h(int(k), float(nval[k])); nrng.pop(int(k), None)
            if int(k) in ag.slide_sis:
                nrng[int(k)] = (float(q_lo[k]), float(q_hi[k]))
    nvis_l, naim, nin = float(st["vis_logit"]), st["aim"].copy(), bool(st["in_cont"])
    if chg[ag.mov_si]:
        nvis_l = float(tr["vis_next_logit"][0])
        if nvis_l >= 0.0:
            naim = tr["aim_next"][0].cpu().numpy().copy(); nin = bool(naim[2] > ag.z_cut)
    return dict(h=nh, val=nval, vis_logit=nvis_l, p_vis=float(1.0 / (1.0 + np.exp(-nvis_l))), aim=naim, in_cont=nin, rng=nrng), p_chg


def slide_same(ag, st, goal, k):
    if hasattr(ag, "slide_same"):
        return ag.slide_same(st, goal, k)
    return abs(st["val"][k] - goal["val"][k]) <= ag.tol_imag[k]      # card 170's rule


def probe(ag, st, goal):
    dr = ag.meta["cont_si"]
    lk = int(ag.meta["cause_of"][dr])
    d_open = "lo" if ag.put_req[dr][2] < ag.slide_ext[dr]["mid"] else "hi"
    d_shut = "hi" if d_open == "lo" else "lo"

    def pick(st, kind, slot, dir=None):
        for op in ag.candidate_ops(st, goal):
            if op["kind"] == kind and op["slot"] == slot and (dir is None or op.get("dir") == dir):
                return op
        return None

    links = [("unlock", "press", lk, None), ("open", "slide", dr, d_open), ("put in", "put_in", ag.mov_si, None), ("shut", "slide", dr, d_shut)]
    if (goal["val"][lk] > 0.0) != (st["val"][lk] <= 0.0):       # the goal shows the lock as it was at the start
        links.append(("lock again", "press", lk, None))
    for name, kind, slot, d in links:
        op = pick(st, kind, slot, d)
        if op is None:
            return f"{name}: not offered"
        st, p_chg = step(ag, st, op)
        if p_chg[op["slot"]] < 0.5:
            return f"{name}: predicted to do nothing"
        if name == "open" and not all(lo <= st["val"][s] <= hi for s, (lo, hi, _) in ag.put_req.items()):
            return "open: imagined drawer reading outside the put-in range"
        if name == "put in" and not st["in_cont"]:
            return "put in: cube not predicted in the drawer"
        if name == "shut":
            if st["p_vis"] >= 0.5:
                return "shut: cube predicted still in view"
            if not slide_same(ag, st, goal, dr):
                return "shut: imagined drawer reading not the goal's"
    if (st["val"][lk] > 0.0) != (goal["val"][lk] > 0.0):
        return "end: lock not as the goal"
    return "chain reaches the goal (drawer, its lock and the cube)"


def main():
    arm, n = P.arg("--arm", "both"), int(P.arg("--n", 60))
    env = gym.make("visual-scene-v0", terminate_at_goal=False, width=P.R, height=P.R)
    ag = A1.build(arm)
    rng = np.random.default_rng(101)
    env.reset(seed=101)
    run = P.Runner(env)
    spots = P.load_hidden_spots_for_problem_gen()
    out = []
    for e in range(n):
        ag.reset_episode()
        run.reset(task_info=P.make_locked_put_problem(run.u, rng, ag.table_lo, ag.table_hi, spots))
        goal = ag.encode_frame(run.goal)
        try:
            run.rest(ag.mem, ag.rest)
        except P.Done:
            continue
        st = ag.encode_frame(run.frame)
        st["in_cont"] = bool(st["p_vis"] < 0.5) or bool(st["aim"][2] > ag.z_cut)
        goal["in_cont"] = bool(goal["p_vis"] < 0.5) or bool(goal["aim"][2] > ag.z_cut)
        st["rng"] = {}
        if hasattr(ag, "_tol"):
            ag._tol = getattr(ag, "tol_imag", ag.cuts)
        out.append(probe(ag, st, goal))
        print(e, out[-1], flush=True)
    cnt = Counter(out)
    res = dict(arm=arm, n=len(out), first_break={k: v for k, v in cnt.most_common()})
    json.dump(res, open(OUT / f"probe_{arm}.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
