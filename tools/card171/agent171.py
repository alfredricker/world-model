"""Card 171: card 170's search over this card's scene model, reading a pull's predicted range.

A subclass of card 170's agent (effects from the model, every kind of touch offered, no guards). Two differences:
  - the weights are this card's (runs/171/scene_wnet_<seed>.pt);
  - two readings of a slide are the same if they differ by at most the slot's threshold, or if both lie in the range
    the model predicted for the pull that put the slide there (widened by the threshold). In the search the range is
    the imagined pull's; after the agent's own pull it is the range predicted before that pull, kept while the
    reading stays inside it. This replaces card 170's tol_imag.

Modes:
  --dev 1 [--arm both|rng]          mixed + locked put-away, problem seeds 102 / 101 (development) -> runs/171/dev_<arm>.json
  --gate 1 --arm both|rng|card170|og5   fresh problems, seeds 402 / 401, 100 each -> runs/171/gate_b_<arm>.json
  --task <1..5> --seed <s>          frozen benchmark run -> runs/171/plan_task<t>_s<s>.json
"""
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

import plan as P          # card 169
import agent as A170      # card 170
import model171 as M1

ROOT = P.ROOT
OUT = ROOT / "runs/171"
OWN = ROOT / "runs/169/own.json"
arg = P.arg


class RangeAgent(A170.ModelEffectsAgent):
    def __init__(self, rng, ckpt_path=None, **kw):
        base_cls = P.M168.SceneWNet
        P.M168.SceneWNet = M1.SceneWNet171          # card 169's constructor builds the net by this name
        try:
            super().__init__(rng, ckpt_path=ckpt_path or (OUT / "scene_wnet_0.pt"), **kw)
        finally:
            P.M168.SceneWNet = base_cls
        self.use_rng = bool(self.meta.get("card171", {}).get("rng", False))
        self.real_rng = {}

    def reset_episode(self):
        super().reset_episode()
        self.real_rng = {}          # s_k -> (lo, hi) predicted before the agent's own last pull on that slide

    # --- "the same reading" ---
    def _rng_of(self, st):
        if "rng" in st:
            return st["rng"]
        return {k: r for k, r in self.real_rng.items() if r[0] - self.cuts[k] <= st["val"][k] <= r[1] + self.cuts[k]}

    def slide_same(self, st, goal, s_k):
        if abs(st["val"][s_k] - goal["val"][s_k]) <= self.cuts[s_k]:
            return True
        r = self._rng_of(st).get(s_k)
        return bool(r is not None and r[0] - self.cuts[s_k] <= goal["val"][s_k] <= r[1] + self.cuts[s_k])

    def goal_met(self, st, goal):
        for c_k in self.lock_sis:
            if (st["val"][c_k] > 0.0) != (goal["val"][c_k] > 0.0):
                return False
        for s_k in self.slide_sis:
            if not self.slide_same(st, goal, s_k):
                return False
        if goal["p_vis"] < 0.5:
            return bool(st["p_vis"] < 0.5 and st["in_cont"])
        if st["p_vis"] < 0.5:
            return False
        return bool(np.linalg.norm(st["aim"] - goal["aim"]) * 0.1 <= self.cuts[self.mov_si])

    def _progress_score(self, st, goal):
        sc = 0.0
        if goal["p_vis"] < 0.5:
            if st["p_vis"] < 0.5 and st["in_cont"]:
                sc += 3.0
            elif st["in_cont"]:
                sc += 1.5
        elif st["p_vis"] >= 0.5 and np.linalg.norm(st["aim"] - goal["aim"]) * 0.1 <= self.cuts[self.mov_si]:
            sc += 2.0
        for s_k in self.slide_sis:
            if self.slide_same(st, goal, s_k):
                sc += 1.0
        for c_k in self.lock_sis:
            if (st["val"][c_k] > 0.0) == (goal["val"][c_k] > 0.0):
                sc += 0.4
        return sc

    def state_key(self, st, goal):
        lk, sl, mv = super().state_key(st, goal)
        return (lk, sl, mv, tuple(bool(self.slide_same(st, goal, s)) for s in self.slide_sis))

    def decide(self, now_raw, goal_raw, nocond=False):
        # the agent's own last pull: did the reading land inside the range predicted before it?
        if self.pending is not None and self.pending[0]["kind"] == "slide" and self.pending[0].get("rng") is not None:
            p_op = self.pending[0]
            s_k, (lo, hi) = p_op["slot"], p_op["rng"]
            v = float(now_raw["val"][s_k])
            if lo - self.cuts[s_k] <= v <= hi + self.cuts[s_k]:
                self.real_rng[s_k] = (lo, hi)
                p_op["tgt"] = v                    # card 169's "reached" test then holds: the pull ended as predicted
            else:
                self.real_rng.pop(s_k, None)       # fell short of the predicted range: plan again from the reading
        act, log = super().decide(now_raw, goal_raw, nocond=nocond)
        log["slide_same"] = {int(s): bool(self.slide_same(now_raw, goal_raw, s)) for s in self.slide_sis}
        return act, log

    @torch.no_grad()
    def bfs_plan(self, root_st, goal_st, max_depth=8, max_exp=250, nocond=False, relax=False):
        if nocond:
            return P.SceneBFSAgent.bfs_plan(self, root_st, goal_st, max_depth=max_depth, max_exp=max_exp, nocond=True)
        if relax:
            return None, 0, 0.0
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000.0, 3)
        root_st = dict(root_st)
        root_st["rng"] = dict(self._rng_of(root_st))
        if self.goal_met(root_st, goal_st):
            return None, 0, ms()
        mov = self.mov_si
        visited = {self.state_key(root_st, goal_st)}
        frontier = [(root_st, [])]
        expansions = 0
        best_partial, best_score = None, self._progress_score(root_st, goal_st)
        for _ in range(max_depth):
            if not frontier or expansions >= max_exp:
                break
            nodes, ops, chains = [], [], []
            for st_node, chain in frontier:
                if expansions >= max_exp:
                    break
                expansions += 1
                for op in self.candidate_ops(st_node, goal_st):
                    nodes.append(st_node); ops.append(op); chains.append(chain)
            if not ops:
                break
            tr = self.net.forward_transition(
                torch.from_numpy(np.stack([n["h"] for n in nodes])).to(self.dev),
                torch.tensor([n["vis_logit"] for n in nodes], dtype=torch.float32, device=self.dev),
                torch.from_numpy(np.stack([n["aim"] for n in nodes]).astype(np.float32)).to(self.dev),
                torch.from_numpy(np.stack([o["u_act"] for o in ops])).to(self.dev),
            )
            p_chg = tr["p_chg"].cpu().numpy()
            val_next = tr["val_next"].cpu().numpy()
            vis_next = tr["vis_next_logit"].cpu().numpy()
            aim_next = tr["aim_next"].cpu().numpy()
            q_lo = tr["q_lo"].cpu().numpy() if self.use_rng else val_next
            q_hi = tr["q_hi"].cpu().numpy() if self.use_rng else val_next

            nxt_frontier = []
            for b, (st_node, op, chain) in enumerate(zip(nodes, ops, chains)):
                if p_chg[b, op["slot"]] < 0.5:
                    continue  # the model predicts this touch does nothing to its own slot
                chg = p_chg[b] >= 0.5
                nh, nval, nrng = st_node["h"].copy(), st_node["val"].copy(), dict(st_node["rng"])
                for k in np.where(chg)[0]:
                    if k != mov:
                        nval[k] = val_next[b, k]
                        nh[k] = self.snap_slide_h(int(k), float(nval[k]))
                        nrng.pop(int(k), None)
                        if int(k) in self.slide_sis:
                            nrng[int(k)] = (float(q_lo[b, k]), float(q_hi[b, k]))
                nvis_l, naim, nin = float(st_node["vis_logit"]), st_node["aim"].copy(), bool(st_node["in_cont"])
                if chg[mov]:
                    nvis_l = float(vis_next[b])
                    if nvis_l >= 0.0:
                        naim = aim_next[b].copy()
                        nin = bool(naim[2] > self.z_cut)
                nxt = dict(h=nh, val=nval, vis_logit=nvis_l, p_vis=float(1.0 / (1.0 + np.exp(-nvis_l))), aim=naim, in_cont=nin, rng=nrng)
                op = dict(op)
                if op["kind"] == "slide":
                    op["tgt"] = float(nval[op["slot"]])
                    op["rng"] = nrng.get(int(op["slot"]))
                new_chain = chain + [op]
                if self.goal_met(nxt, goal_st):
                    return new_chain, expansions, ms()
                key = self.state_key(nxt, goal_st)
                if key not in visited:
                    visited.add(key)
                    nxt_frontier.append((nxt, new_chain))
                    sc = self._progress_score(nxt, goal_st)
                    if sc > best_score:
                        best_score, best_partial = sc, new_chain
            frontier = nxt_frontier
        return best_partial, expansions, ms()


def accepted_slides(res, var_map):
    """Scoring only (simulator truth): of the slides the agent held to be at the goal at its last decision of an
    episode, the share truly within the benchmark's 4 cm of the goal."""
    n = ok = 0
    for ep in res["log"]:
        if not ep["touches"] or "slide_same" not in ep["touches"][-1]:
            continue
        last = ep["touches"][-1]
        for name in ("drawer", "window"):
            if last["slide_same"].get(str(var_map[name]), last["slide_same"].get(var_map[name])):
                n += 1
                ok += int(abs(ep["end"][name] - ep["goal"][name]) <= 0.04)
    return dict(accepted=n, within_4cm=ok, pct=round(100.0 * ok / n, 1) if n else None)


def build(arm, seed=0):
    rng = np.random.default_rng(seed)
    if arm == "og5":
        return P.SceneBFSAgent(rng, own_path=OWN, practice=False)
    if arm == "card170":
        return A170.ModelEffectsAgent(rng, own_path=OWN, practice=False)
    path = OUT / (f"scene_wnet_{seed}.pt" if arm == "both" else f"scene_wnet_{arm}_{seed}.pt")
    return RangeAgent(rng, ckpt_path=path, own_path=OWN, practice=False)


def brief(r):
    return {k: v for k, v in r.items() if k != "log"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = gym.make("visual-scene-v0", terminate_at_goal=False, width=P.R, height=P.R)
    arm = arg("--arm", "both")
    if int(arg("--dev", 0)) or int(arg("--gate", 0)):
        dev = bool(int(arg("--dev", 0)))
        n = int(arg("--n", 30 if dev else 100))
        s_mix, s_put = (102, 101) if dev else (402, 401)
        ag = build(arm)
        res = dict(mix=P.run_episodes(ag, env, "mix", n, seed=s_mix), locked_put=P.run_episodes(ag, env, "put", n, seed=s_put))
        if isinstance(ag, RangeAgent):
            for k in ("mix", "locked_put"):
                res[k]["accepted_slides"] = accepted_slides(res[k], ag.var_map)
        json.dump(res, open(OUT / f"{'dev' if dev else 'gate_b'}_{arm}.json", "w"), indent=1)
        print(json.dumps({k: brief(v) for k, v in res.items()}, indent=1), flush=True)
    else:
        task, seed = int(arg("--task", 1)), int(arg("--seed", 0))
        ag = build(arm, seed=0)
        ag.rng = np.random.default_rng(seed)
        res = P.run_episodes(ag, env, task, int(arg("--n", 50)), seed=seed)
        json.dump(res, open(OUT / f"plan_task{task}_s{seed}.json", "w"), indent=1)
        print(json.dumps(brief(res), indent=1), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
