"""Card 170: the search imagines with SceneWNet's own predicted next state.

A subclass of card 169's agent. Only how the search imagines changes:
  - every remembered kind of touch is offered at every node (no written tests of the goal, no guards);
  - the next state of every slot is the model's prediction where it predicts a change, else kept exactly;
  - a pull is offered by recall (remembered pulls in that direction that started near the node's reading).
Skills, own-tries counts, failed-try handling and the scene model's weights are og5's.

Modes:
  --dev 1                       mixed + locked put-away problems, seeds 102 / 101 (development)
  --gate 1                      fresh problems, seeds 202 / 201: this agent and og5's planner -> runs/170/gate.json
  --task <1..5> --seed <s>      frozen benchmark run -> runs/170/plan_task<t>_s<s>.json
  --summarize 1                 -> experiments/170-.../results.json
"""
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

import plan as P  # card 169

ROOT = P.ROOT
OUT = ROOT / "runs/170"
OWN = ROOT / "runs/169/own.json"
arg = P.arg


class ModelEffectsAgent(P.SceneBFSAgent):
    def __init__(self, rng, **kw):
        super().__init__(rng, **kw)
        # Tolerance for an imagined slide value: the model's own error on remembered pulls (90th percentile).
        Z_m = np.load(ROOT / "runs/132/z_m400.npy").astype(np.float32)
        self.tol_imag = dict(self.cuts)
        with torch.no_grad():
            for s_k in self.slide_sis:
                idx = np.concatenate([self.slide_cands[s_k]["lo"], self.slide_cands[s_k]["hi"]])
                eo = self.net.encode(torch.from_numpy(Z_m[idx]).to(self.dev))
                tr = self.net.forward_transition(eo["h"], eo["vis_logit"], eo["aim"], torch.from_numpy(self.u_act_m[idx]).to(self.dev))
                err = np.abs(tr["val_next"][:, s_k].cpu().numpy() - self.Val_m[idx + 1, s_k])
                self.tol_imag[s_k] = float(max(self.cuts[s_k], np.quantile(err, 0.90)))
        self._tol = self.cuts
        del Z_m

    # --- comparisons: readings use the slot's threshold, imagined slide values the model's own error ---
    def goal_met(self, st, goal):
        for c_k in self.lock_sis:
            if (st["val"][c_k] > 0.0) != (goal["val"][c_k] > 0.0):
                return False
        for s_k in self.slide_sis:
            if abs(st["val"][s_k] - goal["val"][s_k]) > self._tol[s_k]:
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
            if abs(st["val"][s_k] - goal["val"][s_k]) <= self._tol[s_k]:
                sc += 1.0
        for c_k in self.lock_sis:
            if (st["val"][c_k] > 0.0) == (goal["val"][c_k] > 0.0):
                sc += 0.4
        return sc

    # --- every remembered kind of touch, offered at every node ---
    def candidate_ops(self, st, goal, nocond=False, relax=False):
        if nocond:
            return super().candidate_ops(st, goal, nocond=True)
        ops = []
        lk = tuple(int(st["val"][c] > 0.0) for c in self.lock_sis)
        for c_k in self.lock_sis:
            b = lk[self.lock_sis.index(c_k)]
            ops.append(dict(kind="press", slot=c_k, b_from=b, tgt=1 - b, u_act=self.press_u[c_k][b]))
        for s_k in self.slide_sis:
            for d_tag in ("lo", "hi"):
                if ("slide", s_k, d_tag, lk) in self.refuted_ops:
                    continue
                pool = self.slide_cands[s_k][d_tag]
                dist = np.abs(self.Val_m[pool, s_k] - st["val"][s_k])
                near = pool[np.argsort(dist)[:8]]
                if int((dist <= 2.0 * self.cuts[s_k]).sum()) < 3:
                    continue  # no remembered pull of this kind starts from a reading like this one
                u = np.median(self.u_act_m[near], axis=0).astype(np.float32)
                ops.append(dict(kind="slide", slot=s_k, dir=d_tag, tgt=None, u_act=u))
        if st["p_vis"] >= 0.5:
            put_ok = all(lo <= st["val"][s_k] <= hi for s_k, (lo, hi, _) in self.put_req.items())
            if goal["p_vis"] >= 0.5 and np.linalg.norm(st["aim"] - goal["aim"]) * 0.1 > self.cuts[self.mov_si]:
                if goal["aim"][2] <= self.z_cut or put_ok:
                    u = np.r_[st["aim"], self.med_open0, goal["aim"], 1.0].astype(np.float32)
                    ops.append(dict(kind="move_goal", slot=self.mov_si, tgt=goal["aim"].copy(), u_act=u))
            if not st["in_cont"] and put_ok:
                ctx = tuple(int(round(st["val"][s_k] / 0.25)) for s_k in self.put_req)
                if ("put_in", ctx) not in self.refuted_ops:
                    u = np.r_[st["aim"], self.med_open0, self.B_cont_med, 1.0].astype(np.float32)
                    ops.append(dict(kind="put_in", slot=self.mov_si, tgt=self.B_cont_med.copy(), u_act=u, ctx=ctx))
        return ops

    @torch.no_grad()
    def bfs_plan(self, root_st, goal_st, max_depth=8, max_exp=250, nocond=False, relax=False):
        if nocond:
            return super().bfs_plan(root_st, goal_st, max_depth=max_depth, max_exp=max_exp, nocond=True)
        if relax:  # og5's fallback with its guard relaxed: there is no guard here
            return None, 0, 0.0
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000.0, 3)
        self._tol = self.cuts
        if self.goal_met(root_st, goal_st):
            return None, 0, ms()
        mov = self.mov_si
        try:
            self._tol = self.tol_imag
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

                nxt_frontier = []
                for b, (st_node, op, chain) in enumerate(zip(nodes, ops, chains)):
                    if p_chg[b, op["slot"]] < 0.5:
                        continue  # the model predicts this touch does nothing to its own slot
                    chg = p_chg[b] >= 0.5
                    nh, nval = st_node["h"].copy(), st_node["val"].copy()
                    for k in np.where(chg)[0]:
                        if k != mov:
                            # the model's predicted value; its vector is recalled from memory for that value, so the
                            # next query sees a state of the kind the model was trained on (not its own output)
                            nval[k] = val_next[b, k]
                            nh[k] = self.snap_slide_h(int(k), float(nval[k]))
                    nvis_l, naim, nin = float(st_node["vis_logit"]), st_node["aim"].copy(), bool(st_node["in_cont"])
                    if chg[mov]:
                        nvis_l = float(vis_next[b])
                        if nvis_l >= 0.0:  # in view after: position from the model; out of view: belief keeps the last one
                            naim = aim_next[b].copy()
                            nin = bool(naim[2] > self.z_cut)
                    nxt = dict(h=nh, val=nval, vis_logit=nvis_l, p_vis=float(1.0 / (1.0 + np.exp(-nvis_l))), aim=naim, in_cont=nin)
                    op = dict(op)
                    if op["kind"] == "slide":
                        op["tgt"] = float(nval[op["slot"]])
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
        finally:
            self._tol = self.cuts


def make(cls, seed):
    return cls(np.random.default_rng(seed), own_path=OWN, practice=False)


def brief(r):
    return {k: v for k, v in r.items() if k != "log"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if int(arg("--summarize", 0)):
        return summarize()
    env = gym.make("visual-scene-v0", terminate_at_goal=False, width=P.R, height=P.R)
    if int(arg("--dev", 0)):
        ag = make(ModelEffectsAgent, 0)
        print("tol_imag", ag.tol_imag, "cuts", ag.cuts, flush=True)
        res = dict(mix30=P.run_episodes(ag, env, "mix", 30, seed=102), locked_put30=P.run_episodes(ag, env, "put", 30, seed=101))
        json.dump(res, open(OUT / "dev.json", "w"), indent=1)
        print(json.dumps({k: brief(v) for k, v in res.items()}, indent=1), flush=True)
    elif int(arg("--gate", 0)):
        ag, og5 = make(ModelEffectsAgent, 0), make(P.SceneBFSAgent, 0)
        res = dict(
            mix30=brief(P.run_episodes(ag, env, "mix", 30, seed=202)),
            locked_put30=brief(P.run_episodes(ag, env, "put", 30, seed=201)),
            og5_mix30=brief(P.run_episodes(og5, env, "mix", 30, seed=202)),
            og5_locked_put30=brief(P.run_episodes(og5, env, "put", 30, seed=201)),
        )
        json.dump(res, open(OUT / "gate.json", "w"), indent=1)
        print(json.dumps(res, indent=1), flush=True)
    elif int(arg("--compare", 0)):  # after the gate: a larger like-for-like sample on further fresh self-made problems
        n = int(arg("--compare", 0))
        ag, og5 = make(ModelEffectsAgent, 0), make(P.SceneBFSAgent, 0)
        res = {}
        for name, a in (("card170", ag), ("og5", og5)):
            res[f"{name}_mix"] = P.run_episodes(a, env, "mix", n, seed=302)
            res[f"{name}_locked_put"] = P.run_episodes(a, env, "put", n, seed=301)
        json.dump(res, open(OUT / "compare.json", "w"), indent=1)
        print(json.dumps({k: brief(v) for k, v in res.items()}, indent=1), flush=True)
    else:
        task, seed = int(arg("--task", 1)), int(arg("--seed", 0))
        ag = make(ModelEffectsAgent, seed)
        res = P.run_episodes(ag, env, task, int(arg("--n", 50)), seed=seed)
        json.dump(res, open(OUT / f"plan_task{task}_s{seed}.json", "w"), indent=1)
        print(json.dumps(brief(res), indent=1), flush=True)
    os._exit(0)


def summarize():
    og5 = json.load(open(ROOT / "experiments/169-zero-heuristic-event-bfs-on-ogbench/results.json"))
    per, ms, ex = {}, [], []
    for t in range(1, 6):
        d = [json.load(open(OUT / f"plan_task{t}_s{s}.json")) for s in (0, 1, 2)]
        vals = [x["success_pct"] for x in d]
        ms += [x["mean_bfs_ms"] for x in d]; ex += [x["mean_expansions"] for x in d]
        per[f"task{t}"] = dict(
            mean_pct=round(float(np.mean(vals)), 2), per_seed_pct=vals, og5_mean_pct=og5["per_task"][f"task{t}"]["mean_pct"],
            mean_steps_when_success=round(float(np.mean([x["mean_steps_when_success"] for x in d if x["mean_steps_when_success"]])), 1),
            mean_bfs_ms=round(float(np.mean([x["mean_bfs_ms"] for x in d])), 3),
            p95_bfs_ms=round(float(np.max([x["p95_bfs_ms"] for x in d])), 3),
            mean_expansions=round(float(np.mean([x["mean_expansions"] for x in d])), 2),
            p14=int(sum(x["p14_recoveries"] for x in d)),
        )
    res = dict(card="170", per_task=per,
               five_task_mean_pct=round(float(np.mean([per[f"task{t}"]["mean_pct"] for t in range(1, 6)])), 2),
               og5_five_task_mean_pct=og5["five_task_mean_pct"],
               mean_bfs_ms=round(float(np.mean(ms)), 3), mean_expansions=round(float(np.mean(ex)), 2))
    for name in ("effects_gate", "effects_test", "gate"):
        f = OUT / f"{name}.json"
        if f.exists():
            res[name] = json.load(open(f))
    json.dump(res, open(ROOT / "experiments/170-effects-imagined-by-the-scene-model/results.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("effects_gate", "effects_test", "gate")}, indent=1))


if __name__ == "__main__":
    main()
