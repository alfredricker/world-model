"""Card 167: Partial-observability ZeroDijkstraAgent with unified Goal BFS + Epistemic Frontier BFS.

Uses Card 166's frozen unified world model (WNet) and Card 164's frozen System 1 (q_reach):
  - Pass 1 (Goal BFS): exact Card 166 batched event-level BFS over known space (alpha[UNK] = 0).
  - Pass 2 (Epistemic Frontier BFS, when Pass 1 finds no known chain to any visible goal):
      runs the EXACT same batched event-level BFS with epistemic target "make any UNK cell satisfy r_act[UNK] >= 0.5":
      * Depth 0: nearest reachable UNK frontier cell -> walk/face toward it to reveal unseen space.
      * Depth >= 1: when all open-space UNK frontiers are exhausted (remaining UNK lies behind closed/locked doors),
        BFS automatically plans "open closed door -> reveal UNK" or "pick up key -> unlock door -> reveal UNK"
        with ZERO room or door heuristics!
"""
import os, importlib.util, numpy as np
from grid import NA, UNK, OUT
from partial import PartialBelief

_sp166 = importlib.util.spec_from_file_location("plan166_core", os.path.abspath(os.path.join(os.path.dirname(__file__), "../card166/plan.py")))
plan166 = importlib.util.module_from_spec(_sp166); _sp166.loader.exec_module(plan166)

HOLD = int(os.environ.get("HOLD", 40))
MAX_EXP = int(os.environ.get("MAX_EXP", 250))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", 7))
R_TH = float(os.environ.get("R_TH", 0.5))


class PartialAgent(plan166.ZeroDijkstraAgent):
    """Zero-Dijkstra System 2 + System 1 under 7x7 occluded partial observability."""
    def __init__(s, qmodel, wmodel, goals, epistemic=True, hold=HOLD, max_exp=MAX_EXP, max_depth=MAX_DEPTH):
        super().__init__(qmodel, wmodel, goals, hold=hold, max_exp=max_exp, max_depth=max_depth)
        s.epistemic = epistemic
        # Ensure UNK is strictly non-traversable in pessimistic propagation (alpha[UNK] = 0.0)
        assert float(wmodel.al[UNK]) < R_TH, f"Expected alpha[UNK] < {R_TH}, got {wmodel.al[UNK]}"
        # Learn an inert non-traversable code directly from WNet (alpha < 0.5 and no observed cell effects, e.g. wall)
        # so System 1 (q_reach) sees UNK as impassable — matching System 2's pessimistic alpha[UNK]=0!
        inert = [c for c in range(2, wmodel.NC) if wmodel.al[c] < R_TH and c not in wmodel.eff_by_code]
        s._unk_block = inert[0] if inert else OUT

    def reset_partial(s, obs_map, vis_mask, pose):
        s.bel = PartialBelief(obs_map, vis_mask, pose)
        s.taken = {}; s.noplan = 0; s.held = None; s.age = 0
        s.first = None; s.plans = 0; s.expansions = 0
        s.epistemic_plans = 0; s.is_frontier = False
        s.fail = {}; s.WM.clear_cache()

    def observe_partial(s, a, obs_map, vis_mask, pose):
        items = s.bel.items()
        ev, revealed = s.bel.step(a, obs_map, vis_mask, pose)
        if ev or s.bel.items() != items:
            s.held = None; s.age = 0; s.is_frontier = False
        elif s.held is not None and s.held[1] != s.Q.ARR and a == s.held[2]:
            X, o, ea = s.held; c = int(s.bel.map[X])
            px, py, pd = s.bel.pose
            dx, dy = X[0] - px, X[1] - py
            if c != UNK and (s.WM.p_at(c, o, ea, dx, dy, pd, items) >= 0.5 or s.WM.p_at(c, o, ea, dx, dy, pd, frozenset()) >= 0.5):
                k = (X, o, ea, items); s.fail[k] = s.fail.get(k, 0) + 1; s.held = None; s.age = 0; s.is_frontier = False
        # Replan immediately when new information arrives:
        #   - if pursuing an UNK frontier target, replan as soon as that target is revealed OR any non-background token appears
        #   - if pursuing a goal/event chain, replan if any non-background token (e.g. newly seen wall, door, key, goal) appears
        if revealed and s.held is not None:
            if s.is_frontier:
                if s.bel.map[s.held[0]] != UNK or any(c != s.Q.bg for _, _, c in revealed):
                    s.held = None; s.age = 0; s.is_frontier = False
            elif any(c != s.Q.bg for _, _, c in revealed):
                s.held = None; s.age = 0
        return ev, revealed

    def _ask_q(s, ev):
        """Query frozen System 1 q_reach with boundary pessimistic UNK masking:
        only UNK cells orthogonally adjacent to a seen cell appear as the learned inert blocker
        (matching WNet's alpha[UNK]=0 without bloating q_reach's token list with deep interior UNK on 16x16 maps),
        except the target cell X itself when X is an UNK frontier arrival target (treated as bg)."""
        X, o, ea = ev
        raw = s.bel.map
        mp = raw.copy()
        seen = (raw != UNK)
        seen[s.bel.pose[0], s.bel.pose[1]] = True
        # Boundary UNK: UNK cells with at least one 4-neighbor already seen
        adj = np.zeros_like(seen)
        adj[1:, :] |= seen[:-1, :]; adj[:-1, :] |= seen[1:, :]
        adj[:, 1:] |= seen[:, :-1]; adj[:, :-1] |= seen[:, 1:]
        mp[(raw == UNK) & adj] = s._unk_block
        mp[s.bel.pose[0], s.bel.pose[1]] = s.Q.bg
        if o == s.Q.ARR and raw[X] == UNK:
            mp[X] = s.Q.bg
        return s.Q.ask([(mp, s.bel.pose, s.bel.items(), X, o, ea)])[0].copy()

    def act(s):
        key = (s.bel.map.tobytes(), s.bel.pose, s.bel.items())
        taken = s.taken.setdefault(key, set())
        if len(taken) >= NA:
            taken.clear()
        if s.held is None or s.age >= s.hold:
            s.held = s.choose_event(); s.age = 0; s.plans += 1
            if s.first is None:
                s.first = s.held
        ev = s.held
        if ev is None:
            G = s.goal_asks(s.bel.map)
            if G:
                qs = s.Q.ask([(s.bel.map, s.bel.pose, s.bel.items(), *e) for e in G])
                qa = qs[int(qs.max(1).argmax())].copy()
                qa[list(taken)] = -1; a = int(qa.argmax()); s.noplan += 1
            else:
                a = min(set(range(NA)) - taken); s.noplan += 1
        else:
            qa = s._ask_q(ev)
            a = int(qa.argmax())
            if a in taken:
                s.held = s.choose_event(); s.age = 0; s.plans += 1
                ev2 = s.held or ev
                qa = s._ask_q(ev2)
                a = int(qa.argmax())
                if a in taken:
                    s.noplan += 1; qa[list(taken)] = -1; a = int(qa.argmax())
        s.age += 1; taken.add(a)
        return a

    def find_unk_frontier(s, mp, pxy, r_act):
        """Find nearest reachable UNK frontier cell (mp == UNK and r_act >= R_TH, excluding agent's own cell)."""
        ux, uy = np.nonzero((mp == UNK) & (r_act >= R_TH))
        if not len(ux):
            return None
        best = None; best_d = 1e9
        for x, y in zip(ux, uy):
            X = (int(x), int(y))
            if X == pxy:
                continue
            d = abs(X[0] - pxy[0]) + abs(X[1] - pxy[1])
            if d < best_d:
                best_d = d; best = X
        if best is None:
            return None
        return (best, s.Q.ARR, s.Q.ARR_ACT)

    def choose_epistemic_event(s):
        """Pass 2 (P15): exact same batched event-level BFS over WNet with epistemic goal 'reveal any UNK cell'."""
        mp0, pxy0, items0 = s.bel.map, s.bel.pose[:2], s.bel.items()
        layer = [(mp0, pxy0, items0, None, {})]
        seen = {(mp0.tobytes(), items0)}
        exp = 0
        for depth in range(s.max_depth + 1):
            if not layer or exp >= s.max_exp:
                break
            reach_res = s.WM.query_reach([(st[0], st[1]) for st in layer])
            next_layer = []
            for (mp, pxy, items, first, chg), (r_arr, r_act) in zip(layer, reach_res):
                exp += 1
                f_ev = s.find_unk_frontier(mp, pxy, r_act)
                if f_ev is not None:
                    s.expansions += exp
                    s.epistemic_plans += 1
                    s.is_frontier = (first is None)
                    return f_ev if first is None else first
                if depth == s.max_depth or exp >= s.max_exp:
                    continue
                for e in s.expand_events(mp, pxy, items, r_arr, r_act, depth == 0, chg):
                    mp2, pxy2, it2 = s.imagine(mp, pxy, items, r_arr, *e)
                    sk = (mp2.tobytes(), it2)
                    if sk in seen:
                        continue
                    seen.add(sk)
                    chg2 = dict(chg); chg2[e[0]] = chg.get(e[0], ()) + (int(mp[e[0]]),)
                    next_layer.append((mp2, pxy2, it2, e if first is None else first, chg2))
            layer = next_layer
        s.expansions += exp
        return None

    def choose_event(s):
        s.is_frontier = False
        # Pass 1: Goal-directed BFS over known space (exact Card 166 planner!)
        ev = super().choose_event()
        if ev is not None:
            return ev
        # Pass 2: Epistemic Frontier BFS over UNK (only when epistemic=True and no known chain reaches a goal)
        if s.epistemic:
            return s.choose_epistemic_event()
        return None
