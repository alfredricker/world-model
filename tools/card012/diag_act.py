"""Card 012 acting diagnostic on a saved run: learned chooser, and each of
(trigger, walk) learned or exact. Also scores learned walking moves against
the exact distance to the chosen way's ready states."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from worldmodel import discover_logic as dl
from worldmodel.envs import logicdoor as ld


def main():
    run = Path(sys.argv[1]); n_lay = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    res = json.loads((run / "result.json").read_text())
    world = res["world"]
    torch, _, _ = dl._torch()
    dev = torch.device("cuda")
    ck = torch.load(run / "nets.pt", weights_only=False)
    model = dl.make_model(); model.load_state_dict(ck["model"]); model.eval()
    z = np.zeros((1, 9, 8), np.uint8)
    trainer = dl.Trainer(model, {"c0": z, "c1": z, "act": np.zeros(1, np.int8), "term1": np.zeros(1, bool)}, dev)
    tl = dl.Tree.from_report(res["learned"]["tree"])
    te = dl.Tree.from_report(res["gate"]["tree"])
    lt = dl.LearnedTree(tl, trainer, model.val, {int(k): v for k, v in ck["way_index"].items()})
    pers = res["learned"]["persistence"]
    persist_l = [float(pers.get("/".join(tl.path(n)), np.nan)) if n else np.nan for n in range(len(tl.parent))]
    lpath_e = {te.path(c): c for c in range(1, len(te.parent)) if te.accepted(c)}
    rng_l = np.random.default_rng(777)
    layouts = [ld.make_layout(8, world, rng_l) for _ in range(n_lay)]
    out = {}
    for trig in ("learned",):
        for walk in ("learned",):
            rng = np.random.default_rng(0)
            states = [ld.start_state(l) for l in layouts]
            exs = [dl.Exact(l, te.parent, te.action) for l in layouts]
            done = np.zeros(n_lay, bool); moves = Counter(); chosen_fail = Counter(); tr_log = {}
            for step in range(200):
                live = np.flatnonzero(~done)
                if not len(live):
                    break
                o = dl.learned_on_states(lt, [(layouts[i], states[i]) for i in live], trainer)
                L, P, W, R, Q = [o[k].float().cpu().numpy() for k in ("L", "P", "W", "R", "Q")]
                for j, i in enumerate(live):
                    s, ex = states[i], exs[i]
                    wi = lt.way_index
                    c = dl.choose(tl, lambda n: bool(L[j, n]), lambda n: R[j, wi[n]], lambda n: W[j, wi[n]], persist_l)
                    a = None
                    if c is not None:
                        g, aw, k = tl.parent[c], tl.action[c], wi[c]
                        ce = lpath_e.get(tl.path(c))
                        if trig == "learned":
                            fire = P[j, g, aw] > 0.5 and not L[j, g]
                        else:
                            fire = ce is not None and ex.ready(te.parent[ce], aw, s)
                        if fire:
                            a = aw
                        elif walk == "exact" and ce is not None:
                            a = ex.move_towards(ce, s)
                        elif rng.random() < 0.05:
                            a = int(rng.choice(dl.MOVES))
                        else:
                            a = int(Q[j, k].argmax())
                            if ce is not None:
                                d0 = ex.distances(ce, s).get(s[:3])
                                s1 = ld.step(layouts[i], s, a)[0]
                                d1 = ex.distances(ce, s1).get(s1[:3]) if s1[3:] == s[3:] else None
                                moves["/".join(tl.path(c)) + ": " + ("unreachable" if d0 is None else "closer" if d1 is not None and d1 < d0
                                      else "same" if d1 == d0 else "farther")] += 1
                            else:
                                moves["no exact way"] += 1
                    if a is None:
                        a = int(rng.integers(len(dl.ACTIONS)))
                    if "log" in sys.argv and i < 12:
                        tr_log.setdefault(i, []).append(f"{step:3d} pos=({s[0]},{s[1]}) dir={s[2]} carry={s[3]} door={s[8]} way={'/'.join(tl.path(c)) if c is not None else None} act={dl.ACTIONS[a]} L={''.join('1' if L[j, n] else '0' for n in range(len(tl.parent)))}")
                    states[i], end = ld.step(layouts[i], s, a)
                    if end:
                        done[i] = True
            for i in np.flatnonzero(~done):     # what the agent pursued at the end of failures
                o = dl.learned_on_states(lt, [(layouts[i], states[i])], trainer)
                L1_ = o["L"].cpu().numpy()[0]; R1 = o["R"].float().cpu().numpy()[0]; W1 = o["W"].float().cpu().numpy()[0]
                wi = lt.way_index
                c = dl.choose(tl, lambda n: bool(L1_[n]), lambda n: R1[wi[n]], lambda n: W1[wi[n]], persist_l)
                ex = exs[i]
                truth = [("/".join(te.path(n))) for n in range(1, len(te.parent)) if te.accepted(n) and ex.holds(n, states[i]) and not ex.holds(te.parent[n], states[i])][:2]
                chosen_fail[("/".join(tl.path(c)) if c is not None else "none") + " | exact frontier: " + ",".join(truth)] += 1
            for i, lines in tr_log.items():
                if not done[i]:
                    lay = layouts[i]
                    print(f"--- layout {i} FAILED  goal={lay.goal} wall_x={lay.wall_x} door_y={lay.door_y} key={lay.key}")
                    print("\n".join(lines[:30] + ["..."] + lines[-12:]))
            out[f"trigger_{trig}_walk_{walk}"] = {"success": float(done.mean()), "learned_moves": dict(moves),
                                                  "pursued_at_failure": dict(chosen_fail.most_common(6))}
            print(f"trigger {trig} walk {walk}: {json.dumps(out[f'trigger_{trig}_walk_{walk}'])}", flush=True)


if __name__ == "__main__":
    main()
