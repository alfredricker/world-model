"""Card 161: the evaluation.  Card 160's code with the kind "shared" and this card's agent.

usage: ev.py fit <experience> shared                         fits and saves arm B's network (card 160's evaluate.py fit)
       ev.py run <arm> <experience> <split> <tasks...>       arm: base (card 160's agent, perceptron), A (this agent, perceptron), B (this agent, shared)
       ev.py hands <kind> <experience>                       criterion 3: the model's answer to "will the pickup happen", against MiniGrid's rules
       ev.py statements <experience> <kind>                  scores a saved model's statements under the present GUARD
Every agent runs with card 160's planner repair (FIX=1)."""
import os, sys, json, time
os.environ["FIX"] = "1"
import numpy as np
import model, evaluate, agent, agent161, grid, rules, tasks
from grid import gid

evaluate.SEEDS.update(fresh=range(5000, 5100), old=range(1000, 1100), smoke=range(2000, 2003))
ARMS = dict(base=("mlp", agent.Agent), A=("mlp", agent161.Agent), B=("shared", agent161.Agent))
os.makedirs("runs/161", exist_ok=True)


def run(arm, exp, split, names):
    kind, cls = ARMS[arm]; T = evaluate.load(exp); M = evaluate.model(exp, kind, T); M._st = {}
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes); ex, tried = evaluate.examples(t, T.codes); A = cls(M, agent.goal_events(ex)); gc = {g[1] for g in A.goals}; out = []; lost = 0; t0 = time.time()
        for s in evaluate.SEEDS[split]:
            g, p = w.reset(s); A.reset(g, p); done = False; n = 0; gone = False
            while not done:
                a = A.act(); g, p, succ, done = w.step(a); ev = A.observe(a, g, p); n += 1
                gone |= any(c in gc and a == rules.TOGGLE for _, _, c, c2 in ev)
            out.append((succ, n, A.noplan)); lost += gone and not succ
        ok = [r for r in out if r[0]]; steps = sum(r[1] for r in out)
        res = dict(success=100.0 * len(ok) / len(out), steps_when_solved=float(np.mean([r[1] for r in ok])) if ok else None, no_plan_share=100.0 * sum(r[2] for r in out) / steps,
                   ms_per_step=1000 * (time.time() - t0) / steps, failed=[s for s, r in zip(evaluate.SEEDS[split], out) if not r[0]], failed_with_goal_token_destroyed=int(lost), examples=len(ex))
        print(arm, exp, split, t, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items() if k != "failed"}, flush=True)
        json.dump(res, open("runs/161/run_%s_%s_%s_%s.json" % (arm, exp, split, t), "w"), indent=1)


def hands(kind, exp):
    T = evaluate.load(exp); M = evaluate.model(exp, kind, T); Rl = rules.Rules(T.codes); e = T.codes.of((rules.EMPTY, 0, 0)); front = gid(1, 0, 0); P = rules.PICK
    objs = [c for c in M.cell if e in M.cell[c] and Rl.cell(c, P, front, frozenset()) == e]; held = [it for it in M.items if it[1] == P]; rows = {}
    for label, cases in (("hands empty", [frozenset()]), ("one token held", [frozenset([it]) for it in held])):
        right = n = never = never_right = 0
        for c in objs:
            sa, sg, si, Y, _, _ = M.sit[c]
            for items in cases:
                truth = Rl.cell(c, P, front, items) == e; says = M.p_cell(c, P, front, items).get(e, 0.0) >= 0.5; tried = Y[(sa == P) & (sg == front) & np.array([M.sets[i] == items for i in si])].sum()
                n += 1; right += says == truth; never += tried == 0; never_right += (says == truth) and tried == 0
        rows[label] = dict(cases=n, right=int(right), never_tried=int(never), never_tried_right=int(never_right))
    tot = sum(r["cases"] for r in rows.values()); rt = sum(r["right"] for r in rows.values())
    out = dict(kind=kind, experience=exp, objects=len(objs), tokens_that_can_be_held=len(held), percent_right=100.0 * rt / tot, by_case=rows)
    json.dump(out, open("runs/161/hands_%s_%s.json" % (kind, exp), "w"), indent=1); print(out, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif cmd == "hands":
        hands(sys.argv[2], sys.argv[3])
    elif cmd == "statements":
        exp, kind = sys.argv[2:4]; T = evaluate.load(exp); M = evaluate.model(exp, kind, T); M._st = {}; rows = evaluate.statements(M, T); v = [r["verdict"] for r in rows]
        print(exp, kind, "guard", model.GUARD, "statements right %d of %d (none %d, place %d, hands %d)" % (v.count("right"), len(rows), v.count("none"), v.count("place"), v.count("hands")))
    else:
        sys.argv = ["evaluate.py"] + sys.argv[1:]; exec(open(evaluate.__file__).read().split('if __name__ == "__main__":')[1].replace("\n    ", "\n"), dict(vars(evaluate), sys=sys))
