"""Card 161, after the test: KeyCorridor with the learned effect on tokens but MiniGrid's rule for the agent's own move.
usage: dev3.py <experience> <task> <first seed> <episodes>     (layouts outside every test; a diagnostic, not a result)"""
import os, sys
os.environ["FIX"] = "1"
import model, evaluate, agent, agent161, grid, rules, tasks
exp, t, s0, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); T = evaluate.load(exp); codes = T.codes; Rl = rules.Rules(codes)
ex, _ = evaluate.examples(t, codes); goals = agent.goal_events(ex); w = grid.World(tasks.ALL[t], codes)
for label in ("learned move", "the rule for the agent's move"):
    M = evaluate.model(exp, "mlp", T); M._st = {}
    if label != "learned move":
        M.agent_pred = Rl.agent_pred; M.agent_one = Rl.agent_one; M.move_actions = Rl.move_actions
    A = agent161.Agent(M, goals); ok = 0; nop = 0; steps = 0
    for seed in range(s0, s0 + n):
        succ, k, noplan = agent.episode(w, seed, A); ok += succ; nop += noplan; steps += k
    print(exp, t, "%-30s solved %d of %d, steps without a plan %.0f%%" % (label, ok, n, 100.0 * nop / steps), flush=True)
