"""Card 161, after the test: does the learned move effect let the agent through a door it imagines open, in KeyCorridor?
usage: dev2.py <experience> <layouts>     for every closed door of the first maps and each heading through it"""
import os, sys
os.environ["FIX"] = "1"
import numpy as np
from collections import Counter
import model, evaluate, grid, rules, tasks
from grid import DIRV, OFF8
exp, n = sys.argv[1], int(sys.argv[2]); T = evaluate.load(exp); M = evaluate.model(exp, "mlp", T); M._st = {}; codes = T.codes; nm = lambda c: rules.name(codes, int(c)); FWD = 2
seen = Counter(); tried = Counter()
for (d, a, nb, it, out), cnt in T.agent.items():                        # experience: forward into an open door, by colour and heading
    if a == FWD:
        c = nb[OFF8.index(DIRV[d])]; t3 = codes.triples[c] if c < len(codes.triples) else None
        if t3 and t3[0] == rules.DOOR and t3[2] == 0:
            tried[(nm(c), d)] += cnt; seen[(nm(c), d)] += cnt * (out[0] != 0)
w = grid.World(tasks.ALL["KeyCorridorS3R3"], codes); res = Counter(); tot = Counter(); bad = Counter()
for seed in range(2000, 2000 + n):
    mp, pose = w.reset(seed)
    for c in np.unique(mp):
        t3 = codes.triples[int(c)]
        if not (t3 and t3[0] == rules.DOOR and t3[2] == 1):
            continue
        outs = M.outcomes(int(c)); o = outs[0] if outs else None
        for x, y in zip(*np.nonzero(mp == c)):
            for d, (dx, dy) in enumerate(DIRV):
                px, py = int(x) - dx, int(y) - dy
                if mp[px, py] == codes.of((rules.WALL, 0, 0)) if hasattr(rules, "WALL") else nm(mp[px, py]) == "wall":
                    continue
                tot[d] += 1
                if o is None:
                    bad["no known change of " + nm(c)] += 1; continue
                nb = list(grid.neighbours(mp, px, py)); nb[OFF8.index((dx, dy))] = o; a = M.statement(int(c), o)["a"]
                with_mem = M.agent_one(d, FWD, tuple(nb), frozenset([(int(c), a)]))[0] != 0; without = M.agent_one(d, FWD, tuple(nb), frozenset())[0] != 0
                res[(d, "passes with the memory of opening it")] += with_mem; res[(d, "passes without it")] += without
                if not with_mem:
                    bad[(nm(o), "heading %d" % d, "experience: %d of %d forward moves went through" % (seen[(nm(o), d)], tried[(nm(o), d)]))] += 1
print("experience", exp, "closed doors approached, by heading:", dict(tot)); [print("   ", k, v) for k, v in sorted(res.items())]
print("  not passable in imagination:"); [print("   %3d %s" % (v, k)) for k, v in sorted(bad.items(), key=lambda q: -q[1])[:30]]
