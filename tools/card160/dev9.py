"""after the test: does the learned move effect let the agent through an open door, with and without the memory of having opened it?"""
import sys, numpy as np
import grid, rules, evaluate
from grid import OFF8
for exp in sys.argv[1:]:
    T = evaluate.load(exp); M = evaluate.model(exp, "sparse", T); codes = T.codes; E, Wl = codes.of((1, 0, 0)), codes.of((2, 5, 0)); res = {}
    for col in range(6):
        op, lk, cl, key = codes.of((4, col, 0)), codes.of((4, col, 2)), codes.of((4, col, 1)), codes.of((5, col, 0))
        nb = [E] * 8; nb[OFF8.index((1, 0))] = op; nb[OFF8.index((1, -1))] = Wl; nb[OFF8.index((1, 1))] = Wl; nb = tuple(nb)
        res[rules.CNAME[col]] = ["yes" if M.agent_one(0, 2, nb, frozenset(it))[0] else "no" for it in ([], [(lk, 5), (key, 3)], [(cl, 5)])]
    print("experience", exp, "forward, heading east, an open door ahead in a wall: moves? [no memory, after unlocking it with the key, after opening it closed]", res)
