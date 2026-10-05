"""Card 058: walking compares a route with clearing the way.

Card 045's walking takes a route to the need's placements whenever one exists, and counts a token walkable (a key
picked up, a door opened: the condition ("walk", j)) only when none exists. Card 056's forks (traced 2026-10-05):
a key lies between the agent and the switch; picking it up is the fastest first action, and the agent walks round.

Here, when a route exists, the chains that clear one token on the way are priced as well: the route's cost with
that token counted walkable, plus one step for the act that clears it. A clearing chain cheaper than the route is
pursued as ("walk", j); if its conditions cannot be met, the route stands. Comparing candidate chains by their cost
is GOAL P21's System 2. Everything else is version 10's agent.

The runner installs this into walk2 (card 051) and runs another tool with its arguments:
  bin/prun python tools/card058/shorter.py tools/card056/goals.py runs/054/b_m0.5_399.pt --out runs/058/goals_399.json
  bin/prun python tools/card058/shorter.py tools/card053/planner_check.py runs/054/b_m0.5_399.pt --gates 0 --codes noise ...
"""
import runpy
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "card051"))
import walk2                                           # noqa: E402

CM = walk2.CM
MV = CM.S7.MV
G, F = MV.G, MV.F
INF = MV.INF
STATS = {"priced": 0, "cleared": 0}


def install():
    route_walk = MV.Walk.walk

    def walk(self, need, st, depth, chain, protect, faces):
        pid = self.face_pid(st[0], need)
        r, _ = self.reach(st, pid)
        if r is not None and r != G.HERE and depth + 1 <= G.MAXD:
            fid = st[0]
            w, op = self.walkable(fid), self.openable(fid)
            if op.any() and w[F.M.cidx[st[1]]]:
                n0 = self.cost(st, pid)
                pid2 = self.face_open(fid, need)
                cands = []
                for j in np.flatnonzero(op).tolist():
                    w2 = w.copy()
                    w2[j] = True
                    ch = self.plan(fid, pid2, w2, ("opened", j))
                    n = self.cost_of(ch, st[1])
                    if n + 1 < n0 and j in self.stepped(ch, st[1]):
                        cands.append((n, j))
                STATS["priced"] += 1
                for _, j in sorted(cands):
                    c = ("walk", j)
                    if c in chain:
                        continue
                    res = self.solve(c, st, depth + 1, chain + [need], protect, faces + [need])
                    if res is not None:
                        STATS["cleared"] += 1
                        res.trace = [need, ("walk", "cleared")] + res.trace
                        return res
        return route_walk(self, need, st, depth, chain, protect, faces)

    MV.Walk.walk = walk


def main():
    w2_install = walk2.install

    def install_both():
        w2_install()
        install()

    walk2.install = install_both
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
