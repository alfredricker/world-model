"""Card 051, step 4b: the chain's choices kept between steps (the persistent half of change 5).

Card 029 rebuilds the chain from the top every step, so the achiever chosen for a condition can flip from one step
to the next. Card 050's cluttered layout 66 (seed 403, traced 2026-10-04): after the purple key fails at the red
door, step 9 chooses the blue key (drop the purple key first), step 10 the red key, whose route the dropped purple
key now blocks (pick it up), and the two alternate for the rest of the episode.

Here each step records, for every condition in the chain it acts on, the achiever it chose; the next step tries
that achiever first for the same condition, and searches as before only when it no longer gives a plan (its needs
can no longer be met, or the act failed and recall now predicts it does not work). Intentions persist until they
fail (Bratman 1987; the kept chain of card 051's change 5, without its cost margin). Step 4a's threats are kept.

  bin/prun python tools/card051/commit.py --clutter --seed 403 --n 100 --out runs/051/s4b_clutter_403.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threats as TH                                   # noqa: E402

S7, G = TH.S7, TH.G
STATS = {"kept": 0, "released": 0}


def opkey(op):
    return (getattr(op, "a", None), int(op.u) if getattr(op, "u", None) is not None else None, getattr(op, "j", None))


def solve(self, c, st, depth, chain, protect, faces):
    alts = []
    for op, needs in self.achievers(c, st):
        if any(n in chain or n == c for n in needs if n[0] != "face"):
            continue
        alts.append((sum(not self.hold(n, st) for n in needs), op, needs))
    commit = getattr(self, "commit", {}).get(c)
    if commit is not None:
        for _, op, needs in alts:
            if opkey(op) == commit:
                res = self.pursue(c, op, needs, st, depth, chain, protect, faces)
                if res is not None:
                    STATS["kept"] += 1
                    res.trace = [c] + res.trace
                    res.ops = [(c, opkey(op))] + list(getattr(res, "ops", []))
                    return res
                STATS["released"] += 1
                break
    alts.sort(key=lambda x: x[0])
    i = 0
    while i < len(alts):
        grp = [x for x in alts if x[0] == alts[i][0]]
        found = []
        for _, op, needs in grp:
            res = self.pursue(c, op, needs, st, depth, chain, protect, faces)
            if res is not None:
                res.ops = [(c, opkey(op))] + list(getattr(res, "ops", []))
                found.append(res)
        if found:
            best = min(found, key=lambda r: r.near)
            best.trace = [c] + best.trace
            return best
        i += len(grp)
    return None


_choose = S7.Plan047.choose


def choose(self, st):
    res = _choose(self, st)
    self.commit = dict(getattr(res, "ops", [])) if res is not None else {}
    return res


def install():
    TH.install()
    S7.Plan047.solve = solve
    S7.Plan047.choose = choose


def main():
    install()
    sys.argv[1:1] = ["--recall", "own"]
    TH.IX.CD.main()


if __name__ == "__main__":
    main()
