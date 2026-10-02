"""Card 046: card 045's planner, with recall judging what can be made walkable at its two levels.

Per pick up and toggle, as recall's templates do (card 042): when memory holds tries on the same thing as u
(equal codes, the same-thing level), u can be made walkable if one of those tries made it walkable; only when
memory holds none is u judged by things like it (card 038's rule, the similar-thing level). Card 045's filter
on the door candidates is removed: every token judged openable is tried. Everything else is card 045's.

  bin/prun python tools/card046/openable.py --check --seeds 399-404 --worlds key --out runs/046_openable_check.json
  bin/prun python tools/card046/openable.py --dev --arm A --seeds 399-399 --layouts 30 --out runs/046_dev.json
  bin/prun python tools/card046/openable.py --arm A --seeds 399-399 --layouts 30 --layouts-b 30 --out runs/046_shakedown_399.json
  bin/prun python tools/card046/openable.py --arm A --seeds 400-404 --layouts-b 100 --out runs/046_armA.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card045"))
import movement as MV                                  # noqa: E402

VP, CR, F = MV.VP, MV.CR, MV.F
SP = CR.SP
EVERY = frozenset(range(MV.TK.NT))


def similar(self, a, u):
    """Card 038's rule for one action: memory holds a try that made a thing like u walkable."""
    kd = self.kinds[a]
    fr = self.openedc.get(a)
    if fr is None:
        out = set()
        for c in (1, 3):
            for t in np.flatnonzero(kd.aw[:, c] > 0):
                b = kd.keys[t][0]
                if not self.M.free[b] and self.M.free[self.S.add(kd.after_mat(c, 0)[t])]:
                    out.add(b)
        fr = self.openedc[a] = np.array(sorted(out), np.int64)
    return bool(len(fr)) and np.exp(-(np.abs(self.S.arr[fr] - self.S.arr[u]) @ kd.lam[:kd.D])).max() >= VP.KMIN


def same_level(self, a, u):
    """The tries on the same thing as u (equal codes), and whether one of them made it walkable; (0, False)
    when there are none."""
    kd = self.kinds[a]
    n = len(kd.keys)
    same = np.flatnonzero(np.asarray(kd.fid[:n]) == CR.code_id(self.S, u))
    opened = any(self.M.free[self.S.add(kd.after_mat(c, 0)[t])]
                 for c in (1, 3) for t in same[kd.aw[same, c] > 0])
    return len(same), opened


def openable(self, u):
    """Memory holds a pick up or toggle that made u walkable: read on the same thing when memory has tried it,
    on things like it otherwise."""
    r = self.openc.get(u)
    if r is None:
        r = False
        for a in (VP.PICK, VP.TOG):
            n, opened = same_level(self, a, u)
            if opened or (n == 0 and similar(self, a, u)):
                r = True
                break
        self.openc[u] = r
    return r


def install():
    SP.SlotWorld.openable = openable
    MV.Walk.may_open = lambda self, fid, pid, c: EVERY          # card 045's filter removed


def check(seeds, out, worlds):
    """Per seed and world: which things in memory are judged openable, by card 038's rule and by card 046's,
    with the same-thing tries behind each."""
    CS = MV.CS
    CS.install()
    MV.install("045")
    SP.install()
    VP.Kind, VP.vectors_of = CR.kind, CR.vectors_of
    VP.T.configure()
    cache = pickle.loads(VP.T.MEMORY.read_bytes())
    groups = VP.T.use_groups(cache["mem"], "cuda")
    data = pickle.loads(VP.T.DATA.read_bytes())["data"]
    old = VP.World.openable
    res = {"note": "Card 046's gate: things judged openable, card 038's rule against card 046's", "seeds": {}}
    for seed in seeds:
        z, parts, _ = VP.vectors_of("A", seed, cache["tiles"], cache["pairs"], groups, "cuda", lambda *a: None)
        S = VP.Store(z)
        lut = np.zeros(256, np.uint8)
        lut[:len(S.hof)] = S.hof
        VP.Kd.APP = lut
        res["seeds"][seed] = {}
        for world in worlds:
            W = VP.World(S, parts, data[world], "cuda", lambda *a: None)
            VP.WORLD, F.M = W, W.M
            nm = lambda h: W.judge.name(int(h))
            things = sorted({int(k[0]) for a in (VP.PICK, VP.TOG) for k in W.kinds[a].keys})
            rows = {}
            for u in things:
                if W.M.free[u]:
                    continue
                W.openc.clear()
                o = bool(old(W, u))
                W.openc.clear()
                nw = bool(openable(W, u))
                tries = {VP.KNAME[a]: same_level(W, a, u)[0] for a in (VP.PICK, VP.TOG)}
                rows[nm(u)] = {"card038": o, "card046": nw, "same_thing_tries": tries}
            W.openc.clear()
            res["seeds"][seed][world] = rows
            print(seed, world, {k: (v["card038"], v["card046"]) for k, v in rows.items() if v["card038"] or v["card046"]},
                  flush=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    if "--check" in args:
        a, b = get("--seeds", "399-404").split("-")
        check(range(int(a), int(b) + 1), get("--out", "runs/046_openable_check.json"),
              get("--worlds", ",".join(VP.WORLDS)).split(","))
        return
    install()
    MV.main()
    out = Path(get("--out", ""))
    if out.is_file() and "--dev" not in args:
        r = json.loads(out.read_text())
        r["note"] = "Card 046, tools/card046/openable.py (card 045's walking, recall's openable at two levels)"
        out.write_text(json.dumps(r, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
