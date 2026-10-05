"""Card 068, criterion 3 (and, at the end, card 047's openable test situation by situation): with a tier's memory (card 068's, play starts), per door colour: what recall predicts for
toggling the locked door with its key, with the other five keys and holding nothing; whether the locked door is
openable (card 047); whether the open door is walkable; and forward steps onto the open door among the stored rows.

  bin/prun python tools/card068/openable.py 2
"""
import sys
tier = int(sys.argv[1])
sys.argv += ["--tier", str(tier)]
sys.path.insert(0, "tools/card066"); sys.path.insert(0, "tools/card068")
import play as P
T = P.T
import numpy as np
W, info = T.setup(tier, lambda *a: None)
VP, KR, Kd = T.VP, T.KR, T.Kd
h = lambda o: int(Kd.APP[KR.code(o)])
pl = T.S7.Plan047(W)
env = T.make(tier); env.reset(seed=T.SEED_TEST + 1000 * tier)
codes = T.now_codes(env); V = T.PV.crop(Kd.APP[codes], codes)
facts, st, _, _ = pl.observe(None, V)
ctx = pl.ctx_id(st[0], st[1])
rows = []
for c in T.HUES:
    door = h(("door", c, 0))
    match = W.outcome(VP.TOG, door, h(("key", c)), ctx)
    others = [W.outcome(VP.TOG, door, h(("key", c2)), ctx)[0] for c2 in T.HUES if c2 != c]
    empty = W.outcome(VP.TOG, door, h(None), ctx)[0]
    closed = W.outcome(VP.TOG, h(("door", c, 1)), h(None), ctx)[0]
    rows.append((c, match[0], int(sum(o == 0 for o in others)), empty, closed, bool(W.openable(door))))
print("tier", tier, "toggle a locked door: (colour, category with its key [1 = opens], other keys predicted unchanged of 5, "
      "holding nothing, closed unlocked door holding nothing, openable)")
for r in rows:
    print(" ", r)
print("tier", tier, "per colour: (colour, open door walkable (recall's forward kind), opened situations held this key, "
      "forward tries onto the open door in memory)")
F = T.F
kd = W.kinds[VP.TOG]
sit = T.S7.opened_in(W, VP.TOG)
D = T.memory(tier)[0]
lut = Kd.APP
for c in T.HUES:
    od = h(("door", c, 2))
    n_sit = sum(1 for s in sit if s[0] == h(("key", c)))
    fw = (D["act"] == VP.FWD) & (lut[D["ego0"][:, T.FRONT]] == od)
    print(" ", c, bool(F.M.free[od]), n_sit, int(fw.sum()))
print("tier", tier, "per colour, card 047's test: each situation where memory saw a thing made walkable, with the locked "
      "door in front: (action, held, category, predicted front after, walkable)")
for c in T.HUES:
    door = h(("door", c, 0))
    for a in (VP.PICK, VP.TOG):
        kd = W.kinds[a]
        qs = [(door, hh, v) for hh, v in T.S7.opened_in(W, a)]
        for q, cat in zip(qs, kd.cat_of(qs) if qs else []):
            f = kd.result(q, cat)[0] if cat and cat & 1 else None
            print(" ", c, VP.KNAME[a], T.name(q[1]), int(cat), None if f is None else T.name(f),
                  None if f is None else bool(F.M.free[int(f)]))
