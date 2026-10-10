"""Card 094 in the agent: belief as object files and a layout, and which readers read which.

  object files  per attended token id (attend.py): its token (what), its id (where at first sight), the step it was
                last seen, and where it is expected now (in this card: where it was last seen)
  layout        per token id: walkable, blocked or never seen (walking's own walkability, M.free)

Readers:
  the planner's candidate tokens (Plan047.things: card 038's achievers, card 056's goal achievers, walking's
  blockers) and card 072's guess candidates read the object files, plus the layout's walkable token (a place to
  stand or to drop onto); recall's view, as the router reads it, holds only attended tokens, for stored keys and
  queries alike (the declared exception: stored views re-read through the same filter; recall's admitted view
  conditions read only keys and balls, every one attended). Walking and exploration read the layout, as before.

The planner's imagined states still carry a token per place; the layout and the files are read from it (in MiniGrid
the two hold the same information). Card 095 will need them apart.

  WM_FILES=1 | oracle (gate: object files chosen by hand, every door, key, ball, box and the goal square)
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card094"))
import attend as AT                                    # noqa: E402

STATS = {"files_steps": 0, "files_sum": 0, "layout_seen_sum": 0, "things_before_sum": 0, "things_after_sum": 0}
STATE = {"mode": os.environ.get("WM_FILES", "1")}
PARTS = set(os.environ.get("WM_FILES_PARTS", "things,router").split(","))   # which readers ("guess": diagnosis only)
WALK, BLOCK, UNSEEN = 0, 1, 2


def _name(W, h):
    inv = STATE.setdefault("inv", {int(hh): t for t, hh in reversed(list(enumerate(W.S.hof.tolist())))})
    t = inv.get(int(h))
    return STATE["VP"].CO.name_of_code(t) if t is not None else "?"


def static(W, h):
    """Attended whatever the goal and the kept choices: what stored keys can be filtered by."""
    VP = STATE["VP"]
    if STATE["mode"] == "oracle":
        n = _name(W, h)
        return n.split()[0] in ("door", "key", "ball", "box", "goal")
    return AT.changeable(W, VP, h) or AT.ends(W, VP, h)


def attended(pl, st, h):
    W, VP = pl.W, STATE["VP"]
    if STATE["mode"] == "oracle":
        return static(W, h)
    return AT.attended(W, VP, pl, st, h)


def install(T):
    if STATE.get("installed"):
        return
    STATE["installed"] = True
    VP, TK, PL = T.VP, T.TK, T.S7.Plan047
    STATE["VP"] = VP
    things0 = PL.things

    def things(self, f):
        th = things0(self, f)
        st = STATE.get("st")
        M = T.F.M
        keep = [u for u in th if (attended(self, st, u) if st is not None else static(self.W, u))
                or (M.free[u] and not static(self.W, u))]         # the layout's walkable token: stand, drop onto
        return keep

    if "things" in PARTS:
        PL.things = things
    choose0 = PL.choose

    def choose(self, st):
        STATE["st"] = st
        f = self.facts[st[0]]
        lat = f[TK.LAT]
        seen = lat[lat != TK.ABSENT]
        files = getattr(self, "files", {})
        STATS["files_steps"] += 1
        STATS["files_sum"] += len(files)
        STATS["layout_seen_sum"] += len(seen)
        STATS["things_before_sum"] += len(things0(self, f))
        STATS["things_after_sum"] += len(things(self, f))
        return choose0(self, st)

    PL.choose = choose
    observe0 = PL.observe

    def observe(self, world, V, *a, **k):
        r = observe0(self, world, V, *a, **k)
        f, st = r[0], r[1]
        t = getattr(self, "files_t", -1) + 1
        self.files_t = t
        M = T.F.M
        Ap = M.A[st[1]]
        vis = np.asarray(V) >= 0
        ids = Ap[(Ap >= 0) & vis[:len(Ap)]]
        files = getattr(self, "files", {})
        for i in ids.tolist():
            h = int(f[i])
            if h != TK.ABSENT and attended(self, st, h):
                files[i] = {"what": h, "seen": t, "expected_where": i}   # expected where: where last seen
            else:
                files.pop(i, None)
        self.files = files
        lay = np.full(len(f), UNSEEN, np.int8)
        known = f != TK.ABSENT
        hs = f[known]
        u, inv = np.unique(hs, return_inverse=True)
        lay[known] = np.array([WALK if M.free[int(h)] else BLOCK for h in u], np.int8)[inv]
        self.layout = lay
        return r

    PL.observe = observe
    TRY = sys.modules.get("trying")
    if TRY is not None and "guess" in PARTS:
        cand0 = TRY.candidates

        def candidates(W, pl, st):
            out = cand0(W, pl, st)
            keep = [x for x in out if attended(pl, st, x[1]["u"])]
            if os.environ.get("WM_FILES_DEBUG") == "1" and len(keep) < len(out):
                print("   files: guess dropped", [(round(p, 3), _name(W, x["u"]), _name(W, x["h"]), x["a"])
                                                   for p, x in out if not attended(pl, st, x["u"])][:6], flush=True)
            return keep

        TRY.candidates = candidates


def hook(T):
    """Before setup: the router's view tokens filtered (its stored keys are embedded during setup)."""
    R = sys.modules.get("routed")
    if R is not None and "router" in PARTS:
        kt0 = R.key_tokens

        def key_tokens(W, a, keys):
            if a == R.STATE["fwd"]:
                return kt0(W, a, keys)
            SP = sys.modules["slot_planner"]
            filt = []
            for k in keys:
                s = [int(h) for h in SP.SETS[int(k[2])] if static(W, h)]
                filt.append((k[0], k[1], SP.set_of(s)))
            return kt0(W, a, filt)

        R.key_tokens = key_tokens
    STATE["VP"] = T.VP
    T.INSTALL.append(lambda *a, **k: install(T))
