"""Card 161: what must not be held is asked of the effect model in the present situation.

Card 160 read "this effect fails while X is out of view" from a statement made in advance, and stated it only for a code
that had itself been tried at least three times in that case.  Here the agent asks the same model that predicts every
tick: with what is out of view now, will the effect happen?  If not, would it happen with one of those tokens back in
view?  Then that token is brought back first.  Nothing else in card 160's agent changes.
"""
import agent
from agent import FAIL, HERE, holds
from grid import FAR, gid_inv


class Agent(agent.Agent):
    def p(s, c, o, a, G, items):
        return max(s.M.p_cell(c, a, g, items).get(o, 0.0) for g in G)

    def want_effect(s, X, o, a_req, stack, chain):
        c = int(s.bel.map[X]); items = s.bel.items()
        if (X, o) in chain or len(chain) > 8 or s.fail[(X, o, items)] >= 2:
            return FAIL
        st = s.M.statement(c, o, a_req)
        if st is None:
            return FAIL
        free = any(g < 0 or g == FAR for g in st["G"])
        targets = [] if free else [(X[0] - dx, X[1] - dy, d) for dx, dy, d in map(gid_inv, st["G"])]
        stack2 = stack + (tuple(targets),) if targets else stack; chain2 = chain + ((X, o),)
        for f in st["need"]:
            if not holds(f, items):
                return s.want_in(f, stack2, chain2)
        unmet = []
        if s.p(c, o, st["a"], st["G"], items) < 0.5:                   # the model says: not as things are now
            unmet = [("it", it) for it in sorted(items) if s.p(c, o, st["a"], st["G"], items - {it}) >= 0.5][:1]
        if unmet and (free or s.free_reach(targets)):
            return s.want_out(unmet[0], stack2, chain2)
        r = HERE if free else s.go(targets, stack, chain2)
        if r == HERE:
            s.tried = (X, o, items)
            return int(st["a"])
        return r
