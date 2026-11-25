"""what does a model say about picking up, by what is out of view?  usage: dev1.py <experience> <kind>"""
import sys, numpy as np, model, evaluate, rules
from grid import gid
exp, kind = sys.argv[1:3]; T = evaluate.load(exp); M = evaluate.model(exp, kind, T); M._st = {}; codes = T.codes; nm = lambda c: rules.name(codes, int(c)); e = codes.of((rules.EMPTY, 0, 0))
front = gid(1, 0, 0)                                                    # the token lies one cell east of an agent heading east
names = {nm(c): c for c in M.cell}; pick = rules.PICK
ctx = [("nothing out of view", frozenset())]
for it in M.items:
    if nm(it[0]) in ("yellow key", "red ball", "locked yellow door", "red box", "empty") :
        ctx.append((nm(it[0]) + " gone by " + rules.ANAME[it[1]], frozenset([it])))
for name in ["yellow key", "blue key", "red ball", "red box", "grey box"]:
    c = names[name]; sa, sg, si, Y, outs, _ = M.sit[c]; io = outs.index(e); print(name, "statement", {k: v for k, v in (M.statement(c, e, pick) or {}).items() if k != "G"})
    for label, items in ctx:
        m = (sa == pick) & (sg == front) & np.array([M.sets[i] == items for i in si]); tries = Y[m].sum(); ok = Y[m][:, io].sum()
        print("    %-34s model p(picked up) %.2f   experience: %d tries, %d picked up" % (label, M.p_cell(c, pick, front, items).get(e, 0), tries, ok))
