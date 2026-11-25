"""Card 162, gate: "will the pickup happen", with empty hands and with one token held, against MiniGrid's rules.  usage: dev1.py <arm> <experience>"""
import os, sys
os.environ["FIX"] = "1"
import model, evaluate, net, rules
from grid import gid
arm, exp = sys.argv[1:3]; T = evaluate.load(exp); M = net.load(exp, arm, T); B = M if arm == "base" else M.base; Rl = rules.Rules(T.codes); e = T.codes.of((rules.EMPTY, 0, 0)); front = gid(1, 0, 0); P = rules.PICK
objs = [c for c in B.cell if e in B.cell[c] and Rl.cell(c, P, front, frozenset()) == e]; cases = [frozenset()] + [frozenset([it]) for it in B.items if it[1] == P]
right = sum((M.p_cell(c, P, front, it).get(e, 0.0) >= 0.5) == (Rl.cell(c, P, front, it) == e) for c in objs for it in cases)
print("pickup answers right", arm, exp, "%d of %d (%.1f%%)" % (right, len(objs) * len(cases), 100.0 * right / (len(objs) * len(cases))), flush=True)
