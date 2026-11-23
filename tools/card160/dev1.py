"""development: fit the energy on one experience and print the statements against MiniGrid's rules"""
import sys, pickle, time, numpy as np
import grid, rules, energy
T = pickle.load(open("runs/160/exp_%s.pkl" % sys.argv[1], "rb")); kind = sys.argv[2]; t0 = time.time()
M = energy.Energy(T, kind); print(kind, "fitted in %.0f s, nats %.0f" % (time.time() - t0, M.nll), "inputs of a token", M.F, "of the agent", M.AF, "move actions", M.move_actions)
R = rules.Rules(T.codes); cnt = rules.effect_counts(T); ctx = rules.contexts(R); c_, a_, g_, it_, o_, n_ = T.rows(); ok = 0; tot = 0
for c, o, a in rules.truth_effects(T.codes, sorted(set(int(x) for x in np.unique(c_)))):
    if cnt.get((c, o, a), 0) < 3:
        continue
    st = M.statement(c, o, a); gs = {int(g) for g in np.unique(g_[c_ == c]) if n_[(c_ == c) & (g_ == g)].sum() >= 3}
    bad = [(x, rules.says(st, *x), x[0] == a and R.cell(c, *x) == o) for x in ctx if x[1] in gs and rules.says(st, *x) != (x[0] == a and R.cell(c, *x) == o)]
    tot += 1; ok += not bad
    if bad or "-v" in sys.argv:
        x = bad[0] if bad else None
        print("  %-5s %-20s -> %-20s %-7s n %4d" % ("WRONG" if bad else "ok", rules.name(T.codes, c), rules.name(T.codes, o), rules.ANAME[a], cnt[(c, o, a)]), st and dict(a=st["a"], G=[grid.gid_inv(g) for g in st["G"]], need=st["need"], avoid=st["avoid"]),
              x and ("differs at", rules.ANAME[x[0][0]], grid.gid_inv(x[0][1]) if x[0][1] < grid.NG else "far", [rules.name(T.codes, k) for k, _ in x[0][2]], "says", x[1], "rules", x[2], len(bad)))
print("statements right: %d of %d" % (ok, tot))
