import sys, pickle, numpy as np
import grid, rules
T = pickle.load(open("runs/160/exp_1.pkl", "rb")); c, a, g, it, o, n = T.rows(); codes = T.codes; sets = T.item_sets()
for nm in sys.argv[1:]:
    ci = [k for k in range(2, len(codes.triples)) if rules.name(codes, k) == nm][0]
    for act in (3, 5):
        for d in range(4):
            gg = grid.gid(*grid.DIRV[d], d); m = (c == ci) & (a == act) & (g == gg)
            held = np.array([any(x[1] == 3 for x in sets[i]) for i in it[m]], bool) if m.any() else np.zeros(0, bool)
            print("%-18s %-7s facing %d: tries %4d changed %4d | with something picked up and not put back: tries %3d changed %3d" % (nm, rules.ANAME[act], d, n[m].sum(), n[m][o[m] != ci].sum(), n[m][held].sum(), n[m][held & (o[m] != ci)].sum()))
