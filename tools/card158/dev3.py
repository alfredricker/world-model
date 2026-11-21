"""development check: the class beliefs and the per-class answers behind the take test.  dev3.py <by>"""
import sys, torch, numpy as np, importlib.util
_s = importlib.util.spec_from_file_location("events158", "tools/card158/events.py"); Ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(Ev)
R = Ev.R; by = sys.argv[1]; X = torch.load("runs/158/exp_script_%s_0.pt" % by, weights_only=False)
L, cl, te, views, fs = torch.load("runs/158/fit_script_%s_0.pt" % by, weights_only=False)
rows = Ev.own_sequences(X, L); D = Ev.stack(rows); m = Ev.States(L["dyn"].K).fit(D)
print("table", cl["classes_by_true_behaviour_then_ground_hand"], "sizes", cl["class_sizes"], "share", cl["appearance_share"])
acts = [(g, mv) for g in (0, 1) for mv in (0, 1)]
for k in range(m.K):
    print("class", k, "p(taken | free, contact, (grip, moved))", {a: (round(float(m.p1(k, 0, 1, *a)), 2), int(m.N[k, 0, 1, a[0], a[1]])) for a in acts}, "sequences", int((D["cls"] == k).sum()))
W = {0: [], 1: [], 2: []}
for (e, i), v in views.items():
    W[te[e]["objs"][i]["kind"]].append(R.kind_belief(L, [L["app"][v["p"]]] if v["p"] >= 0 else [v["v"]])[0])
for b, nm in ((0, "takeable"), (1, "fixed"), (2, "sticky")):
    w = np.array(W[b]); print(nm, len(w), "mean belief over classes", w.mean(0).round(2), "likeliest class counts", np.bincount(w.argmax(1), minlength=m.K))
