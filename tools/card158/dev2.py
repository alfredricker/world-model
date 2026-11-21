"""development: the sparsity penalty (nats per unit of coupling), on own and on the simulator's tokens.  dev2.py <by> <l1...>"""
import sys, torch, numpy as np, importlib.util
_s = importlib.util.spec_from_file_location("events158", "tools/card158/events.py"); Ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(Ev)
C, R, W = Ev.C, Ev.R, Ev.W
by = sys.argv[1]; X = torch.load("runs/158/exp_script_%s_0.pt" % by, weights_only=False)
L, cl, te, views, fs = torch.load("runs/158/fit_script_%s_0.pt" % by, weights_only=False)
rows = Ev.own_sequences(X, L); D = Ev.stack(rows); srows = Ev.sim_sequences(X); Ds = Ev.stack(srows); table = np.array(cl["classes_by_true_behaviour_then_ground_hand"])
app = lambda e, i, o: None if views.get((e, i)) is None else R.kind_belief(L, [L["app"][views[e, i]["p"]]] if views[e, i]["p"] >= 0 else [views[e, i]["v"]])[0]
def missed(rows, h, up):
    out = dict(taken_with_move=[0, 0], taken_without=[0, 0], released_with_move=[0, 0], released_without=[0, 0])
    for r, hs in zip(rows, h):
        if r["obj"] < 0: continue
        st = (hs == up).astype(int); held = (X["tr"][r["e"]]["held"][1:] == r["obj"]).astype(int)
        for t in range(r["t0"] + 1, Ev.T):
            if held[t] != held[t - 1]:
                k = ("taken" if held[t] else "released") + ("_with_move" if r["moved"][t] else "_without"); out[k][1] += 1; out[k][0] += int(st[t] == held[t] and st[t - 1] == held[t - 1])
    return {k: "%d/%d" % tuple(v) for k, v in out.items()}
for l1 in [float(v) for v in sys.argv[2:]]:
    m = Ev.States(L["dyn"].K).fit(D, l1=l1); ms = Ev.States(3).fit(Ds, l1=l1); h = m.viterbi(D); hs = ms.viterbi(Ds)
    print(by, "l1", l1, "| own F1", Ev.events_f1(rows, h, X, m.up)["f1"], "take", Ev.test_take(te, views, L, m, app), "| sim F1", Ev.events_f1(srows, hs, X, ms.up)["f1"], "take", Ev.test_take(te, views, L, ms, lambda e, i, o: np.eye(3)[o["kind"]])["all"])
    print("    own found", missed(rows, h, m.up)); print("    sim found", missed(srows, hs, ms.up))
    print("    own conditions", {nm: {q: v for q, v in Ev.conditions(m, int(table[:, b].argmax())).items()} for b, nm in ((0, "takeable"), (2, "sticky"))})
    print("    sim conditions", {nm: Ev.conditions(ms, b) for b, nm in ((0, "takeable"), (2, "sticky"))})
    print("    sim weights", (ms.w0 + ms.dw).round(1).tolist(), "emit", ms.emit.round(3).tolist())
