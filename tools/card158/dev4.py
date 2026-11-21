"""check: what the own tokens show at the steps where an object is truly taken, with and without a move.  dev4.py <exp> <by> <seed>"""
import sys, torch, numpy as np, importlib.util, collections
_s = importlib.util.spec_from_file_location("events158", "tools/card158/events.py"); Ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(Ev)
exp, by, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
X = torch.load(("runs/158/exp_script_%s_%d.pt" if exp == "script" else "runs/157/exp_%s_%d.pt") % (by, seed), weights_only=False)
L = torch.load("runs/158/fit_%s_%s_%d.pt" % (exp, by, seed), weights_only=False)[0]
rows = Ev.own_sequences(X, L); n = collections.Counter(); fc = collections.Counter()
for r in rows:
    if r["obj"] < 0: continue
    ep = X["tr"][r["e"]]; held = (ep["held"][1:] == r["obj"]).astype(int); o = ep["objs"][r["obj"]]; right = 1 if o["kind"] == 0 else 0
    for t in range(r["t0"] + 1, Ev.T):
        if held[t] and not held[t - 1]:
            k = "taken with a move" if r["moved"][t] else "taken without a move"
            n[k, "seen" if r["valid"][t] else "pattern not seen", "contact" if r["contact"][t] else "no contact", "came" if r["came"][t] else "did not come"] += 1
        if r["valid"][t] and r["contact"][t] and not held[t - 1] and r["grip"][t] == right and o["kind"] != 1:      # own contact, right grip, free before
            sim = Ev.W.touching(ep["hand"][t], ep["pos"][t, r["obj"]], o["size"])
            fc["moved" if r["moved"][t] else "still", "taken" if held[t] else ("not taken: no contact in the simulator" if not sim else "not taken: another object held" if ep["held"][t] >= 0 else "not taken: other")] += 1
for k, v in sorted(n.items()): print(exp, by, seed, k, v)
for k, v in sorted(fc.items()): print("   own contact, right grip, free:", k, v)
