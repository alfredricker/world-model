"""development check: the latent-state model on the simulator's tokens, by the sparsity penalty and the fitting steps"""
import sys, torch, numpy as np, importlib.util
_s = importlib.util.spec_from_file_location("events158", "tools/card158/events.py"); Ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(Ev)
by = sys.argv[1]; X = torch.load("runs/158/exp_script_%s_0.pt" % by, weights_only=False)
rows = Ev.sim_sequences(X); D = Ev.stack(rows)
for l1, steps, iters in ((1e-9, 300, 30), (1.0, 300, 30), (1.0, 1000, 30), (0.1, 300, 30), (0.1, 1000, 30)):
    m = Ev.States(3); m.fit(D, iters=iters, l1=l1, steps=steps)
    print(by, "l1", l1, "steps", steps, Ev.events_f1(rows, m.viterbi(D), X, m.up), "ll", m.log[-1], "emit", m.emit.round(3).tolist())
    print("    weights by class (bias, state before, contact, grip, moved)", (m.w0 + m.dw).round(1).tolist())
    print("    conditions", {nm: Ev.conditions(m, b) for b, nm in ((0, "takeable"), (2, "sticky"))})
