"""development check: how often the own contact token equals the simulator's, by the tolerance of the box rule"""
import sys, torch
import importlib.util
_spec = importlib.util.spec_from_file_location("classes157", "tools/card157/classes.py"); C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
for by in ("shape", "colour"):
    X = torch.load("runs/157/exp_%s_0.pt" % by, weights_only=False)
    for tol in (0, 1, 2):
        a = C.own_tokens(X, tol)[4]
        print(by, "tol", tol, "sim contacts caught %.1f%%" % (100 * a["contact_both"] / (a["contact_both"] + a["contact_sim_only"])), "own contacts that are not the simulator's %.1f%%" % (100 * a["contact_own_only"] / (a["contact_both"] + a["contact_own_only"])))
