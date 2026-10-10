"""Card 094.2: the router's view, cut to what recall's admitted conditions read.

WM_ROUTER_VIEW=admitted   a view token stays when an admitted view condition of that action names its appearance
                          (card 049's ("view", code tuple) candidates, admitted in that world at setup)
WM_ROUTER_VIEW=none       no view token stays (the baseline: front and held alone)
unset                     version 20 (every token in the believed view)

Read by tools/card091/routed.py (the agent's queries and stored keys) and tools/card091/keys.py (the router's
training keys), so the router is trained and queried on the same cut.
"""
import os

import numpy as np

MODE = os.environ.get("WM_ROUTER_VIEW")
_ADM = {}


def admitted(kd):
    """The appearances (code tuples) of kd's admitted view conditions."""
    t = _ADM.get(id(kd))
    if t is None:
        t = _ADM[id(kd)] = frozenset(kd.cand[ci][1] for ci in kd.adm if kd.cand[ci][0] == "view")
    return t


def view_of(W, a, s):
    """The view set s (handles) as the router reads it for action a."""
    if MODE is None:
        return s
    if MODE == "none":
        return np.asarray(s, np.int64)[:0]
    kd = W.kinds[a]
    t = admitted(kd)
    return np.array([h for h in s if kd.codes(int(h)) in t], np.int64)
