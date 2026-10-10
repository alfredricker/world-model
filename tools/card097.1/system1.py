"""Card 097.1, stage 2: the agent walks by System 1 (WM_WALK_S1=1).

Where the planner's chosen step walks toward a need (a trace [..., need, ("walk", "closer")]), the move is System 1's:
the three copies trained in learn.py (--revise) each give steps to go per move over the need's target placements in
the believed 13 x 13 view (rounded; forward, left, right among equals), and their move is taken when all three pick
it. Otherwise the field's move stays: no target in view, or the copies disagree. The field still says whether a
need can be reached and how far (cost, crossings), and exploration (card 057) still walks by it.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learn as L                                      # noqa: E402

STATS = {"s1_walk_steps": 0, "s1_moves": 0, "s1_out_of_view": 0, "s1_disagree": 0, "s1_differs_from_field": 0,
         "s1_seconds": 0.0}
STATE = {}


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        VP = T.VP
        F = VP.F
        torch.set_num_threads(1)
        nets = []
        for seed in (97, 98, 99):
            n = L.Net(bins=L.K)
            n.load_state_dict(torch.load(L.OUT / f"walk_net_r1_s{seed}.pt", map_location="cpu"))
            nets.append(n.eval())
        arr = torch.as_tensor(L.vectors(tier))
        dirs = [tuple(d) for d in F.DIRS]
        cls = T.PLANNER["cls"] or T.S7.Plan047
        choose0 = cls.choose

        def rel(M, c, t):
            ahead, right = dirs[M.pd[c]], dirs[(M.pd[c] + 1) % 4]
            d = (M.px[t] - M.px[c], M.py[t] - M.py[c])
            return (int(d[0] * right[0] + d[1] * right[1]), int(d[0] * ahead[0] + d[1] * ahead[1]),
                    int((M.pd[t] - M.pd[c]) % 4))

        @torch.no_grad()
        def move(pl, st, need):
            M = F.M
            pid = pl.face_pid(st[0], need)
            tg = sorted({rel(M, int(st[1]), int(p)) for p in pl.psets[pid]})
            tg = [g for g in tg if L.cell(g[0], g[1]) >= 0]
            if not tg:
                STATS["s1_out_of_view"] += 1
                return None
            v = torch.as_tensor(np.asarray(pl.view(st)[:L.NV], np.int64))
            V = arr[v][None].expand(len(tg), L.NV, 32)
            G = torch.zeros(len(tg), 4, L.NV)
            for k, (x, y, h) in enumerate(tg):
                G[k, h, L.cell(x, y)] = 1.0
            picks = {L.pick_int(n(V, G).min(0).values.numpy()) for n in nets}
            if len(picks) > 1:
                STATS["s1_disagree"] += 1
                return None
            return picks.pop()

        def choose(self, st):
            res = choose0(self, st)
            tr = list(getattr(res, "trace", None) or [])
            k = next((i for i, c in enumerate(tr) if c == ("walk", "closer")), -1)
            if res is None or k < 1 or not (isinstance(tr[k - 1], tuple) and tr[k - 1][0] == "face"):
                return res
            STATS["s1_walk_steps"] += 1
            t0 = time.perf_counter()
            a = move(self, st, tr[k - 1])
            STATS["s1_seconds"] += time.perf_counter() - t0
            if a is not None:
                STATS["s1_moves"] += 1
                if a != res.action:
                    STATS["s1_differs_from_field"] += 1
                    res.action = a
            return res

        cls.choose = choose
        log("card 097.1: walking toward needs by System 1 (three copies)")
        return W, info

    T.setup = setup
