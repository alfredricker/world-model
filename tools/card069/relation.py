"""Card 069: the relation rel:P in recall, refitted online; card 069's encoder; the decoy world.

Installed into card 066's harness (version 15) as a later card's component (tiers.INSTALL):
  - the encoder is card 069's (runs/069/encoder.pt), trained with the relation (train.py);
  - recall's candidate conditions for pick up, toggle and drop lose card 049's four relations rel:p (the front and
    held tiles' L1 distance within part p) and gain one, rel:P = ||P z_front - P z_held||_1, with P the encoder's
    relation projection (8 x 32), shared by both tiles; it is admitted like any other candidate;
  - online (ONLINE["on"]): after a pick up or toggle whose outcome recall predicted wrong, P is refitted for 50 Adam
    steps (in numpy) on the admission's leave-one-out likelihood of that action's stored tries (card 049's, over card 051's
    groups, every other admitted weight held), starting from the current P; each episode starts from the
    encoder's P, as memory starts from its stored tries (World.reset).

The decoy world (DECOY): MiniGrid's DoorKey-8x8 with the door and its key in one hue and a second key of another
hue in the first room; the dividing wall at column 3 or more, so the first room is at least two tiles wide and the
decoy can never block the way (a hand problem, not a relation). Hues are drawn from DECOY["door"] and
DECOY["decoy"].
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ENCODER = Path(os.environ.get("WM_REL_ENCODER", ROOT / "runs" / "069" / "encoder.pt"))   # card 070 sets its own
ONLINE = {"on": True, "steps": 50, "lr": 0.01}
REL = {"P": None, "P0": None}
STATS = {"wrong_predictions_pick_toggle": 0, "refits": 0, "refit_seconds": 0.0}
CAND = ("rel", 4)
T = None


def _kinds(W):
    IX = sys.modules["index"]
    return [kd for kd in W.kinds.values() if isinstance(kd, IX.IndexKind)]


def _clear(W):
    for kd in _kinds(W):
        kd.ix["feats"] = None
        kd._fk = None
    W.openc.clear()
    W.openedc.clear()


def loo_grad(P, Dz, Df, lam, gt, Ct, Cg, L, other=None):
    """Card 049's leave-one-out log likelihood (a = 1) over card 051's groups as a function of P, and its gradient,
    in numpy (episodes run in forked workers, where autograd cannot run once the parent has used it).
    r_g = ||P Dz_g||_1; w_gh = exp(-(Df_gh + lam |r_g - r_h|)); key i in group g: P_i = (w_g. Cg - C_i + 1/L) /
    (w_g. Tg - |C_i| + 1). With `other` (card 071), w is masked to other (front, held) combinations and nothing
    is subtracted."""
    s = Dz @ P.T
    r = np.abs(s).sum(1)
    sg = np.sign(r[:, None] - r[None])
    w = np.exp(-(Df + lam * np.abs(r[:, None] - r[None])))
    tot, Tg = Ct.sum(1), Cg.sum(1)
    if other is not None:
        w = w * other
        num = (w @ Cg)[gt] + 1.0 / L
        den = (w @ Tg)[gt] + 1.0
    else:
        num = (w @ Cg)[gt] - Ct + 1.0 / L
        den = (w @ Tg)[gt] - tot + 1.0
    ll = float((Ct * np.log(np.maximum(num / den[:, None], 1e-300))).sum())
    Qg = np.zeros_like(Cg)
    np.add.at(Qg, gt, Ct / num)
    qg = np.zeros(len(Cg))
    np.add.at(qg, gt, tot / den)
    dW = Qg @ Cg.T - qg[:, None] * Tg[None]            # d ll / d w
    dE = -dW * w                                       # d ll / d E, E the distance
    dr = lam * ((dE + dE.T) * sg).sum(1)
    dP = (dr[:, None] * np.sign(s)).T @ Dz
    return ll, dP


def refit(W, kd):
    """Raise card 049's leave-one-out likelihood of this action's stored tries by moving P alone (50 Adam steps)."""
    ci = kd.cand.index(CAND)
    if ci not in kd.adm:
        return
    t0 = time.monotonic()
    ix = kd.ix
    n = len(kd.keys)
    reps = ix["rep"]
    Fg = kd.feats([kd.keys[i] for i in reps])
    Df = np.zeros((len(reps), len(reps)))
    for c, lam in zip(kd.adm, kd.lamc):
        if c != ci:
            Df += lam * kd.cand_dist(c, Fg, Fg)
    lam_p = float(kd.lamc[kd.adm.index(ci)])
    Dz = (Fg[0] - Fg[1]).reshape(len(reps), -1)
    gt = np.asarray(ix["gof"][:n])
    Ct = kd.lc[:n]
    Cg = np.zeros((len(reps), Ct.shape[1]))
    np.add.at(Cg, gt, Ct)
    other = None
    if sys.modules["index"].HOLD["by"] == "combination":       # card 071: the admission's own likelihood
        fc = np.asarray(kd.fid)[reps]
        hc = np.asarray(kd.hid)[reps]
        other = ((fc[:, None] != fc[None]) | (hc[:, None] != hc[None])).astype(np.float64)
    P = REL["P"].copy()
    m, v = np.zeros_like(P), np.zeros_like(P)
    b1, b2, lr = 0.9, 0.999, ONLINE["lr"]
    for k in range(1, ONLINE["steps"] + 1):
        _, g = loo_grad(P, Dz, Df, lam_p, gt, Ct, Cg, Ct.shape[1], other)
        g = -g                                          # minimise the negative log likelihood
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        P -= lr * (m / (1 - b1 ** k)) / (np.sqrt(v / (1 - b2 ** k)) + 1e-8)
    REL["P"] = P
    _clear(W)
    STATS["refits"] += 1
    STATS["refit_seconds"] += time.monotonic() - t0


def install():
    global T
    T = sys.modules["tiers"]
    VP = T.VP
    IX, CD = sys.modules["index"], sys.modules["conditions"]
    ck = torch.load(ENCODER)
    REL["P0"] = ck["P"].double().numpy()
    REL["P"] = REL["P0"].copy()
    _encoder = T.encoder
    T.encoder = lambda path=ENCODER: _encoder(path)

    _feats = CD.CondKind.feats

    def feats(self, keys):
        f, h, rel, v = _feats(self, keys)
        rp = np.abs((f - h).reshape(len(keys), -1) @ REL["P"].T).sum(1)
        return f, h, np.c_[rel, rp], v

    _admit = IX.IndexKind.admit

    def admit(self):
        if CAND not in self.cand:
            self.cand = [c for c in self.cand if c[0] != "rel"]
            self.cand.insert(sum(c[0] in ("front", "held") for c in self.cand), CAND)
        return _admit(self)

    _cname = CD.CondKind.cname

    def cname(self, ci):
        return "rel:P" if self.cand[ci] == CAND else _cname(self, ci)

    CD.CondKind.feats, CD.CondKind.cname, IX.IndexKind.admit = feats, cname, admit

    World = VP.World
    _learn, _reset = World.learn_try, World.reset

    def learn_try(self, V, a, V2, end):
        pred = None
        if a in (VP.PICK, VP.TOG):
            pred = self.outcome(a, int(V[VP.FRONT]), int(V[VP.HELD]), self.ctx_of_view(V))[0]
        cat = _learn(self, V, a, V2, end)
        if pred is not None and pred != cat:
            STATS["wrong_predictions_pick_toggle"] += 1
            if ONLINE["on"]:
                refit(self, self.kinds[a])
        return cat

    def reset(self):
        _reset(self)
        if not np.array_equal(REL["P"], REL["P0"]):
            REL["P"] = REL["P0"].copy()
            _clear(self)

    World.learn_try, World.reset = learn_try, reset


# ---------------------------------------------------------------- the decoy world

DECOY = {"door": None, "decoy": None}


def register():
    import gymnasium as gym
    from minigrid.core.grid import Grid
    from minigrid.core.world_object import Door, Goal, Key
    from minigrid.envs.doorkey import DoorKeyEnv

    class DoorKeyDecoy(DoorKeyEnv):
        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            self.put_obj(Goal(), width - 2, height - 2)
            split = self._rand_int(3, width - 2)
            self.grid.vert_wall(split, 0)
            self.place_agent(size=(split, height))
            c = self._rand_elem(DECOY["door"])
            c2 = self._rand_elem([h for h in DECOY["decoy"] if h != c])
            self.put_obj(Door(c, is_locked=True), split, self._rand_int(1, height - 2))
            self.place_obj(obj=Key(c), top=(0, 0), size=(split, height))
            self.place_obj(obj=Key(c2), top=(0, 0), size=(split, height))
            self.mission = "use the key to open the door and then get to the goal"
            self.decoy_tries = 0

        def step(self, action):
            fr = self.grid.get(*self.front_pos)
            if (action == self.actions.toggle and fr is not None and fr.type == "door" and fr.is_locked
                    and self.carrying is not None and self.carrying.type == "key" and self.carrying.color != fr.color):
                self.decoy_tries += 1
            return super().step(action)

    gym.register(id="WM-DoorKeyDecoy-8x8-v0", entry_point=DoorKeyDecoy, kwargs={"size": 8})
    return "WM-DoorKeyDecoy-8x8-v0"
