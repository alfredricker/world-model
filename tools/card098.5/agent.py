"""Card 098.5 in the agent (WM_TRACK=1): files.py as the second level of the agent's placing step (card 057's observe).

Level 1 is the agent's observe, unchanged except that places where files stand are no evidence for the placement.
Level 2 is files.py on the observed places: positions are the lattice coordinates of their token ids (where each was
first seen, in the frame of the episode's start), vectors are the encoder's (the adapter's catalogue renders), the
identity radius is half the smallest distance between two different appearances, and the step length is forward's
learned translation (card 061). A place a file stands on keeps what belief held there.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card098.5"))
import files as FL                                     # noqa: E402

STATS = {"track_births": 0, "track_files_seen": 0, "track_explained": 0}
STATE = {"files": None}


def euclid(A, B):
    """Exact differences: the expanded form leaves ~1e-8 between identical vectors, which set the radius (a fix)."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    return np.linalg.norm(A[:, None] - B[None], axis=-1)


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        TK, VP = T.TK, T.VP
        M = T.F.M
        cls = T.PLANNER["cls"] or T.S7.Plan047
        arr = np.asarray(W.S.arr, np.float64)
        D = euclid(arr, arr)
        radius = 0.5 * float(D[D > 1e-9].min())
        pos = np.zeros((TK.NT, 2))
        for (x, y), i in TK.ID_AT.items():
            pos[i] = (x, y)
        step_len = 1.0                                  # forward's learned translation, b = (0, -1) (card 061)
        own = [(0, -step_len), (step_len, 0), (0, step_len), (-step_len, 0)]   # forward under the four headings
        observe0 = cls.observe

        def observe(self, world, V, *a, **k):
            if world is None or STATE["files"] is None:
                STATE["files"] = FS = STATE["files"] or FL.Files(euclid, radius, step_len, own=own)
                FS.reset()
                self._track_t = 0
                if world is None:
                    return observe0(self, world, V, *a, **k)
            FS = STATE["files"]
            held = [int(np.argmin(np.abs(pos - f["pos"]).sum(1))) for f in FS.files]
            w0 = np.array(world, np.int32)
            if held:                                    # places files stand on are no evidence for the placement
                w0[held] = TK.ABSENT
            r = observe0(self, w0, V, *a, **k)
            f, st = r[0], r[1]
            if held:
                back = [i for i in held if f[i] == TK.ABSENT]
                f[back] = np.asarray(world)[back]
            self._track_t += 1
            Vv = np.asarray(V)
            Ap = M.A[st[1]]
            sel = (Ap >= 0) & (Vv[:len(Ap)] >= 0)
            sel[VP.CENTRE] = False
            sel[VP.NV:] = False                         # carried tokens (the hand) are not given to the files level
            ids = Ap[sel]
            if not len(ids):
                return r
            wv = np.asarray(world)[ids]
            known = wv != TK.ABSENT
            B = np.where(known[:, None], arr[np.maximum(wv, 0)], 0.0)
            P = pos[ids]
            explained, seen = FS.step(self._track_t, P, arr[Vv[:len(Ap)][sel]], B, known, (P.min(0), P.max(0)))
            STATS["track_births"] = FS.stats["pair_births"] + FS.stats["kind_births"]
            STATS["track_files_seen"] += len(seen)
            if explained.any() or held:
                STATS["track_explained"] += int(explained.sum())
                f = np.array(f, np.int32)
                f[ids[explained]] = np.asarray(world)[ids[explained]]
                r = (f, (self.intern(f), st[1], st[2])) + tuple(r[2:])
            return r

        cls.observe = observe
        info["track"] = {"radius": round(radius, 4), "step_len": step_len}
        log(f"card 098.5: object files in the placing step (radius {radius:.4f})")
        return W, info

    T.setup = setup
