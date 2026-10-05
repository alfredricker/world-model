"""Card 054: identity up to observation noise, in place of a learned codebook.

Per part k, the noise scale tau_k is 1.25 x the largest part-distance between an untouched cell's two renders a step
apart (1,000 pairs from the generator's stream; agent-observable, as card 053's pixel threshold). Codes are read
online: a piece takes the nearest class mean within tau_k; otherwise it starts a new class (leader clustering; a
class mean follows its members). There is no codebook to fit, no radius rule and no fresh codes: a new appearance
simply starts a class.

Revision (IDENTITY=local, card 054's declared revision): noise is not the same for every appearance (doors are
several times noisier than floor in the encoder's space), so each class has its own scale: 1.25 x the largest noise
distance among the 20 untouched pairs (of 20,000) whose first render lies nearest the piece that starts the class,
and never below tau_k.

install(tau, z, books=None) fills code_recall's CC so that version 10's reading (nearest used entry within
ALPHA x radius, else "new") is exactly this one: the entries are the class means, each radius its scale / ALPHA.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card053"))
import recall_probe as RP                              # noqa: E402

EF = RP.EF
CR, NV = RP.CR, RP.NV
K, D = 4, 8
LOCAL = os.environ.get("IDENTITY", "") == "local"
N_LOCAL, KNN = 20_000, 20
PAIRS = {}


def noise_scale(enc, n=1000, seed=399):
    """(tau per part, the part-distances of the n untouched pairs). Under IDENTITY=local, N_LOCAL pairs are drawn
    and kept for the classes' own scales; tau is still read from the first n."""
    EF.VIEWS = True
    s = EF.Stream(seed)
    m = N_LOCAL if LOCAL else n
    pairs = []
    while len(pairs) < m:
        s.step()
        pairs += list(s.views)
    a = RP.vectors(enc, [p[0] for p in pairs[:m]]).reshape(m, K, D)
    b = RP.vectors(enc, [p[1] for p in pairs[:m]]).reshape(m, K, D)
    d = np.linalg.norm(a - b, axis=-1)
    if LOCAL:
        PAIRS.update(A=a.astype(np.float64), D=d)
    return 1.25 * d[:n].max(0), d[:n]


class Identity:
    def __init__(self, tau):
        self.tau = np.asarray(tau, np.float64)
        self.means = [np.zeros((0, D)) for _ in range(K)]
        self.count = [np.zeros(0) for _ in range(K)]
        self.rad = [np.zeros(0) for _ in range(K)]
        self.local = LOCAL and "A" in PAIRS

    def scale(self, k, p):
        """The noise scale for a class started by piece p in part k."""
        if not self.local:
            return self.tau[k]
        dd = np.linalg.norm(PAIRS["A"][:, k] - p, axis=1)
        idx = np.argpartition(dd, KNN)[:KNN]
        return max(self.tau[k], 1.25 * float(PAIRS["D"][idx, k].max()))

    def learn(self, P):
        """P: (n, K, D) unit parts, read in order; classes grow and their means follow their members."""
        for k in range(K):
            M, C, Rd = list(self.means[k]), list(self.count[k]), list(self.rad[k])
            for p in P[:, k]:
                if M:
                    d = np.linalg.norm(np.asarray(M) - p, axis=1)
                    j = int(d.argmin())
                    if d[j] <= Rd[j]:
                        C[j] += 1
                        M[j] = M[j] + (p - M[j]) / C[j]
                        continue
                M.append(np.array(p, np.float64))
                C.append(1.0)
                Rd.append(self.scale(k, p))
            self.means[k], self.count[k], self.rad[k] = np.asarray(M).reshape(-1, D), np.asarray(C), np.asarray(Rd)
        return self

    def install(self, z):
        """code_recall's CC for these classes; z (tiles x 32) are the renderer's tiles (catalogue order), or a
        single dummy row."""
        CR.CC.clear()
        CR.CC.update({"z": np.asarray(z, np.float64), "books": [m.copy() for m in self.means],
                      "used": [np.arange(len(m)) for m in self.means],
                      "rad": [r / CR.ALPHA for r in self.rad],
                      "fresh": {}, "cut": {k: 0.0 for k in range(K)}, "hcodes": {}, "ids": {},
                      "ca": np.full((len(z), K), NV.NEW)})
        CR.CC["ca"] = np.array([CR.code_of_vec(v) for v in np.asarray(z, np.float64)], np.int64).reshape(len(z), K)
        return CR.CC

    def report(self):
        r = {"tau": [round(float(t), 4) for t in self.tau], "classes_per_part": [len(m) for m in self.means]}
        if self.local:
            r["scale_range_per_part"] = [[round(float(x.min()), 4), round(float(x.max()), 4)] if len(x) else None
                                         for x in self.rad]
        return r
