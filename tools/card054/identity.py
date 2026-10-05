"""Card 054: identity up to observation noise, in place of a learned codebook.

Per part k, the noise scale tau_k is 1.25 x the largest part-distance between an untouched cell's two renders a step
apart (1,000 pairs from the generator's stream; agent-observable, as card 053's pixel threshold). Codes are read
online: a piece takes the nearest class mean within tau_k; otherwise it starts a new class (leader clustering; a
class mean follows its members). There is no codebook to fit, no radius rule and no fresh codes: a new appearance
simply starts a class.

install(tau, z, books=None) fills code_recall's CC so that version 10's reading (nearest used entry within
ALPHA x radius, else "new") is exactly this one: the entries are the class means, every radius tau_k / ALPHA.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card053"))
import recall_probe as RP                              # noqa: E402

EF = RP.EF
CR, NV = RP.CR, RP.NV
K, D = 4, 8


def noise_scale(enc, n=1000, seed=399):
    """(tau per part, the part-distances of the n untouched pairs)."""
    EF.VIEWS = True
    s = EF.Stream(seed)
    pairs = []
    while len(pairs) < n:
        s.step()
        pairs += list(s.views)
    a = RP.vectors(enc, [p[0] for p in pairs[:n]]).reshape(n, K, D)
    b = RP.vectors(enc, [p[1] for p in pairs[:n]]).reshape(n, K, D)
    d = np.linalg.norm(a - b, axis=-1)
    return 1.25 * d.max(0), d


class Identity:
    def __init__(self, tau):
        self.tau = np.asarray(tau, np.float64)
        self.means = [np.zeros((0, D)) for _ in range(K)]
        self.count = [np.zeros(0) for _ in range(K)]

    def learn(self, P):
        """P: (n, K, D) unit parts, read in order; classes grow and their means follow their members."""
        for k in range(K):
            M, C = list(self.means[k]), list(self.count[k])
            for p in P[:, k]:
                if M:
                    d = np.linalg.norm(np.asarray(M) - p, axis=1)
                    j = int(d.argmin())
                    if d[j] <= self.tau[k]:
                        C[j] += 1
                        M[j] = M[j] + (p - M[j]) / C[j]
                        continue
                M.append(np.array(p, np.float64))
                C.append(1.0)
            self.means[k], self.count[k] = np.asarray(M).reshape(-1, D), np.asarray(C)
        return self

    def install(self, z):
        """code_recall's CC for these classes; z (tiles x 32) are the renderer's tiles (catalogue order), or a
        single dummy row."""
        CR.CC.clear()
        CR.CC.update({"z": np.asarray(z, np.float64), "books": [m.copy() for m in self.means],
                      "used": [np.arange(len(m)) for m in self.means],
                      "rad": [np.full(len(m), self.tau[k] / CR.ALPHA) for k, m in enumerate(self.means)],
                      "fresh": {}, "cut": {k: 0.0 for k in range(K)}, "hcodes": {}, "ids": {},
                      "ca": np.full((len(z), K), NV.NEW)})
        CR.CC["ca"] = np.array([CR.code_of_vec(v) for v in np.asarray(z, np.float64)], np.int64).reshape(len(z), K)
        return CR.CC

    def report(self):
        return {"tau": [round(float(t), 4) for t in self.tau], "classes_per_part": [len(m) for m in self.means]}
