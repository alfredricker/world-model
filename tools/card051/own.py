"""Card 051: card 049's recall, with a query's own tries first.

  P(class | q) = (N_own + alpha P_nb) / (|N_own| + alpha)

- N_own: the outcome counts of the stored tries in q's own situation: the front and held tiles with the same code
  tuples (codes for identity; card 042's same-thing level) and the same values on the admitted view conditions.
  Tokens in view that no condition reads are still ignored.
- P_nb: card 049's prediction from the neighbours, weighted by the admitted conditions.
- alpha: fitted by leaving one try out (card 038's back-off; MacKay and Peto 1995), after card 049's admission.

A failed try is the situation's own evidence: after one failure in a situation the neighbours called a success,
the prediction there is alpha P_nb / (1 + alpha).

  bin/prun python tools/card051/own.py --clutter --seed 403 --n 100 --out runs/051_trial_clutter_403.json
  bin/prun python tools/card051/own.py --dev --arm A --seeds 403-403 --layouts 30 --out runs/051_trial_familiar_403.json
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card049"))
import conditions as CD                                # noqa: E402

CR = CD.CR
ALPHAS = np.exp(np.linspace(-9.0, 9.0, 181))


class OwnKind(CD.CondKind):
    def admit(self):
        super().admit()
        self.a = self.beta                               # card 049's prior strength inside the neighbours' vote
        n = len(self.keys)
        Fk = self.key_feats()
        d = np.zeros((n, n))
        for ci, l in zip(self.adm, self.lamc):
            d += l * self.cand_dist(ci, Fk, Fk)
        w = np.exp(-d)
        np.fill_diagonal(w, 0.0)
        lc = self.lc[:n]
        L = lc.shape[1]
        tot = lc.sum(1)
        Pn = (w @ lc + self.a / L) / ((w @ tot)[:, None] + self.a)
        sig = self.signatures(np.arange(n), Fk)
        groups = {}
        for i, g in enumerate(sig):
            groups.setdefault(g, []).append(i)
        G = np.zeros_like(lc)
        for idx in groups.values():
            G[idx] = lc[idx].sum(0)
        Gt = G.sum(1, keepdims=True)
        m = lc > 0

        def ll(alpha):
            P = (G - 1.0 + alpha * Pn) / (Gt - 1.0 + alpha)
            return float((lc[m] * np.log(np.maximum(P[m], 1e-300))).sum())

        vals = [ll(al) for al in ALPHAS]
        j = int(np.argmax(vals))
        self.beta = float(ALPHAS[j])
        self.report["conditions"]["own"] = {"alpha": round(self.beta, 6), "ll": round(vals[j], 2),
                                            "ll_neighbours_only": round(ll(1e9), 2), "situations": len(groups)}
        print(self.name, "own tries first: alpha", self.report["conditions"]["own"], flush=True)

    def view_cols(self):
        return [self.vidx[ci] for ci in self.adm if self.cand[ci][0] == "view"]

    def signatures(self, idx, Fk):
        cols = self.view_cols()
        return [(int(self.fid[i]), int(self.hid[i]), tuple(Fk[3][i, cols].tolist())) for i in idx]

    def weights(self, qs):
        w = self.cond_weights(qs)
        n = len(self.keys)
        lc = self.lc[:n]
        L = lc.shape[1]
        share = lc / lc.sum(1, keepdims=True)
        Nn = w @ lc
        s = (Nn + self.a / L) / (Nn.sum(1, keepdims=True) + self.a)
        Fk, Fq = self.key_feats(), self.feats(qs)
        cols = self.view_cols()
        own = np.stack([(self.fid[:n] == CR.code_id(self.S, q[0])) & (self.hid[:n] == CR.code_id(self.S, q[1]))
                        & (Fk[3][:, cols] == Fq[3][i, cols]).all(1) for i, q in enumerate(qs)]).astype(np.float64)
        N = own @ lc
        z = w.sum(1, keepdims=True) + 1.0
        return own, w, z, share, N, s


_kind = CR.kind


def own_kind(name, S, parts, nkey, ncat, keys, cats, w, after=None, other=None, lam=None):
    if nkey == 3:
        return OwnKind(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)
    return _kind(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)


def main():
    CR.kind = own_kind
    sys.argv[1:1] = ["--recall", "own"]                # card 049's main leaves CR.kind alone for "own"
    CD.main()


if __name__ == "__main__":
    main()
