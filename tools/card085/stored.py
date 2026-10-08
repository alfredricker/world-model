"""Recall's fitted weights stored on disk, so that a memory's fits are paid once (card 085.3's revision (a)).

Each fit (card 039's lambda and alpha, card 042's two levels) is keyed by a hash of everything it reads: the stored
keys' front and held vectors, the stored view sets' vectors, the outcome counts and groups, its settings, the card 085
sampling flags and the source of the files that fit. A new encoder, a different memory (another tier, a fold that
removes tries, a fraction) or a change to the fitting code gives a new key, and the fit runs again. Online tries are
added after setup and never refit, so they do not enter. Nothing about the agent changes: a stored fit returns the
same values the fit returned when it was stored.

  WM_STORED_FITS=1 ...        before T.setup(); install() after card 085's (tools/card069/run.py)
Files: runs/fits/<hash>.pkl (a few KB to a few MB each).
"""
import hashlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "runs" / "fits"
SP = sys.modules["slot_planner"]
CR = sys.modules["code_recall"]
STATS = {"stored_fits_loaded": 0, "stored_fits_fitted": 0}   # counters (the runner diffs them per episode)


def _src():
    h = hashlib.sha256()
    for m in ("slot_planner", "code_recall", "sampled"):
        f = getattr(sys.modules.get(m), "__file__", None)
        if f:
            h.update(Path(f).read_bytes())
    for k in sorted(os.environ):
        if k.startswith("WM_SAMPLE"):
            h.update(f"{k}={os.environ[k]}".encode())
    return h.digest()


def _arr(h, x):
    x = np.ascontiguousarray(x)
    h.update(f"{x.dtype}{x.shape}".encode())
    h.update(x.tobytes())


def _sets(h, S, ulist):
    """The view sets' vectors, exactly as the fits read them."""
    idx = [np.asarray(SP.SETS[s], np.int64) for s in ulist]
    _arr(h, np.array([len(i) for i in idx], np.int64))
    _arr(h, S.arr[np.concatenate(idx)] if idx else np.zeros(0))


def _key(name, S, ulist, arrays, scalars):
    h = hashlib.sha256(_src())
    h.update(name.encode())
    _sets(h, S, ulist)
    for a in arrays:
        _arr(h, np.asarray(a))
    h.update(repr(scalars).encode())
    return h.hexdigest()[:32]


def _stored(name, key, fit):
    p = DIR / f"{key}.pkl"
    if p.exists():
        STATS["stored_fits_loaded"] += 1
        return pickle.loads(p.read_bytes())
    out = fit()
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(out))
    tmp.replace(p)
    STATS["stored_fits_fitted"] += 1
    return out


def _wrap_lambda(f):
    def fit_lambda_sets(S, Xfh, kpos, ulist, Rk, group, steps=1500, lr=0.02):
        k = _key("lambda", S, ulist, [Xfh, kpos, Rk, group], (steps, lr))
        return _stored("lambda", k, lambda: f(S, Xfh, kpos, ulist, Rk, group, steps, lr))
    return fit_lambda_sets


def _wrap_alpha(f):
    def fit_alpha_dist(Dist, counts, *a, **kw):
        kd = getattr(Dist, "kind", None)
        if kd is None:                                   # the exact path: the distance matrix itself
            h = hashlib.sha256(_src())
            for x in (Dist, counts):
                _arr(h, np.asarray(x))
            k = h.hexdigest()[:32]
        else:                                            # card 085's lazy distances: what they are computed from
            k = _key("alpha", kd.S, kd.ulist, [kd.X, kd.kpos, kd.lam, counts], (a, sorted(kw.items())))
        return _stored("alpha", k, lambda: f(Dist, counts, *a, **kw))
    return fit_alpha_dist


def _wrap_codes(f, name):
    def fit(S, Xfh, kpos, ulist, counts, *rest, **kw):
        arrays = [Xfh, kpos, counts] + [r for r in rest if isinstance(r, (np.ndarray, list))]
        scalars = ([r for r in rest if not isinstance(r, (np.ndarray, list))], sorted(kw.items()))
        k = _key(name, S, ulist, arrays, scalars)
        return _stored(name, k, lambda: f(S, Xfh, kpos, ulist, counts, *rest, **kw))
    return fit


def install():
    """Wrap whichever fits are in place (version 18's, or card 085's when installed before this)."""
    if getattr(SP, "_stored_fits", False):
        return
    SP.fit_lambda_sets = _wrap_lambda(SP.fit_lambda_sets)
    SP.fit_alpha_dist = _wrap_alpha(SP.fit_alpha_dist)
    CR.fit_codes = _wrap_codes(CR.fit_codes, "codes")
    SF = getattr(CR, "SAMPLED_FIT", None)
    if SF is not None:
        SF.fit_codes_sampled = _wrap_codes(SF.fit_codes_sampled, "codes_sampled")
    SP._stored_fits = True
