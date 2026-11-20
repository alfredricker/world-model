"""Scoring as in AKOrN's eval_obj.py: FG-ARI over foreground pixels and mean best overlap."""
import numpy as np
from sklearn.metrics import adjusted_rand_score


def upsample(lab, size):
    r = size // lab.shape[-1]
    return np.repeat(np.repeat(lab, r, -2), r, -1)


def fgari(gt, pred):
    return np.array([adjusted_rand_score(g[g > 0], p[g > 0]) for g, p in zip(gt, pred)])


def mbo(gt, pred):
    out = []
    for g, p in zip(gt, pred):
        ious = []
        for o in np.unique(g):
            if o == 0:
                continue
            gm = g == o
            ious.append(max((np.logical_and(gm, p == q).sum() / np.logical_or(gm, p == q).sum()) for q in np.unique(p)))
        out.append(np.mean(ious))
    return np.array(out)


def patch_majority(gt, psize):
    B, H, W = gt.shape
    g = gt.reshape(B, H // psize, psize, W // psize, psize).transpose(0, 1, 3, 2, 4).reshape(B, H // psize, W // psize, -1)
    return np.array([[[np.bincount(c, minlength=5).argmax() for c in row] for row in img] for img in g])


def patch_fg_majority(gt, psize):
    """Each patch takes the object that covers most of its foreground pixels (background if it has none).
    FG-ARI scores foreground pixels only, so this is the best an assignment per patch can do."""
    B, H, W = gt.shape
    g = gt.reshape(B, H // psize, psize, W // psize, psize).transpose(0, 1, 3, 2, 4).reshape(B, H // psize, W // psize, -1)
    def pick(c):
        f = c[c > 0]
        return np.bincount(f).argmax() if len(f) else 0
    return np.array([[[pick(c) for c in row] for row in img] for img in g])
