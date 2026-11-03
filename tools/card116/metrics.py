"""Card 116: segmentation metrics against the pixel truth.

- Segments given per patch (14 x 14 soft masks or hard ids) are upsampled to 64 x 64: each segment's mask bilinearly,
  then each pixel goes to its highest segment.
- FG-ARI: adjusted Rand index on foreground pixels (truth label > 0), per frame, averaged (frames with fewer than two
  foreground objects included; chance 0).
- mBO per object: for each true object present, the best IoU over predicted segments; averaged over frames where the
  object is present.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

NAMES = ['arm', 'cube', 'button0', 'button1', 'drawer', 'window']


def upsample(masks):
    """masks: [n, K, G*G] soft or one-hot (G = 14 or 32) → pixel segment ids [n, 64, 64]."""
    G = int(round(masks.shape[-1] ** 0.5))
    m = torch.as_tensor(masks, dtype=torch.float32).reshape(len(masks), -1, G, G)
    up = F.interpolate(m, size=(64, 64), mode='bilinear', align_corners=False)
    return up.argmax(1).numpy().astype(np.int16)


def onehot(ids, K):
    """ids: [n, 196] hard segment ids → [n, K, 196]."""
    return np.eye(K, dtype=np.float32)[ids].transpose(0, 2, 1)


def scores(seg, truth):
    """seg, truth: [n, 64, 64] → dict with fg_ari, mbo per object, mean_mbo."""
    aris, best = [], {k: [] for k in NAMES}
    for s, t in zip(seg, truth):
        fg = t > 0
        aris.append(adjusted_rand_score(t[fg], s[fg]) if fg.sum() > 1 else 1.0)
        ids = np.unique(s)
        for j, name in enumerate(NAMES, start=1):
            obj = t == j
            if not obj.any():
                continue
            iou = max(((s == i) & obj).sum() / ((s == i) | obj).sum() for i in ids)
            best[name].append(iou)
    mbo = {k: float(np.mean(v)) for k, v in best.items()}
    return dict(fg_ari=float(np.mean(aris)), mbo={k: round(v, 4) for k, v in mbo.items()},
                mean_mbo=round(float(np.mean(list(mbo.values()))), 4))


def patch_of_pixels(G=14):
    p = (np.arange(64) * G) // 64
    return p[:, None] * G + p[None, :]
