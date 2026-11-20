"""The motion front end for card 147: for every pixel of a frame, where it went in the next frame.

Block matching with shiftable windows: the cost of a displacement d at a pixel is the lowest mean
squared difference over the four 2 x 2 windows that contain the pixel, each compared with the window
d away in the next frame.  (A window that may sit on either side of the pixel keeps a pixel at the
edge of a moving object from being matched together with the background beside it.)  A pixel's motion
is `valid` when its best match is as good as sensor noise allows, for its window and for the pixel
itself.  No learning.
"""
import numpy as np, torch
import torch.nn.functional as F

H = 64
R = 5
DISP = [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)]
ZERO = DISP.index((0, 0))
THR = 0.0008          # a 2 x 2 window that matches: sensor noise alone gives 0.0002 +- 0.00008
THR_P = 0.0015        # a single pixel that matches: sensor noise alone gives 0.0002, under 0.0011 in 999 of 1000


def cost_volume(a, b):
    """a, b: [B, 3, H, W] in 0..1.  C[B, D, H, W]."""
    B, _, Hh, W = a.shape
    bp = F.pad(b, (R, R, R, R), value=10.0)
    out, own = [], []
    for dy, dx in DISP:
        e = ((a - bp[:, :, R + dy:R + dy + Hh, R + dx:R + dx + W]) ** 2).mean(1, keepdim=True).clamp(max=1.0)
        w = F.pad(F.avg_pool2d(e, 2, 1), (1, 1, 1, 1), value=10.0)          # windows by their top-left corner
        out.append(-F.max_pool2d(-w, 2, 1))                                 # the best window that contains the pixel
        own.append(e)
    return torch.cat(out, 1), torch.cat(own, 1)


def flow(C, P):
    """index of the best displacement per pixel [B, H, W] (near-ties go to 'stays put'), and validity:
    the window matches, and so does the pixel itself (the window proposes, the pixel confirms)."""
    Cz = C.clone(); Cz[:, ZERO] -= 0.5 * THR
    idx = Cz.argmin(1)
    return idx, (C.gather(1, idx[:, None])[:, 0] < THR) & (P.gather(1, idx[:, None])[:, 0] < THR_P)


def true_flow(masks, hand, obj_pos):
    """the same two arrays from the simulator: every pixel's true displacement, valid when it is still seen."""
    T = len(masks) - 1
    m = masks.astype(np.int64)
    d = np.zeros((T, H, H, 2), np.int64)
    dh = np.diff(hand.astype(np.int64), axis=0); do = np.diff(obj_pos.astype(np.int64), axis=0)
    for t in range(T):
        d[t][m[t] == 1] = dh[t]
        for i in range(obj_pos.shape[1]):
            d[t][m[t] == 3 + i] = do[t, i]
    ys, xs = np.mgrid[0:H, 0:H]
    ty, tx = ys[None] + d[..., 0], xs[None] + d[..., 1]
    inb = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < H)
    nxt = np.stack([m[t + 1][ty[t].clip(0, H - 1), tx[t].clip(0, H - 1)] for t in range(T)])
    return (d[..., 0] + R) * (2 * R + 1) + (d[..., 1] + R), inb & (nxt == m[:-1])
