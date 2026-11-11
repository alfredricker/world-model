"""Card 122: cards 119, 120 and 121's signals from 256 x 256 renders (DINOv2 at 448 px input, 32 x 32 patches).

Per episode of OGBench's state dataset scene-play-v0 (which stores contact): render every frame at 256 x 256 in
colour and as a segmentation, encode (blocks 2 and 12), and keep only
- masks [n, 6, 1024]: which of the 32 x 32 patches each object covers (any pixel);
- card 119's fixed-window signal on block 2 (numerator, denominator) at every frame;
- card 120's before/after signal per touch on block 2, and the arm's share of each window;
- for card 121's gate: blocks 2 and 12 of 4 frames per episode.
Frames are not stored.

  MUJOCO_GL=egl ... bin/prun python tools/card122/compute.py --set val|tail [--episodes N] → runs/122/computed_<set>.npz
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

for d in ['card114', 'card115', 'card119', 'card120']:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / d))
from encode import MEAN, STD, load_encoder  # noqa: E402
from masks import OBJECTS, object_of_body  # noqa: E402
from gate import in_range, transitions  # noqa: E402
from touch import CLEAR, W, touches  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
OUT = ROOT / 'runs/122'
EP, R, RES, GRID = 1001, 256, 448, 32
P = GRID * GRID
G, LO, HI = 5, 15, EP - 16
ARM = OBJECTS.index('arm')
DEV = 'cuda'
SAMPLE = [125, 375, 625, 875]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


@torch.no_grad()
def encode(enc, frames):
    b2, b12 = [], []
    for i in range(0, len(frames), 96):
        x = torch.from_numpy(frames[i:i + 96]).to(DEV).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(RES, RES), mode='bilinear', align_corners=False)
        x = (x - MEAN.to(DEV)) / STD.to(DEV)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            a, b = enc.get_intermediate_layers(x, n=[1, 11], norm=True)
        b2.append(a.float()); b12.append(b.float()[[j - i for j in SAMPLE if i <= j < i + 96]])
    return torch.cat(b2), torch.cat(b12)


def stats(x, lo0, lo1, hi0, hi1):
    """Windows x[lo0:lo1] and x[hi0:hi1] of one patch chunk → num, den per patch."""
    b, a = x[lo0:lo1].double(), x[hi0:hi1].double()
    mb, ma = b.mean(0), a.mean(0)
    den = ((b - mb) ** 2).sum(-1).mean(0) + ((a - ma) ** 2).sum(-1).mean(0)
    return ((ma - mb) ** 2).sum(-1), den


def fixed(x):
    """Card 119's signal for one episode, x [1001, P, d] → num, den [971, P], in patch chunks."""
    tau = torch.arange(LO, HI + 1, device=x.device)
    num, den = [], []
    for c in range(0, P, 128):
        y = x[:, c:c + 128].double()
        S = torch.cat([torch.zeros_like(y[:1]), y.cumsum(0)])
        Q = torch.cat([torch.zeros_like(y[:1, :, 0]), (y * y).sum(-1).cumsum(0)])
        mb = (S[tau - G] - S[tau - G - W]) / W
        ma = (S[tau + G + W + 1] - S[tau + G + 1]) / W
        vb = (Q[tau - G] - Q[tau - G - W]) / W - (mb * mb).sum(-1)
        va = (Q[tau + G + W + 1] - Q[tau + G + 1]) / W - (ma * ma).sum(-1)
        num.append(((ma - mb) ** 2).sum(-1).float()); den.append((va + vb).clamp(min=0).float())
        del y, S, Q, mb, ma
    return torch.cat(num, 1).cpu().numpy(), torch.cat(den, 1).cpu().numpy()


def main():
    name = arg('--set', 'val')
    neps = int(arg('--episodes', '100'))
    z = np.load(DATA / ('scene-play-v0-val.npz' if name == 'val' else 'scene-play-v0.npz'))
    cut = 0 if name == 'val' else int(arg('--first', '900')) * EP      # --first E: any 100 training episodes (card 125)
    n = neps * EP
    qpos, btn = z['qpos'][cut:cut + n], z['button_states'][cut:cut + n]
    c = z['observations'][cut:cut + n, 18]
    s = dict(np.load(ROOT / f"runs/114/sstate_{'val' if name == 'val' else 'train'}.npz"))
    s = {k: v[cut:cut + n] for k, v in s.items() if len(v) == len(s['buttons'])}
    st, en, dropped = touches(c)
    A, B, K, O = transitions(s)
    ok = in_range(A) & in_range(B)

    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection', width=R, height=R)
    env.reset(seed=0)
    u = env.unwrapped
    m = u._model
    geom_obj = np.array([object_of_body(m.body(m.geom_bodyid[g]).name) for g in range(m.ngeom)])
    patch_of = (np.arange(R) * GRID) // R
    pid = (patch_of[:, None] * GRID + patch_of[None, :]).reshape(-1)
    qvel0 = np.zeros(m.nv)
    enc = load_encoder(DEV)

    masks = np.zeros((n, 6, P), bool)
    num = np.full((n, P), np.nan, np.float16)
    den = np.full((n, P), np.nan, np.float16)
    T = {k: np.zeros((len(st), P), np.float32) for k in ['tnum', 'tden', 'share']}
    s2, s12, sidx = [], [], []
    rgb = np.zeros((EP, R, R, 3), np.uint8)
    t0 = time.time()
    for ep in range(neps):
        for t in range(EP):
            r = ep * EP + t
            u.set_state(qpos[r].astype(np.float64), qvel0, btn[r].astype(np.int64))
            seg = u.render(segmentation=True)
            gid, typ = seg[..., 0].reshape(-1), seg[..., 1].reshape(-1)
            obj = np.where((typ == 5) & (gid >= 0), geom_obj[np.clip(gid, 0, m.ngeom - 1)], -1)
            hit = obj >= 0
            masks[r, obj[hit], pid[hit]] = True
            rgb[t] = u.render()
        t1 = time.time()
        x, d12 = encode(enc, rgb)
        sl = slice(ep * EP + LO, ep * EP + HI + 1)
        a, b = fixed(x)
        num[sl], den[sl] = np.minimum(a, 6e4), np.minimum(b, 6e4)
        arm = torch.from_numpy(masks[ep * EP:(ep + 1) * EP, ARM]).float()
        for i in np.where(st // EP == ep)[0]:
            ls, le = st[i] % EP, en[i] % EP
            w = (ls - CLEAR - W + 1, ls - CLEAR + 1, le + CLEAR, le + CLEAR + W)
            tn, td = stats(x, *w)
            T['tnum'][i], T['tden'][i] = tn.cpu().numpy(), td.cpu().numpy()
            T['share'][i] = torch.maximum(arm[w[0]:w[1]].mean(0), arm[w[2]:w[3]].mean(0)).numpy()
        s2.append(x[SAMPLE].half().cpu().numpy()); s12.append(d12.half().cpu().numpy())
        sidx += [ep * EP + j for j in SAMPLE]
        if ep % 10 == 0 or ep == neps - 1:
            print(f'{name} episode {ep}: render {t1 - t0:.0f}s total {time.time() - t0:.0f}s', flush=True)
    np.savez(OUT / f'computed_{name}.npz', c=c, masks=masks, st=st, en=en, dropped=dropped, num=num, den=den,
             A=A[ok], B=B[ok], TK=K[ok], TO=O[ok], s2=np.concatenate(s2), s12=np.concatenate(s12),
             sidx=np.array(sidx), **T)
    print(f'{name}: {neps} episodes in {time.time() - t0:.0f}s; mean patches per object:',
          dict(zip(OBJECTS, masks.sum(2).mean(0).round(1))), flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
