"""Card 129: one read-out of where a hand can close with an effect, for every kind of touch.

Training pairs, no object named:
- every touch of sets m400, m600, m800 across which a change was seen (card 125's outcome);
- target: the gripper's position at contact (proprioception; observation units, 1 = 10 cm);
- input: one frame from before the touch, after the previous touch's release, with the gripper at least 15 cm
  from the target (drawn at random among such frames), 256 px, blocks 2 and 12 per patch.
Read-out: card 128's network (per patch a score and an offset; a position per patch from its row and column).
New loss: the softmax weights are a mixture over patches with centres (position + offset); the loss is the negative
log-likelihood of the target under that mixture of round Gaussians, whose width is annealed from 10 cm to 0.6 cm.
Peaks: patches taken in order of weight; each absorbs the patches whose centre lies within 3 cm; a peak is kept
if its summed weight is at least MASS.
Score (held-out: episodes 80-99 of each set; the truth only scores): per kind of place of the touch that followed,
the share of frames with a peak within 1 cm and 3 cm of where the gripper then closed; for every frame, a peak
within 1 cm of the true cube; baseline: the place's mean contact point.

  MUJOCO_GL=egl ... bin/prun python tools/card129/readout.py build|train|score → runs/129/
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

HERE = Path(__file__).resolve().parent
for d in ('card127', 'card125', 'card124', 'card115'):
    sys.path.insert(0, str(HERE.parent / d))
from gate import DATA, EP, ROOT, otsu  # noqa: E402  (card 125's)

OUT = ROOT / 'runs/129'
SETS = {'m400': 400, 'm600': 600, 'm800': 800}
R, FAR, MASS, JOIN = 256, 1.5, 0.02, 0.3
CEN = np.array([0.425, 0, 0])
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def touches():
    k = np.load(ROOT / 'runs/127/know.npz')
    z = np.load(DATA / 'scene-play-v0.npz')
    obs400 = z['observations'][400 * EP:500 * EP]
    centres = np.stack([obs400[k['st'][k['place'] == c] + 2, 12:15].mean(0) for c in range(int(k['place'].max()) + 1)])
    rng = np.random.default_rng(0)
    out = {}
    for name, first in SETS.items():
        sl = slice(first * EP, (first + 100) * EP)
        obs, qpos = z['observations'][sl], z['qpos'][sl]
        c = np.load(ROOT / f'runs/122/computed_{name}.npz')
        st, en = c['st'], c['en']
        s = np.log(np.sort(c['tnum'] / (c['tden'] + np.median(c['tden'])), 1)[:, -4:].mean(1))
        ym = s > otsu(s)
        pos = obs[st + 2, 12:15]
        place = np.linalg.norm(pos[:, None] - centres[None], axis=-1).argmin(1)
        rows = []
        for i in np.where(ym)[0]:
            lo = en[i - 1] + 3 if i and st[i - 1] // EP == st[i] // EP else (st[i] // EP) * EP
            t = np.arange(lo, st[i] - 2)
            t = t[np.linalg.norm(obs[t, 12:15] - pos[i], axis=1) >= FAR] if len(t) else t
            if len(t):
                r = rng.choice(t)
                rows.append((r, *pos[i], *obs[en[i], 12:15], *((qpos[r, 14:17] - CEN) * 10), place[i], st[i] // EP, first))
        out[name] = np.array(rows)
    return out


def build():
    import gymnasium as gym
    import ogbench  # noqa: F401
    from train import features
    from encode import load_encoder
    T = touches()
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection', width=R, height=R)
    env.reset(seed=0)
    u = env.unwrapped
    qvel0 = np.zeros(u._model.nv)
    enc = load_encoder('cuda')
    z = np.load(DATA / 'scene-play-v0.npz')
    n = sum(len(v) for v in T.values())
    X = np.lib.format.open_memmap(OUT / 'feats.npy', 'w+', np.float16, (n, 1024, 768))
    j = 0
    for name, first in SETS.items():
        qpos, btn = z['qpos'][first * EP:(first + 100) * EP], z['button_states'][first * EP:(first + 100) * EP]
        for a in range(0, len(T[name]), 32):
            rows = T[name][a:a + 32, 0].astype(int)
            fr = np.zeros((len(rows), R, R, 3), np.uint8)
            for i, r in enumerate(rows):
                u.set_state(qpos[r].astype(np.float64), qvel0, btn[r].astype(np.int64))
                fr[i] = u.render()
            f = features(enc, torch.from_numpy(fr).to('cuda'))
            X[j:j + len(f)] = f.half().cpu().numpy()
            j += len(f)
            if (a // 32) % 8 == 7:                                         # pace the write (memory: throttle-big-disk-writes)
                X.flush(); time.sleep(2)
    X.flush()
    np.save(OUT / 'meta.npy', np.concatenate([v[:, 1:] for v in T.values()]))   # target 3, release 3, cube 3, place, episode, set
    print('touches', n, {k: len(v) for k, v in T.items()}, flush=True)
    os._exit(0)


class ReadOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU())
        self.score, self.offset = nn.Linear(128, 1), nn.Linear(128, 3)
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
        self.where = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
        g = torch.linspace(-1, 1, 32)
        self.register_buffer('grid', torch.stack(torch.meshgrid(g, g, indexing='ij'), -1).reshape(1024, 2))

    def forward(self, f):                                                  # [B, 1024, 768] → centres [B, 1024, 3], log weights
        h = self.trunk(f)
        return self.where(self.grid) + self.offset(h), torch.log_softmax(self.score(h).squeeze(-1), -1)


def peaks(mu, logw):
    """One frame's centres [1024, 3] and log weights → peaks [(position, mass)], heaviest first."""
    w = np.exp(logw)
    left = np.ones(len(w), bool)
    out = []
    while left.any():
        p = np.argmax(np.where(left, w, -1))
        if w[p] < 1e-4:
            break
        near = left & (np.linalg.norm(mu - mu[p], axis=1) < JOIN)
        m = w[near].sum()
        if m >= MASS:
            out.append(((mu[near] * w[near, None]).sum(0) / m, float(m)))
        left &= ~near
    return sorted(out, key=lambda x: -x[1])


def train():
    X = np.load(OUT / 'feats.npy')                                         # into memory
    M = np.load(OUT / 'meta.npy')
    tr = np.where(M[:, 10] < 80)[0]
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ReadOut().cuda()
    steps = int(arg('--steps', 6000))
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    Y = torch.from_numpy(M[:, :3]).float().cuda()
    for s in range(steps):
        sig = 1.0 * (0.06 / 1.0) ** min(1.0, s / (0.8 * steps))
        b = rng.choice(tr, 64, replace=False)
        mu, lw = model(torch.from_numpy(X[b]).cuda().float())
        nll = -torch.logsumexp(lw - ((mu - Y[b, None]) ** 2).sum(-1) / (2 * sig ** 2), -1).mean()
        opt.zero_grad(); nll.backward(); opt.step(); sched.step()
        if s % 500 == 0:
            print(s, round(sig, 3), round(nll.item(), 3), flush=True)
    torch.save(model.state_dict(), OUT / 'readout.pt')
    score(model, X, M)


def score(model=None, X=None, M=None):
    if model is None:
        X, M = np.load(OUT / 'feats.npy', mmap_mode='r'), np.load(OUT / 'meta.npy')
        model = ReadOut().cuda()
        model.load_state_dict(torch.load(OUT / 'readout.pt'))
    model.eval()
    k = np.load(ROOT / 'runs/127/know.npz')
    slide = {int(n[3:]) for n in k.files if n.startswith('eff')}
    cause = {int(k[n]) for n in k.files if n.startswith('cause')}
    kind = lambda c: 'slide place' if c in slide else 'cause place' if c in cause else 'free place'
    te = np.where(M[:, 10] >= 80)[0]
    tr = M[:, 10] < 80
    by, cube, npk = {}, [], []
    for i in te:
        with torch.no_grad():
            mu, lw = model(torch.from_numpy(np.asarray(X[i])[None]).cuda().float())
        pk = peaks(mu[0].cpu().numpy(), lw[0].cpu().numpy())
        P = np.stack([p for p, _ in pk]) if pk else np.zeros((0, 3))
        d = np.linalg.norm(P - M[i, :3], axis=1).min() * 10 if len(P) else 99
        base = np.linalg.norm(M[tr & (M[:, 9] == M[i, 9]), :3].mean(0) - M[i, :3]) * 10
        by.setdefault(kind(int(M[i, 9])), []).append((d, base))
        cube.append(np.linalg.norm(P[:, :2] - M[i, 6:8], axis=1).min() * 10 if len(P) else 99)
        npk.append(len(P))
    res = dict(test_frames=int(len(te)), peaks_per_frame_median=float(np.median(npk)), by_kind={})
    for kd, v in by.items():
        v = np.array(v)
        res['by_kind'][kd] = dict(n=len(v), peak_within_1cm=round(float((v[:, 0] <= 1).mean()), 3),
                                  peak_within_3cm=round(float((v[:, 0] <= 3).mean()), 3),
                                  median_cm=round(float(np.median(v[:, 0])), 2),
                                  baseline_place_mean_median_cm=round(float(np.median(v[:, 1])), 2))
    cube = np.array(cube)
    res['any_frame_peak_near_true_cube'] = dict(within_1cm=round(float((cube <= 1).mean()), 3), within_3cm=round(float((cube <= 3).mean()), 3),
                                                median_cm=round(float(np.median(cube)), 2))
    json.dump(res, open(OUT / 'readout.json', 'w'), indent=1)
    sys.stdout.write(json.dumps(res, indent=1) + '\n'); sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    dict(build=build, train=train, score=score)[sys.argv[1]]()
