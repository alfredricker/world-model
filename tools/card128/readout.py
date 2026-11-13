"""Card 128, gate part 2: read from a frame where the gripper will close on the cube.

Training pairs come from the agent's own touches, no cube label:
- touches of sets m400, m600, m800 at places that have no variable and are no cause place (card 127's knowledge;
  a touch's place is the nearest of m400's place centres), across which a change was seen (card 125's outcome);
- target: the gripper's position at contact (proprioception, observation units: 1 = 10 cm);
- input: two frames from before the touch, after the previous touch's release, with the gripper at least 15 cm
  from the target (the latest such frame and a random one), 256 px, blocks 2 and 12 per patch.
Read-out: per patch, a score and a small offset from its features (768 → 128 → 1 and 3); softmax of the scores over
the 1,024 patches; the position is the weighted mean of (a position given by a small network on the patch's row
and column + the patch's offset). The first version had no offset: 2.1 cm median on held-out touches.
Tests (the truth, the cube's position, only scores):
- held-out touches: the last 20 episodes of each set; error against the target and against the cube's true place;
- live: visual-scene-v0, the cube put at random on the table, the arm at card 127's rest pose.
Baseline: the mean target of the training touches.

  MUJOCO_GL=egl ... bin/prun python tools/card128/readout.py build|train|live → runs/128/
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

HERE = Path(__file__).resolve().parent
for d in ('card127', 'card125', 'card124', 'card115'):
    sys.path.insert(0, str(HERE.parent / d))
from gate import DATA, EP, ROOT, otsu  # noqa: E402  (card 125's)
from train import features  # noqa: E402
from encode import load_encoder  # noqa: E402

OUT = ROOT / 'runs/128'
SETS = {'m400': 400, 'm600': 600, 'm800': 800}
R, FAR = 256, 1.5
CEN = np.array([0.425, 0, 0])


def cube_touches():
    """→ per set: rows (frame index within the set) of the two input frames, target, true cube position, episode."""
    k = np.load(ROOT / 'runs/127/know.npz')
    z = np.load(DATA / 'scene-play-v0.npz')
    obs400 = z['observations'][400 * EP:500 * EP]
    centres = np.stack([obs400[k['st'][k['place'] == c] + 2, 12:15].mean(0) for c in range(int(k['place'].max()) + 1)])
    used = {int(n[3:]) for n in k.files if n.startswith('eff')} | {int(k[n]) for n in k.files if n.startswith('cause')}
    free = np.array([c not in used for c in range(len(centres))])
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
        for i in np.where(free[place] & ym)[0]:
            lo = en[i - 1] + 3 if i and st[i - 1] // EP == st[i] // EP else (st[i] // EP) * EP
            t = np.arange(lo, st[i] - 2)
            t = t[np.linalg.norm(obs[t, 12:15] - pos[i], axis=1) >= FAR] if len(t) else t
            if len(t):
                rows.append((t[-1], rng.choice(t), *pos[i], *qpos[st[i] - 2, 14:17], st[i] // EP))
        out[name] = np.array(rows)
    return out


def build():
    T = cube_touches()
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection', width=R, height=R)
    env.reset(seed=0)
    u = env.unwrapped
    qvel0 = np.zeros(u._model.nv)
    enc = load_encoder('cuda')
    z = np.load(DATA / 'scene-play-v0.npz')
    n = sum(len(v) for v in T.values())
    X = np.lib.format.open_memmap(OUT / 'cube_feats.npy', 'w+', np.float16, (n, 2, 1024, 768))
    meta, j = [], 0
    for name, first in SETS.items():
        qpos, btn = z['qpos'][first * EP:(first + 100) * EP], z['button_states'][first * EP:(first + 100) * EP]
        for a in range(0, len(T[name]), 16):
            rows = T[name][a:a + 16, :2].astype(int).reshape(-1)
            fr = np.zeros((len(rows), R, R, 3), np.uint8)
            for i, r in enumerate(rows):
                u.set_state(qpos[r].astype(np.float64), qvel0, btn[r].astype(np.int64))
                fr[i] = u.render()
            f = features(enc, torch.from_numpy(fr).to('cuda')).reshape(-1, 2, 1024, 768)
            X[j:j + len(f)] = f.half().cpu().numpy()
            j += len(f)
        meta.append(np.c_[T[name][:, 2:], np.full(len(T[name]), first)])
    X.flush()
    np.save(OUT / 'cube_meta.npy', np.concatenate(meta))                 # target xyz, true cube xyz, episode, set
    print('touches', n, {k: len(v) for k, v in T.items()})
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

    def forward(self, f):                                                  # [B, 1024, 768] → [B, 3], weights
        h = self.trunk(f)
        w = torch.softmax(self.score(h).squeeze(-1), -1)
        return ((self.where(self.grid) + self.offset(h)) * w[..., None]).sum(1), w


def train():
    X = np.load(OUT / 'cube_feats.npy', mmap_mode='r')
    M = np.load(OUT / 'cube_meta.npy')
    tgt, cube, ep = M[:, :3], (M[:, 3:6] - CEN) * 10, M[:, 6]
    te = ep >= 80
    tr = np.where(~te)[0]
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ReadOut().cuda()
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    Y = torch.from_numpy(tgt).float().cuda()
    steps = int(next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == '--steps'), 3000))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    for s in range(steps):
        b = np.sort(rng.choice(tr, 48, replace=False))
        f = torch.from_numpy(X[b, rng.integers(0, 2)].astype(np.float32)).cuda()
        loss = ((model(f)[0] - Y[b]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 500 == 0:
            print(s, round(float(loss), 4), flush=True)
    model.eval()
    pred = []
    with torch.no_grad():
        for i in np.where(te)[0]:
            pred.append(model(torch.from_numpy(X[i].astype(np.float32)).cuda())[0].cpu().numpy())
    pred = np.stack(pred)                                                  # [n, 2 frames, 3]
    cm = lambda a, b: np.linalg.norm(a[..., :2] - b[..., :2], axis=-1) * 10
    base = tgt[tr].mean(0)
    q = lambda e: [round(float(x), 2) for x in np.quantile(e, [0.5, 0.75, 0.9])]
    res = dict(train_touches=int(len(tr)), test_touches=int(te.sum()),
               error_cm_median_p75_p90=dict(
                   latest_frame_vs_target=q(cm(pred[:, 0], tgt[te])), random_frame_vs_target=q(cm(pred[:, 1], tgt[te])),
                   latest_frame_vs_true_cube=q(cm(pred[:, 0], cube[te])), baseline_mean_vs_target=q(cm(base[None], tgt[te])),
                   target_vs_true_cube=q(cm(tgt[te], cube[te]))))
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / 'readout.json', 'w'), indent=1)
    torch.save(model.state_dict(), OUT / 'readout.pt')
    os._exit(0)


def live():
    from plan import Agent, Runner
    n = 60
    rng = np.random.default_rng(0)
    ag = Agent(rng)
    model = ReadOut().cuda()
    model.load_state_dict(torch.load(OUT / 'readout.pt'))
    model.eval()
    env = gym.make('visual-scene-v0', terminate_at_goal=False, width=R, height=R)
    env.reset(seed=0)
    run = Runner(env)
    err, errg, pts = [], [], []
    for _ in range(n):
        run.reset(task_id=3)
        c = rng.uniform([0.27, -0.2], [0.47, 0.15])
        run.u._data.joint('object_joint_0').qpos[:2] = c
        try:
            run.rest(ag.mem, ag.rest)
        except Exception:
            continue
        true = run.u._data.joint('object_joint_0').qpos[:2].copy()
        with torch.no_grad():
            p, w = model(torch.from_numpy(ag.read(run.frame))[None].cuda())
            p, w = p[0].cpu().numpy(), w[0].cpu().numpy()
            g = model(torch.from_numpy(ag.read(run.goal))[None].cuda())[0][0].cpu().numpy()
        err.append(np.linalg.norm(p[:2] / 10 + CEN[:2] - true) * 100)
        pts.append((*true, *(p[:2] / 10 + CEN[:2]), float(w.max())))
        errg.append(np.linalg.norm(g[:2] / 10 + CEN[:2] - run.u.cur_task_info['goal']['block_xyzs'][0][:2]) * 100)
    q = lambda e: [round(float(x), 2) for x in np.quantile(e, [0.5, 0.75, 0.9])]
    res = dict(trials=len(err), live_rest_pose_error_cm_median_p75_p90=q(err), goal_frame_error_cm_median_p75_p90=q(errg))
    print(json.dumps(res))
    json.dump(res, open(OUT / 'readout_live.json', 'w'), indent=1)
    np.save(OUT / 'live_pts.npy', np.array(pts))
    os._exit(0)


if __name__ == '__main__':
    dict(build=build, train=train, live=live)[sys.argv[1]]()
