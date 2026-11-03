"""Card 114: train the predictor by teacher-forced squared error, within a fixed time budget.

--input patch: frozen DINOv2 patches of visual-scene-play-v0 (training episodes 0-899, every fifth frame).
--input state: the upper bound, OGBench's 40-number state observation of scene-play-v0 (same episodes and frames),
standardised per number. Windows are H+1 = 4 consecutive model steps inside one episode. The learning rate follows
a cosine over the time budget. A held-out loss (validation episodes) is printed as it goes.

  bin/prun python tools/card114/train.py --input patch|state --minutes 25 → runs/114/pred_<input>.pt
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import H, Predictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/114'
DATA = Path.home() / '.ogbench/data'
EP, SKIP = 1001, 5
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def chunks(actions, frames):
    """The 5 actions after each stored frame, joined into 25 numbers."""
    return np.stack([actions[frames + j] for j in range(SKIP)], 1).reshape(len(frames), 25).astype(np.float32)


def state_stats():
    o = np.load(DATA / 'scene-play-v0.npz')['observations'][:900 * EP]
    mu, sd = o.mean(0), o.std(0) + 1e-3
    return mu.astype(np.float32), sd.astype(np.float32)


def load(which, split):
    """→ features [n, N, D] (memmap or array), the frame numbers, the action chunks after them."""
    if which == 'patch':
        st = np.load(OUT / f"state_{'val' if split == 'val' else 'train'}.npz")
        name = {'train': 'train', 'val': 'val', 'tail': 'tail'}[split]
        feat = np.load(OUT / f'feat_{name}.npy', mmap_mode='r')
        frames = np.load(OUT / f'frames_{name}.npy')
    else:
        st = np.load(OUT / f"sstate_{'val' if split == 'val' else 'train'}.npz")
        o = np.load(DATA / f"scene-play-v0{'-val' if split == 'val' else ''}.npz")['observations']
        mu, sd = state_stats()
        frames = {'train': np.arange(0, 900 * EP, SKIP), 'val': np.arange(len(o)), 'tail': np.arange(900 * EP, 1000 * EP)}[split]
        feat = ((o[frames] - mu) / sd)[:, None, :].astype(np.float32)
    acts = st['actions']
    pad = np.concatenate([acts, np.zeros((SKIP, 5), np.float32)])
    return feat, frames, chunks(pad, frames)


def windows(frames, step):
    """Start positions i such that stored frames i, i+step, ..., i+H*step lie in one episode."""
    i = np.arange(len(frames) - H * step)
    return i[frames[i] // EP == frames[i + H * step] // EP]


def main():
    which = arg('--input', 'patch')
    minutes = float(arg('--minutes', 25))
    dev = 'cuda'
    torch.manual_seed(0)
    feat, frames, acts = load(which, 'train')
    vfeat, vframes, vacts = load(which, 'val')
    N, D = feat.shape[1:]
    model = Predictor(N, D).to(dev)
    print(which, 'parameters', sum(p.numel() for p in model.parameters()) / 1e6, 'M', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    # held-out windows: every fifth validation frame as the model step
    vw = windows(vframes, SKIP)
    vw = vw[np.random.default_rng(0).choice(len(vw), 512, replace=False)]
    vidx = vw[:, None] + SKIP * np.arange(H + 1)
    vz = torch.from_numpy(np.asarray(vfeat[vidx.ravel()]).reshape(512, H + 1, N, D)).float()
    va = torch.from_numpy(vacts[vidx[:, :H].ravel()].reshape(512, H, 25))

    def held_out():
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            tot = 0
            for b in range(0, 512, 64):
                z, a = vz[b:b + 64].to(dev), va[b:b + 64].to(dev)
                tot += ((model(z[:, :H], a).float() - z[:, 1:]) ** 2).mean().item() * len(z)
        model.train()
        return tot / 512

    B = 32 if which == 'patch' else 1024
    CH = 4096 if which == 'patch' else len(frames)
    t0, step, warm = time.time(), 0, 500
    budget = minutes * 60
    starts = np.arange(0, len(frames), CH)
    rng = np.random.default_rng(0)
    done = False
    epoch = 0
    while not done:
        rng.shuffle(starts)
        for s0 in starts:
            sl = slice(s0, min(s0 + CH + H, len(frames)))
            f = torch.from_numpy(np.asarray(feat[sl])).to(dev)
            a = torch.from_numpy(acts[sl]).to(dev)
            w = windows(frames[sl], 1)
            w = torch.from_numpy(w[rng.permutation(len(w))]).to(dev)
            for b in w.split(B):
                frac = (time.time() - t0) / budget
                if frac >= 1:
                    done = True
                    break
                lr = 3e-4 * min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * frac))
                for g in opt.param_groups:
                    g['lr'] = lr
                ix = b[:, None] + torch.arange(H + 1, device=dev)
                z = f[ix].float()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = ((model(z[:, :H], a[ix[:, :H]]).float() - z[:, 1:]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                if step % 500 == 0:
                    print(f'step {step} epoch {epoch} {time.time() - t0:.0f}s lr {lr:.2e} train {loss.item():.4f}'
                          f' held-out {held_out():.4f}', flush=True)
            if done:
                break
        epoch += 1
    print(f'final: step {step}, {epoch} epochs, {time.time() - t0:.0f}s, held-out {held_out():.4f}')
    torch.save(model.state_dict(), OUT / f'pred_{which}.pt')


if __name__ == '__main__':
    main()
