"""Card 145: train the state layer, or one of the gate's decoders.

  --mode state   the state layer (model.py) on segments of --T steps
  --mode truth   the decoder alone, given the true scene state (cube pose 7, two openings, two locks) and the pose
  --mode pose    the decoder alone, given the pose only (the trivial baseline)
  the state layer's reconstruction is scored on a random quarter of the patches per step (memory)
  --lam λ        cost of an open gate;  --nopose 1  the decoder is not given the pose (arm C);  --S state size
  --minutes m    training time;  --seed;  --out name → runs/145/<name>.pt, <name>.log.json
  --carry 1      each row of a batch walks through one whole episode in chunks of T, the state carried over, so a
                 gate is free only at an episode's first step (without it: at every segment's first step)
  --init name    start from runs/145/<name>.pt
  --free m --ramp r   no cost for the first m minutes, then λ rises from 0 over r minutes (learn first, then make it sparse)
  --resume 1     continue from <name>.part.pt, saved every 5 minutes (a GPU fault stopped the first 30-minute run)

  bin/prun python tools/card145/train.py --mode state --lam 0.001 --minutes 9 --out pilot_1e-3
"""
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import Decoder, State, open_count  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / 'runs/145'
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def norms():
    """Mean frame per patch, one overall scale, and the pose's mean and spread; from 2,000 training steps."""
    f = OUT / 'norm.npz'
    if not f.exists():
        X, M = np.load(OUT / 'feat_train.npy', mmap_mode='r'), np.load(OUT / 'meta_train.npz')
        rng = np.random.default_rng(0)
        e, t = rng.integers(0, X.shape[0], 2000), rng.integers(0, X.shape[1], 2000)
        o = np.lexsort((t, e))
        S = np.stack([np.asarray(X[a, b], np.float32) for a, b in zip(e[o], t[o])])
        mean = S.mean(0)
        scale = float(np.sqrt(((S - mean) ** 2).sum(-1).mean()))             # a patch's deviation from its mean has length 1 on average
        P = M['pose'].reshape(-1, 12)
        np.savez(f, mean=mean, scale=scale, pmean=P.mean(0), pstd=P.std(0) + 1e-6)
    return np.load(f)


def true_state(M):
    s = np.concatenate([M['cube'], M['drawer'][..., None], M['window'][..., None], M['locks']], -1).astype(np.float32)   # [E, 201, 11]
    return (s - s.reshape(-1, 11).mean(0)) / (s.reshape(-1, 11).std(0) + 1e-6)


def main():
    mode, lam, minutes, seed = arg('--mode', 'state'), float(arg('--lam', 1e-3)), float(arg('--minutes', 9)), int(arg('--seed', 0))
    T, B, S, name = int(arg('--T', 32)), int(arg('--B', 16)), int(arg('--S', 32)), arg('--out', 'smoke')
    nz = norms()
    X, M = np.load(OUT / 'feat_train.npy', mmap_mode='r'), np.load(OUT / 'meta_train.npz')
    E, L = X.shape[:2]
    pose = ((M['pose'] - nz['pmean']) / nz['pstd']).astype(np.float32)
    truth = true_state(M)
    mean, scale = torch.from_numpy(nz['mean']).cuda(), float(nz['scale'])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if mode == 'state':
        model = State(S, use_pose=not int(arg('--nopose', 0))).cuda()
    else:
        model = Decoder(11).cuda()
    if arg('--init', ''):                                                  # start from another run's weights (learn first, then make it sparse)
        model.load_state_dict(torch.load(OUT / f"{arg('--init', '')}.pt")['state'])
    ramp = float(arg('--ramp', 0)) * 60                                    # λ rises linearly from 0 over this many minutes
    free = float(arg('--free', 0)) * 60                                    # after this many minutes with no cost at all
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)

    carry = int(arg('--carry', 0))
    slot = [None] * B                                                      # --carry 1: each row of the batch walks through one episode, chunk after chunk

    def batch():
        if mode == 'state' and carry:
            first = np.zeros(B, bool)
            for i in range(B):
                if slot[i] is None or slot[i][1] + T > L:
                    slot[i], first[i] = [int(rng.integers(0, E)), int(rng.integers(0, (L % T) + 1))], True
            out = (np.stack([X[e, s:s + T] for e, s in slot]), np.stack([pose[e, s:s + T] for e, s in slot]), first)
            for i in range(B):
                slot[i][1] += T
            return out
        if mode == 'state':
            e, s = rng.integers(0, E, B), rng.integers(0, L - T + 1, B)
            o = np.argsort(e)
            return (np.stack([X[a, b:b + T] for a, b in zip(e[o], s[o])]), np.stack([pose[a, b:b + T] for a, b in zip(e[o], s[o])]), None)
        e, s = rng.integers(0, E, 128), rng.integers(0, L, 128)
        o = np.lexsort((s, e))
        return np.stack([X[a, b] for a, b in zip(e[o], s[o])]), pose[e[o], s[o]], truth[e[o], s[o]]
    q = queue.Queue(6)

    def feed():
        while True:
            q.put(batch())
    threading.Thread(target=feed, daemon=True).start()
    t0, step, log, done, last, zc = time.time(), 0, [], 0.0, time.time(), None
    budget = minutes * 60
    part = OUT / f'{name}.part.pt'                                         # a checkpoint every 5 minutes; --resume 1 continues from it
    if int(arg('--resume', 0)) and part.exists():
        ck = torch.load(part)
        model.load_state_dict(ck['state']); opt.load_state_dict(ck['opt'])
        step, log, done = ck['step'], ck['log'], ck['seconds']
        print('resumed at step', step, 'after', round(done), 's', flush=True)
    while done + time.time() - t0 < budget:
        frac = (done + time.time() - t0) / budget
        if time.time() - last > 300:
            torch.save(dict(state=model.state_dict(), opt=opt.state_dict(), step=step, log=log, seconds=done + time.time() - t0), part)
            last = time.time()
        for g in opt.param_groups:
            g['lr'] = 3e-4 * min(1.0, (step + 1) / 300) * (0.1 + 0.9 * (1 - frac))
        x, p, tr = q.get()
        x = (torch.from_numpy(x).cuda().float() - mean) / scale
        p = torch.from_numpy(p).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if mode == 'state':
                idx = torch.rand(B * T, 1024, device='cuda').topk(256, dim=1).indices      # a random quarter of the patches is scored
                if carry:
                    fm = torch.from_numpy(tr).cuda()
                    zin = torch.where(fm[:, None], torch.zeros_like(zc), zc) if zc is not None else None
                    rec, zs, gs = model(x, p, z=zin, idx=idx, first=fm if zin is not None else None)
                    zc = zs[:, -1].detach().float()
                    paid = torch.ones(B, T, device='cuda'); paid[:, 0] = (~fm).float() if zin is not None else 0.0
                else:
                    rec, zs, gs = model(x, p, idx=idx)
                    paid = torch.ones(B, T, device='cuda'); paid[:, 0] = 0.0
                tgt = x.view(B * T, 1024, -1).gather(1, idx[..., None].expand(-1, -1, x.shape[-1])).view(B, T, 256, -1)
                err = ((rec.float() - tgt) ** 2).sum(-1).mean()
                n_open = (open_count(gs.float()) * paid).sum() / paid.sum()
                el = done + time.time() - t0 - free
                lam_now = 0.0 if el < 0 else lam * min(1.0, el / ramp) if ramp else lam
                loss = err + lam_now * n_open
            else:
                st = torch.from_numpy(tr).cuda() * float(mode == 'truth')
                rec = model(st, p)
                err = ((rec.float() - x) ** 2).sum(-1).mean()
                loss, n_open, gs = err, torch.zeros(()), None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            r = dict(step=step, s=round(done + time.time() - t0), err=round(err.item(), 4), open_per_step=round(float(n_open), 3))
            if gs is not None:
                r['steps_with_a_gate_open'] = round(float((((gs > 0).any(-1).float()) * paid).sum() / paid.sum()), 3)
            log.append(r)
            if step % 500 == 0:
                print(json.dumps(r), flush=True)
                json.dump(log, open(OUT / f'{name}.log.json', 'w'))
    torch.save(dict(state=model.state_dict(), mode=mode, S=S, lam=lam, nopose=int(arg('--nopose', 0)), steps=step), OUT / f'{name}.pt')
    json.dump(log, open(OUT / f'{name}.log.json', 'w'))
    print('done', step, 'steps', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
