"""Card 114, gate step 2: can the frozen encoder see the lock? And the read-out used later for scoring.

One linear layer from all 196 x 384 patch numbers to the evaluator's state: both button states (logistic), the
drawer's and window's slide positions and the cube's position (squared error, standardised). Trained on training
episodes 0-899 (every fifth frame), tested on the validation episodes (every fifth frame). A measuring device only.

  bin/prun python tools/card114/probes.py → runs/114/probe.pt and the gate numbers
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/114'
NAMES = ['button0', 'button1', 'drawer', 'window', 'cube_x', 'cube_y', 'cube_z']


def targets(state, frames):
    s = state
    return np.concatenate([s['buttons'][frames].astype(np.float32), s['drawer'][frames, None], s['window'][frames, None],
                           s['cube'][frames]], 1)


class Probe(torch.nn.Module):
    def __init__(self, mu=None, sd=None):
        super().__init__()
        self.lin = torch.nn.Linear(196 * 384, len(NAMES))
        self.register_buffer('mu', torch.zeros(len(NAMES)) if mu is None else mu)
        self.register_buffer('sd', torch.ones(len(NAMES)) if sd is None else sd)

    def forward(self, f):                       # f: [..., 196, 384] → raw outputs (logits for buttons, z for rest)
        return self.lin(f.flatten(-2).float())

    def read(self, f):                          # evaluator units: button probabilities, positions in metres
        z = self(f)
        return torch.cat([torch.sigmoid(z[..., :2]), z[..., 2:] * self.sd[2:] + self.mu[2:]], -1)


def load(device='cuda'):
    p = Probe()
    p.load_state_dict(torch.load(OUT / 'probe.pt', map_location=device))
    return p.to(device).eval()


def main():
    torch.manual_seed(0)
    dev = 'cuda'
    st = np.load(OUT / 'state_train.npz')
    sv = np.load(OUT / 'state_val.npz')
    Ftr = np.load(OUT / 'feat_train.npy', mmap_mode='r')
    Fva = np.load(OUT / 'feat_val.npy', mmap_mode='r')
    ytr = targets(st, np.load(OUT / 'frames_train.npy'))
    fva = np.arange(0, len(Fva), 5)
    yva = targets(sv, fva)
    mu, sd = torch.tensor(ytr.mean(0)), torch.tensor(ytr.std(0))
    mu[:2], sd[:2] = 0, 1
    probe = Probe(mu, sd).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-4, weight_decay=1e-2)
    chunk, B, epochs = 8192, 512, 3
    starts = np.arange(0, len(Ftr), chunk)
    ytr_t = torch.tensor(ytr)
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(starts)
        tot = 0
        for s0 in starts:
            f = torch.from_numpy(np.asarray(Ftr[s0:s0 + chunk])).to(dev)
            y = ytr_t[s0:s0 + chunk].to(dev)
            for b in torch.randperm(len(f), device=dev).split(B):
                z = probe(f[b])
                yz = (y[b] - probe.mu) / probe.sd
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z[:, :2], y[b, :2]) + \
                    ((z[:, 2:] - yz[:, 2:]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
        print(f'epoch {ep} loss {tot / len(Ftr):.4f}', flush=True)
    torch.save(probe.state_dict(), OUT / 'probe.pt')
    probe.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(fva), 4096):
            preds.append(probe.read(torch.from_numpy(np.asarray(Fva[fva[i:i + 4096]])).to(dev)).cpu().numpy())
    p = np.concatenate(preds)
    res = {}
    for j, n in enumerate(NAMES):
        if j < 2:
            acc = float(((p[:, j] > 0.5) == (yva[:, j] > 0.5)).mean())
            maj = float(max(yva[:, j].mean(), 1 - yva[:, j].mean()))
            res[n] = dict(accuracy=round(acc, 4), majority=round(maj, 4))
        else:
            r2 = 1 - ((p[:, j] - yva[:, j]) ** 2).mean() / yva[:, j].var()
            res[n] = dict(r2=round(float(r2), 4), rmse_cm=round(float(np.sqrt(((p[:, j] - yva[:, j]) ** 2).mean()) * 100), 2))
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / 'probe_gate.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
