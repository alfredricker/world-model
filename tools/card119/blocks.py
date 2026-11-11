"""Card 119, main run step 1: the lasting-change signal on the outputs of the frozen encoder's blocks 2, 4, 8, 12.

Frames are encoded as in card 114 (64 → 196 px, DINOv2 ViT-S/14); each tapped block's 196 patch tokens pass through
the encoder's final norm (block 12 then equals card 114's stored features). The signal is computed per episode and
only its numerator and denominator are stored.

  bin/prun python tools/card119/blocks.py → runs/119/signal_b<k>_<set>.npz for k in 2, 4, 8, 12 and set in tail, val
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card114'))
from encode import MEAN, RES, STD, load_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/119'
EP, W, G = 1001, 10, 5
LO, HI = G + W, EP - 1 - G - W
BLOCKS = [2, 4, 8, 12]
DEV = 'cuda'


def lasting_ep(x):
    """x [1001, 196, d] → num, den [HI - LO + 1, 196]."""
    tau = torch.arange(LO, HI + 1, device=x.device)
    x = x.double()
    S = torch.cat([torch.zeros_like(x[:1]), x.cumsum(0)])
    Q = torch.cat([torch.zeros_like(x[:1, :, 0]), (x * x).sum(-1).cumsum(0)])
    mb = (S[tau - G] - S[tau - G - W]) / W
    ma = (S[tau + G + W + 1] - S[tau + G + 1]) / W
    vb = (Q[tau - G] - Q[tau - G - W]) / W - (mb * mb).sum(-1)
    va = (Q[tau + G + W + 1] - Q[tau + G + 1]) / W - (ma * ma).sum(-1)
    return ((ma - mb) ** 2).sum(-1).float().cpu().numpy(), (va + vb).clamp(min=0).float().cpu().numpy()


@torch.no_grad()
def encode(enc, frames):
    outs = [[] for _ in BLOCKS]
    for i in range(0, len(frames), 512):
        x = torch.from_numpy(np.ascontiguousarray(frames[i:i + 512])).to(DEV).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(RES, RES), mode='bilinear', align_corners=False)
        x = (x - MEAN.to(DEV)) / STD.to(DEV)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            toks = enc.get_intermediate_layers(x, n=[b - 1 for b in BLOCKS], norm=True)
        for o, t in zip(outs, toks):
            o.append(t.float())
    return [torch.cat(o) for o in outs]


def main():
    enc = load_encoder(DEV)
    t0 = time.time()
    for name, fname, cut in [('tail', 'visual-scene-play-v0', 900 * EP), ('val', 'visual-scene-play-v0-val', 0)]:
        obs = np.load(ROOT / f'runs/114/obs_{fname}.npy', mmap_mode='r')[cut:cut + 100 * EP]
        n = len(obs)
        num = {b: np.full((n, 196), np.nan, np.float32) for b in BLOCKS}
        den = {b: np.full((n, 196), np.nan, np.float32) for b in BLOCKS}
        for e in range(n // EP):
            for b, x in zip(BLOCKS, encode(enc, obs[e * EP:(e + 1) * EP])):
                sl = slice(e * EP + LO, e * EP + HI + 1)
                num[b][sl], den[b][sl] = lasting_ep(x)
            if e % 20 == 0:
                print(name, e, f'{time.time() - t0:.0f}s', flush=True)
        share = np.load(OUT / f'signal_{name}.npz')['share']
        for b in BLOCKS:
            np.savez(OUT / f'signal_b{b}_{name}.npz', num=num[b], den=den[b], share=share)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
