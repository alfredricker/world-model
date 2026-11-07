"""Card 116.1: frozen DINOv2 ViT-S/14 patch features at 448 x 448 input (32 x 32 = 1,024 patches of 384 numbers, fp16).

--set train: every fifth frame of training episodes 0-449 (90,090 frames).
--set val:   the validation frames card 116.1 scores: the 2,000 truth frames with 4 model steps of history each, and the
             100 clips with a 4-step lead-in.
--part i/n:  encode only part i of n of the frames (each part its own capped run); parts write into one memmap.
--src rgb256: read card 116.3's 256-px renders (runs/116/rgb256_<set>.npy) instead of the dataset's 64-px frames
             → runs/116/feat256_<set>.npy.

  bin/prun python tools/card116/encode448.py --set train|val [--part 0/2] → runs/116/feat448_<set>.npy, frames448_<set>.npy
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card114'))
from encode import MEAN, STD, frames_file, load_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/116'
EP, SKIP, RES = 1001, 5, 448
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


@torch.no_grad()
def encode(enc, frames_uint8, device):
    x = torch.from_numpy(np.ascontiguousarray(frames_uint8)).to(device).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(RES, RES), mode='bilinear', align_corners=False)
    x = (x - MEAN.to(device)) / STD.to(device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        tok = enc.forward_features(x)['x_norm_patchtokens']
    return tok.float().half().cpu().numpy()


def frame_list(which):
    if which == 'train':
        return np.arange(0, 450 * EP, SKIP)
    t = np.load(OUT / 'truth.npz')
    need = [t['frames'][:, None] - SKIP * np.arange(4, -1, -1)]
    s = t['clips'][:, :1]
    need.append((s - SKIP * np.arange(4, 0, -1)).reshape(len(s), -1))
    need.append(t['clips'])
    return np.unique(np.concatenate([a.ravel() for a in need]))


def main():
    which = arg('--set', 'val')
    part, nparts = (int(x) for x in arg('--part', '0/1').split('/'))
    idx = frame_list(which)
    src = arg('--src', 'dataset')
    if src == 'rgb256':
        rgb = np.load(OUT / f'rgb256_{which}.npy', mmap_mode='r')
        obs = type('ByRow', (), {'__getitem__': lambda self, fr: rgb[np.searchsorted(idx, fr)]})()
    else:
        obs = frames_file('visual-scene-play-v0' + ('-val' if which == 'val' else ''))
    path = OUT / (f'feat256_{which}.npy' if src == 'rgb256' else f'feat448_{which}.npy')
    if path.exists():
        feat = np.load(path, mmap_mode='r+')
    else:
        feat = np.lib.format.open_memmap(path, 'w+', np.float16, (len(idx), 1024, 384))
    if src != 'rgb256':
        np.save(OUT / f'frames448_{which}.npy', idx)
    lo, hi = len(idx) * part // nparts, len(idx) * (part + 1) // nparts
    enc = load_encoder('cuda')
    t0, B = time.time(), 128
    for i in range(lo, hi, B):
        j = min(i + B, hi)
        feat[i:j] = encode(enc, obs[idx[i:j]], 'cuda')
        if (i - lo) % (B * 100) == 0:
            print(f'{which} part {part}: {i - lo}/{hi - lo} {time.time() - t0:.0f}s', flush=True)
    feat.flush()
    print(f'{which} part {part}/{nparts}: {hi - lo} frames in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
