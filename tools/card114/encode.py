"""Card 114: frozen DINOv2 ViT-S/14 patch features of visual-scene-play-v0 frames.

Each 64x64 frame is upsampled (bilinear) to 196x196, normalised with ImageNet statistics, and passed through the
frozen encoder; the 196 normalised patch tokens (14 x 14, 384 numbers each) are stored as fp16 (DINO-WM's
representation). Training episodes 0-899: every fifth frame (the model step). Held-out frames (validation, and
training episodes 900-999): every frame, so a rollout can start anywhere.

  bin/prun python tools/card114/encode.py --set train|val|tail [--limit N]
  → runs/114/obs_<file>.npy (uncompressed frames, once) and runs/114/feat_<set>.npy + frames_<set>.npy
"""
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
OUT = ROOT / 'runs/114'
EP, SKIP, RES = 1001, 5, 196
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def frames_file(name):
    """Stream the npz member 'observations' to an uncompressed .npy once, so it can be memory-mapped."""
    path = OUT / f'obs_{name}.npy'
    if path.exists():
        return np.load(path, mmap_mode='r')
    with zipfile.ZipFile(DATA / f'{name}.npz') as z, z.open('observations.npy') as f:
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(f)
        out = np.lib.format.open_memmap(path.with_suffix('.tmp.npy'), 'w+', dtype, shape)
        per = int(np.prod(shape[1:]))
        for i in range(0, shape[0], 20000):
            n = min(20000, shape[0] - i)
            out[i:i + n] = np.frombuffer(f.read(n * per), dtype).reshape(n, *shape[1:])
        out.flush()
        del out
    path.with_suffix('.tmp.npy').rename(path)
    return np.load(path, mmap_mode='r')


def load_encoder(device):
    hub = Path.home() / '.cache/torch/hub/facebookresearch_dinov2_main'
    enc = torch.hub.load(str(hub), 'dinov2_vits14', source='local', pretrained=False)
    sd = torch.load(Path.home() / '.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth', map_location='cpu')
    enc.load_state_dict(sd)
    return enc.eval().to(device)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def encode(enc, frames_uint8, device):
    x = torch.from_numpy(np.ascontiguousarray(frames_uint8)).to(device).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(RES, RES), mode='bilinear', align_corners=False)
    x = (x - MEAN.to(device)) / STD.to(device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        tok = enc.forward_features(x)['x_norm_patchtokens']
    return tok.float().half().cpu().numpy()


def main():
    which = arg('--set', 'val')
    limit = int(arg('--limit', 0))
    OUT.mkdir(parents=True, exist_ok=True)
    if which == 'val':
        obs = frames_file('visual-scene-play-v0-val')
        idx = np.arange(len(obs))
    else:
        obs = frames_file('visual-scene-play-v0')
        if which == 'train':
            idx = np.arange(0, 900 * EP, SKIP)
        else:
            idx = np.arange(900 * EP, 1000 * EP)
    if limit:
        idx = idx[:limit]
    device = 'cuda'
    enc = load_encoder(device)
    feat = np.lib.format.open_memmap(OUT / f'feat_{which}.npy', 'w+', np.float16, (len(idx), 196, 384))
    t0, B = time.time(), 512
    for i in range(0, len(idx), B):
        feat[i:i + B] = encode(enc, obs[idx[i:i + B]], device)
        if i % (B * 40) == 0:
            print(f'{which} {i}/{len(idx)} {time.time() - t0:.0f}s', flush=True)
    feat.flush()
    np.save(OUT / f'frames_{which}.npy', idx)
    print(f'{which}: {len(idx)} frames in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
