"""Card 116.1, gate step 3: SAM 2.1-small's automatic masks, the supplied reference (reported, not judged).

On 300 of the truth frames: each 64 x 64 frame is upsampled (bicubic) to 512 x 512, SAM 2's automatic mask generator
(32 x 32 point grid) proposes masks, and masks are painted largest first so smaller ones lie on top; the label image is
downsampled back to 64 x 64 (nearest) and scored like the slots.

  bin/prun python tools/card116/sam2_ref.py → runs/116/sam2_ref.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from metrics import scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/116'
CKPT = Path.home() / '.cache/wm-sam2/sam2/checkpoints/sam2.1_hiera_small.pt'


def main():
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    model = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', str(CKPT), device='cuda')
    gen = SAM2AutomaticMaskGenerator(model, points_per_side=32, pred_iou_thresh=0.7, stability_score_thresh=0.85,
                                     min_mask_region_area=0)
    t = np.load(OUT / 'truth.npz')
    frames, labels = t['frames'][:300], t['labels'][:300]
    obs = np.load(ROOT / 'runs/114/obs_visual-scene-play-v0-val.npy', mmap_mode='r')
    segs, n_masks, t0 = [], [], time.time()
    for f in frames:
        x = torch.from_numpy(np.asarray(obs[f])).permute(2, 0, 1)[None].float()
        big = F.interpolate(x, size=(512, 512), mode='bicubic', align_corners=False).clamp(0, 255)
        img = big[0].permute(1, 2, 0).byte().numpy()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            masks = gen.generate(img)
        lab = np.zeros((512, 512), np.int16)
        for k, m in enumerate(sorted(masks, key=lambda m: -m['area']), start=1):
            lab[m['segmentation']] = k
        segs.append(lab[4::8, 4::8])
        n_masks.append(len(masks))
    res = scores(np.stack(segs), labels)
    res['masks_per_frame'] = float(np.mean(n_masks))
    res['seconds_per_frame'] = round((time.time() - t0) / len(frames), 2)
    print(res)
    json.dump(res, open(OUT / 'sam2_ref.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
