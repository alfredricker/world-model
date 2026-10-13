"""Card 104 smoke test: does SAM 2.1 segment and follow Crafter's creatures at all, and how fast?

  1. automatic masks on a few frames (day and night) with a creature in view: is each creature covered by a mask of
     about its own size?
  2. video propagation over a short clip, prompted on its first frame by the automatic masks on creatures (an oracle
     prompt, for the smoke test only): does each track stay on its creature, and at how many frames per second?

The view (49 x 63 px, 7 px per tile) is enlarged without smoothing to SAM's 1024 x 1024 input.

  bin/prun python tools/card104/smoke.py [--model t|s|b+|l] [--clip 40]
      → runs/104/smoke_<model>.json and overlays in runs/104/
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
MODEL, CLIP = arg("--model", "t"), int(arg("--clip", 40))
CKPT = {"t": "tiny", "s": "small", "b+": "base_plus", "l": "large"}[MODEL]
OUT = ROOT / "runs" / "104"
HW, WW, UNIT, CX, CY = 49, 63, 7, 31, 24
S = 16                                                  # enlargement for the overlays and SAM's output masks
KINDS = {1: "cow", 2: "zombie", 3: "skeleton", 4: "arrow"}
DEV = torch.device("cuda")


def big(f):
    return np.kron(f, np.ones((S, S, 1), np.uint8))


def small(m):
    """A mask at S x the view's size, back to view pixels by majority."""
    return m.reshape(HW, S, WW, S).mean((1, 3)) > 0.5


def creatures(obj, t):
    return [(int(r[1]), KINDS[int(r[2])], CX + UNIT * int(r[3]), CY + UNIT * int(r[4]))
            for r in obj[obj[:, 0] == t] if int(r[2]) in KINDS and abs(r[3]) <= 4 and abs(r[4]) <= 3]


def covering(masks, x, y):
    """The smallest mask covering the creature's tile centre, with its area in view pixels."""
    best = None
    for j, m in enumerate(masks):
        if m[y, x] and (best is None or m.sum() < best[1]):
            best = (j, int(m.sum()))
    return best


def overlay(f, masks, path):
    rng = np.random.default_rng(0)
    img = big(f).astype(np.float32)
    for m in sorted(masks, key=lambda m: -m.sum()):
        c = rng.integers(0, 255, 3)
        mb = np.kron(m, np.ones((S, S), bool))
        img[mb] = 0.45 * img[mb] + 0.55 * c
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(path)


def frames_tensor(fs, size):
    x = torch.from_numpy(np.stack(fs)).permute(0, 3, 1, 2).float().div(255)
    x = F.interpolate(x, size=(size, size), mode="nearest")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).to(DEV)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    import sam2.sam2_video_predictor as VP
    cfg = f"configs/sam2.1/sam2.1_hiera_{MODEL}.yaml"
    ck = str(ROOT / "runs" / "sam2" / "ckpt" / f"sam2.1_hiera_{CKPT}.pt")
    z = np.load(ROOT / "runs" / "098.5" / "play_w8.npz")
    obs, obj, light, ep, sleep = z["obs"], z["obj"], z["light"], z["ep"], z["sleep"]
    rep = {"model": CKPT}

    # 1. automatic masks on frames with creatures, two by day and two by night
    gen = SAM2AutomaticMaskGenerator(build_sam2(cfg, ck, device=DEV), points_per_side=32, pred_iou_thresh=0.7,
                                     stability_score_thresh=0.85, min_mask_region_area=0)
    with_c = sorted({int(t) for t in obj[np.isin(obj[:, 2], list(KINDS)), 0] if not sleep[int(t)]})
    day = [t for t in with_c if light[t] >= 0.5][::997][:2]
    night = [t for t in with_c if light[t] < 0.5][::331][:2]
    rows, secs = [], []
    for t in day + night:
        f = obs[t, :HW, :WW]
        t0 = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = gen.generate(big(f))
        torch.cuda.synchronize()
        secs.append(time.monotonic() - t0)
        masks = [small(o["segmentation"]) for o in out]
        masks = [m for m in masks if m.any()]
        overlay(f, masks, OUT / f"auto_{CKPT}_t{t}.png")
        Image.fromarray(big(f)).save(OUT / f"frame_t{t}.png")
        for i, k, x, y in creatures(obj, t):
            c = covering(masks, x, y)
            rows.append({"t": t, "night": bool(light[t] < 0.5), "kind": k, "masks in frame": len(masks),
                         "covering mask area (px; a tile is 49)": c[1] if c else None})
    rep["automatic masks"] = {"frames": day + night, "seconds per frame": [round(s, 3) for s in secs],
                              "creatures": rows}

    # 2. propagation over a clip that starts with a creature in view (awake, one episode)
    t0c = next(t for t in with_c[len(with_c) // 3:] if light[t] >= 0.5 and len(creatures(obj, t)) >= 2
               and all(not sleep[u] and ep[u] == ep[t] for u in range(t, t + CLIP)))
    clip = [obs[u, :HW, :WW] for u in range(t0c, t0c + CLIP)]
    pred = build_sam2_video_predictor(cfg, ck, device=DEV)
    imgs = frames_tensor(clip, pred.image_size)
    VP.load_video_frames = lambda **kw: (imgs, HW * S, WW * S)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        first = [small(o["segmentation"]) for o in gen.generate(big(clip[0]))]
        state = pred.init_state(video_path="(in memory)")
        prompts = {}
        for i, k, x, y in creatures(obj, t0c):
            c = covering(first, x, y)
            if c:
                prompts[i] = k
                pred.add_new_mask(state, frame_idx=0, obj_id=i, mask=np.kron(first[c[0]], np.ones((S, S), bool)))
        torch.cuda.synchronize()
        t1 = time.monotonic()
        on, n = {i: 0 for i in prompts}, {i: 0 for i in prompts}
        for fi, ids, logits in pred.propagate_in_video(state):
            t = t0c + fi
            here = {i: (x, y) for i, k, x, y in creatures(obj, t)}
            for oid, lg in zip(ids, logits):
                m = small((lg[0] > 0).cpu().numpy())
                if oid in here:
                    n[oid] += 1
                    x, y = here[oid]
                    on[oid] += bool(m[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4].any())
            if fi in (0, CLIP // 2, CLIP - 1):
                overlay(clip[fi], [small((lg[0] > 0).cpu().numpy()) for lg in logits],
                        OUT / f"track_{CKPT}_t{t}.png")
        torch.cuda.synchronize()
        dt = time.monotonic() - t1
    rep["propagation"] = {"clip start": t0c, "frames": CLIP, "objects prompted": prompts,
                          "frames per second": round(CLIP / dt, 1),
                          "frames with the creature in view, and with its track on it":
                              {f"{prompts[i]} {i}": [n[i], on[i]] for i in prompts}}
    rep["GPU memory peak (GB)"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    (OUT / f"smoke_{CKPT}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
