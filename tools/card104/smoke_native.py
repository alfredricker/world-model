"""Card 104 smoke test, part 2: can SAM 2.1 give a Crafter creature one mask of its own?

Scored against each creature's exact outline (its sprite's alpha, as Crafter draws it), at Crafter's native 16 px per
tile and at the stored 7 px, by intersection over union (IoU):
  - "everything" mode: the best IoU among the automatic masks;
  - a point on the creature (an oracle point, the smoke test only): the best IoU among SAM's three answers, and the IoU
    of the answer SAM scores highest. If pointing works, discovery only has to say where to point.

Frames come from a fresh random-play world (seed 2100, no card's data). The 16 px render's night noise draws from its
own stream, so the world's random stream is the one the 64 px observation uses.

  bin/prun python tools/card104/smoke_native.py [--model t|s|b+|l]  → runs/104/native_<model>.json, overlays
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
MODEL = arg("--model", "t")
CKPT = {"t": "tiny", "s": "small", "b+": "base_plus", "l": "large"}[MODEL]
OUT = ROOT / "runs" / "104"
KINDS = ("Cow", "Zombie", "Skeleton", "Arrow")
DEV = torch.device("cuda")


def outline(env, o, p, unit):
    """The creature's pixels in the view (rows, columns), from its sprite's alpha at this unit."""
    tex = env._textures.get(o.texture, (unit, unit))
    m = np.zeros((7 * unit, 9 * unit), bool)
    x, y = (o.pos[0] - p.pos[0] + 4) * unit, (o.pos[1] - p.pos[1] + 3) * unit
    m[y:y + unit, x:x + unit] = (tex[..., 3] > 127).T
    return m


def frames(n_day=6, n_night=6, seed=2100):
    import crafter
    env = crafter.Env(seed=seed, size=(64, 64))
    rng, own = np.random.default_rng(seed), np.random.RandomState(seed + 1)
    obs, day, night, t = env.reset(), [], [], 0
    while (len(day) < n_day or len(night) < n_night) and t < 100000:
        p, dl = env._player, float(env._world.daylight)
        cre = [o for o in env._world.objects if type(o).__name__ in KINDS
               and abs(o.pos[0] - p.pos[0]) <= 4 and abs(o.pos[1] - p.pos[1]) <= 3]
        bucket = night if dl < 0.35 else day if dl > 0.8 else None
        if cre and not p.sleeping and bucket is not None and len(bucket) < (n_night if bucket is night else n_day) \
                and t % 23 == 0:
            wr, env._world.random = env._world.random, own
            hi = env.render((144, 144))
            env._world.random = wr
            bucket.append({"t": t, "daylight": round(dl, 2), "lo": obs[:49, :63], "hi": hi[:112, :144],
                           "cre": [(type(o).__name__, outline(env, o, p, 7), outline(env, o, p, 16)) for o in cre]})
        obs, _, done, _ = env.step(int(rng.integers(17)))
        t += 1
        if done:
            obs = env.reset()
    return day + night


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(f"configs/sam2.1/sam2.1_hiera_{MODEL}.yaml",
                       str(ROOT / "runs" / "sam2" / "ckpt" / f"sam2.1_hiera_{CKPT}.pt"), device=DEV)
    gen = SAM2AutomaticMaskGenerator(model)
    pred = SAM2ImagePredictor(model)
    fr = frames()
    rep = {"model": CKPT, "frames (t, daylight, creatures)":
           [(f["t"], f["daylight"], [k for k, *_ in f["cre"]]) for f in fr]}
    for res, unit, scale, key, ci in (("7 px", 7, 16, "lo", 1), ("16 px", 16, 7, "hi", 2)):
        rows, secs = [], []
        for f in fr:
            img = np.kron(f[key], np.ones((scale, scale, 1), np.uint8))
            t0 = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                auto = [o["segmentation"] for o in gen.generate(img)]
                torch.cuda.synchronize()
                secs.append(time.monotonic() - t0)
                pred.set_image(img)
                for c in f["cre"]:
                    gt = np.kron(c[ci], np.ones((scale, scale), bool))
                    ys, xs = np.nonzero(gt)
                    j = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)    # the outline's pixel nearest its centre
                    ms, sc, _ = pred.predict(point_coords=np.array([[xs[j], ys[j]]]), point_labels=np.array([1]),
                                             multimask_output=True)
                    ms = ms > 0.5
                    rows.append({"t": f["t"], "night": f["daylight"] < 0.35, "kind": c[0],
                                 "everything: best IoU": round(max((iou(m, gt) for m in auto), default=0), 3),
                                 "point: best of 3": round(max(iou(m, gt) for m in ms), 3),
                                 "point: SAM's top": round(iou(ms[int(np.argmax(sc))], gt), 3)})
            if f is fr[0] or f is fr[-1]:
                rgb = img.astype(np.float32)
                r = np.random.default_rng(0)
                for m in sorted(auto, key=lambda m: -m.sum()):
                    rgb[m] = 0.45 * rgb[m] + 0.55 * r.integers(0, 255, 3)
                Image.fromarray(rgb.astype(np.uint8)).save(OUT / f"native_{CKPT}_{unit}px_t{f['t']}.png")
                Image.fromarray(img).save(OUT / f"native_frame_{unit}px_t{f['t']}.png")
        summ = {}
        for part, sel in (("day", lambda r: not r["night"]), ("night", lambda r: r["night"])):
            for k in KINDS:
                rs = [r for r in rows if sel(r) and r["kind"] == k]
                if rs:
                    summ[f"{part} {k} (n={len(rs)})"] = {m: round(float(np.median([r[m] for r in rs])), 2) for m in
                                                         ("everything: best IoU", "point: best of 3", "point: SAM's top")}
        rep[res] = {"everything mode: seconds per frame": round(float(np.mean(secs)), 3),
                    "median IoU by kind": summ, "creatures": rows}
    (OUT / f"native_{CKPT}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"model": CKPT, **{r: {k: v for k, v in rep[r].items() if k != "creatures"}
                                        for r in ("7 px", "16 px")}}, indent=1))


if __name__ == "__main__":
    main()
