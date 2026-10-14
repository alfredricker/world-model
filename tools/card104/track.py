"""Card 104: SAM 2.1 for correspondence. Each tracked thing is SAM 2.1's mask, carried from frame to frame by SAM's
own memory (its per-object track step over the last frames and the prompted one), online: objects are added as they
are found and memory older than SAM's window is pruned, so GPU memory stays bounded over any number of frames.

Discovery (mode "discover"): where the 16 px frame changed in the world's frame (the last frame moved by the view's
motion, card 103.4's origins), outside the camera's pixels (card 103.1, enlarged) and the masks already tracked, SAM
is pointed at each changed piece's pixel nearest its centre and its top-scored mask taken. A candidate is confirmed
at birth when its content is found in the last frame at a displacement whose source has since changed (the pair test,
card 103.5), or later when it moves against the world; a tentative one not confirmed within 3 steps is dropped.
Mode "oracle" (the gate only): SAM is pointed at each creature's tile centre the first frame it is in view, using the
evaluator's positions; this measures SAM's own correspondence with discovery perfect.

Reported: confirmed tracks whose mask is present (object score > 0, mask not empty), at the mask's centroid, as tile
offsets from the agent for card 098.5's IDF1 (scoring only).

  bin/prun python tools/card104/track.py --worker W --mode discover|oracle --origins 1034|true [--n N] [--model t]
      → runs/104/track_<mode>_<origins>_w<W>[_n<N>].json
"""
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
W, MODE, ORIG = int(arg("--worker", 10)), arg("--mode", "discover"), arg("--origins", "1034")
N_LIM, MODEL = int(arg("--n", 0)) or None, arg("--model", "t")
START = int(arg("--start", 0))
CKPT = {"t": "tiny", "s": "small", "b+": "base_plus", "l": "large"}[MODEL]
OUT = ROOT / "runs" / "104"
DEV = torch.device("cuda")
H16, W16, U16 = 112, 144, 16                           # the 16 px view; one agent step is 16 px here
CX16, CY16 = 4 * U16 + U16 // 2, 3 * U16 + U16 // 2    # the agent's tile centre (scoring only)
MIN_PIX, MAX_PIX = 30, 640                            # a piece worth pointing at; a mask larger than 2.5 steps^2 is ground
WINDOW = 24                                           # the pair test's window: 1.5 of the agent's own steps
TENTATIVE, GONE, KEEP = 3, 20, 15                     # steps to confirm; steps absent before a track ends; memory kept
MAX_TRACKS, MAX_NEW = 24, 6
OVERLAP = arg("--overlap", "dup")                      # overlapping tracks: the younger ends (dup) or pixels by logit
RESIZE = arg("--resize", "nearest")                    # upscaling to SAM's 1024: nearest or bicubic (SAM's own loader)
PROMPT = arg("--prompt", "point")                      # a point (the default) or a box around the piece
SELECT = "--select" in sys.argv                         # SAMURAI's memory selection (Yang et al. 2024)
GATE = float(arg("--gate", 0.5))                        # appearance gate: L1 distance of exposure-normalised mean colour

# 103.5's files module, for its origins and its extra measures (its argv parsed at import)
argv = sys.argv
sys.argv = ["files.py", "--origins", ORIG, "--workers", str(W)]
spec = importlib.util.spec_from_file_location("files1035", ROOT / "tools" / "card103.5" / "files.py")
FL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FL)
sys.argv = argv
TR = FL.TR


def dilate(m, r):
    t = torch.as_tensor(m, dtype=torch.float32)[None, None]
    return (F.max_pool2d(t, 2 * r + 1, 1, r)[0, 0] > 0).numpy()


def erode(m, r):
    return ~dilate(~m, r)


def components(m):
    """Connected pieces (4-neighbour) of a boolean mask, as lists of (y, x)."""
    seen = np.zeros_like(m)
    out = []
    for y0, x0 in zip(*np.nonzero(m)):
        if seen[y0, x0]:
            continue
        stack, piece = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            piece.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] and m[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        out.append(np.array(piece))
    return out


def shifted(prev, dy, dx):
    """prev moved so that out[y, x] = prev[y + dy, x + dx]; NaN outside."""
    out = np.full(prev.shape, np.nan, np.float32)
    ys, yd = slice(max(0, -dy), min(H16, H16 - dy)), slice(max(0, dy), min(H16, H16 + dy))
    xs, xd = slice(max(0, -dx), min(W16, W16 - dx)), slice(max(0, dx), min(W16, W16 + dx))
    out[ys, xs] = prev[yd, xd]
    return out


class Track:
    def __init__(self, tid, k, out, mask, confirmed, wpos, col):
        self.id, self.born, self.mask, self.confirmed, self.col = tid, k, mask, confirmed, col
        self.od = {"cond_frame_outputs": {k: out}, "non_cond_frame_outputs": {}}
        self.wpos0, self.absent, self.big, self.mem = wpos, 0, 0, []


class Tracker:
    def __init__(self):
        from sam2.build_sam import build_sam2_video_predictor
        self.m = build_sam2_video_predictor(f"configs/sam2.1/sam2.1_hiera_{MODEL}.yaml",
                                            str(ROOT / "runs" / "sam2" / "ckpt" / f"sam2.1_hiera_{CKPT}.pt"), device=DEV)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
        self.stats = defaultdict(int)

    def features(self, f):
        x = torch.as_tensor(f, device=DEV).permute(2, 0, 1)[None].float() / 255
        x = (F.interpolate(x, size=(self.m.image_size,) * 2, mode=RESIZE, **({} if RESIZE == "nearest" else
                                                                            {"align_corners": False})).clamp(0, 1)
             - self.mean) / self.std
        _, vf, vpe, fs = self.m._prepare_backbone_features(self.m.forward_image(x))
        return vf, vpe, fs

    def step(self, feats, k, od, init, point=None):
        vf, vpe, fs = feats
        pin = None
        if point is not None and len(point) == 4:      # a box (y0, x0, y1, x1): SAM's corner labels 2 and 3
            sy, sx = self.m.image_size / H16, self.m.image_size / W16
            pin = {"point_coords": torch.tensor([[[point[1] * sx, point[0] * sy], [point[3] * sx, point[2] * sy]]],
                                                device=DEV, dtype=torch.float32),
                   "point_labels": torch.tensor([[2, 3]], device=DEV, dtype=torch.int32)}
        elif point is not None:
            pin = {"point_coords": torch.tensor([[[point[1] * self.m.image_size / W16, point[0] * self.m.image_size / H16]]],
                                                device=DEV, dtype=torch.float32),
                   "point_labels": torch.ones(1, 1, device=DEV, dtype=torch.int32)}
        # SAM 2's track step, written out to keep the chosen mask's predicted IoU (SAMURAI's mask affinity)
        out, so, _, _ = self.m._track_step(k, init, vf, vpe, fs, pin, None, od, 10 ** 7, False, None)
        _, _, ious, low, high, ptr, osl = so
        out.update(pred_masks=low, pred_masks_high_res=high, obj_ptr=ptr, object_score_logits=osl)
        self.m._encode_memory_in_output(vf, fs, pin, True, high, osl, out)
        self.last_iou = float(ious.float().max())
        compact = {"maskmem_features": out["maskmem_features"].to(torch.bfloat16), "maskmem_pos_enc": out["maskmem_pos_enc"],
                   "pred_masks": out["pred_masks"], "obj_ptr": out["obj_ptr"],
                   "object_score_logits": out["object_score_logits"]}
        logit = F.interpolate(out["pred_masks"].float(), size=(H16, W16), mode="bilinear", align_corners=False)[0, 0]
        present = bool(out["object_score_logits"].float().item() > 0)
        if not present:
            logit = torch.full_like(logit, -1e4)
        self.last_logit = logit.cpu().numpy()
        return compact, self.last_logit > 0


_SPRITE = {}
AREAS = []
STATES = []


def sprite_point(kind, fx, fy):
    """Oracle only: the sprite's opaque pixel nearest its opaque centre (row, column within its 16 px tile), from
    Crafter's own sprite alpha, as the smoke test pointed (a tile's centre can fall on ground: a cow covers 39%)."""
    import importlib.metadata
    from PIL import Image
    key = (kind, fx, fy)
    if key not in _SPRITE:
        assets = Path(importlib.metadata.distribution("crafter").locate_file("crafter")) / "assets"
        name = {1: "cow", 2: "zombie", 3: "skeleton"}.get(kind) or "arrow-" + {(-1, 0): "left", (1, 0): "right",
                                                                               (0, -1): "up", (0, 1): "down"}.get((fx, fy), "up")
        a = np.array(Image.open(assets / f"{name}.png").convert("RGBA").resize((U16, U16), Image.NEAREST))[..., 3] > 127
        ys, xs = np.nonzero(a)
        j = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
        _SPRITE[key] = (int(ys[j]), int(xs[j]))
    return _SPRITE[key]


def colour(fn, mask):
    """The mask's mean colour, divided by the frame's mean brightness (exposure, card 103.1)."""
    return fn[mask].mean(0) if mask.any() else None


def pair_test(fk, prev_al, prev, mask, dy0, dx0):
    """Its content found in the last frame at a displacement whose source has since changed: it moved by itself."""
    ys, xs = np.nonzero(mask)
    cur = fk[ys, xs].astype(np.float32)
    best, err0 = None, None
    for dy in range(-WINDOW, WINDOW + 1, 2):
        for dx in range(-WINDOW, WINDOW + 1, 2):
            yy, xx = ys + dy0 + dy, xs + dx0 + dx
            ok = (yy >= 0) & (yy < H16) & (xx >= 0) & (xx < W16)
            if ok.mean() < 0.7:
                continue
            e = float(np.abs(cur[ok] - prev[yy[ok], xx[ok]]).mean())
            if dy == 0 and dx == 0:
                err0 = e
            elif best is None or e < best:
                best = e
    if best is None or err0 is None:
        return False
    return best < 20 and best < 0.5 * err0


def main():
    t0 = time.monotonic()
    z = np.load(ROOT / "runs" / "098.5" / f"play_w{W}.npz")
    d = {k: z[k] for k in ("obs", "act", "done", "pos", "sleep", "light", "ep", "obj")}
    hi = np.load(OUT / f"hi_w{W}.npy", mmap_mode="r")
    if N_LIM:
        d = {k: (v[:N_LIM] if k != "obj" else v[v[:, 0] < N_LIM]) for k, v in d.items()}
    if START:                                          # frames from START on (frames before it tuned the tracker)
        assert ORIG == "true", "the learned origins are read from frame 0"
        o = d["obj"][d["obj"][:, 0] >= START].copy()
        o[:, 0] -= START
        d = {k: (v[START:] if k != "obj" else o) for k, v in d.items()}
        hi = hi[START:]
    seg = TR.segments(d)
    org16 = np.rint(FL.origins(d, seg, W) * U16 / 7).astype(np.int64)
    cam7 = np.load(ROOT / "runs" / "103.1" / "pi_crafter_world_learned.npy")[:49, :63] > 0.5
    iy = np.minimum((np.arange(H16) * 7) // U16, 48)
    ix = np.minimum((np.arange(W16) * 7) // U16, 62)
    cam = dilate(cam7[iy][:, ix], 2)
    by_t = defaultdict(list)
    for row in d["obj"]:
        if row[2] in (1, 2, 3, 4):
            by_t[int(row[0])].append(row)
    T = Tracker()
    rec, tracks, next_id, last, k = [], [], 0, None, 0
    prompted = set()
    n_frames, t_track = 0, 0.0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(len(seg)):
            if seg[t] < 0:
                rec.append([])
                continue
            if last is None or seg[t] != seg[last]:
                tracks, k, last, prompted = [], 0, None, set()
            t1 = time.monotonic()
            fc = np.asarray(hi[t], dtype=np.float32)
            fk = fc.mean(-1)
            fn_ = fc / max(float(fc.mean()), 1.0)
            feats = T.features(hi[t])
            # carry every track
            for tr in tracks:
                if SELECT:                                      # the last selected frames, as SAM's recent memory
                    tr.od["non_cond_frame_outputs"] = {k - 1 - j: o for j, o in enumerate(reversed(tr.mem[-KEEP:]))}
                out, mask = T.step(feats, k, tr.od, False)
                if not SELECT:
                    tr.od["non_cond_frame_outputs"][k] = out
                    for old in [j for j in tr.od["non_cond_frame_outputs"] if j < k - KEEP]:
                        del tr.od["non_cond_frame_outputs"][old]
                tr.out, tr.iou = out, T.last_iou
                tr.logit = T.last_logit
                T.stats["carried"] += 1
            if len(tracks) > 1 and OVERLAP == "argmax":
                L = np.stack([tr.logit for tr in tracks])
                win = L.argmax(0)
                for j, tr in enumerate(tracks):
                    tr.logit = np.where(win == j, tr.logit, -1e4)
            for tr in tracks:
                mask = tr.logit > 0
                T.stats["carried: SAM says absent"] += int(tr.logit.max() <= -1e3)
                T.stats["carried: present, mask lost to another track"] += int(tr.logit.max() > -1e3 and not mask.any())
                c = colour(fn_, mask)
                if c is not None and tr.col is not None and np.abs(c - tr.col).sum() > GATE:
                    mask = np.zeros_like(mask)                 # not its own look: absent here, not drifted onto it
                    T.stats["masks refused by the appearance gate"] += 1
                elif c is not None:
                    tr.col = c if tr.col is None else 0.9 * tr.col + 0.1 * c
                tr.state = ("sam absent" if tr.logit.max() <= -1e3 else "lost to another track" if not (tr.logit > 0).any()
                            else "refused by gate" if not mask.any() else "reported")
                if SELECT and mask.any() and tr.iou >= 0.5:      # SAMURAI: only confident, accepted frames enter memory
                    tr.mem.append(tr.out)
                    del tr.mem[:-KEEP]
                tr.mask = mask
                tr.absent = 0 if mask.any() else tr.absent + 1
                tr.big = tr.big + 1 if mask.sum() > MAX_PIX else 0
            alive = []
            for tr in tracks:
                if tr.absent > GONE or tr.big >= 2 or (not tr.confirmed and k - tr.born >= TENTATIVE):
                    T.stats["ended"] += 1
                    continue
                if not tr.confirmed and tr.mask.any():
                    ys, xs = np.nonzero(tr.mask)
                    wp = np.array([xs.mean(), ys.mean()]) + org16[t]
                    if np.abs(wp - tr.wpos0).max() >= U16 / 2:
                        tr.confirmed = True
                        T.stats["confirmed by a later move"] += 1
                alive.append(tr)
            tracks = alive
            if OVERLAP == "dup":                                # duplicates: the younger of two overlapping tracks ends
                tracks.sort(key=lambda tr: tr.born)
                kept = []
                for tr in tracks:
                    if any((tr.mask & o.mask).sum() > 0.5 * max(min(tr.mask.sum(), o.mask.sum()), 1) for o in kept):
                        T.stats["duplicates ended"] += 1
                        continue
                    kept.append(tr)
                tracks = kept
            # new tracks
            claimed = np.zeros((H16, W16), bool)
            for tr in tracks:
                claimed |= tr.mask
            if MODE == "oracle":
                live = {tr.id for tr in tracks}
                for row in by_t.get(t, []):
                    g = int(row[1])
                    if g in live:
                        continue
                    y, x = sprite_point(int(row[2]), int(row[5]), int(row[6]))
                    y, x = y + CY16 - U16 // 2 + U16 * int(row[4]), x + CX16 - U16 // 2 + U16 * int(row[3])
                    if not (0 <= y < H16 and 0 <= x < W16):
                        continue
                    T.stats["re-prompted (oracle)" if g in prompted else "prompted (oracle)"] += 1
                    prompted.add(g)
                    od = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
                    if PROMPT == "box":
                        y0, x0 = (3 + int(row[4])) * U16, (4 + int(row[3])) * U16
                        out, mask = T.step(feats, k, od, True, point=(y0, x0, y0 + U16, x0 + U16))
                    else:
                        out, mask = T.step(feats, k, od, True, point=(y, x))
                    tr = Track(g, k, out, mask, True, None, colour(fn_, mask))
                    tracks.append(tr)
            elif last is not None and len(tracks) < MAX_TRACKS:
                dy0, dx0 = (org16[t] - org16[last])[::-1]
                prev = np.asarray(hi[last], dtype=np.float32).mean(-1)
                prev_al = shifted(prev, int(dy0), int(dx0))
                diff = np.abs(fk - prev_al)
                ok = ~np.isnan(diff) & ~cam & ~dilate(claimed, 3)
                if ok.any():
                    thr = max(15.0, 3 * float(np.percentile(diff[ok], 75)))
                    ch = np.zeros((H16, W16), bool)
                    ch[ok] = diff[ok] > thr
                    ch = dilate(erode(ch, 1), 1) & ok
                    pieces = sorted((p for p in components(ch) if len(p) >= MIN_PIX), key=len, reverse=True)[:MAX_NEW]
                    for p in pieces:
                        c = p.mean(0)
                        y, x = p[np.argmin(((p - c) ** 2).sum(1))]
                        od = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
                        out, mask = T.step(feats, k, od, True, point=(int(y), int(x)))
                        a = int(mask.sum())
                        T.stats["pointed"] += 1
                        if a < MIN_PIX or a > MAX_PIX or (mask & cam).sum() > 0.3 * a or (mask & claimed).sum() > 0.2 * a:
                            T.stats["rejected (size, camera or claimed)"] += 1
                            continue
                        ys, xs = np.nonzero(mask)
                        wp = np.array([xs.mean(), ys.mean()]) + org16[t]
                        moved = pair_test(fk, prev_al, prev, mask, int(dy0), int(dx0))
                        T.stats["born confirmed (pair test)" if moved else "born tentative"] += 1
                        tracks.append(Track(next_id, k, out, mask, moved, wp, colour(fn_, mask)))
                        next_id += 1
                        claimed |= mask
            torch.cuda.synchronize()
            t_track += time.monotonic() - t1
            n_frames += 1
            seen = []
            for tr in tracks:
                if tr.confirmed and tr.mask.any():
                    ys, xs = np.nonzero(tr.mask)
                    seen.append((tr.id, ((xs.mean() - CX16) / U16, (ys.mean() - CY16) / U16)))
                    AREAS.append((t, tr.id, int(tr.mask.sum())))
            rec.append(seen)
            vmove = last is not None and bool((org16[t] != org16[last]).any())
            for tr in tracks:
                pre = getattr(tr, "logit", None)
                pm = pre > 0 if pre is not None else tr.mask
                cyx = tuple(np.argwhere(pm).mean(0).round(1)) if pm.any() else None
                STATES.append((t, tr.id, getattr(tr, "state", "new"), vmove, k - tr.born, getattr(tr, "iou", None),
                               float(tr.out["object_score_logits"].float().item()) if hasattr(tr, "out") else None,
                               int(pm.sum()), cyx))
            last, k = t, k + 1
            if t % 2000 == 0:
                print(f"  step {t} tracks {len(tracks)} ({n_frames / max(t_track, 1e-9):.1f} frames/s; "
                      f"{time.monotonic() - t0:.0f} s)", flush=True)
    kinds = {1: "cow", 2: "zombie", 3: "skeleton", 4: "arrow"}
    idtp, ngt, npred, ik, nk = TR.idf1(rec, d["obj"], d["ep"], kinds)
    ex = FL.extra(rec, d, kinds)
    rep = {"worker": W, "frames from": START, "mode": MODE, "origins": ORIG, "model": CKPT, "gate": GATE, "memory selection": SELECT, "prompt": PROMPT, "overlap": OVERLAP, "resize": RESIZE,
           "IDF1 (movers pooled)": round(2 * idtp / max(ngt + npred, 1), 4),
           "ID recall / ID precision": [round(idtp / max(ngt, 1), 4), round(idtp / max(npred, 1), 4)],
           "object frames / detections": [int(ngt), int(npred)],
           "ID recall by kind": {kinds[q]: round(ik.get(q, 0) / max(nk.get(q, 1), 1), 4) for q in kinds},
           "detections on no creature: share": round(ex["detections, on no creature"][1]
                                                     / max(ex["detections, on no creature"][0], 1), 4),
           "night creature frames: share with a track": round(ex["night frames, with a file"][1]
                                                              / max(ex["night frames, with a file"][0], 1), 4),
           "still since seen: share with a track": round(ex["still since seen: frames, with a file"][1]
                                                        / max(ex["still since seen: frames, with a file"][0], 1), 4),
           "frames per second (tracking and discovery)": round(n_frames / max(t_track, 1e-9), 1),
           "events": dict(T.stats), "GPU memory peak (GB)": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
           "seconds": round(time.monotonic() - t0, 1)}
    tag = f"{MODE}_{ORIG}_w{W}" + (f"_from{START}" if START else "") + (f"_n{N_LIM}" if N_LIM else "") + (f"_g{GATE}" if "--gate" in sys.argv else "") + ("_sel" if SELECT else "") + ("_box" if PROMPT == "box" else "") + (f"_{OVERLAP}" if "--overlap" in sys.argv else "") + (f"_{RESIZE}" if "--resize" in sys.argv else "")
    if "--dump" in sys.argv:
        import pickle
        pickle.dump({"rec": rec, "areas": AREAS, "states": STATES}, open(OUT / f"dump_{tag}.pkl", "wb"))
    (OUT / f"track_{tag}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
