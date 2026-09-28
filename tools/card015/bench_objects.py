"""Card 015: which self-supervised signal makes the encoder's map show objects.

Arms (argv[1]): R = per-tile reconstruction, C = per-tile change prediction
(given the action), input = the gate's upper bound (read-out on the tiles'
own pixels), run5 = control: run 5's encoder with no added loss. Both
arms start from card 012 run 5's encoder; episodes with the wall in
column 5 are never trained on.

Run: PYTHONPATH=tools/card012:tools/card013:tools/card014:tools/card015 bin/prun python -c \
       "import sys; sys.argv=['x','C','logit']; import bench_objects; bench_objects.main()"
"""
import copy, json, sys, time
from pathlib import Path
import numpy as np
from worldmodel import discover_logic as dl
from worldmodel.envs.keydoor_render import OBJECTS
import bench_vin as bv

NAMES = ["empty", "wall", "goal", "door closed", "door open", "key", "switch off", "switch on", "vase"]


def cls(o):
    if o is None: return 0
    if o == "wall": return 1
    if o == "goal": return 2
    if o[0] == "door": return 4 if o[2] == 2 else 3
    if o[0] == "key": return 5
    if o == ("ball", "grey"): return 6
    if o == ("ball", "yellow"): return 7
    return 8


def tiles_of(img, F):
    return F.pixel_unshuffle(img, 8)                 # (B, 192, 9, 8): each tile's own pixels


def readout(S, feat, tag, rows_ok, updates=4000):
    """Linear one-tile read-out (1x1 conv) of each tile's object class from frozen features.
    feat: img -> (B, C, 9, 8). Trained on rows_ok; tested on held-out frames, wall column seen / unseen."""
    torch, nn, F = dl._torch()
    lut = torch.as_tensor([cls(o) for o in OBJECTS], device=S.dev)
    with torch.no_grad():
        C = feat(bv.images(S.trainer.c0[:2].long(), S.trainer.tiles)).shape[1]
    head = nn.Conv2d(C, len(NAMES), 1).to(S.dev)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    gen = torch.Generator(device=S.dev).manual_seed(0)
    rows = torch.nonzero(rows_ok).squeeze(1)
    for u in range(updates):
        r = rows[torch.randint(len(rows), (256,), device=S.dev, generator=gen)]
        codes = S.trainer.c0[r].long()
        with torch.no_grad():
            m = feat(bv.images(codes, S.trainer.tiles)).float()
        y = lut[codes // 5]
        # classes are very unequal (one door per frame): weight each class equally
        cnt = torch.bincount(y.flatten(), minlength=len(NAMES)).float().clamp_min(1)
        loss = F.cross_entropy(head(m), y, weight=(cnt.sum() / cnt) / len(NAMES))
        opt.zero_grad(); loss.backward(); opt.step()
    out = {}
    for name, sel in (("wall_column_seen", S.test_wall != bv.HELD_WALL), ("wall_column_unseen", S.test_wall == bv.HELD_WALL)):
        idx = np.flatnonzero(sel)[:40000]
        codes = torch.as_tensor(S.tseq["codes"][idx], device=S.dev).long()
        with torch.no_grad():
            pred = torch.cat([head(feat(bv.images(codes[s:s + 2048], S.trainer.tiles)).float()).argmax(1)
                              for s in range(0, len(codes), 2048)])
        y = lut[codes // 5]
        out[name] = {NAMES[c]: round(float((pred[y == c] == c).float().mean()), 4) for c in range(len(NAMES)) if (y == c).any()}
    S.log(f"RESULT read-out ({tag}): {json.dumps(out)}")
    return out


def train_arm(S, arm, updates, rows_ok, bs=256):
    torch, nn, F = dl._torch()
    trunk = copy.deepcopy(S.trainer.model.enc[:6]).train()
    if arm == "R":
        head = nn.Conv2d(64, 192, 1)
    else:
        head = nn.ModuleDict({"act": nn.Embedding(len(dl.ACTIONS), 16),
                              "net": nn.Sequential(nn.Conv2d(64 + 16, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))})
    head = head.to(S.dev)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(head.parameters()), lr=3e-4)
    gen = torch.Generator(device=S.dev).manual_seed(0)
    rows = torch.nonzero(rows_ok).squeeze(1)
    t0 = time.monotonic()
    for u in range(updates):
        r = rows[torch.randint(len(rows), (bs,), device=S.dev, generator=gen)]
        img0 = bv.images(S.trainer.c0[r], S.trainer.tiles)
        m = trunk(img0)
        if arm == "R":
            loss = F.mse_loss(head(m).float(), tiles_of(img0, F))
        else:
            img1 = bv.images(S.trainer.c1[r], S.trainer.tiles)
            y = (tiles_of(img1, F) != tiles_of(img0, F)).any(1).float()            # (B, 9, 8): tile changed
            a = head["act"](S.trainer.act[r].long())[:, :, None, None].expand(-1, -1, 9, 8)
            logit = head["net"](torch.cat([m, a], 1)).float().squeeze(1)
            pos = y.mean().clamp_min(1e-4)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=(1 - pos) / pos)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if u % 5000 == 0 or u == updates - 1:
            S.log(f"  arm {arm} {u}: loss {float(loss.detach()):.4f} {(u + 1) / (time.monotonic() - t0):.0f} upd/s")
    return trunk.eval()


def walking_upper(S, trunk, updates=8000, train_frames=False):
    torch = S.torch
    main_ = [j for j, c in enumerate(S.ways) if S.tree.path(c) in (("forward",), ("forward", "toggle"))]
    up = bv.make_vin(trunk, len(S.ways)).to(S.dev)
    rows, dm, sm = bv.train_fullmap(S, up, updates, main_)
    held = bv.walk_score(S, up, {j: f for j, f in bv.pick_frames(S).items() if j in main_}, given=True, tag="upper, full map, held-out")
    if not train_frames:
        return held
    # the same measure on frames the module was trained on
    out = {}
    st = S.tr["s0"][rows]
    x, y, d = [torch.as_tensor(st[:, k].astype(np.int64), device=S.dev) for k in (0, 1, 2)]
    for jj, j in enumerate(main_):
        n_ = torch.arange(len(rows), device=S.dev)
        here = dm[n_, jj, d, y, x].long()
        idx = torch.nonzero(here >= 1).squeeze(1)[:3000]
        picks = []
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                b_ = idx[s:s + 1024]
                img = bv.images(S.trainer.c0[torch.as_tensor(rows, device=S.dev)[b_]], S.trainer.tiles)
                _, _, Q, _ = up(img, torch.full((len(b_),), j, device=S.dev),
                                given=bv.agent_onehot(st[b_.cpu().numpy()], S.dev, torch))
                picks.append(Q.sum(1).argmax(1))
        pick = torch.cat(picks)
        after = sm[idx, jj, pick, d[idx], y[idx], x[idx]].long()
        ok = (after >= 0) & (after < here[idx])
        out["/".join(S.tree.path(S.ways[j]))] = {"frames": int(len(idx)), "moves_closer": round(float(ok.float().mean()), 4)}
    S.log(f"RESULT walking upper, full map, training frames: {json.dumps(out)}")
    return {"held_out": held, "training_frames": out}


def main():
    arm = sys.argv[1]
    updates = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 20000
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    S = bv.Setup(log)
    torch, nn, F = dl._torch()
    rows_ok = S.row_wall != bv.HELD_WALL
    door = torch.as_tensor(S.tr["s0"][:, 8], device=S.dev) == 1
    w0 = torch.as_tensor(S.tr["w0"], device=S.dev)
    log(f"data: door open in {float((w0 * door).sum() / w0.sum()):.4f} of frames; "
        f"door changes {int((S.tr['s0'][:, 8] != S.tr['s1'][:, 8]).sum())}, "
        f"switch changes {int((S.tr['s0'][:, 9] != S.tr['s1'][:, 9]).sum())}, "
        f"vase changes {int((S.tr['s0'][:, 10] != S.tr['s1'][:, 10]).sum())}")
    res = {"arm": arm}
    if arm == "run5":
        # control: run 5's encoder, no added loss, same walking upper bound
        wu = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 8000
        res["walking_upper"] = walking_upper(S, S.trainer.model.enc[:6], wu, train_frames="trainfit" in sys.argv)
    elif arm == "input":
        res["readout"] = readout(S, lambda img: tiles_of(img, F), "tiles' own pixels (upper bound)", rows_ok)
    else:
        res["readout_before"] = readout(S, S.trainer.model.enc[:6], "run 5 encoder", rows_ok) if arm == "R" else None
        trunk = train_arm(S, arm, updates, rows_ok)
        res["readout"] = readout(S, trunk, f"arm {arm}", rows_ok)
        wu = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 8000
        res["walking_upper"] = walking_upper(S, trunk, wu, train_frames="trainfit" in sys.argv)
    out = Path(f"runs/015_{arm}{'_trainfit' if 'trainfit' in sys.argv else ''}.json")
    out.write_text(json.dumps(res, indent=1) + "\n")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
