"""Card 016: card 014's local-step walking on an egocentric frame.

The walking module sees the room shifted so the agent is at the centre and
rotated so it faces up (13 x 13 tiles, outside the room drawn as wall, plus
a row whose first tile shows the held item); its readout is the fixed
centre cell, facing up. Run 5's conditions still read the top-down frame.

Stages (argv[1]): check (rendering and label checks, a picture), gate
(upper bound, every cell supervised), main (criteria 1-3, learned), maps (a trained
module's maps at every cell against the exact ones).

Run: PYTHONPATH=tools/card012:tools/card013:tools/card014:tools/card016 bin/prun python -c \
       "import sys; sys.argv=['x','gate','30000','logit']; import bench_ego; bench_ego.main()"
"""
import json, sys, time
from pathlib import Path
import numpy as np
from worldmodel import discover_logic as dl
from worldmodel.envs import logicdoor as ld
from worldmodel.envs.keydoor_render import code
import bench_vin as bv

R = 6                    # view radius: offsets -6 .. +6 cover the 8 x 8 room from any inside position
H, W = 2 * R + 2, 2 * R + 1
UP = 3                   # MiniGrid's facing "up"
WALL = code("wall")
CARD = "016"            # output names; card 017 reuses these stages


class Ego:
    """Top-down tile codes (B, 9, 8) and agent (x, y, facing) -> egocentric codes (B, H, W)."""

    def __init__(self, torch, dev):
        self.torch, self.dev = torch, dev
        r, c = np.meshgrid(np.arange(2 * R + 1), np.arange(2 * R + 1), indexing="ij")
        f, rt = R - r, c - R                                   # steps ahead, steps to the right
        DX, DY = [], []
        for d in range(4):
            fx, fy = ld.DIR_VEC[d]
            rx, ry = ld.DIR_VEC[(d + 1) % 4]
            DX.append(f * fx + rt * rx); DY.append(f * fy + rt * ry)
        self.DX = torch.as_tensor(np.array(DX), device=dev)
        self.DY = torch.as_tensor(np.array(DY), device=dev)

    def where(self, st):
        st = self.torch.as_tensor(np.asarray(st)[:, :3].astype(np.int64), device=self.dev)
        x, y, d = st[:, 0], st[:, 1], st[:, 2]
        X = x[:, None, None] + self.DX[d]
        Y = y[:, None, None] + self.DY[d]
        ok = (X >= 0) & (X < 8) & (Y >= 0) & (Y < 8)
        return X.clamp(0, 7), Y.clamp(0, 7), ok, d

    def codes(self, codes, st, chunk=262144):
        torch = self.torch
        out = []
        for s in range(0, len(codes), chunk):
            c = codes[s:s + chunk].long()
            X, Y, ok, _ = self.where(st[s:s + chunk])
            n = torch.arange(len(c), device=self.dev)[:, None, None]
            e = torch.where(ok, c[n, Y, X], torch.full_like(X, WALL))
            e[:, R, R] = (e[:, R, R] // 5) * 5 + UP + 1           # the agent, drawn facing up
            held = torch.zeros(len(c), 1, W, dtype=e.dtype, device=self.dev)
            held[:, 0, 0] = c[:, 8, 0]
            out.append(torch.cat([e, held], 1).to(torch.uint8))
        return torch.cat(out)

    def maps(self, dm, sm, st):
        """Exact distance maps (N, nw, 4, 9, 8) and successor maps (N, nw, 3, 4, 9, 8), top-down,
        into the egocentric frame: facing e relative to the view is world facing (d + e - UP) % 4."""
        torch = self.torch
        X, Y, ok, d = self.where(st)
        n = torch.arange(len(dm), device=self.dev)[:, None, None]
        D, S_ = dm.permute(0, 2, 3, 4, 1), sm.permute(0, 3, 4, 5, 1, 2)
        de, se = [], []
        for e in range(4):
            wf = ((d + e - UP) % 4)[:, None, None]
            a = D[n, wf, Y, X].masked_fill(~ok[..., None], -1)                # (N, 13, 13, nw)
            b = S_[n, wf, Y, X].masked_fill(~ok[..., None, None], -1)         # (N, 13, 13, nw, 3)
            de.append(a.permute(0, 3, 1, 2)); se.append(b.permute(0, 3, 4, 1, 2))
        de, se = torch.stack(de, 2), torch.stack(se, 3)                       # (N,nw,4,13,13), (N,nw,3,4,13,13)
        pad = lambda t: torch.cat([t, torch.full((*t.shape[:-2], 1, W), -1, dtype=t.dtype, device=self.dev)], -2)
        return pad(de), pad(se)


def to_ego(S):
    """Replace the walking module's frames by egocentric ones; returns the Ego converter."""
    torch = S.torch
    ego = Ego(torch, S.dev)
    S.trainer.c0 = ego.codes(S.trainer.c0, S.tr["s0"])
    S.trainer.c1 = ego.codes(S.trainer.c1, S.tr["s1"])
    S.tseq["codes_topdown"] = S.tseq["codes"]
    S.tseq["codes"] = ego.codes(torch.as_tensor(S.tseq["codes"], device=S.dev), S.tseq["st"]).cpu().numpy()
    fixed = torch.zeros(4, H, W, device=S.dev)
    fixed[UP, R, R] = 1.0
    S.fixed = fixed
    bv.agent_onehot = lambda st, dev, torch_: fixed[None].expand(len(st), -1, -1, -1)
    return ego


def new_vin(S):
    return bv.make_vin(S.trainer.model.enc[:6], len(S.ways), H, W, fixed=S.fixed).to(S.dev)


def write_png(path, rgb):
    import struct, zlib
    raw = b"".join(b"\x00" + row.tobytes() for row in rgb)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", rgb.shape[1], rgb.shape[0], 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def check(S, ego):
    """Rendering and label checks before any training."""
    torch = S.torch
    res = {}
    main_ = [j for j, c in enumerate(S.ways) if S.tree.path(c) in (("forward",), ("forward", "toggle"))]
    rows = np.sort(np.random.default_rng(0).choice(np.flatnonzero(S.walking.cpu().numpy()), 2000, replace=False))
    parts = S.pool.map(bv._dmaps, [({int(e): S.layouts[e] for e in np.unique(S.tr["ep"][ch])}, list(S.tree.parent),
                                    list(S.tree.action), [S.ways[j] for j in main_],
                                    [(int(S.tr["ep"][i]), S.tr["s0"][i]) for i in ch]) for ch in np.array_split(rows, 20)])
    dm = torch.as_tensor(np.concatenate([p[0] for p in parts]), device=S.dev)
    sm = torch.as_tensor(np.concatenate([p[1] for p in parts]), device=S.dev)
    st = S.tr["s0"][rows]
    de, se = ego.maps(dm, sm, st)
    x, y, d = [torch.as_tensor(st[:, k].astype(np.int64), device=S.dev) for k in (0, 1, 2)]
    n = torch.arange(len(rows), device=S.dev)
    # the agent's own entry: centre cell, facing up
    res["centre_equals_agent_entry"] = float((de[:, :, UP, R, R] == dm[n, :, d, y, x]).float().mean())
    # forward from the centre (facing up) lands one row up, still facing up
    fwd = se[:, :, ld.FORWARD, UP, R, R]
    ahead = de[:, :, UP, R - 1, R]
    want = torch.where(ahead >= 0, ahead, de[:, :, UP, R, R])       # blocked (wall, closed door): stays put
    res["forward_successor_is_cell_ahead"] = float((fwd[fwd >= 0] == want[fwd >= 0]).float().mean())
    # turning left from the centre faces "left" in the view: same cell, facing (UP - 1) % 4
    lft = se[:, :, ld.LEFT, UP, R, R]
    res["left_successor_is_facing_left"] = float((lft[lft >= 0] == de[:, :, (UP - 1) % 4, R, R][lft >= 0]).float().mean())
    # frames: one move forward shifts the view by one row (the agent stays at the centre)
    e0 = S.trainer.c0[torch.as_tensor(rows, device=S.dev)]
    e1 = S.trainer.c1[torch.as_tensor(rows, device=S.dev)]
    moved = torch.as_tensor((S.tr["s0"][rows, :2] != S.tr["s1"][rows, :2]).any(1), device=S.dev)
    same = (e1[:, 1:2 * R + 1, :] == e0[:, 0:2 * R, :])
    same[:, R - 1, R] = True; same[:, R, R] = True           # the agent's own tile and the one it left
    res["forward_shifts_view_one_row"] = float(same[moved].all((1, 2)).float().mean()) if moved.any() else None
    res["rows_moved_forward"] = int(moved.sum())
    S.log(f"RESULT checks: {json.dumps(res)}")
    # one picture: top-down and egocentric frames of the same state
    i = int(rows[5])
    td = dl.encode_pairs([(S.layouts[int(S.tr["ep"][i])], dl.to_tup(S.tr["s0"][i]))])
    a = (bv.images(torch.as_tensor(td, device=S.dev), S.trainer.tiles)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    b = (bv.images(S.trainer.c0[i:i + 1], S.trainer.tiles)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    canvas = np.full((max(a.shape[0], b.shape[0]), a.shape[1] + 16 + b.shape[1], 3), 255, np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1] + 16:] = b
    write_png("runs/016_view.png", canvas.repeat(4, 0).repeat(4, 1))
    S.log(f"state {S.tr['s0'][i][:3].tolist()} (x, y, facing) -> runs/016_view.png")
    return res


def gate(S, ego, updates):
    main_ = [j for j, c in enumerate(S.ways) if S.tree.path(c) in (("forward",), ("forward", "toggle"))]
    up = new_vin(S)
    bv.train_fullmap(S, up, updates, main_, relabel=ego.maps)
    frames = {j: f for j, f in bv.pick_frames(S).items() if j in main_}
    return {"upper_bound": bv.walk_score(S, up, frames, tag="(gate) upper bound, egocentric, every cell supervised")}


def main_stage(S, ego, updates, flat):
    torch = S.torch
    res = {"criterion_2_flat_run5": flat}
    lv = new_vin(S)
    bv.train(S, lv, updates, "learned", allowed=S.row_wall != bv.HELD_WALL, tag="learned")
    torch.save(lv.state_dict(), f"runs/{CARD}_vin_main_k{bv.K}_{updates}.pt")
    seen, unseen = bv.pick_frames(S, not_wall=bv.HELD_WALL), bv.pick_frames(S, wall=bv.HELD_WALL)
    res["criterion_1_walking"] = bv.walk_score(S, lv, seen, tag="criterion 1, every way, wall column seen")
    res["criterion_2_unseen_column"] = bv.walk_score(S, lv, unseen, tag="criterion 2, wall column 5 (never trained on)")
    lay_rng = np.random.default_rng(777)
    layouts = [ld.make_layout(8, "key", lay_rng) for _ in range(500)]
    res["criterion_3_acting"] = bv.act_vin(S, lv, layouts, view=lambda c, st: ego.codes(c, st))
    S.log(f"RESULT criterion 3 acting: {json.dumps(res['criterion_3_acting'])}")
    return res


def maps_check(S, ego, path):
    """A trained module's maps at every cell against the exact ones, the agent's own entry (centre,
    facing up) apart from the rest: is "ready here" right only where it was trained?"""
    torch = S.torch
    lv = new_vin(S)
    lv.load_state_dict(torch.load(path, map_location=S.dev)); lv.eval()
    paths = {("forward",): "goal square", ("forward", "toggle"): "door (with key)", ("forward", "pickup"): "key"}
    ways = [j for j, c in enumerate(S.ways) if S.tree.path(c) in paths]
    allowed = (S.walking & (S.row_wall != bv.HELD_WALL)).cpu().numpy()
    rng = np.random.default_rng(3)
    res = {}
    for j in ways:
        rd = S.ready0[:, j].cpu().numpy()
        rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(allowed), 1500, replace=False),
                                       rng.choice(np.flatnonzero(allowed & rd), min(500, int((allowed & rd).sum())), replace=False)]))
        parts = S.pool.map(bv._dmaps, [({int(e): S.layouts[e] for e in np.unique(S.tr["ep"][ch])}, list(S.tree.parent),
                                        list(S.tree.action), [S.ways[j]],
                                        [(int(S.tr["ep"][i]), S.tr["s0"][i]) for i in ch]) for ch in np.array_split(rows, 20)])
        dm = torch.as_tensor(np.concatenate([p[0] for p in parts]), device=S.dev)
        sm = torch.as_tensor(np.concatenate([p[1] for p in parts]), device=S.dev)
        de = ego.maps(dm, sm, S.tr["s0"][rows])[0][:, 0]                          # (N, 4, H, W), -1 = none
        est, T = [], []
        with torch.no_grad():
            for s in range(0, len(rows), 256):
                r = torch.as_tensor(rows[s:s + 256], device=S.dev)
                img = bv.images(S.trainer.c0[r], S.trainer.tiles)
                mp_, _, _ = lv(img, torch.full((len(r),), j, device=S.dev), full=True)   # (B, K+1, 4, H, W) logits
                on = mp_ > 0
                first = torch.where(on.any(1), on.float().argmax(1), torch.full_like(on[:, 0], -1, dtype=torch.long))
                est.append(first); T.append(on[:, 0])
        est, T = torch.cat(est), torch.cat(T)
        centre = torch.zeros_like(de, dtype=torch.bool); centre[:, UP, R, R] = True
        inside = de >= 0
        out = {}
        for name, m in (("agent's own entry", centre & inside), ("every other cell", ~centre & inside)):
            ready, far = m & (de == 0), m & (de > 0)
            close = m & (de > 0) & (de <= bv.K)
            out[name] = {
                "ready_cells": int(ready.sum()),
                "ready_found": round(float(T[ready].float().mean()), 3),
                "false_ready": round(float(T[far].float().mean()), 3),
                "steps_exact": round(float((est[close] == de[close]).float().mean()), 3),
                "steps_within_1": round(float(((est[close] - de[close]).abs() <= 1).float().mean()), 3),
                "says_unreachable": round(float((est[close] < 0).float().mean()), 3)}
        c = torch.as_tensor(rd[rows], device=S.dev)
        out["run5_label_vs_exact_at_agent"] = round(float((c == (de[:, UP, R, R] == 0)).float().mean()), 3)
        res[paths[S.tree.path(S.ways[j])]] = out
        S.log(f"RESULT maps, {paths[S.tree.path(S.ways[j])]}: {json.dumps(out)}")
    return res


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "check"
    updates = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 30000
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    S = bv.Setup(log)
    flat = None
    if stage == "main":     # run 5's own values read the top-down frame: score them before the switch
        flat = bv.flat_score(S, bv.pick_frames(S, wall=bv.HELD_WALL), tag="run 5's flat values, wall column 5")
    log(f"look-ahead steps K = {bv.K}; longest walk in held-out frames: {int(S.Dt.max())}")
    ego = to_ego(S)
    log(f"egocentric frames: training {tuple(S.trainer.c0.shape)}, held-out {S.tseq['codes'].shape}")
    res = {"check": lambda: check(S, ego), "gate": lambda: gate(S, ego, updates),
           "main": lambda: main_stage(S, ego, updates, flat),
           "maps": lambda: maps_check(S, ego, f"runs/{CARD}_vin_main_k{bv.K}_{updates}.pt")}[stage]()
    out = Path(f"runs/{CARD}_{stage}{'' if bv.K == 32 else f'_k{bv.K}'}_{updates}.json")
    out.write_text(json.dumps(res, indent=1) + "\n")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
