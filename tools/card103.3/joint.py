"""Card 103.3: the view's motion read with files explained away. Card 103.1's placing (robust likelihood, the camera's
learned pixels) and card 103.2's files run together, step by step. Before the view's motion is chosen, the world pixels
a file may explain are left out of the evidence: each file's mask at its world position, widened by the largest
displacement a file may make (K pixels). Both hypotheses for a stepping cow ("the cow stepped", "the view stepped the
other way") then lose the cow's pixels alike, and the action's prior decides.

Sleep moves nothing: sleeping frames are skipped and the canvas, the origin and the files carry through. Parameters
(p(s | a), sigma^2, eps) are learned on workers 0-1 with card 103.1's placing; the camera's pixels are 103.1's.

  bin/prun python tools/card103.3/joint.py [--excl on|off] [--workers 8]     → runs/103.3/joint_<excl>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fn

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
EXCL = arg("--excl", "on") == "on"
WORKERS = [int(x) for x in arg("--workers", "8").split(",")]
SMOKE = int(arg("--smoke", 0))
MIN_HITS = int(arg("--min-hits", 3))
sys.argv = ["joint.py", "crafter", "--pi", "learned"] + (["--n", str(SMOKE)] if SMOKE else [])
for p in ("card103.1", "card103.2"):
    sys.path.insert(0, str(ROOT / "tools" / p))
import place as PL                                     # noqa: E402
import track as TR                                     # noqa: E402

OUT, dev = ROOT / "runs" / "103.3", "cuda"
K, NS, HW, WW, HALF = PL.K, PL.NS, PL.HW, PL.WW, 900


def learn():
    tr = PL.load("train")
    pi = np.load(ROOT / "runs" / "103.1" / "pi_crafter_world_learned.npy")
    P = {"s2": 1e-3, "eps": 0.05, "pi": pi}
    cand = None
    for it in range(2):
        pr = np.ones((17, NS * NS))
        _, _, st, _, _ = PL.place(tr, pr, P, learn=True, cand=cand)
        P["s2"] = float(np.clip(st["in_r2"] / (3 * st["in_r"]), 1e-5, 1e-2))
        P["eps"] = float(np.clip(1 - st["in_r"] / st["n"], 1e-3, 0.5))
        q = pr / pr.sum(1, keepdims=True)
        cand = torch.as_tensor((q >= 1e-3).reshape(17, NS, NS), device=dev)
        print("learned pass", it, {k: v for k, v in P.items() if k != "pi"}, flush=True)
    return P, pr, cand


@torch.no_grad()
def run(d, P, prior, cand, cam):
    seg = TR.segments(d)                                  # episodes; sleeping frames -1
    act, obs = d["act"].astype(int), d["obs"]
    n = len(seg)
    org = np.zeros((n, 2), np.int64)
    sh = np.full((n, 2), -99)
    canvas = torch.zeros(3, 2 * HALF, 2 * HALF, device=dev)
    known = torch.zeros(2 * HALF, 2 * HALF, dtype=torch.bool, device=dev)
    pi = torch.as_tensor(P["pi"], device=dev).float().clamp(1e-4, 1 - 1e-4)
    lpi, l1pi = pi.log(), (1 - pi).log()
    write = pi < 0.5
    s2, eps = P["s2"], P["eps"]
    T = TR.Tracker(cam)
    rec, last, prev, s2tr = [], None, None, 1e-4
    excl_px = []
    for t in range(n):
        if seg[t] < 0:
            rec.append([])
            continue
        f = torch.as_tensor(obs[t, :HW, :WW], device=dev).permute(2, 0, 1).float() / 255
        if last is None or seg[t] != seg[last]:
            canvas.zero_()
            known.zero_()
            org[t] = (0, 0)
            T.reset()
        else:
            a = int(act[t - 1]) if last == t - 1 else 0         # after sleep: the last act was a noop
            lp = torch.as_tensor(np.log(prior[a] / prior[a].sum()), device=dev).float().reshape(NS, NS)
            x0, y0 = int(org[last, 0]) + HALF - K, int(org[last, 1]) + HALF - K
            C = canvas[:, y0:y0 + HW + 2 * K, x0:x0 + WW + 2 * K]
            M = known[y0:y0 + HW + 2 * K, x0:x0 + WW + 2 * K].clone()
            if EXCL and T.files:                                # world pixels a file may explain
                E = torch.zeros(1, 1, HW + 2 * K, WW + 2 * K, device=dev)
                for F in T.files:
                    if not (F["moved"] and F["hits"] >= MIN_HITS):    # only files that have moved and been followed
                        continue
                    ys, xs = np.nonzero(F["mask"])
                    ys = ys + F["pos"][1] - (int(org[last, 1]) - K)
                    xs = xs + F["pos"][0] - (int(org[last, 0]) - K)
                    ok = (ys >= 0) & (ys < HW + 2 * K) & (xs >= 0) & (xs < WW + 2 * K)
                    E[0, 0, ys[ok], xs[ok]] = 1
                E = fn.max_pool2d(E, 2 * K + 1, stride=1, padding=K)[0, 0] > 0
                excl_px.append(int(E.sum()))
                M &= ~E
            Cu = C.unfold(1, HW, 1).unfold(2, WW, 1)
            Mu = M.unfold(0, HW, 1).unfold(1, WW, 1)
            cm = cand[a]
            sup = Mu[cm].all(0)
            diff = Cu - f[:, None, None]
            if sup.sum() > 50:
                diff = diff - diff[..., sup].median(-1).values[..., None, None]
            r2 = (diff ** 2).sum(0)
            dc = prev - f
            rc = ((dc - dc.flatten(1).median(1).values[:, None, None]) ** 2).sum(0)
            s2t = s2
            for rep in range(2):
                lw = PL.mix(r2, s2t, eps)
                lc = PL.mix(rc, s2t, eps)
                lq = torch.logaddexp(lpi + lc, l1pi + lw)
                score = torch.where(cm, lp + (lq * sup).sum((-1, -2)), torch.full_like(lp, -1e30))
                k = int(score.argmax())
                dy, dx = divmod(k, NS)
                if rep == 0:
                    m = r2[dy, dx][sup & (pi < 0.5)]
                    if m.numel() < 50:
                        break
                    s2t = max(float(m.median()) / 2.366, s2)
            org[t] = org[last] + (dx - K, dy - K)
            sh[last] = (dx - K, dy - K) if last == t - 1 else (-99, -99)
        x, y = int(org[t, 0]) + HALF, int(org[t, 1]) + HALF
        canvas[:, y:y + HW, x:x + WW] = torch.where(write, f, canvas[:, y:y + HW, x:x + WW])
        known[y:y + HW, x:x + WW] |= write
        # the files, in the same world frame
        fn_ = obs[t, :HW, :WW].astype(np.float32) / 255
        if T.known.any():
            sy, sx = T.view(org[t])
            Kn = T.known[sy, sx] & ~cam
            if Kn.sum() > 50:
                s2tr = max(float(np.median(((fn_ - T.canvas[sy, sx]) ** 2).sum(-1)[Kn])) / 2.366, 1e-4)
        rec.append(T.step(t, fn_, org[t], org[last] if last is not None else org[t], s2tr))
        prev, last = f, t
    return org, sh, rec, T, excl_px


def main():
    t0 = time.monotonic()
    P, prior, cand = learn()
    cam = np.load(ROOT / "runs" / "103.1" / "pi_crafter_world_learned.npy")[:HW, :WW] > 0.5
    rep = {"exclusion of file pixels": EXCL, "workers": WORKERS, "parts": []}
    S = {"right": 0, "n": 0, "moved_right": 0, "moved_n": 0, "still_right": 0, "still_n": 0, "night_right": 0,
         "night_n": 0, "idtp": 0, "gt": 0, "pred": 0}
    for w in WORKERS:
        z = np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz")
        d = {k: z[k] for k in ("obs", "act", "done", "pos", "sleep", "light", "ep", "obj")}
        if SMOKE:
            d = {k: (v[:SMOKE] if k != "obj" else v[v[:, 0] < SMOKE]) for k, v in d.items()}
        org, sh, rec, T, excl_px = run(d, P, prior, cand, cam)
        seg = TR.segments(d)
        st = np.flatnonzero((seg[:-1] >= 0) & (seg[:-1] == seg[1:]))
        true = 7 * (d["pos"][st + 1] - d["pos"][st]).astype(np.int64)
        right = (sh[st] == true).all(1)
        moved = (true != 0).any(1)
        night = d["light"][st] < 0.3
        idtp, ngt, npred, ik, nk = TR.idf1(rec, d["obj"], d["ep"], {1: "cow", 2: "zombie", 3: "skeleton", 4: "arrow"})
        for k_, v in (("right", right.sum()), ("n", len(st)), ("moved_right", right[moved].sum()),
                      ("moved_n", moved.sum()), ("still_right", right[~moved].sum()), ("still_n", (~moved).sum()),
                      ("night_right", right[night].sum()), ("night_n", night.sum()), ("idtp", idtp), ("gt", ngt),
                      ("pred", npred)):
            S[k_] += float(v)
        rep["parts"].append({"worker": w, "excluded world pixels per step, median": int(np.median(excl_px)) if excl_px else 0,
                             "file events": dict(T.stats)})
        print("worker", w, "done", round(time.monotonic() - t0), "s", flush=True)
    rep.update({"view's motion read right": round(S["right"] / S["n"], 4),
                "when it moved, when it did not": [round(S["moved_right"] / S["moved_n"], 4),
                                                   round(S["still_right"] / S["still_n"], 4)],
                "at night": round(S["night_right"] / max(S["night_n"], 1), 4),
                "IDF1 (movers pooled)": round(2 * S["idtp"] / (S["gt"] + S["pred"]), 4),
                "ID recall / ID precision": [round(S["idtp"] / S["gt"], 4), round(S["idtp"] / S["pred"], 4)],
                "seconds": round(time.monotonic() - t0, 1)})
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / f"joint_{'on' if EXCL else 'off'}_w{''.join(map(str, WORKERS))}{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
