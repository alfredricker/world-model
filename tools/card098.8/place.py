"""Card 098.8: the agent's own motion by card 098.5's placing step (level 1), for the walk's world places.

Per step: the candidates are no move and the action's learned shift (card 098.3's); the one whose view best matches
belief (the last vector seen at each world place of the episode, card 098.3's tokens) over places already known is
taken, as tools/card098.6/crafter.py's level 1 does without files. Card 098.2's two-frame check missed about a fifth
of real moves (mostly uniform terrain, where two frames barely differ); the placing step reads them against belief.

  bin/prun python tools/card098.8/place.py      → runs/098.8/place_w*.npz (O: each frame's offset, as card 098.2's)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLAY = ROOT / "runs" / "098.5"
OUT = ROOT / "runs" / "098.8"
SHIFTS = ROOT / "runs" / "098.3" / "results_joint_blocked.json"
CENTRE = 31
REL = np.array([(k % 9 - 4, k // 9 - 3) for k in range(63)], np.float64)
VIEW = np.array([k for k in range(63) if k != CENTRE])


def place(w, shifts):
    z = np.load(PLAY / f"play_w{w}.npz")
    act, ep, done, pos = z["act"], z["ep"], z["done"], z["pos"].astype(np.float64)
    V = np.load(PLAY / f"vec_w{w}.npy").astype(np.float32)
    agent, layout = np.zeros(2), {}
    A = np.zeros((len(act), 2))
    right = []
    for t in range(len(act)):
        if t == 0 or ep[t] != ep[t - 1] or done[t - 1]:
            agent, layout = np.zeros(2), {}
        else:
            cands = [np.zeros(2)]
            if int(act[t - 1]) in shifts:
                cands.append(-shifts[int(act[t - 1])])
            score = []
            for c in cands:
                P = agent + c + REL[VIEW]
                keys = [tuple(p) for p in P.round().astype(int)]
                m = np.array([k in layout for k in keys])
                if not m.any():
                    score.append(np.inf)
                    continue
                Bv = np.stack([layout[k] for k, mm in zip(keys, m) if mm])
                score.append(float(np.linalg.norm(V[t][VIEW][m] - Bv, axis=1).mean()))
            best = cands[int(np.argmin(score))] if np.isfinite(min(score)) else cands[0]
            right.append(bool(np.allclose(best, pos[t] - pos[t - 1])))
            agent = agent + best
        A[t] = agent
        for k, p in zip(VIEW, (agent + REL[VIEW]).round().astype(int)):
            layout[tuple(p)] = V[t][k]
    # card 098.2's convention: O is the world's shift in the view, summed (the agent's motion with its sign turned)
    O = -A.astype(np.int64)
    np.savez(OUT / f"place_w{w}.npz", O=O)
    return float(np.mean(right))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    learned = json.loads(SHIFTS.read_text())["learned shifts (action: dx, dy)"]
    shifts = {int(a): np.array(s, np.float64) for a, s in learned.items()}
    rep = {}
    for w in range(10):
        rep[w] = round(place(w, shifts), 4)
        print("worker", w, "own motion right", rep[w], flush=True)
    (OUT / "place.json").write_text(json.dumps({"own motion right, per worker": rep}, indent=1) + "\n")
