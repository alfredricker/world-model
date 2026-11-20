"""The sprite world of card 147 as a simulator that takes an action, read as state (no pixels; card 148).

64 x 64, a 7 x 7 hand, 2 to 4 square-boxed objects of width 4 to 16.  An action is (dy, dx, grip):
the grip is applied first, where the hand is, then the hand moves by up to 4 pixels a side.
  grip 1, nothing held, the hand touching the side of an object: the hand takes the topmost such object;
  grip 1, nothing touched: nothing (the hand springs open again);  grip 1 while holding: keep holding;
  grip 0: let go.
What is held moves with the hand; the hand and what it holds stay in view.  No button and no lock.
"""
import numpy as np

H, HAND, T = 64, 7, 60
SIZES = (4, 6, 8, 12, 16)


def touching(hand, pos, s):
    hy, hx = hand; y, x = pos
    rows = hy < y + s and y < hy + HAND
    cols = hx < x + s and x < hx + HAND
    return bool((rows and (hx + HAND == x or x + s == hx)) or (cols and (hy + HAND == y or y + s == hy)))


def side_spot(hand, pos, s):
    y, x = pos; c = s // 2 - 3
    cl = lambda v: int(np.clip(v, 0, H - HAND))
    spots = []
    if x - HAND >= 0: spots.append((cl(y + c), x - HAND))
    if x + s + HAND <= H: spots.append((cl(y + c), x + s))
    if y - HAND >= 0: spots.append((y - HAND, cl(x + c)))
    if y + s + HAND <= H: spots.append((y + s, cl(x + c)))
    spots = np.array(spots)
    return spots[np.abs(spots - hand).max(1).argmin()]


def new_state(rng):
    while True:
        n = int(rng.integers(2, 5)); size = rng.choice(SIZES, n)
        boxes = []
        for s in [HAND] + list(size):
            for _ in range(200):
                p = rng.integers(1, H - s - 1, 2)
                if all(p[0] + s + 1 < q[0] or q[0] + t + 1 < p[0] or p[1] + s + 1 < q[1] or q[1] + t + 1 < p[1] for q, t in boxes):
                    boxes.append((p, s)); break
        if len(boxes) == n + 1:
            return dict(hand=boxes[0][0].copy(), pos=np.array([b[0] for b in boxes[1:]]), size=np.array(size),
                        order=[int(i) for i in rng.permutation(n)], held=-1)


def step(st, a):
    hand, pos, held = st["hand"].copy(), st["pos"].copy(), st["held"]
    if a[2] == 0:
        held = -1
    elif held < 0:
        held = next((i for i in reversed(st["order"]) if touching(hand, pos[i], st["size"][i])), -1)
    lo, hi = -hand, (H - HAND) - hand
    if held >= 0:
        lo, hi = np.maximum(lo, -pos[held]), np.minimum(hi, (H - st["size"][held]) - pos[held])
    d = np.clip(np.array(a[:2]), lo, hi)
    hand = hand + d
    if held >= 0:
        pos[held] = pos[held] + d
    return dict(st, hand=hand, pos=pos, held=held)


def collect(episodes, seed, eps=0.2):
    """The agent's own experience: a scripted behaviour (go to an object, take it, carry it somewhere, let go, or
    wander) with a share `eps` of random actions.  Returns a list of (state before, action, state after)."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(episodes):
        st = new_state(rng); n = len(st["size"])

        def pick():
            if rng.random() < 0.3:
                return ("wander", rng.integers(0, H - HAND + 1, 2))
            return ("obj", int(rng.integers(n)))
        goal = pick()
        for t in range(T):
            if st["held"] >= 0 and goal[0] != "carry":
                goal = ("carry", rng.integers(0, H - HAND + 1, 2))
            if st["held"] < 0 and goal[0] == "carry":
                goal = pick()
            g = int(st["held"] >= 0)
            if goal[0] == "obj":
                i = goal[1]
                if touching(st["hand"], st["pos"][i], st["size"][i]):
                    g = 1; goal = ("carry", rng.integers(0, H - HAND + 1, 2)); target = goal[1]
                else:
                    target = side_spot(st["hand"], st["pos"][i], st["size"][i])
            else:
                target = goal[1]
                if np.all(st["hand"] == target):
                    if goal[0] == "carry":
                        g = 0
                    goal = pick()
                    target = goal[1] if goal[0] != "obj" else side_spot(st["hand"], st["pos"][goal[1]], st["size"][goal[1]])
            a = (*np.clip(target - st["hand"], -4, 4), g)
            if rng.random() < eps:
                a = (int(rng.integers(-4, 5)), int(rng.integers(-4, 5)), int(rng.integers(2)))
            a = tuple(int(v) for v in a)
            nxt = step(st, a)
            if goal[0] == "carry" and nxt["held"] >= 0 and np.all(nxt["hand"] == st["hand"]) and not np.all(st["hand"] == goal[1]):
                goal = ("carry", nxt["hand"].copy())               # blocked at the edge: this is where it is put down
            out.append((st, a, nxt)); st = nxt
    return out
