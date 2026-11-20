"""A small sprite world for testing binding from the agent's own interaction (card 147).

64 x 64 pixels, fixed camera.  A hand the agent moves by up to 4 pixels a step; it takes hold of an
object by touching its side and closing, carries it, and lets go.  2 to 4 objects.  In half the
episodes a button changes colour when the hand closes on it and, while on, stops object 0 from being
taken.  Everything is textured, so every pixel says how it moved, and every frame has fresh sensor
noise.  New objects and a new background every episode; the hand looks the same in all of them.
Objects lie over the button and over each other in an order fixed for the episode (an object can be
slid under another), the hand over everything.
The true mask of every pixel is kept for scoring only.

mask ids: 0 background, 1 hand, 2 button, 3 + i object i.
usage: python tools/card147/world.py <out_dir> <split> <episodes> <seed> [small]
       ("small" draws every object 3 or 4 pixels wide: the split that asks about small objects)
"""
import sys, json, numpy as np

H = 64
T = 60
SIZES = (4, 6, 8, 12, 16)
SHAPES = ("square", "disc", "triangle", "ell", "plus")
HAND = 7


def shape_mask(shape, s):
    y, x = np.mgrid[0:s, 0:s]
    c = (s - 1) / 2
    if s <= 4 or shape == "square":
        return np.ones((s, s), bool)
    if shape == "disc":
        return (y - c) ** 2 + (x - c) ** 2 <= (s / 2) ** 2
    if shape == "triangle":
        return np.abs(x - c) <= (y + 1) / 2
    w = max(2, s // 3)
    if shape == "ell":
        return (x < w) | (y >= s - w)
    return (np.abs(x - c) < w / 2 + 0.01) | (np.abs(y - c) < w / 2 + 0.01)        # plus


def hand_sprite():
    y, x = np.mgrid[0:HAND, 0:HAND]
    m = (np.abs(x - 3) <= 1) | (np.abs(y - 3) <= 1)              # a thick cross
    tex = np.random.default_rng(12345).uniform(0.25, 1.0, (HAND, HAND, 1)) * np.ones(3)   # grey speckle, the same in
    return m, tex                                                # every episode; not a pattern that repeats under a shift


def background(rng):
    low = rng.uniform(0.15, 0.55, (9, 9, 3))
    g = np.linspace(0, 8, H)
    i0 = np.clip(g.astype(int), 0, 7); f = g - i0
    rows = low[i0] * (1 - f)[:, None, None] + low[i0 + 1] * f[:, None, None]
    img = rows[:, i0] * (1 - f)[None, :, None] + rows[:, i0 + 1] * f[None, :, None]
    return np.clip(img + rng.normal(0, 0.05, (H, H, 3)), 0, 1)


def touching(hand, o):
    y, x = o["pos"]; s = o["size"]; hy, hx = hand
    rows = hy < y + s and y < hy + HAND
    cols = hx < x + s and x < hx + HAND
    return bool((rows and (hx + HAND == x or x + s == hx)) or (cols and (hy + HAND == y or y + s == hy)))


def side_spot(hand, o):
    """the nearest place for the hand's corner where the hand touches a side of object o and stays in view"""
    y, x = o["pos"]; s = o["size"]; c = s // 2 - 3
    cl = lambda v: int(np.clip(v, 0, H - HAND))
    spots = []
    if x - HAND >= 0: spots.append((cl(y + c), x - HAND))
    if x + s + HAND <= H: spots.append((cl(y + c), x + s))
    if y - HAND >= 0: spots.append((y - HAND, cl(x + c)))
    if y + s + HAND <= H: spots.append((y + s, cl(x + c)))
    spots = np.array(spots)
    return spots[np.abs(spots - hand).max(1).argmin()]


def episode(rng, small=False):
    bg = background(rng)
    hm, ht = hand_sprite()
    n = int(rng.integers(2, 5))
    objs = []
    for i in range(n):
        s = int(rng.choice((3, 4))) if small else int(rng.choice(SIZES))
        shape = str(rng.choice(SHAPES))
        col = rng.uniform(0, 1, 3); col[rng.integers(3)] = rng.uniform(0.7, 1.0)
        m = shape_mask(shape, s)
        objs.append(dict(size=s, shape=shape, mask=m, tex=np.clip(col[None, None] + rng.normal(0, 0.10, (s, s, 3)), 0, 1)))
    alike = bool(rng.random() < 0.3)
    if alike:                                                    # two objects that look exactly the same
        objs[1] = dict(objs[0])
    button = bool(rng.random() < 0.5)
    bm = shape_mask("disc", 6)
    bnoise = rng.normal(0, 0.08, (6, 6, 3))
    btex = [np.clip(np.array(c)[None, None] + bnoise, 0, 1) for c in ((0.85, 0.15, 0.15), (0.15, 0.85, 0.15))]
    boxes = []

    def place(s):                                                # nothing overlaps at the start
        for _ in range(1000):
            p = rng.integers(1, H - s - 1, 2)
            if all(p[0] + s + 1 < q[0] or q[0] + t + 1 < p[0] or p[1] + s + 1 < q[1] or q[1] + t + 1 < p[1] for q, t in boxes):
                boxes.append((p, s)); return p.copy()
        raise RuntimeError("no room")
    hand = place(HAND)
    bpos = place(6) if button else None
    for o in objs:
        o["pos"] = place(o["size"])
    held, on = -1, False
    order = [int(i) for i in rng.permutation(n)]                 # which object lies over which: fixed for the episode
    frames = np.zeros((T + 1, H, H, 3), np.uint8); masks = np.zeros((T + 1, H, H), np.uint8)
    hands = np.zeros((T + 1, 2), np.int16); acts = np.zeros((T, 3), np.int8)
    helds = np.full(T + 1, -1, np.int8); ons = np.zeros(T + 1, bool); opos = np.zeros((T + 1, 4, 2), np.int16)

    def render(t):
        img = bg.copy(); mk = np.zeros((H, H), np.uint8)

        def paint(pos, m, tex, idx):
            s = m.shape[0]
            img[pos[0]:pos[0] + s, pos[1]:pos[1] + s][m] = tex[m]
            mk[pos[0]:pos[0] + s, pos[1]:pos[1] + s][m] = idx
        if button:
            paint(bpos, bm, btex[int(on)], 2)
        for i in order:
            paint(objs[i]["pos"], objs[i]["mask"], objs[i]["tex"], 3 + i)
        paint(hand, hm, ht, 1)
        img = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1)       # sensor noise, new in every frame
        frames[t] = (img * 255).round().astype(np.uint8); masks[t] = mk
        hands[t] = hand; helds[t] = held; ons[t] = on
        for i, o in enumerate(objs):
            opos[t, i] = o["pos"]

    def pick_goal():
        r = rng.random()
        if button and r < 0.15:
            return ("button", bpos.copy())                       # the hand's centre on the button's centre
        if r < 0.30:
            return ("wander", rng.integers(0, H - HAND + 1, 2))
        return ("obj", side_spot(hand, objs[int(rng.integers(n))]))

    goal = pick_goal()
    render(0)
    for t in range(T):
        d = np.clip(goal[1] - hand, -4, 4)
        if rng.random() < 0.15:
            d = np.clip(d + rng.integers(-1, 2, 2), -4, 4)
        lo, hi = -hand, (H - HAND) - hand                        # keep the hand, and what it holds, in view
        if held >= 0:
            p, s = objs[held]["pos"], objs[held]["size"]
            lo, hi = np.maximum(lo, -p), np.minimum(hi, (H - s) - p)
        d = np.clip(d, lo, hi)
        hand = hand + d
        if held >= 0:
            objs[held]["pos"] = objs[held]["pos"] + d
        closed = 0
        if np.all(hand == goal[1]) or np.all(d == 0):
            if goal[0] == "wander":
                goal = pick_goal()
            elif goal[0] == "carry":
                held = -1; goal = pick_goal()
            elif goal[0] == "button":
                c = hand + 3 - bpos; closed = 1
                if np.all(c >= 0) and np.all(c < 6) and bm[c[0], c[1]]:
                    on = not on
                goal = pick_goal()
            else:
                closed = 1
                top = next((i for i in reversed(order) if touching(hand, objs[i])), -1)
                if top >= 0 and not (button and on and top == 0):
                    held = top
                    goal = ("carry", rng.integers(0, H - HAND + 1, 2))
                else:
                    goal = pick_goal()
        acts[t] = (d[0], d[1], int(closed or held >= 0))
        render(t + 1)
    meta = dict(n=n, sizes=[o["size"] for o in objs], shapes=[o["shape"] for o in objs], alike=alike, button=button)
    return frames, masks, hands, acts, helds, ons, opos, meta


def main(out, split, n, seed, small=False):
    rng = np.random.default_rng(seed)
    eps = []
    while len(eps) < n:
        try:
            eps.append(episode(rng, small))
        except RuntimeError:                                     # the draw left no room for an object: draw again
            pass
    for k, name in enumerate(("frames", "masks", "hand", "actions", "held", "button_on", "obj_pos")):
        np.save(f"{out}/{split}_{name}.npy", np.stack([e[k] for e in eps]))
    json.dump([e[7] for e in eps], open(f"{out}/{split}_meta.json", "w"))
    held = np.stack([e[4] for e in eps]); pos = np.stack([e[6] for e in eps])
    moved = np.mean([(np.abs(np.diff(pos[e, :, :m["n"]], axis=0)).sum((0, 2)) > 0).mean() for e, m in enumerate(e[7] for e in eps)])
    print(split, n, "episodes; steps with an object held %.0f%%; objects per episode %.1f; objects that get moved %.0f%%" % (
        100 * (held >= 0).mean(), np.mean([e[7]["n"] for e in eps]), 100 * moved))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), len(sys.argv) > 5)
