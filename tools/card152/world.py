"""Card 152: the sprite world with objects that respond differently to the hand.

64 x 64, a 7 x 7 hand moved by up to 4 pixels a step, 2 to 4 objects.  An action is (dy, dx, grip); the grip is
applied first, where the hand is, then the hand moves and what it holds moves with it.  Three behaviours:
  takeable (0)  taken when the hand touches its side with the grip closed (1); let go when the grip opens
  fixed    (1)  never taken, never moves
  sticky   (2)  taken when the hand touches its side with the grip open (0); shaken off when the grip closes
One object is held at a time.  An object has a shape (5), a colour family (5), a width and its own texture.
In a world "by shape" the behaviour goes with the shape, in a world "by colour" with the colour family; which
shapes or colours go with which behaviour is drawn per world.  A world also has a pool of 40 individual objects
that come back in later episodes; 8 of them are exceptions that do not behave as their looks say.
"""
import importlib.util, numpy as np
_spec = importlib.util.spec_from_file_location("world147", "tools/card147/world.py")      # card 147's sprites and background
_w147 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_w147)
shape_mask, hand_sprite, background, SHAPES = _w147.shape_mask, _w147.hand_sprite, _w147.background, _w147.SHAPES

H, HAND, T = 64, 7, 60
SIZES, NEW_SIZES = (6, 8, 12, 16), (10, 14)
BASE = np.array([(0.9, 0.2, 0.2), (0.2, 0.8, 0.25), (0.25, 0.35, 0.95), (0.9, 0.85, 0.2), (0.85, 0.25, 0.9)])
POOL, EXCEPTIONS = 40, 8


def new_object(rng, world, sizes=SIZES):
    shape, fam, s = int(rng.integers(5)), int(rng.integers(5)), int(rng.choice(sizes))
    col = BASE[fam] + rng.uniform(-0.07, 0.07, 3)
    rule = int(world["kind_of"][shape if world["by"] == "shape" else fam])
    return dict(size=s, shape=shape, fam=fam, mask=shape_mask(SHAPES[shape], s), tex=np.clip(col + rng.normal(0, 0.10, (s, s, 3)), 0, 1),
                kind=rule, rule=rule, uid=-1)


def make_world(seed, by):
    rng = np.random.default_rng(seed)
    world = dict(by=by, seed=seed, kind_of=rng.permutation([0, 0, 1, 1, 2]))
    pool = [new_object(rng, world) for _ in range(POOL)]
    for u, o in enumerate(pool):
        o["uid"] = u
        if u < EXCEPTIONS:
            o["kind"] = int(rng.choice([k for k in range(3) if k != o["rule"]]))
    world["pool"] = pool
    return world


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


def new_state(rng, objs, overlap=False):
    while True:
        boxes = []
        for s in [HAND] + [o["size"] for o in objs]:
            for _ in range(200):
                p = rng.integers(0 if overlap else 1, H - s + 1 if overlap else H - s - 1, 2)
                if overlap or all(p[0] + s + 1 < q[0] or q[0] + t + 1 < p[0] or p[1] + s + 1 < q[1] or q[1] + t + 1 < p[1] for q, t in boxes):
                    boxes.append((p, s)); break
        if len(boxes) == len(objs) + 1:
            return dict(hand=boxes[0][0].copy(), pos=np.array([b[0] for b in boxes[1:]]), order=[int(i) for i in rng.permutation(len(objs))], held=-1)


def step(st, a, objs):
    hand, pos, held = st["hand"].copy(), st["pos"].copy(), st["held"]
    if held >= 0:
        if (objs[held]["kind"] == 0 and a[2] == 0) or (objs[held]["kind"] == 2 and a[2] == 1):
            held = -1
    else:
        want = 0 if a[2] == 1 else 2                              # the behaviour that this grip takes
        held = next((i for i in reversed(st["order"]) if objs[i]["kind"] == want and touching(hand, pos[i], objs[i]["size"])), -1)
    lo, hi = -hand, (H - HAND) - hand
    if held >= 0:
        lo, hi = np.maximum(lo, -pos[held]), np.minimum(hi, (H - objs[held]["size"]) - pos[held])
    d = np.clip(np.array(a[:2]), lo, hi)
    hand = hand + d
    if held >= 0:
        pos[held] = pos[held] + d
    return dict(st, hand=hand, pos=pos, held=held)


def render(rng, bg, st, objs):
    img = bg.copy(); mk = np.zeros((H, H), np.uint8)
    hm, ht = hand_sprite()
    for i in st["order"]:
        p, s, m = st["pos"][i], objs[i]["size"], objs[i]["mask"]
        img[p[0]:p[0] + s, p[1]:p[1] + s][m] = objs[i]["tex"][m]
        mk[p[0]:p[0] + s, p[1]:p[1] + s][m] = 3 + i
    p = st["hand"]
    img[p[0]:p[0] + HAND, p[1]:p[1] + HAND][hm] = ht[hm]; mk[p[0]:p[0] + HAND, p[1]:p[1] + HAND][hm] = 1
    img = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1)
    return (img * 255).round().astype(np.uint8), mk


def draw_objects(rng, world, source, sizes=SIZES):
    n = int(rng.integers(2, 5))
    if source == "pool":
        return [world["pool"][u] for u in rng.choice(POOL, n, replace=False)]
    if source == "fresh":
        return [new_object(rng, world, sizes) for _ in range(n)]
    objs = [world["pool"][u] for u in rng.choice(POOL, n, replace=False)]
    return [o if rng.random() < 0.5 else new_object(rng, world, sizes) for o in objs]


def collect(world, episodes, seed, source="mixed", sizes=SIZES, eps=0.2):
    """The agent's own experience.  A script that does not know the behaviours: go to an object's side, try a grip
    (open or closed, at random), carry what comes along to a place and reverse the grip there; or wander; with a share
    `eps` of random actions.  Returns episodes: the objects, the first frame and its mask, and every state and action."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(episodes):
        objs = draw_objects(rng, world, source, sizes); n = len(objs)
        st = new_state(rng, objs); frame, mask = render(rng, background(rng), st, objs)
        ep = dict(objs=objs, frame=frame, mask=mask, order=st["order"], hand=[st["hand"]], pos=[st["pos"]], held=[-1], acts=[])

        def pick():
            if rng.random() < 0.25:
                return ["wander", rng.integers(0, H - HAND + 1, 2), int(rng.integers(2))]
            return ["obj", int(rng.integers(n)), int(rng.integers(2)), 0]
        goal = pick(); last_g = 0
        for t in range(T):
            if st["held"] >= 0 and goal[0] != "carry":
                goal = ["carry", rng.integers(0, H - HAND + 1, 2), last_g]
            if st["held"] < 0 and goal[0] == "carry":
                goal = pick()
            if goal[0] == "obj":
                i = goal[1]
                if touching(st["hand"], st["pos"][i], objs[i]["size"]):
                    g = int(rng.integers(2)) if goal[3] == 0 else 1 - last_g
                    goal[3] += 1
                    d = rng.integers(-4, 5, 2) if rng.random() < 0.5 else np.zeros(2, int)
                    a = (d[0], d[1], g)
                    if goal[3] >= 2 or rng.random() < 0.4:
                        goal = pick()
                else:
                    a = (*np.clip(side_spot(st["hand"], st["pos"][i], objs[i]["size"]) - st["hand"], -4, 4), goal[2])
            elif goal[0] == "carry":
                if np.all(st["hand"] == goal[1]):
                    a = (0, 0, 1 - goal[2])
                else:
                    a = (*np.clip(goal[1] - st["hand"], -4, 4), goal[2])
            else:
                a = (*np.clip(goal[1] - st["hand"], -4, 4), goal[2])
                if np.all(st["hand"] == goal[1]):
                    goal = pick()
            if rng.random() < eps:
                a = (rng.integers(-4, 5), rng.integers(-4, 5), rng.integers(2))
            a = tuple(int(v) for v in a); last_g = a[2]
            nxt = step(st, a, objs)
            if goal[0] == "carry" and nxt["held"] >= 0 and np.all(nxt["hand"] == st["hand"]) and not np.all(st["hand"] == goal[1]):
                goal[1] = nxt["hand"].copy()                      # blocked at the edge: put it down here
            ep["hand"].append(nxt["hand"]); ep["pos"].append(nxt["pos"]); ep["held"].append(nxt["held"]); ep["acts"].append(a)
            st = nxt
        for k in ("hand", "pos", "held", "acts"):
            ep[k] = np.array(ep[k])
        out.append(ep)
    return out


def tokens(eps):
    """per step: the hand's effect and the action (G); per object: in contact before, held before, its displacement, held after (O)"""
    N = len(eps) * T
    G = np.zeros((N, 5), int); O = np.zeros((N, 4, 5), int); m = np.zeros((N, 4), bool)
    for e, ep in enumerate(eps):
        for t in range(T):
            k = e * T + t; d = ep["hand"][t + 1] - ep["hand"][t]; a = ep["acts"][t]
            G[k] = (d[0] + 4, d[1] + 4, a[0] + 4, a[1] + 4, a[2])
            for i, o in enumerate(ep["objs"]):
                mv = ep["pos"][t + 1, i] - ep["pos"][t, i]
                O[k, i] = (touching(ep["hand"][t], ep["pos"][t, i], o["size"]), ep["held"][t] == i, mv[0] + 4, mv[1] + 4, ep["held"][t + 1] == i)
                m[k, i] = True
    return G, O, m


def scenes(world, n, seed, sizes=SIZES):
    """still scenes for the test of recognition: 4 to 6 objects (pool or new, half and half) and the hand put down
    anywhere, so that they cover each other.  Returns per scene the frame, the mask, the objects, where they are."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        k = int(rng.integers(4, 7))
        objs = [world["pool"][u] if rng.random() < 0.5 else new_object(rng, world, sizes) for u in rng.choice(POOL, k, replace=False)]
        st = new_state(rng, objs, overlap=True)
        frame, mask = render(rng, background(rng), st, objs)
        out.append(dict(objs=objs, frame=frame, mask=mask, pos=st["pos"], hand=st["hand"]))
    return out


if __name__ == "__main__":
    for by in ("shape", "colour"):
        w = make_world(0, by); eps = collect(w, 100, 1)
        G, O, m = tokens(eps)
        kind = np.array([[o["kind"] for o in ep["objs"]] + [-1] * (4 - len(ep["objs"])) for ep in eps]).repeat(T, 0)
        info = (O[..., 0] | O[..., 1]) & m
        print(by, "kind of each", "shape" if by == "shape" else "colour", w["kind_of"], "| steps", len(G), "| object-steps", int(m.sum()), "| in contact or held", int(info.sum()))
        for k, name in enumerate(("takeable", "fixed", "sticky")):
            s = info & (kind == k)
            print("  %-8s in contact or held %5d; taken %4d; carried %5d; let go %4d; touched and not taken: grip closed %4d, open %4d" % (
                name, s.sum(), (s & (O[..., 1] == 0) & (O[..., 4] == 1)).sum(), (s & (O[..., 1] == 1) & (O[..., 4] == 1)).sum(),
                (s & (O[..., 1] == 1) & (O[..., 4] == 0)).sum(), (s & (O[..., 1] == 0) & (O[..., 4] == 0) & (G[:, None, 4] == 1)).sum(),
                (s & (O[..., 1] == 0) & (O[..., 4] == 0) & (G[:, None, 4] == 0)).sum()))
        sc = scenes(w, 200, 2)
        vis = [((s["mask"] == 3 + i).sum() / o["mask"].sum()) for s in sc for i, o in enumerate(s["objs"])]
        print("  still scenes: objects", len(vis), "| wholly seen %.0f%%, a quarter or more hidden %.0f%%, wholly hidden %.0f%%" % (
            100 * np.mean(np.array(vis) == 1), 100 * np.mean(np.array(vis) <= 0.75), 100 * np.mean(np.array(vis) == 0)))
