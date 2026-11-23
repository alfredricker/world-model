"""Card 160: MiniGrid as tokens, the agent's memory of them, and reward-free random play.

What the learner is given (GOAL C1): the full grid as tiles (one token per cell, named by a code the agent makes up for
each distinct tile), its own pose, its own action.  The tile under the agent and what the agent carries are not shown,
as in MiniGrid's full view.  Everything named by MiniGrid's type, colour or state is for the evaluator (rules.py).

usage: grid.py play <seed> <steps per world>     writes runs/160/exp_<seed>.pt and prints the data check
"""
import sys, time, pickle, numpy as np, gymnasium as gym, minigrid
from collections import Counter

DIRV = [(1, 0), (0, 1), (-1, 0), (0, -1)]
R = 2; SIDE = 2 * R + 1; NG = SIDE * SIDE * 4; FAR = NG              # where the agent is relative to a token: a window of two cells, times its heading; else FAR
NA = 7                                                                # MiniGrid's actions: left, right, forward, pickup, drop, toggle, done
OFF8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
UNK, OUT = 0, 1                                                       # codes: never seen (under the agent at the start), outside the grid
PLAY = ["MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-6x6-v0", "MiniGrid-Unlock-v0", "MiniGrid-UnlockPickup-v0", "MiniGrid-BlockedUnlockPickup-v0",
        "MiniGrid-PutNear-6x6-N2-v0", "MiniGrid-MultiRoom-N2-S4-v0"]     # the last two added by the data check: boxes within reach, closed doors on every side


def gid(dx, dy, d):
    """the token lies (dx, dy) from the agent, which heads d"""
    return ((dx + R) * SIDE + (dy + R)) * 4 + d


def gid_inv(g):
    q = g // 4
    return q // SIDE - R, q % SIDE - R, g % 4


class Codes:
    """names for distinct tiles, made up as they are seen"""
    def __init__(s):
        s.lut = np.full(11 * 18, -1, int); s.triples = [None, None]

    def read(s, enc):
        k = enc[..., 0].astype(int) * 18 + enc[..., 1] * 3 + enc[..., 2]; c = s.lut[k]
        if (c < 0).any():
            for kk in np.unique(k[c < 0]):
                s.lut[kk] = len(s.triples); s.triples.append((int(kk) // 18, int(kk) % 18 // 3, int(kk) % 3))
            c = s.lut[k]
        return c

    def of(s, t):                                                     # evaluator and rules only
        return int(s.read(np.array(t)[None])[0])


class World:
    def __init__(s, name, codes):
        s.env = gym.make(name); s.u = s.env.unwrapped; s.codes = codes; s.name = name

    def obs(s):
        return s.codes.read(s.u.grid.encode()), (int(s.u.agent_pos[0]), int(s.u.agent_pos[1]), int(s.u.agent_dir))

    def reset(s, seed):
        s.env.reset(seed=int(seed)); return s.obs()

    def step(s, a):
        _, r, te, tr, _ = s.env.step(int(a)); g, p = s.obs()
        return g, p, bool(r > 0), bool(te or tr)


class Belief:
    """each cell's last seen code, and the tokens out of view, each with the action at which it went"""
    def __init__(s, grid, pose):
        s.map = grid.copy(); s.pose = pose; s.map[pose[0], pose[1]] = UNK; s.B = Counter(); s._items = frozenset()

    def items(s):
        return s._items

    def step(s, a, grid, pose):
        seen = grid.copy(); seen[pose[0], pose[1]] = s.map[pose[0], pose[1]]; ev = []      # the cell under the agent is hidden: keep the belief
        for x, y in np.argwhere(seen != s.map):
            c, c2 = int(s.map[x, y]), int(seen[x, y])
            if c != UNK:
                ev.append((int(x), int(y), c, c2)); s.B[(c, a)] += 1                      # c went out of view at action a ...
                for it in list(s.B):
                    if it[0] == c2 and s.B[it] > 0:                                       # ... and c2 came back into view
                        s.B[it] -= 1; break
        if ev:
            s._items = frozenset(k for k, v in s.B.items() if v > 0)
        s.map = seen; s.pose = pose
        return ev


def cell_rows(before, pose, after, pose2):
    """every token that could be seen before and after the step: its code, where the agent was relative to it, its next code"""
    x0, y0, d = pose; valid = before != UNK; valid[x0, y0] = False; valid[pose2[0], pose2[1]] = False
    xs, ys = np.nonzero(valid); dx, dy = xs - x0, ys - y0
    g = np.where((abs(dx) <= R) & (abs(dy) <= R), ((dx + R) * SIDE + (dy + R)) * 4 + d, FAR)
    return before[xs, ys], g, after[xs, ys], xs, ys


def neighbours(mp, x, y):
    W, H = mp.shape
    return tuple(int(mp[x + dx, y + dy]) if 0 <= x + dx < W and 0 <= y + dy < H else OUT for dx, dy in OFF8)


def agent_outcome(pose, pose2):
    dx, dy = pose2[0] - pose[0], pose2[1] - pose[1]
    return (0 if (dx, dy) == (0, 0) else 1 + DIRV.index((dx, dy)), pose2[2])


def pack(c, a, g, it, o):
    return (((c.astype(np.int64) * 8 + a) * 128 + g) * (1 << 20) + it) * 256 + o


def unpack(k):
    o = k % 256; k = k // 256; it = k % (1 << 20); k = k // (1 << 20); g = k % 128; k = k // 128
    return k // 8, k % 8, g, it, o


class Table:
    """the experience, counted: how often each situation of a token was followed by each next code"""
    def __init__(s):
        s.codes = Codes(); s.item_id = {frozenset(): 0}; s.keys = np.zeros(0, np.int64); s.n = np.zeros(0, np.int64); s.agent = Counter(); s.steps = 0
        s._k, s._n = [], []

    def iid(s, items):
        if items not in s.item_id:
            s.item_id[items] = len(s.item_id)
        return s.item_id[items]

    def add(s, before, pose, items, a, after, pose2):
        c, g, o, _, _ = cell_rows(before, pose, after, pose2); it = s.iid(items)
        k, n = np.unique(pack(c, a, g, it, o), return_counts=True); s._k.append(k); s._n.append(n)
        s.agent[(pose[2], a, neighbours(before, pose[0], pose[1]), it, agent_outcome(pose, pose2))] += 1; s.steps += 1
        if len(s._k) >= 20000:
            s.merge()

    def merge(s):
        k = np.concatenate([s.keys] + s._k); n = np.concatenate([s.n] + s._n); s._k, s._n = [], []
        s.keys, inv = np.unique(k, return_inverse=True); s.n = np.bincount(inv, weights=n).astype(np.int64)

    def rows(s):
        s.merge(); c, a, g, it, o = unpack(s.keys)
        return c, a, g, it, o, s.n

    def item_sets(s):
        out = [None] * len(s.item_id)
        for k, v in s.item_id.items():
            out[v] = k
        return out


def play(seed, steps, names=PLAY, log=True):
    """uniform random actions, no reward, no goal"""
    T = Table(); rng = np.random.default_rng(seed); t0 = time.time(); T.succ = Counter(); T.episodes = Counter()
    for wi, name in enumerate(names):
        w = World(name, T.codes); ep = 0; g, p = w.reset(seed * 1000000 + wi * 100000); bel = Belief(g, p)
        for t in range(steps):
            a = int(rng.integers(NA)); before, pose, items = bel.map, bel.pose, bel.items()
            g, p, succ, done = w.step(a); bel.step(a, g, p); T.add(before, pose, items, a, bel.map, p)
            if done:
                ep += 1; T.succ[name] += succ; T.episodes[name] += 1; g, p = w.reset(seed * 1000000 + wi * 100000 + ep); bel = Belief(g, p)
        if log:
            print("  played", name, steps, "steps,", T.episodes[name], "episodes,", T.succ[name], "successes, %.0f s" % (time.time() - t0), flush=True)
    T.merge()
    return T


if __name__ == "__main__":
    import rules, grid                                             # (the module, not __main__, so the table can be read back)
    seed, steps = int(sys.argv[2]), int(sys.argv[3]); T = grid.play(seed, steps)
    pickle.dump(T, open("runs/160/exp_%d.pkl" % seed, "wb"))
    print("codes", len(T.codes.triples), "sets of tokens out of view", len(T.item_id), "distinct situations of tokens", len(T.keys), "of the agent", len(T.agent))
    rules.data_check(T)
