"""Card 167: True 7x7 occluded egocentric partial-view adapter and persistent world-frame belief map.

What the learner receives (GOAL C1 & P2):
  - MiniGrid's native 7x7 line-of-sight partial observation (env.unwrapped.gen_obs()['image'], occluded by walls and closed doors)
  - Proprioception: the agent's own pose (x, y, heading) in the world frame, placing the 7x7 view into Belief.map
  - The cell directly under the agent (half, S-1) and what the agent carries are NEVER read (as in Cards 160-166).

Everything outside the current 7x7 visibility cone retains its last-seen code (P2); unseen cells stay UNK (0).
UNK -> c is discovery (never fires a world event or corrupts items); c (!= UNK) -> c2 (!= c) is a world event.
"""
import numpy as np, gymnasium as gym, minigrid
from collections import Counter
from grid import DIRV, UNK


def project_partial(obs_img, pose, codes, shape):
    """Project 7x7 egocentric MiniGrid obs_img (S, S, 3) into a world-frame (obs_map, vis_mask) of given shape.
    Pixels with obs_img[lx, ly, 0] == 0 are unseen/occluded; (half, S-1) under the agent is strictly skipped."""
    S = obs_img.shape[0]; half = S // 2
    px, py, pd = pose
    fx, fy = DIRV[pd]
    rx, ry = -fy, fx
    W, H = shape
    obs_map = np.full((W, H), UNK, dtype=int)
    vis_mask = np.zeros((W, H), dtype=bool)

    # Vectorized projection over all S*S local cells
    lx, ly = np.nonzero(obs_img[..., 0] != 0)
    keep = ~((lx == half) & (ly == S - 1))
    lx, ly = lx[keep], ly[keep]
    if len(lx):
        wx = px + (S - 1 - ly) * fx + (lx - half) * rx
        wy = py + (S - 1 - ly) * fy + (lx - half) * ry
        inb = (wx >= 0) & (wx < W) & (wy >= 0) & (wy < H)
        wx, wy, lx, ly = wx[inb], wy[inb], lx[inb], ly[inb]
        if len(wx):
            obs_map[wx, wy] = codes.read(obs_img[lx, ly])
            vis_mask[wx, wy] = True
    return obs_map, vis_mask


class WorldPartial:
    """MiniGrid environment wrapper exposing ONLY the 7x7 occluded egocentric partial view + proprioception (x, y, d)."""
    def __init__(s, name, codes):
        s.env = gym.make(name); s.u = s.env.unwrapped; s.codes = codes; s.name = name

    def _obs(s):
        obs_img = s.u.gen_obs()["image"]
        pose = (int(s.u.agent_pos[0]), int(s.u.agent_pos[1]), int(s.u.agent_dir))
        shape = (int(s.u.width), int(s.u.height))
        obs_map, vis_mask = project_partial(obs_img, pose, s.codes, shape)
        return obs_map, vis_mask, pose

    def reset(s, seed):
        s.env.reset(seed=int(seed))
        return s._obs()

    def step(s, a):
        _, r, te, tr, _ = s.env.step(int(a))
        obs_map, vis_mask, pose = s._obs()
        return obs_map, vis_mask, pose, bool(r > 0), bool(te or tr)


class PartialBelief:
    """Persistent world-frame belief map updated strictly from 7x7 partial views.
    Unseen cells start as UNK; seen cells persist out of view; UNK -> c is discovery (not an event)."""
    def __init__(s, obs_map, vis_mask, pose):
        s.map = np.full(obs_map.shape, UNK, dtype=int)
        s.map[vis_mask] = obs_map[vis_mask]
        s.map[pose[0], pose[1]] = UNK
        s.seen_union = vis_mask.copy()
        s.seen_union[pose[0], pose[1]] = False
        s.pose = pose
        s.B = Counter()
        s._items = frozenset()

    def items(s):
        return s._items

    def step(s, a, obs_map, vis_mask, pose):
        """Update belief strictly at vis_mask (never touching cells outside the 7x7 cone or under the agent).
        Returns (ev, revealed_cells): world events on previously seen cells, and list of newly discovered (x, y, code)."""
        vis = vis_mask.copy()
        vis[pose[0], pose[1]] = False
        s.seen_union |= vis
        ev = []; revealed = []
        xs, ys = np.nonzero(vis)
        for x, y in zip(xs, ys):
            x, y = int(x), int(y)
            c = int(s.map[x, y]); c2 = int(obs_map[x, y])
            if c == UNK:
                s.map[x, y] = c2
                revealed.append((x, y, c2))
            elif c2 != c:
                s.map[x, y] = c2
                ev.append((x, y, c, c2))
                s.B[(c, a)] += 1
                for it in list(s.B):
                    if it[0] == c2 and s.B[it] > 0:
                        s.B[it] -= 1; break
        if ev:
            s._items = frozenset(k for k, v in s.B.items() if v > 0)
        s.pose = pose
        return ev, revealed
