"""Card 053, step 2b: version 10's agent (card 051's walk2, familiar worlds) on a frozen encoder from card 052's
harness, in place of its own per-world encoder.

The agent's 130 catalogue tiles (26 objects x 5 agent states, src/worldmodel/envs/keydoor_render.py) are drawn as
card 052's generator draws them: MiniGrid's object drawing at 8 pixels without the grid lines, no noise, with the
agent's triangle where the agent stands. The frozen encoder maps them to vectors (4 unit parts of 8) and its
codebooks (unit entries) replace the agent's; everything downstream (handles, card 035's codes with radii from the
20 familiar tiles, card 036's fresh codes, recall, the planner) is version 10's own. The generator never draws the
agent on a tile or a goal square: those reach the encoder as unseen appearances.

  bin/prun python tools/card053/planner_check.py runs/053/s1_T0_399.pt --out runs/053/s2b_T0.json [--gates 0]
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fn

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "card053"))
sys.path.insert(0, str(TOOLS / "card051"))
import recall_probe as RP                              # noqa: E402  (loads card 052's encoder classes)
import walk2                                           # noqa: E402
from minigrid.utils.rendering import downsample, fill_coords, point_in_triangle, rotate_fn   # noqa: E402
from worldmodel.envs import keydoor_render as KR      # noqa: E402

NV = RP.NV


def catalogue():
    """(130, 8, 8, 3) uint8: every object and agent state, without grid lines."""
    out = np.zeros((KR.N_CODES, KR.TILE, KR.TILE, 3), np.uint8)
    for o in KR.OBJECTS:
        for a in (None, 0, 1, 2, 3):
            img = np.zeros((KR.TILE * 3, KR.TILE * 3, 3), np.uint8)
            obj = KR._minigrid_obj(o)
            if obj is not None:
                obj.render(img)
            if a is not None:                          # MiniGrid's agent triangle (Grid.render_tile)
                tri = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
                fill_coords(img, rotate_fn(tri, cx=0.5, cy=0.5, theta=0.5 * math.pi * a), (255, 0, 0))
            out[KR.code(o, a)] = downsample(img, 3)
    return out


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    path, out = args[0], get("--out", "runs/053/s2b.json")
    enc = RP.load(path, get("--interaction", "transition"), get("--gates", "0") == "1")
    tiles = catalogue()
    z = RP.vectors(enc, list(tiles)).astype(np.float32)
    books = fn.normalize(enc.books.detach(), dim=-1).cpu().numpy().astype(np.float32)

    def encoder(mu, seed, tr_tiles, pairs, groups, dev, log):
        return {"z": z, "books": books, "rep": {"encoder": "card 053, frozen", "weights": str(path)}}

    NV.encoder = encoder
    sys.argv = ["walk2.py", "--dev", "--arm", "A", "--seeds", "399-399", "--layouts", get("--layouts", "30"),
                "--out", out]
    walk2.main()


if __name__ == "__main__":
    main()
