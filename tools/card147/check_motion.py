"""Gate check: how well the motion front end recovers each pixel's true displacement (validation episodes)."""
import sys, numpy as np, torch
sys.path.insert(0, "tools/card147")
import motion as M
D = "data/sprite147/val_"
fr = np.load(D + "frames.npy")[:50]; mk = np.load(D + "masks.npy")[:50]; hand = np.load(D + "hand.npy")[:50]; op = np.load(D + "obj_pos.npy")[:50]
n = dict(mov=0, mov_valid=0, mov_right=0, sta=0, sta_valid=0, sta_false=0, frames=0)
for e in range(len(fr)):
    f = torch.tensor(fr[e]).cuda().permute(0, 3, 1, 2).float() / 255
    idx, valid = M.flow(*M.cost_volume(f[:-1], f[1:]))
    idx, valid = idx.cpu().numpy(), valid.cpu().numpy()
    tidx, tvalid = M.true_flow(mk[e], hand[e], op[e])
    mov = tvalid & (tidx != M.ZERO); sta = tvalid & (tidx == M.ZERO)
    n["mov"] += mov.sum(); n["mov_valid"] += (mov & valid).sum(); n["mov_right"] += (mov & valid & (idx == tidx)).sum()
    n["sta"] += sta.sum(); n["sta_valid"] += (sta & valid).sum(); n["sta_false"] += (sta & valid & (idx != M.ZERO)).sum(); n["frames"] += len(idx)
print("moving pixels: motion read for %.1f%%, right for %.2f%% of those" % (100 * n["mov_valid"] / n["mov"], 100 * n["mov_right"] / n["mov_valid"]))
print("still pixels:  motion read for %.1f%%; read as moving: %.2f per frame" % (100 * n["sta_valid"] / n["sta"], n["sta_false"] / n["frames"]))
