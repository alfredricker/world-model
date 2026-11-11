"""Card 124, after the first pass: how do the two arms' tokens divide the frame?

On validation segments (25 episodes x 6), with each patch given to the token whose decoder mask is largest:
- fixed: per patch, the share of frames in which it belongs to its most common token, averaged over patches.
  1.0 = the tokens are fixed regions of the image whatever is in them. The same number for the true objects
  (each patch labelled arm, cube, ..., or background) is the reference.
- tokens in use: the effective number by area, exp(entropy of the area shares).
- per object: the share of its patches in its best token (recall), the share of that token on the object
  (precision), and the share of its patches in the token that holds most of the arm.

  MUJOCO_GL=egl ... bin/prun python tools/card124/diagnose.py → runs/124/diagnose.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402,F401
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card115'))
from train import DATA, EP, OUT, ROOT, SKIP, T, TokenModel, features  # noqa: E402
from encode import load_encoder  # noqa: E402
from masks import OBJECTS  # noqa: E402

DEV, R = 'cuda', 256


def fixedness(labels, n):
    """labels [frames, N] ints in 0..n-1 → mean over patches of the modal label's share."""
    counts = np.stack([(labels == k).mean(0) for k in range(n)])
    return float(counts.max(0).mean())


def main():
    enc = load_encoder(DEV)
    z = np.load(DATA / 'scene-play-v0-val.npz')
    qpos, btn, act, con = z['qpos'], z['button_states'], z['actions'], z['observations'][:, 18]
    masks = np.load(ROOT / 'runs/122/computed_val.npz')['masks'].astype(bool)
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection', width=R, height=R)
    env.reset(seed=0)
    u = env.unwrapped
    qvel0 = np.zeros(u._model.nv)
    rng = np.random.default_rng(1)
    segs = [ep * EP + s + SKIP * np.arange(T) for ep in range(1, 100, 4)
            for s in rng.choice(np.arange(0, EP - SKIP * (T + 1), SKIP), 6, replace=False)]
    feats, acts = [], []
    for rows in segs:
        fr = np.zeros((T, R, R, 3), np.uint8)
        for i, r in enumerate(rows):
            u.set_state(qpos[r].astype(np.float64), qvel0, btn[r].astype(np.int64))
            fr[i] = u.render()
        feats.append(features(enc, torch.from_numpy(fr).to(DEV)).cpu())
        acts.append(np.stack([np.r_[act[r:r + SKIP].reshape(-1), con[r]] for r in rows]).astype(np.float32))
    last = np.array([rows[-1] for rows in segs])
    true = masks[last]                                                  # [S, 6, N]
    truelab = np.where(true.any(1), true.argmax(1), 6)                  # background = 6
    res = dict(segments=len(segs), true_objects_fixed=round(fixedness(truelab, 7), 3))
    for name in [n for n in ['armA.pt', 'armB.pt', 'armC.pt'] if (OUT / n).exists()]:
        ck = torch.load(OUT / name, map_location=DEV)
        model = TokenModel(K=ck['K']).to(DEV)
        model.load_state_dict(ck['state'])
        model.eval()
        hard = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for f, a in zip(feats, acts):
                _, m, *_ = model(f[None].to(DEV), torch.from_numpy(a)[None].to(DEV))
                hard.append(m[0, -1].float().argmax(0).cpu().numpy())
        hard = np.stack(hard)                                           # [S, N]
        K = model.K
        area = np.array([(hard == k).mean() for k in range(K)])
        p = area[area > 0]
        entry = dict(fixed=round(fixedness(hard, K), 3), tokens_in_use=round(float(np.exp(-(p * np.log(p)).sum())), 2),
                     area_shares=[round(float(x), 3) for x in sorted(area, reverse=True)])
        per = {o: dict(recall=[], precision=[], with_arm=[]) for o in OBJECTS}
        for s in range(len(segs)):
            arm_tok = np.bincount(hard[s][true[s, 0]], minlength=K).argmax() if true[s, 0].any() else -1
            for oi, o in enumerate(OBJECTS):
                if not true[s, oi].any():
                    continue
                inter = np.array([((hard[s] == k) & true[s, oi]).sum() for k in range(K)])
                union = np.array([((hard[s] == k) | true[s, oi]).sum() for k in range(K)])
                k = int((inter / union).argmax())
                per[o]['recall'].append(inter[k] / true[s, oi].sum())
                per[o]['precision'].append(inter[k] / max((hard[s] == k).sum(), 1))
                per[o]['with_arm'].append((hard[s][true[s, oi]] == arm_tok).mean())
        entry['objects'] = {o: {k: round(float(np.mean(v)), 3) for k, v in d.items()} for o, d in per.items()}
        res[name] = entry
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / 'diagnose.json', 'w'), indent=1)
    os._exit(0)


if __name__ == '__main__':
    main()
