"""Card 124: the three criteria, on validation episodes of the state dataset (256-px frames, 32 x 32 patches).

1. Units. Every validation episode is run in segments of 4 model steps; a token's patches are those where its
   decoder mask is the largest. Per object in view: the best token's IoU with the object's true patches.
2. Consequences (card 114.1's counterfactual). For each held-out pull of the drawer or window (start frame t): the
   four frames t-15 .. t are rendered from the stored joint positions with the true button states (factual) and
   with that slide's lock flipped (counterfactual: the button and the handle are repainted, nothing else). The
   model reads the four frames, then rolls its transition 4 steps on the recorded actions without seeing a frame
   (the contact held at its value at t). Read-out: the mean squared change of the decoded features between t and
   t+20 over the slide's envelope (the patches it ever covers). A threshold on it separates unlocked from locked
   factual pulls (chosen to maximise balanced accuracy on them). Of the pulls the model predicts right, the share
   whose prediction switches side when the lock is flipped is the score. Also: which token carries it — the factual history with one token swapped for the
   counterfactual history's (the token best covering the button; the one best covering the slide; a random other).
3. Self. For each sampled frame, the transition is run under the real action chunk and 8 others from the data;
   a token's influence is the variance of its predicted vector across them. The self is the tokens with at least
   3 times the median influence; scored by IoU of their patches with the arm's.

  MUJOCO_GL=egl ... bin/prun python tools/card124/evaluate.py --model armB.pt → runs/124/eval_<model>.json
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
from train import DATA, EP, FEAT, N, OUT, ROOT, SKIP, T, TokenModel, features  # noqa: E402
from encode import load_encoder  # noqa: E402
from masks import OBJECTS  # noqa: E402

DEV, R, KH = 'cuda', 256, 4
SLIDES = {'drawer': 0, 'window': 1}                      # the button that locks each
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def chunk_at(act, con, t):
    """The transition's input after frame t: the next 5 actions and the contact at t."""
    return np.r_[act[t:t + SKIP].reshape(-1), con[t]].astype(np.float32)


@torch.no_grad()
def run(model, feats, acts):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        post, masks, *_ = model(feats, acts)
    return post.float(), masks.float()


@torch.no_grad()
def rollout(model, slots, acts):
    """slots [B, K, d], acts [B, KH, 26] → decoded features now and after KH steps of the transition."""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        now = model.decode(slots)[0].float()
        for j in range(acts.shape[1]):
            slots = model.step(slots, acts[:, j])
        return now, model.decode(slots)[0].float()


def main():
    name = arg('--model', 'smoke.pt')
    ck = torch.load(OUT / name, map_location=DEV)
    model = TokenModel(K=ck['K']).to(DEV)
    model.load_state_dict(ck['state'])
    model.eval()
    enc = load_encoder(DEV)
    z = np.load(DATA / 'scene-play-v0-val.npz')
    qpos, btn, act, con = z['qpos'], z['button_states'], z['actions'], z['observations'][:, 18]
    masks = np.load(ROOT / 'runs/122/computed_val.npz')['masks'].astype(bool)          # [100100, 6, 1024]
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection', width=R, height=R)
    env.reset(seed=0)
    u = env.unwrapped
    qvel0 = np.zeros(u._model.nv)
    rng = np.random.default_rng(0)

    def render(rows, flip=None):
        out = np.zeros((len(rows), R, R, 3), np.uint8)
        for i, r in enumerate(rows):
            b = btn[r].astype(np.int64).copy()
            if flip is not None:
                b[flip] = 1 - b[flip]
            u.set_state(qpos[r].astype(np.float64), qvel0, b)
            out[i] = u.render()
        return out

    def encode(frames):
        return features(enc, torch.from_numpy(frames).to(DEV))

    res = dict(model=name, step=ck.get('step'), args=ck.get('args'))

    # 1. units, 3. self
    iou = {o: [] for o in OBJECTS}
    self_iou, n_self, moved = [], [], []
    for ep in range(0, 100, 2):
        for s in rng.choice(np.arange(0, EP - SKIP * (T + 1), SKIP), 8, replace=False):
            rows = ep * EP + s + SKIP * np.arange(T)
            f = encode(render(rows))[None]
            a = torch.from_numpy(np.stack([chunk_at(act, con, r) for r in rows]))[None].to(DEV)
            post, m = run(model, f, a)
            hard = m[0, -1].argmax(0).cpu().numpy()                                     # [N] token per patch
            true = masks[rows[-1]]
            for oi, o in enumerate(OBJECTS):
                if true[oi].any():
                    iou[o].append(max(((hard == k) & true[oi]).sum() / ((hard == k) | true[oi]).sum()
                                      for k in range(model.K)))
            others = rng.integers(0, len(act) - SKIP, 8)
            A = np.stack([chunk_at(act, con, rows[-1])] + [np.r_[act[r:r + SKIP].reshape(-1), con[rows[-1]]]
                                                           for r in others]).astype(np.float32)
            with torch.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
                nxt = model.step(post[:, -1].expand(9, -1, -1), torch.from_numpy(A).to(DEV)).float()
            infl = nxt.var(0).sum(-1).cpu().numpy()                                     # [K]
            sel = np.where(infl >= 3 * np.median(infl))[0]
            sm = np.isin(hard, sel)
            self_iou.append((sm & true[0]).sum() / max((sm | true[0]).sum(), 1))
            n_self.append(len(sel))
            moved.append(float((post[0, -1] - post[0, -2]).norm(dim=-1).mean()))
    res['units_best_token_iou'] = {o: round(float(np.mean(v)), 3) for o, v in iou.items()}
    res['units_mean'] = round(float(np.mean(list(res['units_best_token_iou'].values()))), 3)
    res['self'] = dict(iou_with_arm=round(float(np.mean(self_iou)), 3), tokens_chosen=round(float(np.mean(n_self)), 2))
    res['mean_token_change_per_step'] = round(float(np.mean(moved)), 4)

    # 2. consequences
    ev = np.load(ROOT / 'runs/114/sevents.npz')
    cons = {}
    for slide, bi in SLIDES.items():
        oi = OBJECTS.index(slide)
        envl = torch.from_numpy(masks[::50, oi].mean(0) > 0.01).to(DEV)
        ts, locked = ev[f'val/{slide}/t'], ev[f'val/{slide}/locked']
        ok = (ts % EP >= 3 * SKIP) & (ts % EP + SKIP * KH < EP - SKIP)
        ts, locked = ts[ok], locked[ok]
        val = {k: [] for k in ['f', 'cf', 'swap_button', 'swap_slide', 'swap_random']}
        for t in ts:
            rows = t - SKIP * np.arange(T - 1, -1, -1)
            a = torch.from_numpy(np.stack([chunk_at(act, con, r) for r in rows]))[None].to(DEV)
            fut = np.stack([np.r_[act[t + SKIP * j:t + SKIP * (j + 1)].reshape(-1), con[t]] for j in range(KH)])
            fut = torch.from_numpy(fut.astype(np.float32))[None].to(DEV)
            pf, mf = run(model, encode(render(rows))[None], a)
            pc, _ = run(model, encode(render(rows, flip=bi))[None], a)
            hard = mf[0, -1].argmax(0).cpu().numpy()
            true = masks[t]
            best = lambda o: int(np.argmax([((hard == k) & true[o]).sum() / max(((hard == k) | true[o]).sum(), 1)
                                            for k in range(model.K)]))
            kb, ks = best(OBJECTS.index(f'button{bi}')), best(oi)
            kr = int(rng.choice([k for k in range(model.K) if k not in (kb, ks)]))
            variants = {'f': pf[:, -1], 'cf': pc[:, -1]}
            for nm, k in [('swap_button', kb), ('swap_slide', ks), ('swap_random', kr)]:
                s = pf[:, -1].clone()
                s[:, k] = pc[:, -1, k]
                variants[nm] = s
            for nm, s in variants.items():
                now, later = rollout(model, s, fut)
                val[nm].append(float(((later - now) ** 2).sum(-1)[0, envl].mean()))
        v = {k: np.array(x) for k, x in val.items()}
        cands = np.unique(v['f'])
        acc = [((v['f'][~locked] > c).mean() + (v['f'][locked] <= c).mean()) / 2 for c in cands]
        c = cands[int(np.argmax(acc))]
        side = lambda x: x > c                                        # True = predicted to open
        entry = dict(pulls=int(len(ts)), locked=int(locked.sum()),
                     factual_balanced_accuracy=round(float(max(acc)), 3),
                     factual_mean_change=dict(unlocked=round(float(v['f'][~locked].mean()), 4),
                                              locked=round(float(v['f'][locked].mean()), 4)))
        right = side(v['f']) == ~locked                               # the factual prediction matches the lock
        entry['factual_right'] = round(float(right.mean()), 3)
        for nm in ['cf', 'swap_button', 'swap_slide', 'swap_random']:
            # of the pulls predicted right, the share whose prediction switches side under the change
            entry[f'{nm}_flips_when_factual_right'] = round(float((side(v[nm]) != side(v['f']))[right].mean()), 3)
        cons[slide] = entry
    res['consequences'] = cons
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / f'eval_{name}.json', 'w'), indent=1)
    os._exit(0)


if __name__ == '__main__':
    main()
