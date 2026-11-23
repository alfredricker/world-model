"""Card 160: the ordinary baseline.  PPO (Schulman and others 2017) with MiniGrid's reward, with the settings of the
reference implementation for MiniGrid (rl-starter-files: 16 environments, 128 steps each, 4 epochs, batches of 256,
learning rate 0.001, discount 0.99, lambda 0.95, entropy 0.01, clip 0.2).  Two views: MiniGrid's standard 7 x 7 view
ahead of the agent ("ego"), and the full grid with the agent drawn in ("full", what the energy model is given).

usage: ppo.py <task> <ego|full> <seed> <frames at which to test, increasing...>     writes runs/160/ppo_<task>_<view>_<seed>.json
"""
import sys, json, time, numpy as np, torch, torch.nn as nn, gymnasium as gym, minigrid
from minigrid.wrappers import FullyObsWrapper
import tasks

torch.set_num_threads(2)
NE, NS, EPOCHS, BATCH, LR, GAMMA, LAM, ENT, CLIP = 16, 128, 4, 256, 1e-3, 0.99, 0.95, 0.01, 0.2


def make(task, view):
    e = gym.make(tasks.ALL[task])
    return FullyObsWrapper(e) if view == "full" else e


def onehot(img):
    """[n, W, H, 3] of type, colour, state -> [n, 21, W, H]"""
    x = torch.as_tensor(np.asarray(img), dtype=torch.long)
    return torch.cat([nn.functional.one_hot(x[..., 0], 11), nn.functional.one_hot(x[..., 1], 6), nn.functional.one_hot(x[..., 2], 4)], -1).permute(0, 3, 1, 2).float()


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.cnn = nn.Sequential(nn.Conv2d(21, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        s.pi = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 7)); s.v = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(s, x):
        h = s.cnn(x).amax((2, 3))
        return torch.distributions.Categorical(logits=s.pi(h)), s.v(h).squeeze(-1)


@torch.no_grad()
def test(net, task, view, seeds):
    envs = [make(task, view) for _ in seeds]; obs = [e.reset(seed=int(s))[0]["image"] for e, s in zip(envs, seeds)]; live = np.ones(len(envs), bool); ok = np.zeros(len(envs), bool); steps = np.zeros(len(envs), int)
    while live.any():
        ii = np.nonzero(live)[0]; a = net(onehot(np.stack([obs[i] for i in ii])))[0].sample().numpy()
        for i, ai in zip(ii, a):
            o, r, te, tr, _ = envs[i].step(int(ai)); obs[i] = o["image"]; steps[i] += 1
            if te or tr:
                live[i] = False; ok[i] = r > 0
    return float(100 * ok.mean()), float(steps[ok].mean()) if ok.any() else 0.0


def train(task, view, seed, marks):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); envs = [make(task, view) for _ in range(NE)]
    obs = np.stack([e.reset(seed=int(rng.integers(1 << 30)))[0]["image"] for e in envs]); net = Net(); opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-8)
    frames = 0; out = dict(task=task, view=view, seed=seed, tests=[]); t0 = time.time(); marks = list(marks); rets = []
    while marks:
        O, A, LP, V, Rw, Dn = [], [], [], [], [], []
        for _ in range(NS):
            x = onehot(obs)
            with torch.no_grad():
                d, v = net(x); a = d.sample()
            O.append(x); A.append(a); LP.append(d.log_prob(a)); V.append(v); r = np.zeros(NE, np.float32); dn = np.zeros(NE, np.float32)
            for i, e in enumerate(envs):
                o, r[i], te, tr, _ = e.step(int(a[i])); dn[i] = te or tr
                if te or tr:
                    rets.append(r[i]); o = e.reset(seed=int(rng.integers(1 << 30)))[0]
                obs[i] = o["image"]
            Rw.append(torch.as_tensor(r)); Dn.append(torch.as_tensor(dn))
        frames += NE * NS
        with torch.no_grad():
            nv = net(onehot(obs))[1]
        adv = torch.zeros(NS, NE); last = torch.zeros(NE)
        for t in range(NS - 1, -1, -1):
            nxt = nv if t == NS - 1 else V[t + 1]; delta = Rw[t] + GAMMA * nxt * (1 - Dn[t]) - V[t]; last = delta + GAMMA * LAM * (1 - Dn[t]) * last; adv[t] = last
        O = torch.cat(O); A = torch.cat(A); LP = torch.cat(LP); adv = adv.reshape(-1); ret = adv + torch.cat(V)
        for _ in range(EPOCHS):
            for b in torch.randperm(NE * NS).split(BATCH):
                d, v = net(O[b]); ratio = (d.log_prob(A[b]) - LP[b]).exp(); ad = adv[b]
                loss = -torch.min(ratio * ad, ratio.clamp(1 - CLIP, 1 + CLIP) * ad).mean() + 0.5 * (v - ret[b]).pow(2).mean() - ENT * d.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        if frames >= marks[0]:
            s, n = test(net, task, view, range(1000, 1100) if task in tasks.TEST else range(500, 600))
            out["tests"].append(dict(frames=frames, success=s, steps_when_solved=n, train_return=float(np.mean(rets[-200:])) if rets else 0.0, seconds=round(time.time() - t0)))
            print(task, view, seed, out["tests"][-1], flush=True); marks.pop(0); json.dump(out, open("runs/160/ppo_%s_%s_%d.json" % (task, view, seed), "w"), indent=1)
    return out


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4:]])
