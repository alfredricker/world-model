"""Train a card-001 candidate on the chained-rooms collection.

Losses (hub H), all on one learned state:
- transition: ||T(z_t, a_t) - z_{t+1}||^2 (LeWM, no stop-gradient);
- SIGReg on every encoding in the batch (LeWM, weight 0.1);
- reachability: expectile regression of d(z_t, g) toward
  1 + d_ema(z_{t+1}, g) (HILP / GCIVL, action-free, EMA target), where
  g is a later frame of the same episode or any frame of the dataset. For a
  later frame k steps ahead, k is an upper bound on the true distance, so the
  target is min(k, 1 + d_ema): bootstrapping alone grows distances by about
  one step per target-network period. A share of single goals is the next
  frame itself, which pins one-step distances at 1.
- supplied condition goals (card 001, decided 2026-09-26; C1 allows goals
  supplied as tasks until P20): for part of the batch the goal is a
  condition (a key of a colour held, a door of a colour open) shown as 4
  example frames from other training episodes where it holds and is
  visible. The success signal (does the condition hold) comes from the
  evaluator's labels, as a supplied task would; the model is never told
  which other conditions lead to it. Target: 0 where the condition holds,
  else min(steps until it next holds in the episode, 1 + d_ema(z', G)).
  The distance to G uses ``goal_rule`` on the pooled example features.
- goal sets (label-free, tried at the gate, off by default): for part of the batch, the
  goal is 4 frames spread over a later stretch of the same episode, so that
  what they share is what lasted. The distance to a set is the head's
  asymmetric part against the coordinate-wise maximum over the set; it is
  trained like a single goal, with the steps to the stretch's first frame
  as the upper bound.

Balanced sampling (declared for the screen only, card 001 section 4): part of
each batch is anchored at or just before an interaction, chosen with the
evaluator's event labels. The learner never sees the labels themselves.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import write_json
from .envs.rooms import EVENTS
from .envs.rooms_data import load_labels, load_rooms_replay
from .fork_eval import evaluate, load_forks, take
from .models.hub import Hub, HubConfig, count_parameters, expectile, sigreg

ANCHOR_EVENTS = ("pickup", "unlock", "door_open", "box_open", "switch_press")


@dataclass
class TrainConfig:
    out: Path
    data: Path = Path("runs/train")
    probe: Path = Path("runs/sampled_dev")
    extras: Path = Path("runs/sampled_dev_extras")
    obs: str = "frames"
    candidate: str = "H"
    seed: int = 0
    updates: int = 20000
    batch: int = 256
    balanced: float = 0.5          # share of anchors at or just before an interaction
    lr: float = 3e-4
    sigreg: float = 0.1
    tau: float = 0.9               # expectile; > 0.5 pulls d toward the best next state
    ema: float = 0.005
    d_max: float = 128.0           # targets are clipped here; "unreachable" in practice
    d_scale: float = 16.0          # the reachability loss is computed on d / d_scale
    next_goal: float = 0.0         # share of single goals that are the next frame
    future: float = 0.6            # share of goals from later in the same episode
    geometric: float = 0.02        # future offset ~ Geometric(p), mean 50 steps
    qrl_transition: float = 0.0   # QRL's transition loss: T's error measured in the learned quasimetric
    local: float = 0.0            # QRL's local constraint as a penalty: relu(d(z, z') - 1)^2
    conditions: float = 0.0        # share of anchors that also get a supplied condition goal
    goal_rule: str = "asym"        # distance from a state to a pooled goal: "asym", "pooled" or "min"
    ceiling: bool = False          # oracle fit: condition goals regress on true steps (diagnostic only)
    sets: float = 0.0              # share of anchors that also get a goal set
    set_size: int = 4
    set_span: tuple[int, int] = (16, 64)   # the set's frames spread over this many steps
    eval_every: int = 2500
    eval_rows: int = 1500          # fork subset for the learning curve; the end uses all
    log_every: int = 100
    hub: dict = field(default_factory=dict)


class Data:
    """The whole collection on the GPU, with transition indices for sampling."""

    def __init__(self, cfg: TrainConfig, device):
        replay, meta = load_rooms_replay(cfg.data, training=True)
        episodes = replay.episodes
        lengths = np.array([len(e.actions) for e in episodes])
        offsets = np.concatenate([[0], np.cumsum(lengths + 1)])[:-1]           # first frame of each episode
        self.frame_count = int((lengths + 1).sum())
        start = np.repeat(offsets, lengths)
        t = np.concatenate([np.arange(n) for n in lengths])
        self.t_frame = torch.from_numpy(start + t).to(device)
        self.t_end = torch.from_numpy(np.repeat(offsets + lengths, lengths)).to(device)     # last frame
        first_transition = np.concatenate([[0], np.cumsum(lengths)])[:-1]
        self.t_first = torch.from_numpy(np.repeat(first_transition, lengths)).to(device)
        self.t_action = torch.from_numpy(np.concatenate([e.actions for e in episodes])).to(device)
        self.t_term = torch.from_numpy(np.concatenate([e.terminated for e in episodes])).to(device)
        _, events = load_labels(cfg.data)
        events = np.concatenate(events)
        cols = [EVENTS.index(e) for e in ANCHOR_EVENTS]
        self.events = events
        self.anchor_events = torch.from_numpy(np.flatnonzero(events[:, cols].any(1))).to(device)
        if cfg.obs == "frames":
            frames = np.concatenate([e.frames for e in episodes])
            self.obs = torch.from_numpy(frames).to(device)
            del frames
        else:
            from .envs.rooms_symbolic import load_symbolic
            grids, carried = load_symbolic(cfg.data, "ego_" if cfg.obs == "egocentric" else "")
            self.obs = (torch.from_numpy(np.concatenate(grids)).to(device),
                        torch.from_numpy(np.concatenate(carried)).to(device))
        self.transitions = len(self.t_frame)
        self.device = device
        holds, visible = condition_labels(cfg.data)
        self.cond_ids = [c for c in range(holds.shape[1]) if (holds[:, c] & visible[:, c]).sum() >= 8]
        pools = [np.flatnonzero(holds[:, c] & visible[:, c]) for c in range(holds.shape[1])]
        self.pool_all = torch.from_numpy(np.concatenate(pools)).to(device)
        self.pool_start = torch.tensor(np.concatenate([[0], np.cumsum([len(q) for q in pools])])[:-1], device=device)
        self.pool_size = torch.tensor([len(q) for q in pools], device=device)
        self.holds = torch.from_numpy(holds).to(device)
        if cfg.ceiling:
            from .envs.rooms_truth import load_true_distances
            self.true_dist = torch.from_numpy(np.concatenate(load_true_distances(cfg.data))).to(device)
        # next_hold[i, c]: first frame >= i of i's episode where c holds, else a large number.
        nxt = np.full(holds.shape, np.iinfo(np.int64).max // 4, np.int64)
        for o, n in zip(offsets, lengths):
            run = np.full(holds.shape[1], np.iinfo(np.int64).max // 4, np.int64)
            for i in range(o + n, o - 1, -1):
                run = np.where(holds[i], i, run)
                nxt[i] = run
        self.next_hold = torch.from_numpy(nxt).to(device)

    def get(self, idx):
        return take(self.obs, idx)

    def sample(self, cfg: TrainConfig, gen: torch.Generator):
        b = cfg.batch
        nb = int(round(b * cfg.balanced))
        dev = self.device
        uniform = torch.randint(self.transitions, (b - nb,), device=dev, generator=gen)
        ev = self.anchor_events[torch.randint(len(self.anchor_events), (nb,), device=dev, generator=gen)]
        back = torch.where(torch.rand(nb, device=dev, generator=gen) < 0.5, 0,
                           torch.randint(1, 9, (nb,), device=dev, generator=gen))
        ev = torch.maximum(ev - back, self.t_first[ev])
        tr = torch.cat([uniform, ev])
        tr = tr[torch.randperm(b, device=dev, generator=gen)]   # sets and conditions take random anchors
        f = self.t_frame[tr]
        end = self.t_end[tr]
        u = torch.rand(b, device=dev, generator=gen).clamp_min(1e-9)
        k = 1 + torch.floor(torch.log(u) / np.log(1 - cfg.geometric)).long()
        future = torch.minimum(f + k, end)
        anywhere = torch.randint(self.frame_count, (b,), device=dev, generator=gen)
        kind = torch.rand(b, device=dev, generator=gen)
        is_next = kind < cfg.next_goal
        is_future = ~is_next & (kind < cfg.next_goal + cfg.future)
        goal = torch.where(is_next, f + 1, torch.where(is_future, future, anywhere))
        bound = torch.where(is_next | is_future, (goal - f).float(), torch.full((b,), float("inf"), device=dev))
        # Goal sets for the first n_sets anchors: a later stretch of the same episode.
        n_sets = int(round(b * cfg.sets))
        u = torch.rand(n_sets, device=dev, generator=gen).clamp_min(1e-9)
        k = 1 + torch.floor(torch.log(u) / np.log(1 - cfg.geometric)).long()
        first = torch.minimum(f[:n_sets] + k, end[:n_sets])
        span = torch.randint(cfg.set_span[0], cfg.set_span[1] + 1, (n_sets,), device=dev, generator=gen)
        j = torch.arange(cfg.set_size, device=dev)
        members = torch.minimum(first[:, None] + (span[:, None] * j) // (cfg.set_size - 1), end[:n_sets, None])
        # Supplied condition goals for the last n_cond anchors: 4 examples each.
        n_cond = int(round(b * cfg.conditions))
        ids = torch.tensor(self.cond_ids, device=dev)
        cond = ids[torch.randint(len(ids), (n_cond,), device=dev, generator=gen)]
        pick = (torch.rand(n_cond, 4, device=dev, generator=gen) * self.pool_size[cond][:, None]).long()
        examples = self.pool_all[self.pool_start[cond][:, None] + pick]
        return tr, f, goal, bound, members, (first - f[:n_sets]).float(), cond, examples


def condition_labels(data: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, for each probe goal name (key_held / door_open x red, green,
    blue, grey): does the condition hold, and is it visible in the frame (a
    carried key always is; an open door must be in view). Evaluator labels:
    they define supplied tasks and choose examples, never enter the model."""
    from .envs.rooms_data import LABEL_COLUMNS, MAX_DOORS
    labels, _ = load_labels(data)
    rows = np.concatenate(labels)
    col = LABEL_COLUMNS.index
    colours = (0, 1, 2, 5)            # MiniGrid COLOR_TO_IDX: red, green, blue, grey
    holds = np.zeros((len(rows), 8), bool)
    visible = np.zeros((len(rows), 8), bool)
    for j, c in enumerate(colours):
        if c != 5:                    # a carried grey object is a box; keys are never grey
            holds[:, j] = rows[:, col("carrying")] == c
        visible[:, j] = holds[:, j]
        for i in range(MAX_DOORS):
            here = (rows[:, col(f"door{i}_color")] == c) & (rows[:, col(f"door{i}_open")] == 1)
            holds[:, 4 + j] |= here
            visible[:, 4 + j] |= here & (rows[:, col(f"door{i}_in_view")] == 1)
    return holds, visible


def cat_obs(parts):
    if isinstance(parts[0], tuple):
        return tuple(torch.cat([p[k] for p in parts]) for k in range(len(parts[0])))
    return torch.cat(parts)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = Hub(HubConfig(obs=cfg.obs, **cfg.hub)).to(self.device)
        self.target = copy.deepcopy(self.model).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(cfg.seed)

    def losses(self, data: Data):
        cfg, m = self.cfg, self.model
        tr, f, goal, bound, members, set_bound, cond, examples = data.sample(cfg, self.gen)
        b, ns, nc = len(tr), len(members), len(cond)
        obs = cat_obs([data.get(f), data.get(f + 1), data.get(goal), data.get(members.reshape(-1)),
                       data.get(examples.reshape(-1))])
        z_all = m.encode(obs)
        z, z_next, z_goal = z_all[:b], z_all[b:2 * b], z_all[2 * b:3 * b]
        z_set, z_ex = z_all[3 * b:3 * b + 4 * ns], z_all[3 * b + 4 * ns:]
        a = data.t_action[tr]
        pred = m.transition(z, a)
        l_pred = F.mse_loss(pred, z_next)
        l_sig = sigreg(z_all)
        d = m.head.distance(m.head.features(z), m.head.features(z_goal))
        n = cfg.set_size
        fz = m.head.features(z[:ns])
        fs = m.head.features(z_set)
        d_set = m.head.set_distance(fz, (fs[0].view(ns, n, fs[0].shape[-1]), fs[1].view(ns, n, fs[1].shape[-1])), "asym")
        def pooled(head, zs, zg, count):
            fg = head.features(zg)
            k = fg[0].shape[-1]
            return head.set_distance(head.features(zs), (fg[0].view(count, 4, k), fg[1].view(count, 4, k)), cfg.goal_rule)
        fc, ff = f[b - nc:], tr[b - nc:]
        d_cond = pooled(m.head, z[b - nc:], z_ex, nc)
        with torch.no_grad():
            zt = self.target.encode(cat_obs([data.get(f + 1), data.get(goal), data.get(members.reshape(-1)),
                                             data.get(examples.reshape(-1))]))
            zt_ex = zt[2 * b + 4 * ns:]
            d_cond_next = pooled(self.target.head, zt[b - nc:b], zt_ex, nc)
            holds_now = data.holds[fc, cond]
            holds_next = data.holds[fc + 1, cond]
            cond_bound = (data.next_hold[fc + 1, cond] - fc).float()
            cond_target = torch.where(data.t_term[ff] & ~holds_next, torch.full_like(d_cond_next, cfg.d_max),
                                      1 + d_cond_next)
            cond_target = torch.minimum(cond_target, cond_bound).clamp(max=cfg.d_max)
            cond_target = torch.where(holds_now, torch.zeros_like(cond_target), cond_target)
            h = self.target.head
            d_next = h.distance(h.features(zt[:b]), h.features(zt[b:2 * b]))
            ts = h.features(zt[2 * b:2 * b + 4 * ns])
            d_set_next = h.set_distance(h.features(zt[:ns]), (ts[0].view(ns, n, ts[0].shape[-1]),
                                                                ts[1].view(ns, n, ts[1].shape[-1])), "asym")
            set_dead = data.t_term[tr[:ns]] & (set_bound > 1)
            set_target = torch.where(set_dead, torch.full_like(d_set_next, cfg.d_max), 1 + d_set_next)
            set_target = torch.minimum(set_target, set_bound).clamp(max=cfg.d_max)
            reached = goal == f + 1
            dead = data.t_term[tr] & ~reached
            target = torch.where(reached, torch.ones_like(d_next), 1 + d_next)
            target = torch.where(dead, torch.full_like(target, cfg.d_max), target)
            target = torch.minimum(target, bound).clamp(max=cfg.d_max)
        l_d = expectile(d / cfg.d_scale, target / cfg.d_scale, cfg.tau)
        l_set = expectile(d_set / cfg.d_scale, set_target / cfg.d_scale, cfg.tau) if ns else torch.zeros((), device=d.device)
        if cfg.ceiling and nc:
            # Oracle fit (CHARTER rule 7: shows the test is passable, never a result).
            cond_target = data.true_dist[fc, cond].float().clamp(max=cfg.d_max)
            l_cond = F.mse_loss(d_cond / cfg.d_scale, cond_target / cfg.d_scale)
        else:
            l_cond = expectile(d_cond / cfg.d_scale, cond_target / cfg.d_scale, cfg.tau) if nc else torch.zeros((), device=d.device)
        loss = l_pred + cfg.sigreg * l_sig + l_d + l_set + l_cond
        h = m.head
        f_pred, f_next = h.features(pred), h.features(z_next)
        l_qt = 0.5 * (h.distance(f_pred, f_next) ** 2 + h.distance(f_next, f_pred) ** 2).mean()
        l_local = F.relu(h.distance(h.features(z), f_next) - 1).pow(2).mean()
        loss = loss + cfg.qrl_transition * l_qt + cfg.local * l_local
        stats = {"loss": loss, "pred": l_pred, "sigreg": l_sig, "reach": l_d, "reach_set": l_set,
                 "reach_cond": l_cond, "d_cond_mean": d_cond.mean() if nc else l_cond,
                 "cond_target_mean": cond_target.mean() if nc else l_cond,
                 "d_set_mean": d_set.mean() if ns else l_set, "qrl_transition": l_qt,
                 "local": l_local, "d_mean": d.mean(),
                 "target_mean": target.mean(), "z_std": z.std(0).mean()}
        return loss, stats

    @torch.no_grad()
    def ema_update(self):
        for p, q in zip(self.model.parameters(), self.target.parameters()):
            q.lerp_(p, self.cfg.ema)
        for p, q in zip(self.model.buffers(), self.target.buffers()):
            q.copy_(p)

    def scorer(self, rule=None):
        m = self.model
        rule = rule or self.cfg.goal_rule

        def score(obs, goal_obs, n):
            return m.fork_scores(obs, goal_obs, rule)
        return score

    def successor_scorer(self, rule=None):
        """The head alone on given observations (no transition model); n
        goal examples per observation, pooled by ``rule`` when n > 1."""
        m = self.model
        rule = rule or self.cfg.goal_rule

        @torch.no_grad()
        def score(obs, goal_obs, n):
            h = m.head
            fx = h.features(m.encode(obs))
            fg = h.features(m.encode(goal_obs))
            if n == 1:
                d = h.distance(fx, fg)
            else:
                k = fg[0].shape[-1]
                d = h.set_distance(fx, (fg[0].view(-1, n, k), fg[1].view(-1, n, k)), rule)
            return d[:, None].repeat(1, 5), d
        return score

    @torch.no_grad()
    def diagnostics(self, data: Data, n: int = 4096) -> dict:
        """True versus shuffled action (T must depend on the action), latent
        spread, and the one-way gap d(z', z) - d(z, z') by event type."""
        m = self.model
        gen = torch.Generator(device=self.device)
        gen.manual_seed(12345)
        out = {}
        ev = data.events
        kinds = {
            "pickup": ev[:, EVENTS.index("pickup")] > 0,
            "unlock": ev[:, EVENTS.index("unlock")] > 0,
            "box_open": ev[:, EVENTS.index("box_open")] > 0,
            "door_toggle_other": (ev[:, EVENTS.index("door_open")] > 0) & (ev[:, EVENTS.index("unlock")] == 0),
            "forward_no_event": (data.t_action.cpu().numpy() == 2) & (ev.sum(1) == 0),
        }
        for name, mask in kinds.items():
            idx = np.flatnonzero(mask)
            if not len(idx):
                continue
            idx = torch.from_numpy(np.random.default_rng(0).choice(idx, size=min(n, len(idx)), replace=False)).to(self.device)
            f = data.t_frame[idx]
            z, z2 = m.encode(data.get(f)), m.encode(data.get(f + 1))
            a = data.t_action[idx]
            err = (m.transition(z, a) - z2).pow(2).sum(1)
            shuffled = (a + torch.randint(1, 5, a.shape, device=self.device, generator=gen)) % 5
            err_s = (m.transition(z, shuffled) - z2).pow(2).sum(1)
            fz, fz2 = m.head.features(z), m.head.features(z2)
            fwd, back = m.head.distance(fz, fz2), m.head.distance(fz2, fz)
            out[name] = {"n": int(len(idx)), "shuffle_ratio": float(err_s.mean() / err.mean().clamp_min(1e-9)),
                         "null_ratio": float((z - z2).pow(2).sum(1).mean() / err.mean().clamp_min(1e-9)),
                         "one_way_gap": float((back - fwd).mean()), "forward_d": float(fwd.mean())}
        idx = torch.randint(data.frame_count, (n,), device=self.device, generator=gen)
        z = m.encode(data.get(idx))
        s = torch.linalg.svdvals(z - z.mean(0))
        p = s / s.sum()
        out["latent"] = {"std": float(z.std(0).mean()), "effective_rank": float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))}
        return out

    def run(self):
        cfg = self.cfg
        cfg.out.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        write_json(cfg.out / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()})
        data = Data(cfg, self.device)
        forks = load_forks(cfg.probe, cfg.extras, cfg.obs)
        curve_rows = np.random.default_rng(99).choice(len(forks.goal), size=min(cfg.eval_rows, len(forks.goal)), replace=False)
        info = {"parameters": count_parameters(self.model), "transitions": data.transitions,
                "anchor_events": int(len(data.anchor_events)), "load_seconds": time.monotonic() - started}
        write_json(cfg.out / "setup.json", info)
        log = (cfg.out / "metrics.jsonl").open("w")
        t0 = time.monotonic()
        for step in range(1, cfg.updates + 1):
            self.model.train()
            loss, stats = self.losses(data)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.ema_update()
            if step % cfg.log_every == 0 or step == 1:
                row = {"step": step, "seconds": time.monotonic() - t0, "grad_norm": float(gn),
                       **{k: float(v.detach()) for k, v in stats.items()}}
                log.write(json.dumps(row) + "\n")
                log.flush()
            if step % cfg.eval_every == 0 and step < cfg.updates:
                self.model.eval()
                r = evaluate(self.scorer(), forks, self.device, rows=curve_rows, exact=False)
                log.write(json.dumps({"step": step, "eval": {"scored_top1": r["pooled"]["scored_top1"],
                                                             "movement_top1": r["pooled"]["movement_top1"],
                                                             "rank_correlation": r["pooled"]["rank_correlation"],
                                                             "swapped_top1": r["swapped"]["scored_top1"]}}) + "\n")
                log.flush()
        train_seconds = time.monotonic() - t0
        self.model.eval()
        torch.save(self.model.state_dict(), cfg.out / "model.pt")
        result = {"train_seconds": train_seconds, "updates": cfg.updates, **info}
        for rule in ("pooled", "min", "asym"):
            result[rule] = evaluate(self.scorer(rule), forks, self.device, exact=(rule == "pooled"),
                                    successor=self.successor_scorer() if rule == "pooled" else None,
                                    same_world=True)
        result["diagnostics"] = self.diagnostics(data)
        result["total_seconds"] = time.monotonic() - started
        write_json(cfg.out / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--obs", choices=("frames", "egocentric", "symbolic"), default="frames")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--eval-rows", type=int, default=1500)
    parser.add_argument("--ema", type=float, default=TrainConfig.ema)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--qrl-transition", type=float, default=0.0)
    parser.add_argument("--local", type=float, default=0.0)
    parser.add_argument("--d-scale", type=float, default=TrainConfig.d_scale)
    parser.add_argument("--next-goal", type=float, default=0.0)
    parser.add_argument("--future", type=float, default=TrainConfig.future)
    parser.add_argument("--sets", type=float, default=0.0)
    parser.add_argument("--conditions", type=float, default=0.0)
    parser.add_argument("--goal-rule", default=TrainConfig.goal_rule)
    parser.add_argument("--ceiling", action="store_true", help="oracle fit on true distances (diagnostic)")
    args = parser.parse_args()
    cfg = TrainConfig(out=args.out, obs=args.obs, seed=args.seed, updates=args.updates, batch=args.batch,
                      eval_every=args.eval_every, eval_rows=args.eval_rows, ema=args.ema, lr=args.lr,
                      qrl_transition=args.qrl_transition, local=args.local, d_scale=args.d_scale,
                      next_goal=args.next_goal, future=args.future, sets=args.sets,
                      conditions=args.conditions, goal_rule=args.goal_rule, ceiling=args.ceiling)
    result = Trainer(cfg).run()
    brief = {k: result[k] for k in ("train_seconds", "total_seconds", "parameters")}
    for rule in ("pooled", "min", "asym"):
        brief[rule] = {k: result[rule]["pooled"][k] for k in ("scored_top1", "movement_top1", "no_effect_best", "rank_correlation")}
        brief[rule]["swapped_top1"] = result[rule]["swapped"]["scored_top1"]
    brief["exact"] = {k: result["pooled"]["exact"][k] for k in ("scored_top1", "movement_top1", "rank_correlation")}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()


def rescore(run: Path) -> dict:
    """Re-run the final evaluation of a finished run from its checkpoint."""
    raw = json.loads((run / "config.json").read_text())
    for k in ("out", "data", "probe", "extras"):
        raw[k] = Path(raw[k])
    cfg = TrainConfig(**raw)
    trainer = Trainer(cfg)
    trainer.model.load_state_dict(torch.load(run / "model.pt"))
    trainer.model.eval()
    forks = load_forks(cfg.probe, cfg.extras, cfg.obs)
    return {rule: evaluate(trainer.scorer(rule), forks, trainer.device, exact=(rule == "pooled"),
                           successor=trainer.successor_scorer() if rule == "pooled" else None, same_world=True)
            for rule in ("pooled", "asym")}
