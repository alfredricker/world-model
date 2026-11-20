"""Card 149: a sequence of actions by recurrence, from card 148's energy.

Each step: the state now and a wanted effect are clamped, the rest is settled, the action is carried
out, the world answers, and the next step starts from what is then observed.  Between steps the agent
keeps one belief of its own, which object it holds, updated by settling the same energy forward.

  observed each step   where the hand and the objects are, each object's width, which objects the hand touches
  learned beforehand   card 148's couplings (from n steps of the agent's own experience), and the places,
                       relative to an object, where the hand was when it touched one (stored whole, one shot)
  given                subtraction of places (an object's frame; how far a thing is from where it is wanted)

Propose, then verify: the action is settled from the state and the wanted effect; then the effect is
settled forward from the state and that action.  If the wanted effect does not come out, the same two
settles are run as if the object were touched; if it then comes out, touching the object is wanted
first ("to take it, touch it"), and the stored touching places turn that into a wanted move of the hand.

Tasks: (1) hold object X; (2) object X at place G and let go.
"""
import sys, json, numpy as np, torch
sys.path.insert(0, "tools/card148"); sys.path.insert(0, "tools/card149")
import env
import action_token as A

GO, OO, GV, OV = A.GO, A.OO, A.GV, A.OV
H, HAND = env.H, env.HAND
E_Y, E_X, A_Y, A_X, A_G = range(5)
C, HB, O_Y, O_X, H2 = range(5)


class Energy:
    """card 148's pairwise energy over a scene, with any subset of tokens clamped and the rest settled"""
    def __init__(self, model, seed=0):
        self.m = model
        self.gen = torch.Generator().manual_seed(seed)

    def fields(self, Xg, Xo):
        m = self.m
        fg = m.bg + Xg @ m.Wg + (Xo @ m.Wgo.T).sum(1)
        fo = m.bo + Xo @ m.Wo + (Xg @ m.Wgo)[:, None]
        return fg, fo

    def settle(self, g, cg, o, co, mask, iters=30, restarts=4):
        """g [B,5], o [B,4,5] token values; cg, co say which are clamped; mask [B,4] which objects exist.
        Mean-field descent with falling temperature from `restarts` starts (one uniform, the rest random); the
        start that ends lowest in energy is kept.  Returns the settled values and, for every token, the
        probability its settled value has given all the other settled tokens."""
        B0 = len(g); R = restarts
        g, cg, o, co, mask = (np.repeat(v, R, 0) for v in (g, cg, o, co, mask))
        B = len(g)
        start = torch.zeros(B, 1); start[np.arange(B) % R != 0] = 1.0          # the first start of each is uniform
        Xg = torch.zeros(B, GO[-1]); Xo = torch.zeros(B, 4, OO[-1])
        for i in range(5):
            blk = slice(GO[i], GO[i + 1])
            free = torch.softmax(2.0 * start * torch.randn(B, GV[i], generator=self.gen), -1)
            Xg[:, blk] = torch.where(torch.as_tensor(cg[:, i])[:, None], torch.nn.functional.one_hot(torch.as_tensor(g[:, i]), GV[i]).float(), free)
            blk = slice(OO[i], OO[i + 1])
            free = torch.softmax(2.0 * start[:, None] * torch.randn(B, 4, OV[i], generator=self.gen), -1)
            Xo[..., blk] = torch.where(torch.as_tensor(co[..., i])[..., None], torch.nn.functional.one_hot(torch.as_tensor(o[..., i]), OV[i]).float(), free)
        mk = torch.as_tensor(mask)[..., None].float()
        Xo = Xo * mk
        cgt, cot = torch.as_tensor(cg), torch.as_tensor(co)
        for t in range(iters):
            beta = 0.25 * 32 ** (t / (iters - 1))
            for i in range(5):
                if not cg[:, i].all():
                    blk = slice(GO[i], GO[i + 1])
                    new = torch.softmax(beta * self.fields(Xg, Xo)[0][:, blk], -1)
                    Xg[:, blk] = torch.where(cgt[:, i][:, None], Xg[:, blk], new)
            for i in range(5):
                if not co[..., i].all():
                    blk = slice(OO[i], OO[i + 1])
                    new = torch.softmax(beta * self.fields(Xg, Xo)[1][..., blk], -1) * mk
                    Xo[..., blk] = torch.where(cot[..., i][..., None], Xo[..., blk], new)
        gs = np.stack([Xg[:, GO[i]:GO[i + 1]].argmax(1).numpy() for i in range(5)], 1)
        os_ = np.stack([Xo[..., OO[i]:OO[i + 1]].argmax(-1).numpy() for i in range(5)], -1)
        Hg = torch.cat([torch.nn.functional.one_hot(torch.as_tensor(gs[:, i]), GV[i]).float() for i in range(5)], 1)
        Ho = torch.cat([torch.nn.functional.one_hot(torch.as_tensor(os_[..., i]), OV[i]).float() for i in range(5)], -1) * mk
        m = self.m
        score = Hg @ m.bg + 0.5 * ((Hg @ m.Wg) * Hg).sum(1) + ((Ho @ m.bo) + 0.5 * ((Ho @ m.Wo) * Ho).sum(-1) + ((Hg @ m.Wgo)[:, None] * Ho).sum(-1)).sum(1)
        best = score.reshape(B0, R).argmax(1).numpy() + np.arange(B0) * R        # lowest energy
        fg, fo = self.fields(Hg, Ho)                            # each token's field given all the others, hardened
        pg = np.stack([torch.softmax(fg[:, GO[i]:GO[i + 1]], -1).gather(1, torch.as_tensor(gs[:, i])[:, None])[:, 0].numpy() for i in range(5)], 1)
        po = np.stack([torch.softmax(fo[..., OO[i]:OO[i + 1]], -1).gather(-1, torch.as_tensor(os_[..., i])[..., None])[..., 0].numpy() for i in range(5)], -1)
        return gs[best], os_[best], pg[best], po[best]


def touch_memory(data, n):
    """every place of the hand, relative to an object it touched, in the first n steps: rows (width, dy, dx)"""
    rows = set(); lo, hi = np.full(2, 99), np.full(2, -99)
    for s, a, nx in data[:n]:
        lo, hi = np.minimum(lo, s["hand"]), np.maximum(hi, s["hand"])
        for i in range(len(s["size"])):
            if env.touching(s["hand"], s["pos"][i], s["size"][i]):
                rows.add((int(s["size"][i]), int(s["hand"][0] - s["pos"][i][0]), int(s["hand"][1] - s["pos"][i][1])))
    return np.array(sorted(rows)).reshape(-1, 3), lo, hi


def make_tasks(n, seed, kind):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        st = env.new_state(rng); x = int(rng.integers(len(st["size"]))); s = st["size"][x]
        goal = rng.integers(HAND, H - s - HAND + 1, 2) if kind == 2 else None      # reachable with the hand on any side
        if kind == 2 and np.abs(goal - st["pos"][x]).max() < 8:
            continue
        out.append((st, x, goal))
    return out


def done(st, x, goal):
    return st["held"] == x if goal is None else bool(np.all(st["pos"][x] == goal) and st["held"] != x)


def oracle(task, T=60):
    st, x, goal = task
    for t in range(T):
        if done(st, x, goal):
            return t
        if st["held"] == x:
            d = np.clip(goal - st["pos"][x], -4, 4); a = (d[0], d[1], 0 if not d.any() else 1)
        elif env.touching(st["hand"], st["pos"][x], st["size"][x]):
            d = np.zeros(2, int) if goal is None else np.clip(goal - st["pos"][x], -4, 4); a = (d[0], d[1], 1)
        else:
            d = np.clip(env.side_spot(st["hand"], st["pos"][x], st["size"][x]) - st["hand"], -4, 4); a = (d[0], d[1], 0)
        st = env.step(st, tuple(int(v) for v in a))
    return T if done(st, x, goal) else -1


def run(tasks, energy, mem, mode="full", T=60, seed=0):
    """mode: full | reflex (the wanted effect is always acted on, no conditions) | no_belief (holds nothing, as far as it knows)
    | random.  Returns steps to success (-1 if never), and counts."""
    rng = np.random.default_rng(seed)
    B = len(tasks); touch, lo, hi = mem
    st = [t[0] for t in tasks]; X = np.array([t[1] for t in tasks]); goal = [t[2] for t in tasks]
    nobj = np.array([len(s["size"]) for s in st]); mask = np.arange(4)[None] < nobj[:, None]
    belief = np.zeros((B, 4), int)                              # which object it believes it holds
    steps = np.full(B, -1); barred = [set() for _ in range(B)]; aim = [None] * B
    stats = dict(belief_right=0, belief_n=0, approach=0, acted=0, no_way=0, barred=0)
    ar = np.arange(B)
    for t in range(T + 1):
        for b in range(B):
            if steps[b] < 0 and done(st[b], X[b], goal[b]):
                steps[b] = t
        live = steps < 0
        if t == T or not live.any():
            break
        con = np.zeros((B, 4), int)
        for b in range(B):
            for i in range(nobj[b]):
                con[b, i] = env.touching(st[b]["hand"], st[b]["pos"][i], st[b]["size"][i])
        hb = belief if mode != "no_belief" else np.zeros_like(belief)
        act = np.zeros((B, 3), int)
        if mode == "random":
            act = np.stack([rng.integers(-4, 5, B), rng.integers(-4, 5, B), rng.integers(0, 2, B)], 1)
        else:
            want = []                                           # the wanted effect on X, from the goal
            for b in range(B):
                if goal[b] is None:
                    want.append({H2: 1})
                else:
                    d = np.clip(goal[b] - st[b]["pos"][X[b]], -4, 4)
                    want.append({O_Y: d[0] + 4, O_X: d[1] + 4} if d.any() else {O_Y: 4, O_X: 4, H2: 0})

            def inverse(c):
                """state and wanted effect clamped, the action settled"""
                g = np.zeros((B, 5), int); cg = np.zeros((B, 5), bool)
                o = np.zeros((B, 4, 5), int); co = np.ones((B, 4, 5), bool)
                o[..., C], o[..., HB] = c, hb
                o[..., O_Y] = o[..., O_X] = 4; o[..., H2] = hb                 # other objects: as they are
                for b in range(B):
                    co[b, X[b], O_Y:] = False
                    for k, v in want[b].items():
                        o[b, X[b], k] = v; co[b, X[b], k] = True
                gs = energy.settle(g, cg, o, co, mask)[0]
                return np.stack([gs[:, A_Y] - 4, gs[:, A_X] - 4, gs[:, A_G]], 1)

            def forward(c, a):
                """state and action clamped, every effect settled: does the wanted effect come out?"""
                g = np.zeros((B, 5), int); cg = np.zeros((B, 5), bool)
                g[:, A_Y], g[:, A_X], g[:, A_G] = a[:, 0] + 4, a[:, 1] + 4, a[:, 2]; cg[:, A_Y:] = True
                o = np.zeros((B, 4, 5), int); co = np.zeros((B, 4, 5), bool)
                o[..., C], o[..., HB] = c, hb; co[..., C] = co[..., HB] = True
                pred = energy.settle(g, cg, o, co, mask)[1]
                return np.array([all(pred[b, X[b], k] == v for k, v in want[b].items()) for b in range(B)])

            act = inverse(con)
            ok = forward(con, act) if mode != "reflex" else np.ones(B, bool)
            if (~ok & live).any():
                # the action would not do it.  Would it, if X were touched?  Then touching X is wanted first.
                con2 = con.copy(); con2[ar, X] = 1
                need = ~ok & (con[ar, X] == 0) & forward(con2, inverse(con2))
                g3 = np.zeros((B, 5), int); cg3 = np.zeros((B, 5), bool)
                o3 = np.zeros((B, 4, 5), int); co3 = np.ones((B, 4, 5), bool)
                o3[..., C], o3[..., HB], o3[..., H2] = con, hb, hb; o3[..., O_Y] = o3[..., O_X] = 4
                for b in np.nonzero(~ok & live)[0]:
                    act[b] = (0, 0, int(hb[b].any()))               # no way known: stay, keep what is held
                    if not need[b]:
                        stats["no_way"] += 1; continue
                    x = X[b]; s = st[b]["size"][x]; rho = st[b]["hand"] - st[b]["pos"][x]
                    if aim[b] is not None and np.all(rho == aim[b][1:]):            # there, and not touching: not again
                        barred[b].add(tuple(aim[b])); stats["barred"] += 1
                    place = st[b]["pos"][x][None] + touch[:, 1:]
                    cost = ((touch[:, 1:] - rho) ** 2).sum(1) + 1000.0 * (touch[:, 0] - s) ** 2
                    cost = cost + 1e9 * ((place < lo) | (place > hi)).any(1)
                    for k, row in enumerate(touch):
                        if tuple(row) in barred[b]:
                            cost[k] = 1e12
                    if len(touch) == 0 or cost.min() >= 1e9:
                        need[b] = False; stats["no_way"] += 1; continue
                    aim[b] = touch[cost.argmin()]
                    d = np.clip(aim[b][1:] - rho, -4, 4)
                    g3[b, E_Y], g3[b, E_X] = d + 4; cg3[b, E_Y] = cg3[b, E_X] = True   # wanted: the hand moves this way,
                    stats["approach"] += 1                                             # every object stays as it is
                gs3 = energy.settle(g3, cg3, o3, co3, mask)[0]
                sel = ~ok & live & need
                act[sel] = np.stack([gs3[:, A_Y] - 4, gs3[:, A_X] - 4, gs3[:, A_G]], 1)[sel]
        # act; the world answers; the belief of what is held is settled forward from what was seen
        g = np.zeros((B, 5), int); o = np.zeros((B, 4, 5), int)
        for b in range(B):
            if not live[b]:
                continue
            nx = env.step(st[b], tuple(int(v) for v in act[b])); stats["acted"] += 1
            e = nx["hand"] - st[b]["hand"]
            g[b] = (e[0] + 4, e[1] + 4, act[b, 0] + 4, act[b, 1] + 4, act[b, 2])
            for i in range(nobj[b]):
                d = nx["pos"][i] - st[b]["pos"][i]
                o[b, i] = (con[b, i], hb[b, i], d[0] + 4, d[1] + 4, 0)
            st[b] = nx
        if mode not in ("random",):
            co = np.ones((B, 4, 5), bool); co[..., H2] = False
            belief = energy.settle(g, np.ones((B, 5), bool), o, co, mask)[1][..., H2] * mask
            for b in np.nonzero(live)[0]:
                truth = np.arange(4) == st[b]["held"]
                stats["belief_right"] += int((belief[b, :nobj[b]] == truth[:nobj[b]]).all()); stats["belief_n"] += 1
    return steps, stats


def summarise(steps, orc):
    ok = steps >= 0
    return dict(success=round(100 * float(ok.mean()), 1), median_steps=float(np.median(steps[ok])) if ok.any() else None,
                steps_over_oracle=round(float(np.mean(steps[ok] / np.maximum(1, orc[ok]))), 2) if ok.any() else None)


if __name__ == "__main__":
    which, l1, fit_steps = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])       # dev | test
    task_seed, streams = (401, [200]) if which == "dev" else (500, [int(v) for v in sys.argv[4:]])
    out = []
    for kind in (1, 2):
        tasks = make_tasks(200, task_seed + kind, kind)
        orc = np.array([oracle(t) for t in tasks])
        base = dict(task=kind, oracle=summarise(orc, orc))
        print(json.dumps(base), flush=True)
        for seed in streams:
            data = env.collect(50, seed); G, O, m, _ = A.tokens(data)
            for n in (100, 300, 1000, 3000):
                model = A.Pairwise().fit(G[:n], O[:n], m[:n], l1=l1, steps=fit_steps)
                en = Energy(model, seed); mem = touch_memory(data, n)
                modes = ["full"] + (["reflex", "no_belief"] + (["random"] if seed == streams[0] else []) if n == 3000 else [])
                for mode in modes:
                    steps, stats = run(tasks, en, mem, mode, seed=seed)
                    r = dict(task=kind, stream=seed, n=n, mode=mode, touching_places=len(mem[0]), **summarise(steps, orc),
                             belief_right=round(100 * stats["belief_right"] / max(1, stats["belief_n"]), 1),
                             approach_steps=stats["approach"], no_way=stats["no_way"], barred=stats["barred"])
                    out.append(r); print(json.dumps(r), flush=True)
        out.append(base)
    name = "runs/149/%s_l1_%g_%d%s.json" % (which, l1, fit_steps, "" if which == "dev" else "_" + "_".join(sys.argv[4:]))
    json.dump(out, open(name, "w"), indent=1)
