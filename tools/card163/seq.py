"""Card 163: the same random play as card 160, kept in order.

Card 160 counted single steps; a prediction of an event to come needs the steps in sequence.  play() here repeats
grid.play() draw for draw (same seeds, same actions) and keeps, per episode: every version of the believed map, and per step
which version held, the pose, the action, the events (cell, code before, code after) and the tokens out of view.

usage: seq.py play <experience>      writes runs/163/seq_<experience>.pkl, checks the counts against runs/160/exp_<experience>.pkl, prints the data check
"""
import sys, time, pickle, numpy as np
from collections import Counter
import grid
from grid import NA, PLAY, Belief, World, Table


def play(seed, steps, names=PLAY):
    T = Table(); rng = np.random.default_rng(seed); eps = []; sets = {}; sid = lambda x: sets.setdefault(x, len(sets))
    for wi, name in enumerate(names):
        w = World(name, T.codes); ep = 0; g, p = w.reset(seed * 1000000 + wi * 100000); bel = Belief(g, p)
        cur = dict(world=wi, maps=[bel.map.astype(np.int16)], ver=[0], poses=[p], acts=[], evs=[], items=[sid(bel.items())], succ=False, ended=False)
        for t in range(steps):
            a = int(rng.integers(NA)); before, pose, items = bel.map, bel.pose, bel.items()
            g, p, succ, done = w.step(a); ev = bel.step(a, g, p); T.add(before, pose, items, a, bel.map, p)
            if not np.array_equal(before, bel.map):
                cur["maps"].append(bel.map.astype(np.int16))
            cur["ver"].append(len(cur["maps"]) - 1); cur["poses"].append(p); cur["acts"].append(a); cur["evs"].append(ev); cur["items"].append(sid(bel.items()))
            if done:
                cur["succ"] = succ; cur["ended"] = True; eps.append(cur); ep += 1; g, p = w.reset(seed * 1000000 + wi * 100000 + ep); bel = Belief(g, p)
                cur = dict(world=wi, maps=[bel.map.astype(np.int16)], ver=[0], poses=[p], acts=[], evs=[], items=[sid(bel.items())], succ=False, ended=False)
        eps.append(cur)
    T.merge(); T.seq_sets = sorted(sets, key=sets.get)
    return T, eps


def check(eps, T, names=PLAY):
    """how often the stretches a prediction of an event must learn from occur"""
    import rules
    nm = lambda c: rules.name(T.codes, c); kind = lambda c, c2, a: "%s: %s -> %s" % (rules.ANAME[a], " ".join(nm(c).split()[::2] if "door" in nm(c) else nm(c).split()[-1:]), " ".join(nm(c2).split()[::2] if "door" in nm(c2) else nm(c2).split()[-1:]))
    n_ev = Counter(); lens = []; start_far = Counter(); pairs = Counter(); per_ep = []; steps = 0
    for e in eps:
        last = 0; prev = None; k = 0
        for t, (a, ev) in enumerate(zip(e["acts"], e["evs"])):
            steps += 1
            if not ev:
                continue
            x, y, c, c2 = ev[0]; kd = kind(c, c2, a); n_ev[kd] += 1; lens.append(t - last); k += 1
            px, py, _ = e["poses"][last]; d = abs(px - x) + abs(py - y); start_far["1" if d <= 1 else "2-3" if d <= 3 else "4-6" if d <= 6 else "7+"] += 1
            if prev is not None:
                pairs[(prev, kd)] += 1
            prev = kd; last = t + 1
        per_ep.append(k)
    lens = np.array(lens); out = dict(steps=steps, episodes=len(eps), events=int(len(lens)), events_per_episode=float(np.mean(per_ep)), episodes_with_3_or_more_events=int(sum(k >= 3 for k in per_ep)),
                                      stretch_length_quartiles=[int(v) for v in np.percentile(lens, [25, 50, 75, 95])], distance_at_stretch_start=dict(start_far),
                                      events_by_kind=dict(n_ev.most_common()), pairs_in_a_row={"%s  then  %s" % k: v for k, v in pairs.most_common(40)})
    return out


if __name__ == "__main__" and sys.argv[1] == "play":
    import json
    exp = int(sys.argv[2]); t0 = time.time(); T, eps = play(exp, 100000); old = pickle.load(open("runs/160/exp_%d.pkl" % exp, "rb"))
    same = bool(np.array_equal(T.keys, old.keys) and np.array_equal(T.n, old.n) and T.agent == old.agent); print("replayed in %.0f s; counts equal to card 160's experience: %s" % (time.time() - t0, same), flush=True)
    assert same
    pickle.dump(dict(T=T, eps=eps), open("runs/163/seq_%d.pkl" % exp, "wb")); out = check(eps, T); json.dump(out, open("runs/163/datacheck_%d.json" % exp, "w"), indent=1)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k)
            for kk, vv in v.items():
                print("    %6d  %s" % (vv, kk))
        else:
            print(k, v)


# ---- stages S and D (the user, 2026-11-24): structured play from checkpoints, and demonstrations
def staged(exp, stage, steps=100000, demos=500, names=PLAY):
    """S: in the play worlds that have a task, half of the steps are played from checkpoints: the agent told MiniGrid's rules
    acts for the first k steps of its own solution (k drawn evenly), unrecorded, and random play follows.  The other half,
    and the worlds without a task, are card 160's random play.  D: stage S with whole solved episodes added, recorded."""
    import agent, rules, tasks
    base = pickle.load(open("runs/163/seq_%d.pkl" % exp, "rb")); T = base["T"]; sets = {x: i for i, x in enumerate(T.seq_sets)}; sid = lambda x: sets.setdefault(x, len(sets))
    task_of = {v: k for k, v in tasks.ALL.items()}; rng = np.random.default_rng(1000 + exp); eps = []; kept = Counter(); count = Counter()
    for e in base["eps"]:                                              # card 160's play: all of it in worlds without a task, the first half elsewhere
        if names[e["world"]] not in task_of or kept[e["world"]] < steps // 2:
            eps.append(e); kept[e["world"]] += len(e["acts"])
    new = lambda bel, wi: dict(world=wi, maps=[bel.map.astype(np.int16)], ver=[0], poses=[bel.pose], acts=[], evs=[], items=[sid(bel.items())], succ=False, ended=False)
    def step(w, bel, cur, a):
        before = bel.map; g, p, succ, done = w.step(a); ev = bel.step(a, g, p)
        if cur is not None:
            if not np.array_equal(before, bel.map):
                cur["maps"].append(bel.map.astype(np.int16))
            cur["ver"].append(len(cur["maps"]) - 1); cur["poses"].append(p); cur["acts"].append(a); cur["evs"].append(ev); cur["items"].append(sid(bel.items())); cur["succ"] = succ; cur["ended"] = done
        return succ, done
    for wi, name in enumerate(names):
        if name not in task_of:
            continue
        w = World(name, T.codes); A = agent.Agent(rules.Rules(T.codes), tasks.told_goal(task_of[name], T.codes)); n = 0; ep = 0
        while n < steps - kept[wi]:                                    # from a checkpoint, then random play
            seed = exp * 1000000 + wi * 100000 + 50000 + ep; ep += 1; g, p = w.reset(seed); A.reset(g, p); sol = []; done = False
            while not done and len(sol) < 400:
                a = A.act(); sol.append(a); g, p, succ, done = w.step(a); A.observe(a, g, p)
            k = int(rng.integers(len(sol))); g, p = w.reset(seed); bel = Belief(g, p)
            for a in sol[:k]:
                step(w, bel, None, a)
            cur = new(bel, wi); done = False
            while not done and n < steps - kept[wi]:
                succ, done = step(w, bel, cur, int(rng.integers(NA))); n += 1
            eps.append(cur); count["checkpoint episodes"] += 1; count["checkpoint steps"] += len(cur["acts"])
        if stage == "D":
            for i in range(demos):                                     # whole solved episodes, recorded
                seed = exp * 1000000 + wi * 100000 + 80000 + i; g, p = w.reset(seed); A.reset(g, p); bel = Belief(g, p); cur = new(bel, wi); done = False
                while not done and len(cur["acts"]) < 400:
                    a = A.act(); succ, done = step(w, bel, cur, a); A.observe(a, *w.obs())
                eps.append(cur); count["demonstrations"] += 1; count["demonstration steps"] += len(cur["acts"]); count["demonstrations solved"] += int(cur["succ"])
    T.seq_sets = sorted(sets, key=sets.get)
    return dict(T=T, eps=eps), dict(count, random_steps=int(sum(kept.values())))


if __name__ == "__main__" and sys.argv[1] == "staged":
    import json
    exp, stage = int(sys.argv[2]), sys.argv[3]; t0 = time.time(); seq, count = staged(exp, stage); pickle.dump(seq, open("runs/163/seq_%d_%s.pkl" % (exp, stage), "wb"))
    out = dict(check(seq["eps"], seq["T"]), **count); json.dump(out, open("runs/163/datacheck_%d_%s.json" % (exp, stage), "w"), indent=1); print("made in %.0f s" % (time.time() - t0), count)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k)
            for kk, vv in list(v.items())[:30]:
                print("    %6d  %s" % (vv, kk))
        else:
            print(k, v)
