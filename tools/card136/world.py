"""Card 136: a memory world, with the best possible answer.

A symbolic world of 17 two-state objects. A touch tries to flip the touched object; it works if the object's parents
hold the values its rule requires (and, for one object, only 70% of the time). The agent sees only the object it
touches, after the touch. Eight objects are switches (no parents), nine are children, one per factor:
  ch_random      parent starts at random every episode, stays until touched        short-term memory
  ch_carry       parent keeps its state from one episode to the next               long-term, a specific past episode
  ch_default     parent starts every episode at the same value                     long-term, a fixed fact
  ch_drift_slow  parent flips unseen at a low rate (2-4% per step)                 trust that falls slowly with age
  ch_drift_fast  parent flips unseen at a high rate (8-12% per step)               trust that falls fast with age
  ch_timer       parent falls back to its default T steps after being set          a learned timer
  ch_chance      works 70% of the time when its parent allows                      a probability, not a guess
  ch_two         two parents, both must hold                                       a condition of two objects
  ch_rewired     its parent is swapped for another switch every 150 episodes       long-term, recent over old
Object numbers, required values, defaults, rates and T are drawn per world. Behaviour: each step the agent touches,
with probability 0.6, one of the last 5 distinct objects it touched, else any object.

The best possible answer (observer): exact Bayesian belief over all 2^17 joint states given everything the agent has
seen, knowing the world's rules as they are at that moment. For states this is the ceiling of any memory; for rules
it is a reference that no learner reaches at once after a rewiring.

  import world; ev = world.stream(seed)   (events of 500 episodes of 60 touches, with the observer's probability)
"""
import numpy as np

N, EPS, LEN, REWIRE = 17, 500, 60, 150
ROLES = ('sw_rand_a', 'sw_rand_b', 'sw_carry', 'sw_default', 'sw_drift_slow', 'sw_drift_fast', 'sw_timer', 'sw_spare',
         'ch_random', 'ch_carry', 'ch_default', 'ch_drift_slow', 'ch_drift_fast', 'ch_timer', 'ch_chance', 'ch_two', 'ch_rewired')
PARENTS = {8: [0], 9: [2], 10: [3], 11: [4], 12: [5], 13: [6], 14: [1], 15: [0, 1], 16: [1]}
IDX = np.arange(2 ** N)


def stream(seed, observer=True, EPS=EPS, LEN=LEN, REWIRE=REWIRE, back=0.6):
    rng = np.random.default_rng(seed)
    obj = rng.permutation(N)                                                             # role → object number
    role = np.argsort(obj)                                                               # object number → role
    req = rng.integers(0, 2, N)                                                          # value required of a role when it is a parent
    default = rng.integers(0, 2, N)
    rate = {4: rng.uniform(0.02, 0.04), 5: rng.uniform(0.08, 0.12)}
    T = int(rng.integers(4, 8))
    bit = [(IDX >> obj[r]) & 1 for r in range(N)]                                        # by role
    flip = [IDX ^ (1 << obj[r]) for r in range(N)]
    s = rng.integers(0, 2, N)                                                            # true state by role
    b = np.full(2 ** N, 2.0 ** -N) if observer else None
    ev = {k: [] for k in ('ep', 't', 'obj', 'role', 'changed', 'value', 'p_obs', 'cond', 'since_rewire')}

    def setbit(b, r, v=None):                                                            # forget role r's bit, then fix it to v or leave it even
        m = b + b[flip[r]]
        return m * (bit[r] == v) if v is not None else m / 2

    for e in range(EPS):
        parents = {k: list(v) for k, v in PARENTS.items()}
        if (e // REWIRE) % 2 == 1:
            parents[16] = [7]
        for r in range(N):                                                               # start of episode
            if r == 2:
                continue                                                                 # carried over
            if r in (3, 6):
                s[r] = default[r]
            else:
                s[r] = rng.integers(0, 2)
            if observer:
                b = setbit(b, r, default[r] if r in (3, 6) else None)
        recent, due = [], None
        for t in range(LEN):
            pool = recent[-5:]
            r = pool[rng.integers(len(pool))] if pool and rng.random() < back else int(rng.integers(N))
            recent = [x for x in recent if x != r] + [r]
            cond = all(s[q] == req[q] for q in parents.get(r, []))
            p = 0.7 if r == 14 else 1.0
            ch = bool(cond and rng.random() < p)
            if observer:
                c = np.ones(2 ** N, bool)
                for q in parents.get(r, []):
                    c &= bit[q] == req[q]
                ev['p_obs'].append(float(p * b[c].sum()))
                b = b[flip[r]] * c * p if ch else b * (1 - p * c)
            if ch:
                s[r] = 1 - s[r]
            if observer:
                b = b * (bit[r] == s[r]); b = b / b.sum()
            for k, v in (('ep', e), ('t', t), ('obj', obj[r]), ('role', r), ('changed', ch), ('value', s[r]), ('cond', cond), ('since_rewire', e % REWIRE if e >= REWIRE else 999)):
                ev[k].append(v)
            if r == 6:                                                                   # the timer: set → falls back T steps later
                due = t + T if s[6] != default[6] else None
            if due is not None and t + 1 >= due:
                s[6] = default[6]; due = None
                if observer:
                    b = setbit(b, 6, default[6])
            for q, rt in rate.items():                                                   # unseen drift
                if rng.random() < rt:
                    s[q] = 1 - s[q]
                if observer:
                    b = (1 - rt) * b + rt * b[flip[q]]
    out = {k: np.array(v) for k, v in ev.items()}
    out['world'] = dict(obj=obj.tolist(), req=req.tolist(), default=default.tolist(), rate={ROLES[k]: round(v, 3) for k, v in rate.items()}, T=T)
    return out
