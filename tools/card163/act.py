"""Card 163: acting by the one prediction.

The agent takes the action with the highest q for the event it expects.  The expected event is the goal itself, or the
first link of the chain whose prediction is highest: a chain's prediction is the product of its links, each link the same
q asked in the situation the link before it leaves (a known event imagined whole: the token's new code, the old one gone
from view).  Chains are compared to DEPTH links.  The expected event is asked again when a token changes, when an action
it has already taken in this very situation comes up best, or after HOLD steps.

The one rule that is not learned: an action already taken in this exact situation without the event coming is not taken
again; the next best is.
"""
import os, numpy as np, torch
from collections import Counter
from grid import NA, UNK, OUT, Belief
import q as Q

DEPTH, HOLD = int(os.environ.get("DEPTH", 3)), int(os.environ.get("HOLD", 8))


class Model:
    """the fitted prediction, asked in batches of (map, pose, tokens out of view, token asked about, wanted next code)"""
    def __init__(s, path, T):
        ck = torch.load(path, map_location=Q.dev, weights_only=False); s.NC = ck["NC"]; s.ARR = s.NC; s.bg = ck["bg"]; s.known = ck["known"]; s.net = Q.Net(s.NC).to(Q.dev); s.net.load_state_dict(ck["net"]); s.net.eval()
        c, a, g, it, o, n = T.rows(); m = c != o; cnt = Counter(); s.act_of = {}; s._tok = {}
        for ci, oi, ai, ni in zip(c[m], o[m], a[m], n[m]):
            cnt[(int(ci), int(oi), int(ai))] += int(ni)
        for (ci, oi, ai), k in sorted(cnt.items(), key=lambda x: x[1]):
            s.act_of[(ci, oi)] = ai                                    # the action that most often went with this change of code

    def toks(s, mp):
        key = (mp.shape, mp.tobytes())
        if key not in s._tok:
            s._tok[key] = Q.tokens_of(mp, s.bg)
        return s._tok[key]

    def ask(s, rows):
        n = len(rows)
        if not n:
            return np.zeros((0, NA))
        tk = [s.toks(r[0]) for r in rows]; L = max(1, max(len(t[0]) for t in tk)); M = max(1, max(len(r[2]) for r in rows)); z = lambda *sh: np.zeros(sh, np.int64)
        pd, ac, adx, ady, oo, tc, tdx, tdy, ic, ia = z(n), z(n), z(n), z(n), z(n), z(n, L), z(n, L), z(n, L), z(n, M), z(n, M); tm = np.zeros((n, L), bool); im = np.zeros((n, M), bool)
        for i, (mp, pose, items, X, o) in enumerate(rows):
            xs, ys, cs = tk[i]; k = len(xs); tc[i, :k] = cs; tdx[i, :k] = xs - pose[0]; tdy[i, :k] = ys - pose[1]; tm[i, :k] = True
            for j, (kk, a) in enumerate(sorted(items)):
                ic[i, j] = kk; ia[i, j] = a; im[i, j] = True
            pd[i] = pose[2]; ac[i] = mp[X]; adx[i] = X[0] - pose[0]; ady[i] = X[1] - pose[1]; oo[i] = o
        cl = lambda x: np.minimum(x, s.NC); t = lambda x: torch.as_tensor(x, device=Q.dev); out = []
        with torch.no_grad():
            for b in range(0, n, 4096):
                e = slice(b, b + 4096); out.append(torch.sigmoid(s.net(t(pd[e]), t(cl(ac[e])), t(adx[e]), t(ady[e]), t(oo[e]), t(cl(tc[e])), t(tdx[e]), t(tdy[e]), t(tm[e]), t(cl(ic[e])), t(ia[e]), t(im[e]))).cpu().numpy())
        return np.concatenate(out)


class Agent:
    def __init__(s, model, goals, depth=DEPTH, hold=HOLD):
        s.M = model; s.goals = goals; s.depth = depth; s.hold = hold

    def reset(s, g, p):
        s.bel = Belief(g, p); s.taken = {}; s.noplan = 0; s.held = None; s.age = 0; s.first = None; s.asked = 0

    def observe(s, a, g, p):
        items = s.bel.items(); ev = s.bel.step(a, g, p)
        if ev or s.bel.items() != items:
            s.held = None                                              # a token changed: what to expect is asked again
        return ev

    # ---- what the model is asked
    def goal_asks(s, mp):
        out = []
        for g in s.goals:
            out += [((int(x), int(y)), s.M.ARR if g[0] == "arrive" else int(g[2])) for x, y in zip(*np.nonzero(mp == g[1]))]
        return out

    def cands(s, mp, pose, items):
        """the events the agent knows: a token on the map turning into a code its code has turned into; a token out of view coming back beside the agent"""
        M = s.M; out = []; xs, ys, cs = M.toks(mp)
        for x, y, c in zip(xs, ys, cs):
            if c < M.NC:
                out += [((int(x), int(y)), int(o)) for o in np.nonzero(M.known[c, :M.NC])[0] if o != c]
        back = sorted({k for k, a in items if k < M.NC and M.known[M.bg, k]}); W, H = mp.shape; x0, y0, _ = pose
        if back:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    x, y = x0 + dx, y0 + dy
                    if (dx or dy) and 0 <= x < W and 0 <= y < H and mp[x, y] == M.bg:
                        out += [((x, y), k) for k in back]
        return out

    def imagine(s, mp, items, X, o):
        c = int(mp[X]); mp2 = mp.copy(); mp2[X] = o; it = set(items)
        for i in sorted(it):
            if i[0] == o:
                it.discard(i); break
        if (c, o) in s.M.act_of:
            it.add((c, s.M.act_of[(c, o)]))
        return mp2, frozenset(it)

    def expect(s):
        """the first link of the chain with the highest prediction, and that prediction"""
        mp, pose, items = s.bel.map, s.bel.pose, s.bel.items(); M = s.M; G = s.goal_asks(mp); C = [c for c in s.cands(mp, pose, items) if c not in G] if s.depth > 1 else []
        q0 = M.ask([(mp, pose, items, X, o) for X, o in G + C]).max(1); s.asked += 1; best = (-1.0, None, 1)
        for i, g in enumerate(G):
            best = max(best, (float(q0[i]), g, 1), key=lambda b: b[0])
        if C:
            sit = [s.imagine(mp, items, X, o) for X, o in C]; rows = []; ix = []
            for m1, it1 in sit:
                G1 = s.goal_asks(m1); C1 = [c for c in s.cands(m1, pose, it1) if c not in G1] if s.depth > 2 else []; ix.append((len(rows), len(G1), C1)); rows += [(m1, pose, it1, X, o) for X, o in G1 + C1]
            q1 = M.ask(rows).max(1); rows2 = []; ix2 = []
            for ci, (r0, ng, C1) in enumerate(ix):
                for j, c2 in enumerate(C1):
                    m2, it2 = s.imagine(*sit[ci], *c2); G2 = s.goal_asks(m2); ix2.append((ci, r0 + ng + j, len(rows2), len(G2))); rows2 += [(m2, pose, it2, X, o) for X, o in G2]
            q2 = M.ask(rows2).max(1) if rows2 else np.zeros(0); val = [(float(q1[r0:r0 + ng].max()) if ng else 0.0, 2) for r0, ng, C1 in ix]
            for ci, r1, r2, ng in ix2:
                if ng:
                    val[ci] = max(val[ci], (float(q1[r1] * q2[r2:r2 + ng].max()), 3), key=lambda b: b[0])
            for ci, c in enumerate(C):
                best = max(best, (float(q0[len(G) + ci]) * val[ci][0], c, val[ci][1]), key=lambda b: b[0])
        return best

    def act(s):
        key = (s.bel.map.tobytes(), s.bel.pose, s.bel.items()); taken = s.taken.setdefault(key, set())
        if len(taken) >= NA:
            taken.clear()
        for again in (False, True):
            if s.held is None or s.age >= s.hold or again:
                v, s.held, s.links = s.expect(); s.age = 0
                if s.first is None:
                    s.first = (s.held, s.links, v)
            if s.held is None:                                         # nothing on the map to expect: any action not yet taken here
                a = min(set(range(NA)) - taken); break
            qa = s.M.ask([(s.bel.map, s.bel.pose, s.bel.items(), *s.held)])[0]; a = int(qa.argmax())
            if a not in taken:
                break
            if again:
                s.noplan += 1; qa[list(taken)] = -1; a = int(qa.argmax())
        s.age += 1; taken.add(a)
        return a
