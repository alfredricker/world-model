"""Card 151: three feed-forward layers of attractor modules in a world built from parts.

The world.  16 patches (16 numbers each, +1 or -1).  24 objects, each 4 patches; 12 of them are twins that
differ from another object in one patch.  32 scenes, each 4 objects.  Every scene has one of 8 output
tokens.  An image is a scene's 256 numbers plus noise; a patch may be covered (set to zero before the noise).

The stack.  Every module is a memory of stored patterns; its energy for a stored pattern is how badly that
pattern fits its input; all modules of a layer are of one kind and share their stored patterns.
  layer 1  16 patch modules.  Input: 16 numbers.   Stored: patches.
  layer 2  4 object modules.  Input: the messages of its 4 patch modules.   Stored: objects, as 4 patch codes.
  layer 3  1 scene module.    Input: the messages of the 4 object modules.  Stored: scenes, as 4 object codes,
           each with its output token.
Nothing flows downward.  What a module passes up is one of:
  state    which stored pattern fits best
  gated    the same, or nothing when even the best fits too badly (its scalar energy over a threshold)
  belief   its energy for every stored pattern (as a normalised log-belief)
  energy   only the scalar: how badly its best pattern fits
Learning is one shot and local: a module stores its input as a new pattern when nothing it has fits
(the same threshold), and keeps a pattern it has met at least twice.  Tokens are seen at the top only.
"""
import sys, json, numpy as np, torch

P, NO, NS, NT = 16, 24, 32, 8
THETA = 12.0                                   # a patch further than this (squared) from every stored patch is new


def make_world(seed):
    rng = np.random.default_rng(seed)
    while True:
        patches = rng.choice([-1.0, 1.0], (P, 16))
        d = (patches[:, None] != patches[None]).sum(-1) + 99 * np.eye(P)
        if d.min() >= 5:
            break
    base = [tuple(rng.integers(0, P, 4)) for _ in range(NO // 2)]
    objs = list(base)
    for b in base:
        while True:
            t = list(b); k = rng.integers(4); t[k] = int(rng.integers(P))
            if tuple(t) not in objs:
                objs.append(tuple(t)); break
    objs = np.array(objs)
    scenes = set()
    while len(scenes) < 2 * NS:
        scenes.add(tuple(rng.integers(0, NO, 4)))
    scenes = np.array(sorted(scenes))[rng.permutation(2 * NS)]
    tok = rng.integers(0, NT, 2 * NS)
    return dict(patches=torch.tensor(patches, dtype=torch.float32), objs=torch.tensor(objs), scenes=torch.tensor(scenes[:NS]),
                tok=torch.tensor(tok[:NS]), new_scenes=torch.tensor(scenes[NS:]), new_tok=torch.tensor(tok[NS:]))


def render(w, scenes, sigma, cover=0.0, gen=None):
    """scenes [N,4] object ids -> x [N,4,4,16]"""
    x = w["patches"][w["objs"][scenes]]                           # [N,4,4,16]
    if cover > 0:
        x = x * (torch.rand(x.shape[:3], generator=gen) >= cover)[..., None]
    return x + sigma * torch.randn(x.shape, generator=gen)


def sample(w, n, sigma, cover=0.0, gen=None, pool="scenes"):
    idx = torch.randint(0, len(w[pool]), (n,), generator=gen)
    return render(w, w[pool][idx], sigma, cover, gen), idx


def patch_energy(x, protos):
    return 0.5 * ((x[..., None, :] - protos) ** 2).sum(-1)        # [..., n1]


def up(energy, mode):
    """a module's message from its energies over its stored patterns [..., n]"""
    if mode == "belief":
        return torch.log_softmax(-energy, -1)
    hard = torch.nn.functional.one_hot(energy.argmin(-1), energy.shape[-1]).float()
    if mode == "gated":
        hard = hard * (2 * energy.min(-1, keepdim=True).values <= THETA)
    return hard


def stack(x, mem, mode):
    """returns the scene chosen, the object chosen per object module, the patch chosen per patch module"""
    e1 = patch_energy(x, mem["patches"])                          # [N,4,4,n1]
    if mode == "energy":                                          # only the scalar goes up: nothing says which patch
        s = e1.min(-1).values                                     # [N,4,4]
        e2 = ((s[:, :, None] - mem["energy_means"]) ** 2).sum(-1)  # nearest stored object in the space of 4 scalars
        m2 = up(e2, "state")
    else:
        m1 = up(e1, mode)
        n2 = len(mem["objs"])
        e2 = -torch.stack([m1[:, :, c][..., mem["objs"][:, c]] for c in range(4)]).sum(0)      # [N,4,n2]
        m2 = up(e2, "belief" if mode == "belief" else "state")
    e3 = -torch.stack([m2[:, o][..., mem["scenes"][:, o]] for o in range(4)]).sum(0)           # [N,n3]
    return e3.argmin(-1), e2.argmin(-1), e1.argmin(-1)


def flat(x, images):
    return ((x.reshape(len(x), 1, -1) - images.reshape(1, len(images), -1)) ** 2).sum(-1).argmin(-1)


def learn(w, n, sigma, gen):
    """layer by layer, from n unlabelled noisy scenes (their tokens seen at the top only)"""
    x, idx = sample(w, n, sigma, gen=gen)
    protos, counts = [], []
    for p in x.reshape(-1, 16):
        if protos:
            d = ((torch.stack(protos) - p) ** 2).sum(-1); k = int(d.argmin())
        if not protos or d[k] > THETA:
            protos.append(p.clone()); counts.append(1)
        else:
            counts[k] += 1; protos[k] += (p - protos[k]) / counts[k]
    keep = []                                                     # two stored patterns that have come to fit each other are one
    for k in sorted(range(len(protos)), key=lambda k: -counts[k]):
        if counts[k] >= 2 and all(((protos[k] - protos[j]) ** 2).sum() > THETA for j in keep):
            keep.append(k)
    patches = torch.stack([protos[k] for k in keep])
    c1 = patch_energy(x, patches).argmin(-1)                      # [n,4,4]
    tup, cnt = np.unique(c1.reshape(-1, 4).numpy(), axis=0, return_counts=True)
    objs = torch.tensor(tup[cnt >= 2])
    c2 = ((c1[:, :, None, :] != objs[None, None]).sum(-1)).argmin(-1)        # exact where it exists
    known = ((c1[:, :, None, :] != objs[None, None]).sum(-1)).min(-1).values == 0
    rows = c2[known.all(1)].numpy(); toks = w["tok"][idx][known.all(1)].numpy()
    tup3, inv, cnt3 = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
    keep3 = cnt3 >= 2
    tok = np.array([np.bincount(toks[inv == k], minlength=NT).argmax() for k in range(len(tup3))])
    return dict(patches=patches, objs=objs, scenes=torch.tensor(tup3[keep3]), tok=torch.tensor(tok[keep3]))


def given(w):
    return dict(patches=w["patches"], objs=w["objs"], scenes=w["scenes"], tok=w["tok"])


def match(w, mem):
    """are the learned patterns the world's own?  counts, and whether they map one to one"""
    d = ((mem["patches"][:, None] - w["patches"][None]) ** 2).sum(-1)
    to_true = d.argmin(1)
    ok1 = len(mem["patches"]) == P and len(set(to_true.tolist())) == P and bool((d.min(1).values < 2).all())
    o_true = {tuple(t) for t in w["objs"].tolist()}; o_mem = {tuple(to_true[torch.tensor(t)].tolist()) for t in mem["objs"].tolist()}
    obj_id = {tuple(t): k for k, t in enumerate(w["objs"].tolist())}
    s_mem = set()
    for t in mem["scenes"].tolist():
        ids = [obj_id.get(tuple(to_true[mem["objs"][k]].tolist()), -1) for k in t]; s_mem.add(tuple(ids))
    s_true = {tuple(t) for t in w["scenes"].tolist()}
    return dict(patches=len(mem["patches"]), objects=len(mem["objs"]), scenes=len(mem["scenes"]),
                patches_true=ok1, objects_true=o_mem == o_true, scenes_true=s_mem == s_true)


def stream(w, mem, sigma, eta, share, gen, B=400, segs=10, L=16):
    """fresh noise at every tick, a new scene every L ticks; every module low-passes what it is driven by
    (v <- (1 - eta_layer) v + eta_layer * drive) and reads the layer below as it was one tick ago; `share` of the
    modules update at each tick.  eta: one rate per layer (1 = no memory of earlier ticks).
    Returns the share of right tokens at each tick since the scene changed."""
    n1, n2, n3 = len(mem["patches"]), len(mem["objs"]), len(mem["scenes"])
    v1 = torch.zeros(B, 4, 4, n1); v2 = torch.zeros(B, 4, n2); v3 = torch.zeros(B, n3)
    right = torch.zeros(L); stok = None
    for s in range(segs):
        idx = torch.randint(0, NS, (B,), generator=gen)
        for t in range(L):
            x = render(w, w["scenes"][idx], sigma, gen=gen)
            d1 = torch.log_softmax(-patch_energy(x, mem["patches"]), -1)
            m1 = torch.log_softmax(v1, -1)
            d2 = torch.stack([m1[:, :, c][..., mem["objs"][:, c]] for c in range(4)]).sum(0)
            m2 = torch.log_softmax(v2, -1)
            d3 = torch.stack([m2[:, o][..., mem["scenes"][:, o]] for o in range(4)]).sum(0)
            for v, d, et in ((v1, d1, eta[0]), (v2, d2, eta[1]), (v3, d3, eta[2])):
                go = (torch.rand(v.shape[:-1], generator=gen) < share)[..., None]
                v.copy_(torch.where(go, (1 - et) * v + et * d, v))
            if s > 0:
                right[t] += (mem["tok"][v3.argmax(-1)] == w["tok"][idx]).float().mean()
    return (100 * right / (segs - 1)).tolist()


def evaluate(seed, n_test=4000):
    w = make_world(seed); gen = torch.Generator().manual_seed(1000 + seed)
    out = dict(world=seed)
    g = given(w)
    xs, ix = sample(w, 4000, 0.5, gen=gen)                        # for the scalar-only reading: what each object's 4 scalars look like
    s = patch_energy(xs, w["patches"]).min(-1).values
    obj_of = w["scenes"][ix]
    g["energy_means"] = torch.stack([s[obj_of == k].mean(0) if (obj_of == k).any() else torch.zeros(4) for k in range(NO)])
    lm = learn(w, 2000, 0.5, gen); out["learned_2000"] = match(w, lm)
    out["learned_96"] = match(w, learn(w, 96, 0.5, gen))
    clean = render(w, w["scenes"], 0.0)
    acc = lambda pred, tok, idx: round(100 * float((tok[pred] == w["tok"][idx]).float().mean()), 1)
    out["noise"] = {}
    for sigma in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        x, idx = sample(w, n_test, sigma, gen=gen)
        r = {m: acc(stack(x, g, m)[0], g["tok"], idx) for m in ("state", "gated", "belief", "energy")}
        r["flat"] = acc(flat(x, clean), w["tok"], idx)
        if len(lm["scenes"]):
            r["belief, learned"] = acc(stack(x, lm, "belief")[0], lm["tok"], idx)
            r["state, learned"] = acc(stack(x, lm, "state")[0], lm["tok"], idx)
        out["noise"][str(sigma)] = r
    out["covered"] = {}
    for cover in (0.25, 0.5):
        x, idx = sample(w, n_test, 0.5, cover, gen)
        r = {m: acc(stack(x, g, m)[0], g["tok"], idx) for m in ("state", "gated", "belief")}
        r["flat"] = acc(flat(x, clean), w["tok"], idx)
        out["covered"][str(cover)] = r
    # a new scene, made of known objects, shown once with its token: stored at the top as object codes, or flat as the noisy image
    out["new_scene_shown_once"] = {}
    for s_new in (0.5, 1.0, 1.5):
        xn = render(w, w["new_scenes"], s_new, gen=gen)
        codes = stack(xn, dict(g, scenes=w["scenes"]), "belief")[1]           # the object modules' choice for the one view
        mem2 = dict(g, scenes=torch.cat([w["scenes"], codes]), tok=torch.cat([w["tok"], w["new_tok"]]))
        x, idx = sample(w, n_test, 1.5, gen=gen, pool="new_scenes")
        pred = stack(x, mem2, "belief")[0]
        hier = round(100 * float((pred == idx + NS).float().mean()), 1)
        fl = round(100 * float((flat(x, torch.cat([clean, xn])) == idx + NS).float().mean()), 1)
        out["new_scene_shown_once"][str(s_new)] = dict(layers=hier, flat=fl)
    out["stream"] = {}
    for sigma in (4.0, 5.0):
        out["stream"][str(sigma)] = {"single look": stream(w, g, sigma, (1, 1, 1), 1.0, gen),
                                     "memory in the patch layer": stream(w, g, sigma, (0.3, 1, 1), 1.0, gen),
                                     "memory in the scene layer": stream(w, g, sigma, (1, 1, 0.3), 1.0, gen),
                                     "memory in every layer": stream(w, g, sigma, (0.3, 0.3, 0.3), 1.0, gen),
                                     "memory in the scene layer, half the modules per tick": stream(w, g, sigma, (1, 1, 0.3), 0.5, gen)}
    return out


if __name__ == "__main__":
    res = [evaluate(int(s)) for s in sys.argv[2:]]
    json.dump(res, open("runs/151/%s.json" % sys.argv[1], "w"), indent=1)
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k != "stream"}))
        for sg, d in r["stream"].items():
            for k, v in d.items():
                print("  stream sigma", sg, k, [round(a) for a in v], "mean ticks 6-15: %.1f, ticks 0-5: %.1f" % (np.mean(v[6:]), np.mean(v[:6])))
