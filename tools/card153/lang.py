"""Card 153, second domain: a stream of words.  The same rule (tools/card153/kinds.py) must find the word classes.

The world.  Twelve function words ("the a this these some is are will can very and .") and content words of four
classes: singular noun, plural noun, verb, adjective.  Sentences come from a small grammar, e.g.
    the (very) (A) N is (very) A (and A) .      these (A) P are A .      a N will V some (A) P .      this N is a N .
A content word has a form of two attributes, a suffix and an onset (6 values each).  In a world "by suffix" the class
goes with the suffix, in a world "by onset" with the onset; which value goes with which class is drawn per world.
The vocabulary is 100 words per class with Zipf frequencies (the rarest are met once or twice); five frequent words of
each class have the form of another class (as irregular words are the frequent ones).

What the learner gets.  The item is a word (an individual: a word type); a record is one occurrence, written as
tokens: G = the token two back; O = the token before, the token after, both effect tokens ("what the word does" is
where it stands).  A context token is the function word, or "a content word".  Word boundaries and which words are
function words are given.

usage: lang.py <name> <by: suffix|onset> <world seeds...>     writes runs/153/<name>_lang_<by>.json
       "_nNN" in the name sets the number of training sentences (1000)
"""
import sys, json, time, math, numpy as np, torch
sys.path.insert(0, "tools/card153")
import kinds as Kd

FUNC = ["the", "a", "this", "these", "some", "is", "are", "will", "can", "very", "and", "."]
F_ = {w: i for i, w in enumerate(FUNC)}
DOT, OTHER, NV = 11, 12, 13
NOUN, PLUR, VERB, ADJ = 0, 1, 2, 3
PER_CLASS, NA = 100, 6
GV, OV, EFFECT = [NV], [NV, NV], (0, 1)
make = lambda: Kd.Couplings(GV, OV, EFFECT)
pc = lambda x: round(100 * float(np.mean(x)), 1)


def make_world(seed, by):
    rng = np.random.default_rng(seed)
    kind_of = rng.permutation([0, 1, 2, 3, int(rng.integers(4)), int(rng.integers(4))])       # the class of each value of the attribute that matters

    def word(cls, rule=None):
        rule = cls if rule is None else rule
        v = int(rng.choice(np.nonzero(kind_of == rule)[0])); o = int(rng.integers(NA))
        return dict(cls=cls, rule=rule, suffix=v if by == "suffix" else o, onset=o if by == "suffix" else v)
    pool = [word(c, (c + 1 + int(rng.integers(3))) % 4 if rank in (1, 3, 5, 7, 9) else None) for c in range(4) for rank in range(PER_CLASS)]
    return dict(seed=seed, by=by, kind_of=kind_of, pool=pool, word=word)


def clause(rng):
    out = []

    def noun_phrase(pl):
        out.append(F_[str(rng.choice(["the", "these", "some"] if pl else ["the", "a", "this"]))])
        if rng.random() < 0.4:
            if rng.random() < 0.3:
                out.append(F_["very"])
            out.append(-1 - ADJ)
        out.append(-1 - (PLUR if pl else NOUN))
    pl = bool(rng.random() < 0.5); noun_phrase(pl); r = rng.random()
    if r < 0.35:
        out.append(F_["are" if pl else "is"])
        if rng.random() < 0.3:
            out.append(F_["very"])
        out.append(-1 - ADJ)
        if rng.random() < 0.25:
            out += [F_["and"], -1 - ADJ]
    elif r < 0.9:
        out += [F_[str(rng.choice(["will", "can"]))], -1 - VERB]
        if r < 0.75:
            noun_phrase(bool(rng.random() < 0.5))
    else:
        out.append(F_["are" if pl else "is"]); noun_phrase(pl)
    return out + [DOT]


def stream(world, n_sent, seed, source="pool"):
    """Returns the records of every content word of the stream: G [N,1], O [N,1,2], and the word at each."""
    rng = np.random.default_rng(seed); toks = [DOT, DOT]; words = []
    zipf = 1 / np.arange(1, PER_CLASS + 1); zipf = zipf / zipf.sum(); exc = np.array([1, 3, 5, 7, 9])
    for _ in range(n_sent):
        for t in clause(rng):
            if t >= 0:
                toks.append(t); continue
            c = -1 - t
            if source == "pool":
                w = c * PER_CLASS + int(rng.choice(PER_CLASS, p=zipf))
            elif source == "exceptions":
                w = c * PER_CLASS + int(rng.choice(exc))
            else:
                w = world["word"](c)                                                      # a word never met, whose form follows the rule
            words.append((len(toks), w)); toks.append(OTHER)
    toks = np.array(toks + [DOT]); pos = np.array([p for p, _ in words])
    return toks[pos - 2][:, None], np.stack([toks[pos - 1], toks[pos + 1]], 1)[:, None], [w for _, w in words]


def learn(world, n_sent, seed, log):
    G, O, words = stream(world, n_sent, seed); m = np.ones((len(G), 1), bool)
    seen = {}; ind = np.array([seen.setdefault(w, len(seen)) for w in words])[:, None]; n_ind = len(seen); klog = []
    dyn, kind, E = Kd.learn_kinds(make, G, O, m, ind, n_ind, 1e-3, log=klog)
    K = dyn.K; pool = world["pool"]; ws = list(seen)
    looks, w = Kd.fit_looks(np.array([pool[u]["suffix"] for u in ws]), np.array([pool[u]["onset"] for u in ws]), E, [NA, NA, K]) if K > 1 else (None, np.ones(n_ind))
    true = np.array([pool[u]["cls"] for u in ws])
    table = np.array([[float(w[(kind == k) & (true == t)].sum()) for t in range(4)] for k in range(K)])
    count = np.bincount(ind[:, 0], minlength=n_ind)
    log.update(train_sentences=n_sent, records=len(G), words_met=n_ind, words_met_at_most_twice=int((count <= 2).sum()), kinds=K, kind_sizes=np.bincount(kind, minlength=K).tolist(),
               kind_log=klog, kinds_by_true_class=table.round(1).tolist(), kind_purity=pc([table.max(1).sum() / max(table.sum(), 1e-9)]),
               looks_share=dict(zip(("suffix", "onset"), looks.share().values())) if looks else None, looks_strength=round(looks.strength, 2) if looks else None)
    return dict(dyn=dyn, kind=kind, E=E, looks=looks, seen=seen, G=G, O=O, m=m, words=words)


@torch.no_grad()
def after(model, G, O, w):
    """the probability of the token after the word, given the two tokens before it, under a belief w [N, K] over kinds:
    each kind's prediction weighted by the belief"""
    r = model.rows(G, O, np.ones((len(G), 1), bool), np.zeros((len(G), 1), int))
    xo = Kd.onehot(O[:, 0], OV); xo[:, OV[0]:] = 0                                                 # only the token before is given
    f = model.bo[None] + torch.einsum("re,ked->rkd", xo, model.Wo) + torch.einsum("rg,kgd->rkd", r["xg"], model.Wgo)
    return (torch.softmax(f[..., OV[0]:], -1) * torch.as_tensor(w, device=Kd.dev).float()[..., None]).sum(1).cpu().numpy()


def score(p, O):
    t = O[:, 0, 1]
    return dict(nats=round(float(-np.log(p[np.arange(len(t)), t] + 1e-12).mean()), 4), right=pc(p.argmax(1) == t))


def looks_belief(L, forms, known=None):
    K = L["dyn"].K
    if L["looks"] is None:
        return np.ones((len(forms), K), np.float32) / K
    oh = lambda v: torch.nn.functional.one_hot(torch.as_tensor(v), NA).float()
    f = L["looks"].field(oh([q["suffix"] for q in forms]), oh([q["onset"] for q in forms]))
    if known is not None:
        f = f - torch.tensor(L["E"][known], dtype=torch.float32)
    return torch.softmax(f, -1).numpy()


def run_world(seed, by, n_sent):
    t0 = time.time(); world = make_world(seed, by); log = dict(world=seed, by=by, kind_of=world["kind_of"].tolist()); pool = world["pool"]
    L = learn(world, n_sent, seed * 100 + 10, log); G, O, m = L["G"], L["O"], L["m"]
    cls = np.array([pool[u]["cls"] for u in L["words"]])[:, None]
    bound = make().fit(G, O, m, cls, 4); one = make().fit(G, O, m, 0 * cls, 1)
    # no kinds: the form's two attributes are tokens of the record, coupled straight to the context
    form = lambda ws: np.array([[q["suffix"], q["onset"]] for q in ws])
    direct = Kd.Couplings([NV, NA, NA], OV, EFFECT).fit(np.concatenate([G, form([pool[u] for u in L["words"]])], 1), O, m, 0 * cls, 1)
    eye = lambda c, K: np.eye(K, dtype=np.float32)[c]
    # words never met, whose form follows the rule
    Gn, On, new = stream(world, 600, seed * 100 + 30, "new"); cn = np.array([q["cls"] for q in new])
    log["new_words"] = dict(n=len(Gn), **{"the module: kind from the form": score(after(L["dyn"], Gn, On, looks_belief(L, new)), On),
                            "true class given": score(after(bound, Gn, On, eye(cn, 4)), On), "one kind for all": score(after(one, Gn, On, np.ones((len(Gn), 1))), On),
                            "no kinds, form coupled to the context": score(after(direct, np.concatenate([Gn, form(new)], 1), On, np.ones((len(Gn), 1))), On)})
    # the best a reader of forms can do: the true rule from form to class, and the share of known words that break it
    rho = float(np.mean([pool[u]["cls"] != pool[u]["rule"] for u in L["seen"]])); oracle = rho / 3 + (1 - rho - rho / 3) * eye(cn, 4)
    log["new_words"]["the true rule, with the share of known words that break it"] = score(after(bound, Gn, On, oracle), On)
    log["share_of_known_words_that_break_the_rule"] = round(rho, 4)
    kind_right = np.array(log["kinds_by_true_class"]).argmax(1)[looks_belief(L, new).argmax(1)] == cn
    log["new_words"]["kind_from_form_right"] = pc(kind_right)
    # known words whose form misleads
    Ge, Oe, ex = stream(world, 600, seed * 100 + 40, "exceptions"); keep = np.array([u in L["seen"] for u in ex])
    Ge, Oe = Ge[keep], Oe[keep]; ex = [u for u in ex if u in L["seen"]]; fe = [pool[u] for u in ex]; ce = np.array([q["cls"] for q in fe]); ke = np.array([L["seen"][u] for u in ex])
    log["known_exceptions"] = dict(n=len(Ge), **{"form and the word's own records": score(after(L["dyn"], Ge, Oe, looks_belief(L, fe, ke)), Oe),
                                   "form only": score(after(L["dyn"], Ge, Oe, looks_belief(L, fe)), Oe), "true class given": score(after(bound, Ge, Oe, eye(ce, 4)), Oe),
                                   "one kind for all": score(after(one, Ge, Oe, np.ones((len(Ge), 1))), Oe)})
    a = log["new_words"]; gain = a["one kind for all"]["nats"] - a["the module: kind from the form"]["nats"]
    log["gap_closed"] = round(100 * gain / (a["one kind for all"]["nats"] - a["the true rule, with the share of known words that break it"]["nats"]), 1)
    log["gap_closed_to_true_class_given"] = round(100 * gain / (a["one kind for all"]["nats"] - a["true class given"]["nats"]), 1)
    log["seconds"] = round(time.time() - t0)
    return log


if __name__ == "__main__":
    name, by = sys.argv[1], sys.argv[2]
    n_sent = int(name.split("_n")[1].split("_")[0]) if "_n" in name else 1000
    res = []
    for s in sys.argv[3:]:
        res.append(run_world(int(s), by, n_sent)); r = res[-1]
        print(json.dumps({q: v for q, v in r.items() if q not in ("new_words", "known_exceptions", "kind_log")}))
        print("  kind log", json.dumps(r["kind_log"]))
        print("  new words", json.dumps(r["new_words"]))
        print("  known exceptions", json.dumps(r["known_exceptions"]), flush=True)
        json.dump(res, open("runs/153/%s_lang_%s.json" % (name, by), "w"), indent=1)
