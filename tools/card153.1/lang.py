"""Card 153.1 on the word stream: card 153's run with the classes inferred by tools/card153.1/classes.py.
usage: lang.py <name> <by: suffix|onset> <world seeds...>     writes runs/153.1/<name>_lang_<by>.json   ("_nNN": training sentences)
"""
import sys, json, numpy as np
sys.path.insert(0, "tools/card153.1"); sys.path.insert(0, "tools/card153")
import classes as Cl, lang as L
Kd = L.Kd


def learn(world, n_sent, seed, log):
    G, O, words = L.stream(world, n_sent, seed); m = np.ones((len(G), 1), bool)
    seen = {}; ind = np.array([seen.setdefault(w, len(seen)) for w in words])[:, None]; n_ind = len(seen); klog = []
    pool = world["pool"]; ws = list(seen); suf = np.array([pool[u]["suffix"] for u in ws]); ons = np.array([pool[u]["onset"] for u in ws])
    dyn, kind, E = Cl.learn_classes(L.make, G, O, m, ind, n_ind, (suf, ons), (L.NA, L.NA), log=klog)
    K = dyn.K; looks, w = Kd.fit_looks(suf, ons, E, [L.NA, L.NA, K]) if K > 1 else (None, np.ones(n_ind))
    true = np.array([pool[u]["cls"] for u in ws]); table = np.array([[float(w[(kind == k) & (true == t)].sum()) for t in range(4)] for k in range(K)])
    count = np.bincount(ind[:, 0], minlength=n_ind)
    log.update(train_sentences=n_sent, records=len(G), words_met=n_ind, words_met_at_most_twice=int((count <= 2).sum()), kinds=K, kind_sizes=np.bincount(kind, minlength=K).tolist(),
               kind_log=klog, kinds_by_true_class=table.round(1).tolist(), kind_purity=L.pc([table.max(1).sum() / max(table.sum(), 1e-9)]),
               looks_share=dict(zip(("suffix", "onset"), looks.share().values())) if looks else None)
    return dict(dyn=dyn, kind=kind, E=E, looks=looks, seen=seen, G=G, O=O, m=m, words=words)


L.learn = learn
if __name__ == "__main__":
    name, by = sys.argv[1], sys.argv[2]; n_sent = int(name.split("_n")[1].split("_")[0]) if "_n" in name else 1000; res = []
    for s in sys.argv[3:]:
        res.append(L.run_world(int(s), by, n_sent)); r = res[-1]
        print(json.dumps({q: v for q, v in r.items() if q not in ("new_words", "known_exceptions", "kind_log")}))
        print("  log", json.dumps(r["kind_log"])); print("  new words", json.dumps(r["new_words"]), flush=True)
        json.dump(res, open("runs/153.1/%s_lang_%s.json" % (name, by), "w"), indent=1)
