"""check before card 160: at first sight, how the appearance couplings bind an object, by its width.  dev4.py <by>"""
import sys, torch, numpy as np, importlib.util, collections
_s = importlib.util.spec_from_file_location("relation159", "tools/card159/relation.py"); Re = importlib.util.module_from_spec(_s); _s.loader.exec_module(Re)
C = Re.C; Ap, Fb, W = C.Ap, C.Fb, C.W; H = W.H
by = sys.argv[1]; X = torch.load("runs/159/exp_%s_0.pt" % by, weights_only=False); world = W.make_world(0, by)
te = Fb.episodes(world, 40, 777, "fresh", W.SIZES); st = collections.defaultdict(list); yy, xx = np.mgrid[:H, :H]
for e, ep in enumerate(te):
    frame, mask = ep["frames"][0], ep["masks"][0]; lab, rec, tokens, model = Ap.first_sight(frame, X["mem"], X["net"], 5000 + e)
    A = model.A.cpu().numpy() if hasattr(model, "A") else None; bg = np.bincount(lab.ravel()).argmax()
    for i, o in enumerate(ep["objs"]):
        m = mask == 3 + i
        if m.sum() < 0.9 * o["mask"].sum(): continue                      # hidden in part by the hand or another object
        ls = np.bincount(lab[m], minlength=Fb.K); big = ls.argmax(); idx = np.nonzero(m.ravel())[0]
        py, px = yy.ravel()[idx], xx.ravel()[idx]; d = np.abs(py[:, None] - py[None]).clip(min=np.abs(px[:, None] - px[None]))
        a = A[np.ix_(idx, idx)]; ring = (~m) & (np.abs(yy - py.mean()) < o["size"] / 2 + 3) & (np.abs(xx - px.mean()) < o["size"] / 2 + 3); ridx = np.nonzero(ring.ravel())[0]
        st[o["size"]].append(dict(in_ground=ls[bg] / m.sum(), largest=(ls[big] / m.sum()) if big != bg else 0.0, labels=int(((ls >= 0.1 * m.sum()) & (np.arange(Fb.K) != bg)).sum()),
                                  near=a[(d > 0) & (d <= 4)].mean(), mid=a[(d > 4) & (d <= 8)].mean() if (d > 4).any() else np.nan, far=a[d > 8].mean() if (d > 8).any() else np.nan, ground=A[np.ix_(idx, ridx)].mean()))
for s in sorted(st):
    v = st[s]; f = lambda k: np.nanmean([q[k] for q in v])
    print(by, "width %2d: objects %3d | pixels left in the ground %.2f, in the largest label %.2f, labels per object %.2f | coupling inside the object: pairs up to 4 apart %+.2f, 5-8 %+.2f, over 8 %+.2f; to the ground around %+.2f" % (
        s, len(v), f("in_ground"), f("largest"), f("labels"), f("near"), f("mid"), f("far"), f("ground")))
