"""check after card 159: the skill of the appearance couplings episode by episode (card 156's learning unchanged), next to
the quality of what was stored.  dev1.py <by> <episodes>"""
import sys, numpy as np, torch
sys.path.insert(0, "tools/card156")
import learn as Le
Ap, Fb, W, M = Le.Ap, Le.Fb, Le.W, Le.M; dev, T = Le.dev, W.T

@torch.no_grad()
def skill(net, frame, cls, n=2000):
    e = net.embed((torch.as_tensor(frame, device=dev).float() / 255)[None])[0]; acc = []
    for c in (1, 2):
        nz = torch.nonzero(cls == c)
        if len(nz) == 0:
            return None
        nz = nz[torch.randint(len(nz), (n,), device=dev)]; p, q = nz[:, 0], nz[:, 1]; lg = net.logits(e[p], e[q], p, q)
        acc.append(float(((lg[:, 1] > lg[:, 2]) == (c == 1)).float().mean()))
    return max(0.0, 2 * float(np.mean(acc)) - 1)


by, n = sys.argv[1], int(sys.argv[2]); world = W.make_world(0, by); mem = Fb.Memory(); net = Ap.Pairs(); eps = Fb.episodes(world, n, 10); sk = []
for e, ep in enumerate(eps):
    frames = ep["frames"]; img = frames.astype(np.float32) / 255; f = torch.tensor(frames).to(dev).permute(0, 3, 1, 2).float() / 255
    idx, valid = M.flow(*M.cost_volume(f[:-1], f[1:])); idx, valid = idx.cpu().numpy(), valid.cpu().numpy()
    S = Le.Seen(frames[0], mem, net, e)
    for t in range(T):
        S.step(img[t], img[t + 1], idx[t], valid[t])
    sk.append(skill(net, frames[T], Ap.evidence(S.model, S.toks[T])) if net.trained else None)
    Le.finish(ep, e, mem, net, S, "own")
    if e % 10 == 9:
        net.fit(300); v = [s for s in sk[-10:] if s is not None]
        q = [i["iou"] for i in mem.info if i["episode"] > e - 10 and i.get("tid", 0) >= 3]
        print(by, "episodes %2d-%2d: skill before fitting to them %s | overlap of the patterns stored on objects %.2f (%d)" % (e - 9, e, "%.3f (lowest %.3f)" % (np.mean(v), min(v)) if v else "-", np.mean(q) if q else 0, len(q)), flush=True)
