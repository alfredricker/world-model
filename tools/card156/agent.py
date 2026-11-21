"""Card 156: an agent that acts to find out, in card 152's sprite world.

Card 152's script (world.collect) is told where every object is and walks to its side.  This agent is told nothing
about objects.  It knows where its hand is (given, as its own pose always is) and what its binding layer makes of
the frame.  Each episode:
  - at first sight it lists the groups its binding layer sees (stored patterns recognised, and groups bound by the
    appearance couplings), other than the ground and its own hand.  For this list the appearance couplings are
    read at equal odds of the three states (what could be an object); for its beliefs they are read with the
    states' frequencies (what has so far proved to be one);
  - it takes them in turn: goes to the side of the group's box, applies a grip and moves; if the group's points
    moved with the hand it carries the group a few steps and lets go with the other grip; if not it comes back
    and tries the other grip; then the next group;
  - with no group left it babbles: random places, random grips.
Whether a group moved is read from the motion modules at the group's own pixels.
"""
import sys, numpy as np, torch
sys.path.insert(0, "tools/card156")
import learn as Le
Ap, Fb, W, M = Le.Ap, Le.Fb, Le.W, Le.M
H, T, HAND, dev = Fb.H, Fb.T, W.HAND, Ap.dev


def groups(lab, hand):
    """the boxes (y, x, h, w) and masks of the labels that are neither the ground nor the hand"""
    bg = np.bincount(lab.ravel()).argmax(); out = []
    hb = np.zeros((H, H), bool); hb[hand[0]:hand[0] + HAND, hand[1]:hand[1] + HAND] = True
    for k in np.unique(lab):
        m = lab == k
        if k == bg or m.sum() < 8 or (m & hb).sum() > 0.5 * m.sum():
            continue
        m = Fb.in_frame(m); ys, xs = np.nonzero(m)
        out.append(((int(ys.min()), int(xs.min()), int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)), m))
    return out


def side_spot(hand, box):
    y, x, h, w = box; cl = lambda v: int(np.clip(v, 0, H - HAND)); spots = []
    if x - HAND >= 0: spots.append((cl(y + h // 2 - 3), x - HAND))
    if x + w + HAND <= H: spots.append((cl(y + h // 2 - 3), x + w))
    if y - HAND >= 0: spots.append((y - HAND, cl(x + w // 2 - 3)))
    if y + h + HAND <= H: spots.append((y + h, cl(x + w // 2 - 3)))
    spots = np.array(spots)
    return spots[np.abs(spots - hand).max(1).argmin()]


def live_episode(world, e, rng, mem, net, source="mixed", sizes=W.SIZES, learn=True, seed=0, labels="own"):
    objs = W.draw_objects(rng, world, source, sizes); st = W.new_state(rng, objs); bg = W.background(rng)
    frame, mask = W.render(rng, bg, st, objs)
    ep = dict(objs=objs, frame=frame, mask=mask, order=st["order"], hand=[st["hand"]], pos=[st["pos"]], held=[-1], acts=[])
    frames, masks = [frame], [mask.astype(int)]
    S = Le.Seen(frame, mem, net, seed)
    look = S.L[0]
    if net is not None and net.trained:                      # what to act on: every pair at its likeliest state at equal odds,
        net.prior = False                                    # i.e. what could be an object, not only what has so far proved one
        look = Ap.first_sight(frame, mem, net, seed)[0]; net.prior = True
    todo = groups(look, st["hand"]); order = rng.permutation(len(todo)); todo = [todo[i] for i in order]
    plan = dict(mode="next"); tried = 0; came = 0
    for t in range(T):
        hand = st["hand"]
        if plan["mode"] == "next":
            if todo:
                box, m = todo.pop(0); plan = dict(mode="go", box=box, m=m, grip=int(rng.integers(2)), n=0); tried += 1
            else:
                plan = dict(mode="babble", goal=rng.integers(0, H - HAND + 1, 2), grip=int(rng.integers(2)))
        if plan["mode"] == "go":
            spot = side_spot(hand, plan["box"])
            if np.all(hand == spot):
                d = np.sign(np.array([H // 2 - 3, H // 2 - 3]) - hand) * 3; d[d == 0] = 3
                a = (int(d[0]), int(d[1]), plan["grip"]); plan["mode"] = "tried"
            else:
                a = (*np.clip(spot - hand, -4, 4), plan["grip"])
        elif plan["mode"] == "carry":
            a = (*np.clip(plan["goal"] - hand, -4, 4), plan["grip"]); plan["left"] -= 1
            if plan["left"] <= 0 or np.all(hand == plan["goal"]):
                plan = dict(mode="release", grip=1 - plan["grip"])
        elif plan["mode"] == "release":
            a = (0, 0, plan["grip"]); plan = dict(mode="next")
        elif plan["mode"] == "babble":
            a = (*np.clip(plan["goal"] - hand, -4, 4), plan["grip"])
            if np.all(hand == plan["goal"]):
                plan = dict(mode="next")
        a = tuple(int(v) for v in a)
        nxt = W.step(st, a, objs); frame, mask = W.render(rng, bg, nxt, objs)
        f = torch.tensor(np.stack([frames[-1], frame])).to(dev).permute(0, 3, 1, 2).float() / 255
        idx, valid = M.flow(*M.cost_volume(f[:1], f[1:])); idx, valid = idx[0].cpu().numpy(), valid[0].cpu().numpy()
        S.step(frames[-1].astype(np.float32) / 255, frame.astype(np.float32) / 255, idx, valid)
        if plan["mode"] == "tried":                                   # did the group's own points move as the hand did?
            dh = nxt["hand"] - st["hand"]; sel = plan["m"] & valid
            want = (dh[0] + M.R) * (2 * M.R + 1) + (dh[1] + M.R)
            if dh.any() and sel.sum() and (idx[sel] == want).mean() > 0.5:
                came += 1; plan = dict(mode="carry", goal=rng.integers(0, H - HAND + 1, 2), grip=plan["grip"], left=int(rng.integers(3, 8)))
            elif plan["n"] == 0:
                plan = dict(mode="go", box=plan["box"], m=plan["m"], grip=1 - plan["grip"], n=1)
            else:
                plan = dict(mode="next")
        ep["hand"].append(nxt["hand"]); ep["pos"].append(nxt["pos"]); ep["held"].append(nxt["held"]); ep["acts"].append(a)
        frames.append(frame); masks.append(mask.astype(int)); st = nxt
    for k in ("hand", "pos", "held", "acts"):
        ep[k] = np.array(ep[k])
    ep["frames"] = np.stack(frames); ep["masks"] = np.stack(masks)
    log = Le.finish(ep, e, mem, net, S, labels) if learn else {}
    log.update(groups_tried=tried, groups_that_came=came, objects=len(objs), objects_moved=int(sum(np.abs(np.diff(ep["pos"][:, i].astype(int), axis=0)).sum() > 0 for i in range(len(objs)))))
    return ep, S, log
