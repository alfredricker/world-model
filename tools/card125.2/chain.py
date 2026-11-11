"""Card 125.2: the chain, scored by changes of state.

One set of 100 episodes is the memory. Places, outcomes, the steady rule, candidate patches and the two states of
each slide place's condition are card 125.1's (main.py), unchanged. New unit: an interval is two consecutive
touches of one episode that both have a direct (steady) reading of the condition, with the touches from the first
up to before the second as its candidate causes. The model says the condition changed in an interval if the two
readings' states differ.
- truth of a reading: the lock's true state 2 frames before the touch's contact (the end of its before window);
- cause place: over the intervals the model says changed, the place that occurs in most of them;
- truth of the cause place: the place where that slide's button flips most often (flips from 5 frames before
  contact to 2 after release, allowing for the contact sensor's lag).
Reported per slide place: detected changes that are true, true changes detected, changed intervals holding a touch
at the cause place, unchanged intervals holding one, whether the cause place is the true one.

  bin/prun python tools/card125.2/chain.py --mem m400 --first 400 → runs/125.2/chain_<mem>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card125.1'))
from main import K, fisher_steady  # noqa: E402
from gate import DATA, EP, ROOT, SLIDE, TOP, load, otsu  # noqa: E402

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def main():
    mem, first = arg('--mem', 'm400'), int(arg('--first', 400))
    M = load(mem, first)
    btn = np.load(DATA / 'scene-play-v0.npz')['button_states'][first * EP:(first + 100) * EP]
    st, en = M['st'], M['en']
    ep = st // EP
    lo = np.maximum(st - 5, ep * EP)
    flips = btn[np.minimum(en + 2, len(btn) - 1)] != btn[lo]                       # [touches, 2]
    vm = np.log(np.load(ROOT / f'runs/125.1/var_{mem}.npy') + 1e-6)
    steady = vm < otsu(vm.reshape(-1)[::7])
    eps = np.median(M['tden'])
    sm = np.log(np.sort(M['tnum'] / (M['tden'] + eps), 1)[:, -4:].mean(1))
    ym = sm > otsu(sm)
    pm = KMeans(K, n_init=5, random_state=0).fit(M['pos']).labels_
    n = len(sm)
    res = dict(memory=mem, touches=n, places=[])
    for c in range(K):
        im = np.where(pm == c)[0]
        slide = next((s for s in SLIDE if M[f'{s}_moved'][im].mean() >= 0.3), None)
        if not slide:
            continue
        bi = SLIDE[slide][1]
        top = np.sort(np.argsort(fisher_steady(np.asarray(M['before'][im], np.float32), ym[im], steady[im]))[-TOP:])
        F = np.concatenate([np.asarray(M['before'][i:i + 256][:, top], np.float32) for i in range(0, n, 256)])
        ws = steady[:, top]
        centre = (F * ws[..., None]).sum(0) / np.maximum(ws.sum(0), 1)[:, None]
        known = ws.sum(1) >= 3
        read = ((F - centre) * ws[..., None]).sum(1) / np.maximum(ws.sum(1), 1)[:, None]
        state = np.full(n, -1)
        state[known] = KMeans(2, n_init=5, random_state=0).fit_predict(read[known])
        true = btn[st - 2, bi]
        ki = np.where(known)[0]
        iv = [(a, b) for a, b in zip(ki[:-1], ki[1:]) if ep[a] == ep[b]]
        said = np.array([state[a] != state[b] for a, b in iv])
        was = np.array([true[a] != true[b] for a, b in iv])
        holds = np.stack([[bool((pm[a:b] == p).any()) for a, b in iv] for p in range(K)])        # [K, intervals]
        cause = int(holds[:, said].sum(1).argmax())
        true_place = int(np.bincount(pm[flips[:, bi]], minlength=K).argmax())
        at = holds[cause]
        one = np.array([(pm[a:b] == cause).sum() == 1 and b - a == 1 for a, b in iv])
        res['places'].append(dict(
            place=c, slide=slide, direct_readings=round(float(known.mean()), 3), intervals=len(iv),
            said_changed=int(said.sum()), truly_changed=int(was.sum()),
            detected_changes_true=round(float(was[said].mean()), 3),
            true_changes_detected=round(float(said[was].mean()), 3),
            cause_place=cause, true_button_place=true_place,
            changed_intervals_with_cause_place=round(float(at[said].mean()), 3),
            unchanged_intervals_with_cause_place=round(float(at[~said].mean()), 3),
            best_other_place_in_changed=round(float(np.delete(holds[:, said].mean(1), cause).max()), 3),
            single_touch_at_cause_place_changes_it=round(float(said[one].mean()), 3) if one.any() else None,
            single_touch_intervals=int(one.sum())))
    print(json.dumps(res, indent=1))
    json.dump(res, open(ROOT / f'runs/125.2/chain_{mem}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
