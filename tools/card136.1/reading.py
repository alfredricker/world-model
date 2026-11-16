"""Card 136.1: the memory world, scored cold and warm.

Changes from card 136 (tools/card136): 1,000 episodes of 30 touches (was 500 of 60), the agent returns to one of its
last 5 objects with probability 0.4 (was 0.6), a rewiring every 300 episodes, episodes 600-999 scored in blocks of
40. Every scored touch of a child is cold (neither the child nor any of its parents seen yet in this episode: only
memory from before the episode can help) or warm (the rest: memory within the episode).
Memory arms (records(), `persist`):
  episode   card 134.1's record, emptied at the start of every episode (the model now)
  persist   the record is never emptied; time since seen runs on across episodes, and one more item says whether the
            object was last seen in this episode (card 137)

  bin/prun python tools/card136.1/reading.py [gate] [persist] → runs/136.1/<gate|reading|persist>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card136'))
_spec = importlib.util.spec_from_file_location('reading136', Path(__file__).resolve().parent.parent / 'card136' / 'reading.py')
R0 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R0)                 # card 136's reading.py (same file name as this one)
import world as W

ROOT = R0.ROOT
SEEDS, EPS, LEN, REWIRE, BACK, START, BLOCK = (0, 1, 2), 1000, 30, 300, 0.4, 600, 40
CH = [n for n in W.ROLES if n.startswith('ch_')]


def records(ev, persist):
    R = np.zeros((len(ev['ep']), W.N, 5 if persist else 4), np.float32)
    rec, when, ep, clock = np.zeros((W.N, 3), np.float32), np.zeros(W.N), -1, 0
    seen_ep = np.full(W.N, -1)
    cold = np.zeros((len(ev['ep']), W.N), bool)                                          # not yet seen in this episode
    for i, (e, o, c, v) in enumerate(zip(ev['ep'], ev['obj'], ev['changed'], ev['value'])):
        if e != ep:
            ep = e
            if not persist:
                rec[:] = 0
        R[i, :, :3] = rec
        R[i, :, 3] = np.where(rec[:, 0] > 0, np.log1p(np.maximum(clock - when, 0)), 0)
        if persist:
            R[i, :, 4] = seen_ep == e
        cold[i] = seen_ep != e
        rec[o] = (1, v, c); when[o] = clock; seen_ep[o] = e; clock += 1
    return R, cold


def main():
    gate, persist = 'gate' in sys.argv, 'persist' in sys.argv
    out = {}
    for seed in SEEDS:
        ev = W.stream(seed, EPS=EPS, LEN=LEN, REWIRE=REWIRE, back=BACK)
        y, ep, obj, wo = ev['changed'].astype(bool), ev['ep'], ev['obj'], np.array(ev['world']['obj'])
        test = ep >= START
        base, rec = np.zeros(len(y)), np.zeros(len(y))
        R, seen = records(ev, persist)
        chosen = {}
        for b0 in range(START, EPS, BLOCK):
            st, blk = ep < b0, (ep >= b0) & (ep < b0 + BLOCK)
            for o in range(W.N):
                S, Q = np.where(st & (obj == o))[0], np.where(blk & (obj == o))[0]
                base[Q] = y[S].mean()
                if not gate and len(Q):
                    extra = R0.select(R, y, ep, S, o)
                    chosen.setdefault(W.ROLES[ev['role'][S[0]]], []).append([W.ROLES[int(np.where(wo == x)[0][0])] for x in extra])
                    rec[Q] = R0.knn(R0.key(R, S, o, extra), y[S], R0.key(R, Q, o, extra), R0.KP).mean(1)
        o_ = dict(world=ev['world'], observer_calibration_gap=round(100 * R0.calibration(y[test], ev['p_obs'][test]), 2), roles={})
        for r, name in enumerate(W.ROLES):
            if r not in W.PARENTS:
                continue
            q = test & (ev['role'] == r)
            cold = seen[:, wo[r]] & np.all(seen[:, wo[W.PARENTS[r] + ([7] if r == 16 else [])]], axis=1)
            d = {}
            for part, m in (('cold', q & cold), ('warm', q & ~cold)):
                e = dict(n=int(m.sum()), base=round(R0.logloss(y[m], base[m]), 3), observer=round(R0.logloss(y[m], ev['p_obs'][m]), 3))
                e['room'] = round(e['base'] - e['observer'], 3)
                if not gate:
                    e['recall'] = round(R0.logloss(y[m], rec[m]), 3)
                    e['gap_closed'] = round((e['base'] - e['recall']) / max(e['room'], 1e-6), 2) if e['room'] >= 0.1 else None
                d[part] = e
            if not gate:
                d['chosen_last_block'] = chosen[name][-1]
            o_['roles'][name] = d
        out[seed] = o_
        print('world', seed, 'observer calibration gap', o_['observer_calibration_gap'], flush=True)
    mean = lambda v: None if any(x is None for x in v) else round(float(np.mean(v)), 3)
    keys = ('n', 'base', 'observer', 'room') + (() if gate else ('recall', 'gap_closed'))
    out['mean'] = {n: {part: {k: mean([out[s]['roles'][n][part][k] for s in SEEDS]) for k in keys} for part in ('cold', 'warm')} for n in CH}
    if not gate:
        out['gap_closed_per_world'] = {n: {part: [out[s]['roles'][n][part]['gap_closed'] for s in SEEDS] for part in ('cold', 'warm')} for n in CH}
    for n in CH:
        for part in ('cold', 'warm'):
            print(f'  {n:14s} {part}', out['mean'][n][part], '' if gate else out['gap_closed_per_world'][n][part], flush=True)
    json.dump(out, open(ROOT / f'runs/136.1/{"gate" if gate else "persist" if persist else "reading"}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
