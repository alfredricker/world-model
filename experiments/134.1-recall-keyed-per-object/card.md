---
id: "134.1"
title: recall keyed per object — the touched object's own record, plus the records the store shows to matter, found by contrast
rung: 3
serves: [P4, P12, P21]
status: done
verdict: pass
arch_version: og4
date: 2026-11-13
---

# 134.1: recall keyed per object

Roadmap step 3, rung 3. Fifth card of the memory week (true visible
state, narrow view). [Card 134](../134-unseen-state-from-effects/card.md)
failed with the right content in memory: one flat key of 33 items did
not read the item that decides. The user (2026-11-13): "proceed with
what's next for memory".

Biological counterpart: attention selects which remembered objects
enter a judgement. Before pulling a drawer one thinks of its bolt,
not of the window; which objects are worth attending to for an
action is learned from outcomes (selective attention in
reinforcement learning, Niv and others 2015; Leong and others 2017).

## 1. Question

If recall compares the touched object's own record first, and adds
the records of other objects only where stored outcomes show that
they matter, does it read what the flat key missed, and does it find
"the drawer depends on its lock" and "the cube in a reopened drawer
depends on the cube" without being told? P4 (memory), P12
(conditions held in short-term memory), P21 (conditions come from
memory).

## 2. What changes

One component: the recall key. View, memory contents, scored events
and stores are card 134's. Nothing is trained.

```
134:    one key = contact point, touched object, every object's last seen value and last
        effect (33 items, equal weight); one recall answers the whole effect
134.1:  memory = one record per object: seen flag, last seen value, touched flag, effect of
        the last touch on it, and whether that touch changed it
        recall answers the effect on each object B of touching object A separately, among
        stored touches of A:
          key(A, B) = A's record + the records chosen for (A, B)
        choice by contrast, on the store only: a record (or the contact point) is added
        while it raises, by at least 1 point, the balanced accuracy of "B changed or not"
        recalled for stored episodes held out in turn (fifths)
```

"Whether that touch changed it" is the one new item of content. A
pull that moved nothing and a pull that moved the drawer 1 cm are
close as distances and opposite as events; the flag stores the event.
It is reported with and without.

Arms: flat keys of card 134 (baseline, already run); per-object key
without the flag; per-object key with the flag (the declared arm);
the true lock in a flat key (upper bound, already run).

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Perception is perfect for what is in view | no | near | yes | no | declared for the memory week |
| The scene is a set of objects, each with a record | needs tracking | near | needs objects | needs tracking | given by perception, declared |
| The touched object is identified | needs tracking | near | needs objects | needs tracking | given, declared |
| An effect is local: few other objects decide it | mostly | mostly | varies by game | mostly | not assumed: the choice may add every record |
| Stored episodes contain both outcomes for a pair | needs exploration | the same | few tries | the same | at least 10 of each, else the pair keeps the own record |

## 3. Dependencies

- Cards 132 to 134: test bed, recall of effects, contents, groups.
- [Card 127](../127-plans-from-remembered-touches/card.md) and 125:
  conditions by contrast over memory, here used to build the key.
- Per-object memory with a learned choice of parents is the factored
  model of the object-centric literature (graph: object codes with
  per-object memory).

## 4. Data check

Card 134's: 2,226 test pulls in three groups (1,534, 464 and 228)
and 198 reopenings. In the store, each pair (touched, affected) that
is scored has at least 100 events of each outcome.

## 5. Feasibility gate

- **Upper bound:** the true lock in the key (card 134): 95.9, 91.9
  and 98.2 in the three groups; reopenings 88.1. In the middle group
  "same as the last pull" scores 90.3.
- **Trivial baseline:** card 134's flat key: 88.5, 50.2, 68.4 with
  last seen values (57.9 in the middle group with last effects);
  reopenings 93.4.
- The gate passes if, on the store alone with a fifth of its
  episodes held out in turn, the per-object key is above the flat
  key's test scores in the first two groups.

Result of the gate (2026-11-13, `tools/card134.1/factored.py gate`,
10 seconds): passed. Held-out store episodes, with the flag: lock
touched 93.4, pull only 84.8, nothing 58.2, reopenings 89.6. Without
the flag: 96.3, 68.2, 60.0, 89.6. The choice added the drawer's lock
to the drawer, the window's lock to the window and the cube to
"drawer moves the cube".

## 6. Success criteria and prediction

Test episodes, store of 400 episodes, the arm with the flag; no
training and so no seeds.

1. **Reads what the flat key missed:** in the 464 pulls with the
   lock never touched and the slide pulled earlier, balanced accuracy
   at least 80 (flat key 57.9, rule 90.3, true lock 91.9; one pull
   that does not move is worth 1.9 points).
2. **No harm, and better where the lock was seen:** in the 1,534
   pulls with the lock touched earlier, at least 92 (flat 88.5, true
   lock 95.9); reopenings at most 2 points under the flat key's 93.4.
3. **Conditions found, and honest where nothing is known:** the
   choice adds the drawer's lock for the drawer, the window's lock
   for the window and the cube for "drawer moves the cube", and not
   the other lock; and in the 228 pulls with nothing known the log
   loss is at most the base rate's 0.538.

Prediction: 1 passes narrowly; 2 passes on pulls and fails on
reopenings (89.6 at the gate); 3 passes. The 100-episode store is
reported without a bar.

## 7. Result

Run 2026-11-13, `tools/card134.1/factored.py` (10 seconds),
`runs/134.1/factored.json`, numbers in [results.json](results.json).
Store of 400 episodes, balanced accuracy on the test pulls by what
the agent knows of the lock.

| Key | Lock touched earlier (1,534) | Slide pulled earlier only (464) | Nothing (228) | Reopenings (198) |
|---|---|---|---|---|
| Flat, last seen values (card 134) | 88.5 | 50.2 | 68.4 | 93.4 |
| Flat, + last effects (card 134) | 86.3 | 57.9 | 61.3 | 93.4 |
| Per object, without the flag | 95.8 | 70.0 | 64.0 | 94.0 |
| **Per object, with the flag** | **95.3** | **90.3** | **64.0** | **94.0** |
| True lock, flat (upper bound) | 95.9 | 91.9 | 98.2 | 88.1 |

1. **Reads what the flat key missed: pass.** 90.3 against a bar of
   80; the same as the rule "same as the last pull" (90.3), with no
   rule written.
2. **No harm, better where the lock was seen: pass.** 95.3 against a
   bar of 92, beside 95.9 for the true lock; reopenings 94.0 against
   93.4.
3. **Conditions found, honest where nothing is known: pass.** The
   choice added the drawer's lock (and the contact point) for the
   drawer, the window's lock for the window, and the cube for
   "drawer moves the cube"; never the other lock. Log loss in the
   "nothing known" group 0.437 against 0.538 for the base rate; over
   all pulls the stated probability is within 1.5 points of the
   observed share.

Three things to read beside the bars.

- **The flag matters in the middle group only.** Without it that
  group scores 70.0; with it 90.3. Whether a touch changed anything
  has to be stored as an event, not left as a small distance.
- **"Nothing known" stays at 64.0**, as the first check of card 134
  said it must: the answer there is a probability, and the
  probability is good.
- **With 100 stored episodes the middle group is not read** (58.4,
  mean of 3 draws; flat key 48.4). About 9 stored pulls decide it,
  and with so few the choice of records is also less steady (it
  adds the other lock for the drawer in one draw). The other groups
  hold: lock touched 87.9 with the flag and 94.6 without (flat 81.1,
  true lock 92.7), reopenings 92.7.

## 8. Decision

**Keep.** Recall keyed per object, with the records chosen by
contrast over the store, replaces the flat key in the memory test
bed. Known limit: an inference that rests on rare stored cases needs
the larger store. Next, as the user set out: the harder memory test,
with longer delays between seeing and using.
