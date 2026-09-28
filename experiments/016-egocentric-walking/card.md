---
id: "016"
title: egocentric walking
rung: 0
serves: [P12, P9, P21]
status: done   # draft | approved | gated | running | done | abandoned
verdict: fail   # pass | fail | uninterpretable, when done
arch_version: 0
date: 2026-09-27
---

# 016: egocentric walking

## 1. Question

Does card 014's local-step walking, learned from the agent's own
experience, walk almost without error once the frame is seen from the
agent's point of view, so the agent is always at the same place? Before
rung 1; serves P12 (the recursion of conditions reaches single moves), P9
(walking as a skill that works in any layout) and P21 (a short, fixed
computation per move).

Card 014 showed that the local step works (99.97% of moves closer when
supervised at every cell; 96–98% supervised at the agent's own entry only)
but failed when learning from experience (60–88%): in a top-down frame the
agent must first be found, and its learned attention found it in only
17.5% of frames. The look-ahead targets are read through that same
attention, so neither could get started.

## 2. What changes

One component: the frame the walking module sees (the user's decision,
2026-09-27). The charter's world already gives the learner an egocentric
frame; the logic-door worlds used a top-down one as a simplification.

```
before: top-down room, 9 × 8 tiles (8 × 8 room + a row for the held item);
        readout = a learned attention over (facing, cell)
after:  the same tiles, shifted so the agent is at the centre and rotated so
        it faces up: 13 × 13 tiles (offsets −6 … +6, enough to show the whole
        room from any position; outside the room drawn as wall) + the same
        row for the held item = 14 × 13 tiles, 112 × 104 pixels;
        readout = the fixed centre cell, facing up
```

Only the grid is rotated, not the tile images (as in MiniGrid's own
egocentric view). The walking module is otherwise card 014's: target map,
one learned 3 × 3 step applied 32 times in log-odds, whole-frame summary at
every cell, fixed-horizon TD ("within k" = "one step from within k − 1").
Its convolutions start from run 5's encoder and keep training.

Unchanged: run 5's network, conditions and ready labels, which still read
the top-down frame of the same state. The discovery method is not rerun on
egocentric frames here; that is the next card if this one passes.

The whole room stays visible, so walking can still be judged from one
frame. The charter's 7 × 7 partial view needs memory and waits for rung 5.

## 3. Dependencies

- Card 014 (done, revise): the module, its upper bound (99.97%) and its
  at-the-agent diagnostic (96–98%); `tools/card014/bench_vin.py`.
- Value iteration networks (`1602_02867`) and fixed-horizon TD
  (`1909_03906`), as in card 014. VIN was given the agent's position; a
  fixed centre readout gives the same information without supplying it.
- Card 012 run 5 (`runs/012run5_key/nets.pt`): conditions and ready labels.

## 4. Data check

Card 012 run 5's key-world data (30k episodes, a third with play starts),
rendered egocentrically. Report per way the walking steps into a ready
frame. Episodes with the wall in column 5 are removed from training and
used only for criterion 2 (about a third).

## 5. Feasibility gate

- **Upper bound:** the module on egocentric frames supervised at every cell
  with exact distances (card 014's full-map bound, 30k updates), fixed
  readout: moves closer ≥ 99% on held-out frames, or the card stops.
- **Trivial baselines:** random moves 45%; card 014's learned version
  (top-down, learned attention) 60–88%; run 5's own values 53–93%.

Result of the gate, before the main run:

View checks (`stage check`): the agent's own entry is the centre cell
facing up in 100% of frames; forward and turn successors, and the one-row
shift of the view after a forward move, match exactly (100%).

Upper bound (`runs/016gate.out`, 30k updates, every cell supervised, fixed
readout): moves closer **99.8%** (goal square) and **97.7%** (door with the
key); the door way is under 99% at every distance (96.6% at 1–4 steps).
The loss was still falling (0.0021 at 20k, 0.0008 at 30k; top-down reached
0.0002). Top-down, same budget: 99.07% / 99.83%. **Gate not passed** at
the budget set; the user decides between a longer gate and stopping.

## 6. Success criteria and prediction

Card 014's criteria, unchanged, all learned (no exact distances), key world:

1. **Walking:** chosen moves bring the agent closer on ≥ 98% of held-out
   frames (wall column seen), for every way of run 5's tree.
2. **Unseen geometry:** trained without any layout whose wall is in column
   5, walking on those layouts ≥ 95% (run 5's own values reported
   alongside).
3. **Acting:** run 5's conditions choose the way and fire its action, the
   new module walks: the goal square in ≥ 90% of 500 new layouts (run 5:
   30.2%; card 014: 25.0%; exact walking: 95%).

Prediction: the gate passes (the view only moves and turns the map). The
main run should land near card 014's at-the-agent diagnostic (96–98%), since
the readout is now what that diagnostic was given; the gap between those
numbers and 98% is the risk for criterion 1. Budget: the map has 2.5 times
as many cells as before, so about 2.5 times slower per update. Runs are
measured first; any over 30 minutes is handed to the user as a command.

## 7. Result

The user chose to run the main run before settling the gate (2026-09-27).
`runs/016main.out`, 30k updates, fixed-horizon TD from the agent's own
moves, fixed centre readout, no exact distances, trained without wall
column 5. The loss was still falling (0.146 at 10k, 0.099 at 20k, 0.075 at
30k; card 014's top-down run was flat at 0.08–0.09 from 10k).

| Criterion | Egocentric (this card) | Top-down (card 014) | Needed |
|---|---|---|---|
| 1. Walking, every way | 73–97% (goal square 86.6%, door 97.4%, key 73.0%) | 60–88% | ≥ 98% |
| 2. Wall column 5, never trained on | 83–95% (run 5's values, trained with it: 53–93%) | 64–79% | ≥ 95% |
| 3. Acting, 500 new layouts | **48.4%** (mean 30.5 steps) | 25.0% | ≥ 90% |

Run 5 acting: 30.2%; with exact walking 95%; random 0.4%. All three
criteria fail, but every number improves on the top-down view, and
the unseen wall column is as good as the seen ones (a rule, not memorised
layouts).

Longer run (the user's choice, `runs/016main90k.out`): 90k updates, 24
look-ahead steps instead of 32 to save time (the longest walk in held-out
frames is 17 steps, in the data 23). The loss curve matched the 30k run up
to 30k (0.081 vs 0.075), then fell slowly to 0.055 at 90k.

| Criterion | 30k updates, 32 steps | 90k updates, 24 steps |
|---|---|---|
| 1. Walking, every way | 73–97% | 61–98% (goal square 80.1%, door 97.5%, key 60.5%) |
| 2. Wall column 5 | 83–95% | 69–96% |
| 3. Acting | 48.4% (mean 30.5 steps) | 48.4% (mean 32.6 steps) |

Three times the training lowered the loss by a quarter and changed nothing
that matters: acting is the same (242 of 500 layouts both times, different
runs), the door ways gain a point, the goal square and key ways lose 6–12.
Training length is not what holds the learned version back. The weakest
ways are the ones with the fewest ready frames in the data (key: 1,258;
goal square: 1,103; door: 6,285), but the ways built on the goal square
with about 345,000 ready frames also stay at 84–89%, so data volume alone
does not explain it. The gap to explain is between exact labels at the
agent's own entry (96–98%, card 014, top-down) and labels the module makes
for itself from its own look-ahead (61–98%).

Maps check (`runs/016maps.out`, stage `maps`, the saved 90k module, 2,000
training frames per way, wall column 5 excluded): the learned maps against
the exact ones, at the agent's own entry (the centre, facing up, the only
place training reaches) and at every other cell of the room.

| Way | "Ready here" found, agent / elsewhere | Wrongly "ready", agent / elsewhere | Step count within 1, agent / elsewhere |
|---|---|---|---|
| Goal square | 100% / 5% | 38% / 6% | 53% / 3% |
| Door (with key) | 100% / 50% | 0% / 54% | 89% / 13% |
| Key | 96% / 56% | 76% / 50% | 10% / 22% |

Away from the agent the "ready here" map is chance or worse, and the step
counts the recurrence spreads from it are almost all wrong (3–22% within
one step). So the recurrence is not doing the walking: the moves come from
what the readout can fit directly. The likely route is the whole-frame
summary given to every cell. It is a linear layer over the flattened map,
so it can tell what lies at a given place in the frame; with the agent
always at the centre, it can learn "the goal is just ahead of the centre"
and add that to every cell alike. Run 5's labels agree with the exact
ready states at the agent in 86–98.5% of these frames, so the labels are
not the main problem.

## 8. Decision

**Revise** (2026-09-27, with the user). The egocentric view did what it
was for: nothing has to find the agent, and acting rose from 25% to 48.4%.
But no criterion passes, three times the training changes nothing, and the
maps check shows why: the "ready here" map is right only at the agent and
at chance everywhere else, so the recurrence spreads noise and the moves
come from a shortcut. The whole-frame summary, a linear layer over the
flattened map, can say what lies just ahead of the centre and so fit the
agent's cell directly. [Card 017](../017-summary-without-position/card.md)
originally proposed averaging the summary. The user superseded that design
with one shared spatial learner and nonspatial context (2026-09-27).
