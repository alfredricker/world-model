---
id: "014"
title: local-step walking
rung: 0
serves: [P12, P9, P21]
status: gated   # draft | approved | gated | running | done | abandoned
verdict:        # pass | fail | uninterpretable, when done
arch_version: 0
date: 2026-09-27
---

# 014: local-step walking

## 1. Question

Can the agent walk to where an action works almost without error, in
layouts it has not seen, if "can get there within k steps" is computed as
"one more step from within k − 1 steps" by one learned local step applied
across the spatial map? Before rung 1; serves P12 (the recursion of
conditions reaches down to single moves), P9 (walking as a reusable skill
that works in any layout) and P21 (a short, fixed computation per move).

Card 012 found the right conditions (acting on them with exact walking:
95% of new layouts) but walked badly (acting 30%). Card 013 showed why the
fix is not more of the same: its value network flattens the encoder's map
into one vector, so it must learn distances layout by layout (86% of moves
closer alone, 61% shared with the other ways, 80–84% as a chain).

## 2. What changes

One component: the walk and reach values of each way (card 012 run 5's
`MinHead` on the flattened encoder vector). Everything else, including the
conditions, stays as in run 5.

```
before: frame → conv map (64 × 9 × 8) → flatten → 256 → MLP → value per move
after:  frame → conv map (64 × 9 × 8)
          target map  T = conv(map)        (4 facings × 9 × 8): "ready here"
          V_0 = T;  V_k = T or step(V_{k-1}, map),  k = 1 … 32
            step: one small conv (3 × 3), the same weights at every cell
                  and every k, one output per move and facing; V_k = max over moves
          readout: a learned attention over (facing, cell) picks the agent's
                   own entry: reach = V_32 there; walk = the move whose
                   successor is in V_k for the smallest k
```

The map cells line up with the grid's tiles (stride 8). The target map is
trained where it can be checked: at the agent's own entry, against the
agent's own "this action works here" output (card 012's achievement head).
The values are trained by fixed-horizon TD at the agent's entry (card 013:
horizon k bootstraps from k − 1). The step's weights are shared, so one
rule for "one step" is learned from every cell of every frame.

**The locality prior (declared, not learned):** what can be reached in one
step depends only on the 3 × 3 neighbourhood, and the rule is the same
everywhere. This fits walking and nothing else here: a switch that opens a
distant door is not local, which is why the conditions keep their own
(non-local) outputs. It also needs a map of the whole room; the full top
view gives one now, a partial view later will need memory to build it.

**Not supplied:** where the agent is, where walls or doors are, what the
target is. The readout attention must find the agent itself (VIN was given
the position; this is the new, risky part, gated in section 5).

## 3. Dependencies

- Value iteration networks (`1602_02867`, Tamar et al. 2016): a learned
  3 × 3 recurrence on a map plans shortest paths; 8 × 8 grids, supervised:
  99.6% success vs 97.9% for a plain convolutional network; 16 × 16: 99.3%
  vs 87.6%; trained by RL on 16 × 16: 82.5% vs 33.1%. It was given the
  agent's position; we are not.
- Fixed-horizon TD (`1909_03906`); card 013's bench (place conditions
  learned with AUC 0.95–0.998).
- Card 012 run 5 (saved network `runs/012run5_key/nets.pt`): conditions,
  ready frames and the encoder the new values attach to (CHARTER rule 7).

## 4. Data check

Card 012 run 5's key-world data (30k episodes, a third with play starts).
Report per way the walking steps into a ready frame and the frames where
the way's condition is on (door open: about 2% of frames). For criterion 2,
episodes whose wall is in column 5 are removed from training (about a
third) and used only for testing.

## 5. Feasibility gate

Bench, key world, attached to run 5's network (its encoder continues
training through the new values):

- **Upper bound:** the recurrence trained on exact walking distances at the
  agent's entry (supervised), with the attention given the agent's true
  cell and facing: moves closer ≥ 99% on held-out frames, or the card stops.
- **Readout check:** with the learned attention, it peaks at the agent's
  true cell and facing in ≥ 99% of held-out frames. If not, finding the
  agent becomes its own card first.
- **Trivial baselines:** random moves 45%; one flat value alone 86% (card
  013); run 5's pipeline 61%.

Result of the gate, before the main run:

In progress (`tools/card014/bench_vin.py`, `runs/014debug_*.out`), upper
bound, key world, walking to the goal square / to the door with the key:

| Version | Moves closer | "Within 8 steps" right at the agent |
|---|---|---|
| Supervised at the agent's entry only (probabilities) | 84% / 79% | – |
| Same, recurrence in log-odds | 80% / 80% | 89% / 73% |
| Every cell supervised, per-move values not trained | 44% / 45% (move choice untrained) | 90% / 46% |
| Every cell + per-move values + whole-frame summary at every cell | **90% / 94%** | 97% / 96% |

Two design flaws found and fixed: the "ready here" map could not see the
held item (drawn in one corner tile), so each cell now also gets a
whole-frame summary; and the per-move values need their own labels.
Still under 99%. A per-tile linear read-out of the encoder's map finds a
closed door 99.7% of the time but an **open door 12%** (run 5's encoder)
and 14% (after training the walking module); the switch and vase 0–74%.
Walking through a doorway needs exactly that. The map does not show the
objects walking depends on: a dependency that is neither built nor
established here (CHARTER rule 7).

Card 015 then showed that the map was not the limit: with 30k updates
instead of 8k, the upper bound (every cell supervised, card 015 arm C's
map) reaches **99.97%** of moves closer on held-out frames for both ways
(99.8% on training frames). **The upper-bound part of the gate passes.**
Run 5's own encoder, trained the same way, reaches 99.07% / 99.83%
(training frames 99.65% / 99.97%), so card 015's map is not needed.

Readout check (`runs/014readout.out`, 30k updates, run 5's encoder): every
cell supervised as in the upper bound, but the position is never given;
the attention learns only from the per-move values read through it
(labels at the agent's own entry). On 5000 held-out frames it peaks at
the agent's true cell and facing in **99.3%** (wall column seen) and
**99.55%** (wall column 5, never trained on); right cell 99.6% / 99.7%.
Walking through the learned readout: 100% / 99.87% of moves closer, the
same as the same module given the position. **The readout check passes;
the gate passes.** Still to show: the same without exact distance maps
(criteria 1–3).

## 6. Success criteria and prediction

All learned (no exact distances, no given position), key world:

1. **Walking:** chosen moves bring the agent closer on ≥ 98% of held-out
   frames, for every way of run 5's tree (flat value: 86%, 61% in the
   pipeline). This shows the recursion reaches single moves reliably.
2. **Unseen geometry:** trained without any layout whose wall is in column
   5, walking on those layouts ≥ 95%, with a flat value trained the same
   way reported alongside. This separates a rule for "one step" from
   memorised layouts.
3. **Acting:** run 5's learned conditions with the new walking reach the
   goal square in ≥ 90% of 500 new layouts (run 5: 30.2%; with exact
   walking: 95%; random 0.4%).

Prediction: the upper bound passes (VIN reached 99.6% on 8 × 8). The risk
is the learned readout; the flat encoder already has to find the agent to
walk at all, so we expect it to be learnable. If criteria 1–3 pass, the
new values replace `MinHead` in the pipeline and the switch, either and
both worlds are run (a later card). Budget: gate about 10 minutes; bench
runs about 20 minutes, over 10 so they need this card's approval.

## 7. Result

## 8. Decision
