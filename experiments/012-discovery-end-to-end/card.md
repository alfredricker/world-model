---
id: "012"
title: discovery end to end
rung: 0
serves: [P12, P16, P4, P6, P20]
status: running   # draft | approved | gated | running | done | abandoned
verdict:
arch_version: 0
date: 2026-09-27
---

# 012: discovery end to end

## 1. Question

Given only pixels and one supplied goal ("be on the goal square"), can the
agent discover every requirement below it, in its own learned detectors,
including requirements that are alternatives (key **or** switch) and ones
needed together (key **and** switch), and then use them to reach the goal
in new layouts? Card 007 did this for one chain in the key room with
detectors shaped by extra supplied goals; card 008 showed the agent must
keep learning from its own discovered goals; card 010 showed the
requirement structure is learnable from pixels when the simulator's words
are given. Card 011 (detectors with every goal supplied) was skipped by
decision with the user: the one-signal problem is the one that matters.
Serves P12, P16, P4, P6 and, in a first form, P20.

## 2. How a goal's requirements are described

A goal can have one or more **ways to succeed**; each way needs certain
things to be true **together** (card 010). The agent discovers them level
by level, as in cards 007–008:

1. For the current goal, learn from frames which actions achieve it
   (achievement head, card 005). **Every** action for which the evidence
   supports "this action achieves the goal" (card 010's test, not a count)
   is a separate way to succeed. Card 007 kept only the best one.
2. For each way, the **ready states** are the frames where that action
   would work (achievement > 0.5).
3. For each way, the discovered **condition** is "a ready state can be
   reached by walking", read from a reach value that does not shrink with
   distance (undiscounted; card 009's idea), so it does not flicker.
4. **Self-check** before accepting a condition, from the agent's own
   experience only: in training frames where it was on, did walking to a
   ready state and taking the action succeed more often than where it was
   off (evidence test)? If not, the condition is rejected.
5. The condition switching on becomes the next goal. The network keeps
   training on each new goal (card 008).
6. A branch stops when its condition is already on in most (> 50%) of the
   episodes' starting frames (a plan can start there), when no action is
   supported by evidence (a condition to keep, not make), or at depth 6.

**Why this should separate lumped requirements.** In the "both" world the
condition below "door open" will be one lump, "holding the key and the
switch on". The lump itself switches on in two ways: picking up the key
(if the switch is already on) or flipping the switch (if already holding
the key). Step 1 at the next level finds both, and each way's condition
names the other piece. In "either" the lump is "key or switch on"; it
switches on by picking up the key or by flipping the switch: two separate
options. This is the untested idea the card rests on.

**Acting.** From the current frame, find the shallowest level whose
condition holds; at a level with several ways, take the one with the
higher reach value (higher chance of getting there; closer breaks ties);
if none holds, go down the branch whose deeper conditions can be met now.
Then walk to a ready state (discounted walk value, 5% random moves) and
take the action. When several things are needed together, the lasting
ones come first (card 004): persistence measured from the agent's own
data.

## 3. What changes

From card 008 arm C: several ways per goal (step 1), undiscounted reach
(step 3), the self-check (step 4), the stopping rule (step 6), the acting
rule above, and card 010's worlds. Encoder trained on the goal-square
signal only (as card 008's encoder G), then fine-tuned per level
(20k updates, all goals at that depth together). Networks saved.

**Declared, not learned:** the walking actions (turns, forward: a later
card on skills); the 0.5 cut-offs and the 50%-of-starts stop; the
level-by-level procedure; full view (no memory); play starts (below).
**Pinned (user, 2026-09-27):** merging two detectors that mean the same
thing. Comparing yes/no answers on frames is easy here but needs care and
a literature review to scale; duplicates are only counted and reported.
**Pinned:** demonstrations (P16), the planned data source once random play
gets too thin (chained rooms at the latest).

## 4. Dependencies

Cards 005 (achievement from frames), 007–008 (level procedure, continual
learning), 010 (evidence test, worlds; passed). Undiscounted Q-learning on
a deterministic world is standard; its risk (values creeping up through
the max) is caught by the self-check and criterion 2.

## 5. Data check and feasibility gate

Worlds key, switch, either, both (card 010, with drop and the vase).
Goal-square arrivals in 10k random episodes: 176, 648, 760, 55. Target:
≥ 300 per world. Use 30k episodes, and play starts in "both" (a third of
episodes begin with the switch on, the key in hand, or both); report the
counts.

- **Upper bound (exact):** the same procedure with exact detectors,
  achievement and reach. Must give the expected structure (table in the
  appendix). Run first; if not, the card stops.
- **Diagnostic swaps** (reported): exact detectors + learned rest; learned
  detectors + exact reach; all learned.
- **Trivial baselines:** acting on the goal square's own walk value;
  random play.

## 6. Success criteria and prediction

On 500 new layouts per world, all learned:

1. **Structure:** each expected condition (appendix) is found, matching
   its meaning with area under the curve ≥ 0.95; the vase appears in no
   condition.
2. **Conditions behave as conditions:** from frames with a condition on,
   walking (exact) to a ready state and taking the action achieves its
   level's goal ≥ 90%; with it off ≤ 10%.
3. **Acting:** ≥ 90% of new layouts reach the goal square in every world,
   against the goal square's own walk value and random play.

Prediction: the exact gate passes. Key and switch pass (one chain each).
"Either" passes. "Both" is the risk: the lump must split at the next
level, and its data is the thinnest. Runtime: about 20 minutes per world,
80 in all, so it is handed to the user unless approved to run here.

## 7. Result

**In progress (2026-09-27, stopped mid-session; see STATUS.md).** Code:
`src/worldmodel/discover_logic.py`; script `runs/012.sh`.

**Speed changes (user asked for ~5 min per world).** The whole training
step is compiled (render + forward + backward: 630 updates/s alone, ~450
with the target pass, vs ~200 in card 008). Data are collected on 20 CPU
workers (30k episodes in ~6 s). Only transitions where something other
than the agent changed, goal arrivals and a uniform sample of the rest are
stored (declared; weights keep counts natural). Updates: pretrain 30k,
fine-tune 12k per depth, values 15k (card said 20k fine-tune). Learned
part ≈ 5–6 min per world; the exact gate is the slow part (key 10 min on
5000 episodes).

**Exact gate: passes** in key and switch (either/both were running when
the session stopped). It finds the expected structure, and also a real
extra way the table missed: "pick up an object blocking the path" (random
play drops keys in front of the door), which grows extra branches.
Acting with exact detectors: 100% of 100 new layouts, ~16 steps.

**Run 1 (as written; `runs/012run1_*`): fail.** Acting 0.4% (key), 0%
(switch). Cause: the reach value (discount 0.99, one network) creeps up
on unreachable frames (switch: median 0.28, 90th percentile 0.77 against
the 0.5 cut), so conditions flip while walking and turns/steps were
accepted as "ways" (e.g. left 11652 nats). The self-check did not catch it.

**Run 2 (`runs/012_*`), three fixes:** clipped double values (smaller of
two estimates, lr 3e-4; bench: accuracy 74% → 98%, but reachable frames
now undershoot, recall 77%); a condition is set equal on both frames of a
walking step (walking cannot change what walking reaches; removes the
movement "ways"); 1/8 walking data kept instead of 1/16. Key world: tree
shape right at the top (forward → toggle → pickup) but deeper
conditions weak and unstable between repeats (pick up key on at start
19% in this run, 100% in an earlier attempt that ran out of memory);
acting 1% (random 0.4%). Switch/either/both results were still being
written when the session stopped.

**Diagnosis so far:** discovery logic is fine (exact gate); the learned
"can I reach it at all" detector (a thresholded bootstrapped value) is
the bottleneck. **Next idea, not yet tested:** fixed-horizon reach
(De Asis et al. 2020): learn "reachable within k steps" for k = 1…48,
each horizon bootstrapping from the one below, no discount, no drift;
distance = number of horizons on. Test on the bench
(scratch script, reach vs exact on switch data) before any rerun.

## 8. Decision

---

## Appendix: expected structure

| World | Level 1 (goal square) | Level 2 (door open) | Level 3 | Stops at |
|---|---|---|---|---|
| key | door open | holding the matching key | empty hands, key on the floor | on at start |
| switch | door open | switch on | switch off, facing it reachable | on at start |
| either | door open | key **or** switch on (one lump) | two ways: pick up key / flip switch | each on at start |
| both | door open | key **and** switch on (one lump) | two ways: pick up key (needs switch on) / flip switch (needs key) | next level: flip switch / pick up key, then on at start |

Meanings are checked against simulator labels, including combined ones
("key and switch on"). Duplicates (the same meaning found twice) are
counted.
