---
id: "012"
title: discovery end to end
rung: 0
serves: [P12, P16, P4, P6, P20]
status: done   # draft | approved | gated | running | done | abandoned
verdict: partial   # user, 2026-09-27: discovery passes, acting fails at walking
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

**Bench, 2026-09-27 (day): the diagnosis above is wrong.** Setup: 30k
random episodes, encoder pretrained on the goal-square signal and then
fine-tuned on the level-1 goal (as the procedure does, but with exact
labels), ready frames from the agent's own achievement head, reach scored
on 300 held-out episodes (~125k frames) against the simulator. Way 1 =
forward onto the goal square (condition ≈ door open); way 2 = toggle that
brings way 1's condition on (≈ switch on / matching key held). 15k updates.

| Reach learner | switch way 1 | switch way 2 | key way 1 | key way 2 |
|---|---|---|---|---|
| plain (one net, discount 0.99) | 99.6% (recall 100%) | 94.4% (100%) | 99.6% (100%) | 91.2% (100%) |
| double (run 2's) | 98.8% (84%) | 97.1% (100%) | 99.8% (100%) | 91.3% (100%) |
| fixed horizon, 32 steps | 99.0% (88%) | 97.1% (100%) | 99.6% (99%) | 91.4% (100%) |

Percentages are accuracy on frames where the parent goal is off; the
share of those frames that can actually reach a ready state is 5.7%, 25.5%,
1.4% and 7.8%. The learned ready frames nearly match the exact ones (key
way 1: 1065 learned vs 954 exact, all 954 included). Without the level-1
fine-tune, way 2 is at chance for every learner (area under the curve
0.50): the goal-square-only encoder does not show the switch. So the reach
learner is not the bottleneck; fixed horizon is not clearly better (its
distance estimates are off by 3–8 steps) and is not adopted. Run 2's learned
conditions nonetheless had low recall on test frames (key, "forward": on in
0.17% of frames; AUC 0.80). The loss must come from a later stage of the
real pipeline: fine-tuning on learned (not exact) conditions, the
walking-invariance rule, junk goals filling the 16-goal budget, or the final
all-ways value refit.

**Trace (run 2's settings, key, learned vs exact after each depth).** The
loss is at level 1 already: ready frames are exact (recall 100%) but "door
open reachable" is on in only 5–10% of the frames where it truly is; every
deeper level inherits it (recall 1–13%). The walking rule, junk goals and
the refit add nothing measurable. Cause: the encoder has only seen the
goal-square signal, and a value head on its frozen features cannot learn
"door open" (bench without the level-1 fine-tune: AUC 0.50–0.61 for every
reach learner).

**Fix (run 3): way values trained through the encoder.** The (walk, reach)
values of every way become outputs of the main network and train together
with the goal outputs. Bench (key, goal-square-only encoder at the start):
way 1 recall 27% → 98%, AUC 0.57 → 0.998. Two details were needed in the
pipeline: the value loss gets its own uniform batch of 1024 walking steps
(with ~130 walking rows per update, recall stayed at 4%; bench with 128:
4%), and it is a cross-entropy on the logits (a squared error on
probabilities near 0 gave the shared encoder almost no gradient: level-1
recall 38–61% → 98%, false-on 0.4%).

**Run 3 (`runs/012run3_key`), key: structure found, acting 21.8%.** All
expected conditions found; "door open" matches door=open with AUC 0.999,
"holding the key" 0.994; the junk `drop` way at level 2 is gone and the
tree matches the exact gate's (including "pick up a blocking object").
Acting 21.8% of 500 new layouts (run 2: 1%; random 0.4%). Diagnostics:
learned conditions + exact walking 97%; exact conditions + learned walking
25%. Acting probe (`diag_act`, 100 layouts): learned walking moves are
near random (3529 closer vs 2819 farther); exact walking with the learned
"take the action now" trigger: 63%.

**Fix (run 4): walking.** Bench (share of learned moves that bring the
agent closer, exact distance; random moves 45%): walking to the goal
square through the door 53% at walk discount 0.95, 66% at 0.8 or 0.6;
with half of the value batch drawn from walking steps where the way's
condition is on (door-open frames are 1.5% of the data): 83%. Walking to
the door with the key: 92% → 95%. Run 4 uses discount 0.8, the focused
batch, and a final phase that trains every goal and way together once all
conditions are known.

**Run 4 (`runs/012run4_key`): acting 22%, no gain.** On random held-out
frames its walking is decent (goal square 67% of moves closer, door 94%),
but failing episodes show two loops: (a) holding the distractor key, the
agent's "holding the key" condition is on, it walks to the door and waits
there forever (the toggle correctly never fires); (b) after opening the
door it spins in the doorway. "Door open" recall fell from 98% after
level 1 to 70% at the end.

**Run 5 (`runs/012run5_key`): + play starts in every world (a third of
episodes begin holding the matching key, with the switch on, or both;
declared, as the card already allowed in "both").** "Holding the key" now
matches "holding the key of the door's colour" (AUC 0.9975, was
"carrying=key"). Acting 30.2%; exact conditions + learned rest 39%,
learned conditions + exact rest 95%. Goal arrivals only 580 → 645. "Door
open reachable" at the end: recall 71% (first room with the door open:
61%; second room: 97%); walking to the goal square 61% of moves closer.
Later training erases a condition learned earlier.

**Run 6: condition detectors.** One output per condition, trained on the
agent's own labels from the level where the condition was found (the
labels are fixed then), in every later phase. Conditions are read from
these outputs. This is rehearsal of the agent's own earlier conclusions
(card 008: keep training on every goal found).

**Run 6 (`runs/012run6_key`): worse, reverted.** Acting 16.6%; exact
conditions + learned walking fell to 18% (run 5: 39%). The extra outputs
seem to crowd the walking values in the shared network. The code is back
to run 5's method.

**Summary (key world; `results.json`).** Exact gate passes in all four
worlds. Switch, either and both were not run with the fixes (the user
stopped the card for the change in section 8).

| Run | Change | Structure | Acting (500 new layouts) | Exact conditions + learned walking | Learned conditions + exact walking |
|---|---|---|---|---|---|
| 2 | frozen-feature values | partly | 1% | 0% | 5% |
| 3 | values through encoder | all | 21.8% | 25% | 97% |
| 4 | walk discount, focused batch | all | 22.0% | 28% | 98% |
| 5 | play starts | all | **30.2%** | 39% | 95% |
| 6 | condition detectors | all | 16.6% | 18% | 94% |

Random play 0.4%; exact procedure 100%.

| Criterion (key, run 5) | Result | Verdict |
|---|---|---|
| 1. Structure, AUC ≥ 0.95 | against the exact condition: door open 0.9975, matching key 0.994; best simulator meaning: door=open 1.0, key matching the door 0.9975, empty hands (leaf) 1.0; breaking the vase changes a learned condition in ≤ 0.4% of frames | pass |
| 2. Condition on → achieves ≥ 90%, off ≤ 10% | door open 76% / 1%; matching key 57.5% / 2%; key reachable 100% / 15% | fail |
| 3. Acting ≥ 90% | 30.2% (random 0.4%) | fail |

## 8. Decision

**Revise** (the user calls it a partial success). Discovery from one
signal works: the learned tree has the expected conditions with the right
meanings, and acting on them with exact walking reaches the goal in 95% of
new layouts. What fails is getting to where an action works. By design,
the recursion stops at "a ready state can be reached by walking", and a
single flat walk value per way has to cover the whole walk. It is reliable
near common targets (94% of moves closer when walking to the door with the
key) and poor over long walks through rare regions (61% to the goal square
through an opened door; stuck spinning in the doorway). Tuning inside this
card did not fix it (runs 4 and 6). Next, [card 013](../013-walking-in-the-recursion/card.md):
let the recursion continue into walking, so long walks become chains of
short ones.

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
