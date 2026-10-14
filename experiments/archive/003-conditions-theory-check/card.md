---
id: "003"
title: conditions theory check
rung: 0
serves: [P12, P16, P19]
status: done   # draft | approved | gated | running | done | abandoned
verdict: pass
arch_version: 0
date: 2026-09-26
---

# 003: conditions theory check

## 1. Question

Before any learner is built: in a small key-and-door world, does the working
theory of conditions below (section 2) recover the conditions a goal really
needs, from random play, and does a second, simpler formulation (jumps in the
long-range chance of success) do as well? This tests the theory behind
GOAL.md's main insight (P12: how close a goal is, which conditions it needs,
subgoals from conditions; P16: conditions inferred from successes) before we
spend runs on architectures. It uses the simulator's true variables, so it
tests the theory, not a model (C1: a hand-built mechanism is a probe here,
never a result).

## 2. The theory (worked out with the user, 2026-09-26)

**Where conditions show up: at the step that achieves the goal.** An action
achieves a goal almost surely when all of its conditions hold, and almost
never when any one is missing. Toggling a locked door opens it only when the
agent faces the door *and* holds the matching key; putting iron in a furnace
gives an ingot only if there is coal in it; mining diamond needs facing
diamond *and* an iron pickaxe. So the quantity to look at is the
**achievement probability**: the chance that this action, in this state,
achieves the goal now. It is naturally sharp, near 0 or near 1.

**A condition** is something whose removal drops the achievement probability
to near zero while everything else stays the same. Measured by removal, "and"
conditions count fully; measured by how much each raises the chance on its
own, a condition that only matters together with another (the key, if you
never walk to the door) would be undervalued. A condition need not last:
facing the door is the final condition for opening it.

**Evidence comes from failures.** Toggling a locked door without the key, or
toggling while facing a wall, happens constantly in random play. Contrasting
those near-misses with the rare successes reveals the conditions. This is
also a route to learning from few examples (P19): a handful of successes set
against many near-misses. Conditions are found while the agent is still
clumsy, because clumsy play makes the failures; they stay stored as subgoals
after it becomes competent.

**The hierarchy.** Each condition becomes a goal in turn, with its own
achieving action and conditions: holding the key needs a pickup while facing
the key with empty hands. The recursion stops at movement: stepping forward
works almost anywhere, so it has almost no conditions. Order falls out of
persistence: conditions that last (holding a key) can be met early and kept;
positional ones (facing the door) must be met last, because they cannot be
carried. So "key first, then walk to the door" follows from which conditions
last, without being learned separately.

**How close.** Two quantities at two levels. Achievement probability is sharp
and defines the conditions. Steps to get somewhere are smooth, and are the
right measure for movement, where nothing is missing but position. "How
close" to a goal is built from both: how far the agent is from making every
condition true at once.

**Why one smooth distance was not enough.** In a deterministic world every
step on a shortest path lowers the true distance to the goal by exactly 1,
so under competent play a key pickup looks like any other step. Card 001
asked one learned distance to carry the "and" structure, and two-step chains
scored at the level of the goal-swapped control. Jumps in the long-range
chance of success do appear under clumsy (random) play; this card compares
that formulation with the achievement-probability one.

**Open points, not tested here.** (a) Achievement in MiniGrid takes one step;
in Minecraft it can take several (smelting takes time). (b) "Remove one
condition, keep the rest" needs a state whose parts can be varied separately.
Here the simulator's variables supply them; in a learner they must be
learned (P4, P7), and that is the part most likely to be hard. (c) What sets
the threshold for "near zero" once probabilities are estimated from few
examples.

## 3. What changes

No model (arch_version 0). New: a small world and an evaluator-side analysis.

- **World.** One MiniGrid room split by a wall, with a locked door, the key
  of the door's colour, one distractor key of another colour, and a goal
  square behind the door. The same 5 actions as chained rooms (no drop, so
  picking up the distractor key makes the door unreachable). Layouts are
  randomised per episode; 8×8 grid (6×6 inside the outer walls), chosen by
  the data check. Code: `src/worldmodel/envs/keydoor.py` (a test checks the
  symbolic step against MiniGrid), analysis `src/worldmodel/conditions.py`.
- **Data.** Random play from ordinary starts, with the simulator's variables
  recorded each step.
- **Candidate variables** (the probe's vocabulary, evaluator side): what the
  agent faces (type, colour, state), what it carries (type, colour), door
  state, agent position and direction, and whether it is in the goal's half.
- **Method (b), achievement conditions.** For a goal (door open, holding the
  matching key, being on the goal square), take every step where the goal
  becomes true, and every step with the same action where it does not. Find
  the smallest set of candidate-variable values under which that action
  achieves the goal with probability ≥ 0.9, then score each member by
  removal: achievement probability with all members minus with that one
  removed. Recurse on each condition found.
- **Method (a), long-range jumps.** Compute exactly, over the whole state
  space, the chance that random play reaches the goal within H steps (H set
  so that it is neither near 0 nor near 1 from ordinary starts). Along
  successful episodes, mark the steps with the largest rises; a condition is
  what changed at those steps.

## 4. Dependencies

The MiniGrid simulator, and exact computation over a small state space.
Nothing learned; nothing assumed (CHARTER rule 7).

## 5. Data check

Random play, uniform over the 5 actions, episodes of 10 × size² steps (the
MiniGrid DoorKey convention). Episodes per 1000 in which each event happens:

| Size | Unlock | Matching key picked | Distractor picked | Goal reached | Failed toggle at locked door |
|---|---|---|---|---|---|
| 6 | 382 | 506 | 489 | 153 | 509 |
| 7 | 385 | 524 | 476 | 128 | 444 |
| 8 | 332 | 510 | 484 | 102 | 427 |

Enough everywhere, so no play starts; the analysis uses size 8 (the largest)
with 3000 episodes: 985 unlocks, 1522 matching pickups, 309 goal arrivals.

## 6. Success criteria and prediction

1. **Door open.** Method (b) recovers exactly {toggle; facing the locked
   door; carrying the matching key}, each with removal score ≥ 0.9; the
   distractor key scores < 0.1.
2. **Recursion.** For "holding the matching key" it recovers {pickup; facing
   the matching key; carrying nothing}; for "on the goal square" it recovers
   {door open} plus positional conditions only.
3. **Comparison and sample size.** Report method (a)'s top jump steps by
   kind (matching pickup, distractor pickup, door opening, movement), and
   criteria 1–2 when (b) sees only 10, 30, 100 and 300 successes (P19).

Prediction: (b) passes 1–2 from about 30 successes; (a) finds the key pickup
and door opening but also ranks some movement near the door, and scores the
distractor pickup as a large fall rather than ignoring it. Runtime: minutes.
The trivial baseline: variables simply more frequent just before success
than overall, which should also pick up the goal's half and nearby cells.

## 7. Result

Two changes after the first pass, before reading the verdicts, both from
section 2's wording: (1) the removal score first compared "all members"
with "that member dropped from the rule", which mixes in how often the
member holds by chance (holding the key scored 0.83 for unlocking because
random play often faces the door already holding it); it now compares with
the member absent and the others present, as section 2 defines. (2)
Criterion 2's expected set listed "key in front" beside "matching key in
front", which already implies it. Two positional goals (in the doorway, in
the right room) were added to follow the chain below the goal square. Full
numbers: [results.json](results.json).

**Method (b), conditions at the achieving step** (probability that the
action achieves the goal when every condition holds; each condition's
removal score is that probability minus the one with it absent):

| Goal | Action | Conditions (removal score) | Achieves | Share of successes | Action alone |
|---|---|---|---|---|---|
| Door unlocked | toggle | facing the door (1.0), holding the matching key (1.0) | 1.0 | 1.0 | 0.003 |
| Holding the matching key | pickup | facing the matching key (1.0), empty hands (1.0) | 1.0 | 1.0 | 0.007 |
| In the doorway | forward | facing the door (1.0), door open (1.0) | 1.0 | 1.0 | 0.003 |
| In the right room | forward | in the doorway (1.0), facing right (1.0) | 1.0 | 1.0 | 0.002 |
| On the goal square | forward | facing the goal (1.0) | 1.0 | 1.0 | 0.001 |

Toggling while facing the door achieves nothing with the distractor key
(0 of 3607) or empty hands (0 of 1313). "Door open" gives the unlock rule
for all its successes: re-opening a closed door also happens with the key
in hand, because keys cannot be dropped.

**Method (a), jumps in the exact chance of success within 100 steps under
random play.** Averaged per occurrence, the right interactions raise the
chance and the distractor lowers it (for unlocking: matching pickup +0.074,
distractor pickup −0.018, a move +0.0006; for reaching the doorway: unlock
+0.19, closing the door −0.19). But in 99% of successful episodes the
single largest rise is a move, the final approach to face the door or key,
because a random agent facing its target succeeds on the next step with
chance 1/5. Only for "in the doorway" is it the unlock (97%).

**Baseline** (variable values more frequent in the 10 steps before success
than overall): ranks the true conditions first for unlocking (facing the
door, lift 12.3; holding the matching key, 3.1), but also ranks incidental
ones (x = 4, 1.9; facing right, 1.6) with no way to tell them apart; for
the goal square it ranks "in the right room" and "door open", which (b)
places two steps down the chain.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Door: {toggle, facing door, matching key}, each ≥ 0.9; distractor < 0.1 | 1.0, 1.0; distractor 0.0 | pass |
| 2. Recursion: key rule; goal square gives {door open} + positional | key rule exact; goal square gives only "facing the goal", door open appears two positional steps down | pass for the key, fail as written for the goal square |
| 3. Report (a) and successes 10/30/100/300 | exact recovery of the unlock and key rules in 20 of 20 subsamples at every count, including 10 | reported |

Prediction check: (b) needed 10 successes, not 30; (a) behaved as
predicted, except that the distractor showed as a small fall, not a large
one, when averaged over whole episodes.
## 8. Decision

**Keep**, with one amendment to the theory. Conditions found at the
achieving step, by contrasting successes with failed attempts, recovered
the exact conditions of every goal here from as few as 10 successes, while
long-range jumps were dominated by the final approach, as the user argued
(proximity is a condition). The amendment: the chain from the goal square
to the door runs through positional conditions (in the right room, in the
doorway), and these had to be named by us. Which regions are worth
treating as conditions, rather than every cell, is the open question the
theory must answer next, alongside open point (b), a learned state whose
parts can be varied separately. This check used the simulator's variables,
including the relation "matches the door", so it shows the theory is
consistent, not that it can be learned. Next: step 1 of the appendix.

---

## Appendix: the steps after this card

Each replaces one exact part with a learned one, so a failure points at that
part.

1. **Achievement probability from frames.** Learn it and the "how close"
   estimate from pixels; check against the exact values on unseen layouts.
2. **Conditions from a learned state.** The removal test needs a learned
   state whose parts can be varied separately (open point b). Store each
   found condition as examples, in the same form as a supplied goal, and
   check it recognises "holding the matching key" in new layouts.
3. **Acting on subgoals.** Pursue the lowest unmet condition; compare
   success and speed against acting on the final goal alone.

Chained rooms and card 001's goal-conditioned fork come back afterwards as
the harder test.
