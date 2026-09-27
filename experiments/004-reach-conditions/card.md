---
id: "004"
title: reach conditions
rung: 0
serves: [P12, P9]
status: done   # draft | approved | gated | running | done | abandoned
verdict: pass
arch_version: 0
date: 2026-09-26
---

# 004: reach conditions

## 1. Question

Card 003 found every goal's conditions at the step that achieves it, but
the chain from the goal square down to the door only appeared once we named
two places by hand (the right room, the doorway). Here we test the principle
agreed with the user on 2026-09-26: **places are never conditions in
themselves; what matters is whether the thing needed is within reach by
walking.** Walking to X is one more action, with conditions found the same
way as for toggle or pickup. Does that give the chain key → door → goal with
no place named by hand, and does acting on the chain reach the goal? Serves
P12 (conditions and subgoals) and P9 (walking to X is the first skill, with
what it achieves and when it fails). Like card 003, it uses the simulator's
variables and tests the theory, not a learner (C1).

## 2. The principle

Everything, at every timescale, is an action plus the conditions under which
it works. "Walk to X" (turns and forward steps only, the shortest way) is an
action whose conditions say when X can be reached at all; step counts say how
far it is. A room is then a byproduct, the set of spots from which the same
things can be walked to, and a doorway is a point walking passes through.
Movement is the base level because it is the set of actions whose success
depends on almost nothing (card 003: forward needs only what is in front).
Why it should generalise: a place matters only through what is reachable
from it (deep in a cave: diamonds are reachable; by a table: the table is).
Limits, not tested here: walking must be competent, not random; a target out
of view needs memory (P2); hazards on the path need a notion of failure;
in Minecraft "reach X" also uses digging and building.

## 3. What changes

No model (arch_version 0). Card 003's world, data (8×8, 3000 random-play
episodes, seed 3) and condition finder, plus:

- **Walk labels.** For a state and a target (standing on the goal square;
  facing the door, the matching key or the distractor key), whether some
  sequence of turns and forward steps reaches the target, computed exactly
  per layout. States where the target is already reached or no longer exists
  (the key is carried) are left out.
- **Vocabulary.** Card 003's variables minus "side" (left, doorway, right),
  the one place we named. Raw position and direction stay. A second run keeps
  "side", for comparison.
- **Acting on the chain.** A planner that uses only the found rules: to make
  a condition true, pick the rule that achieves it (walk to X for "facing X"
  or "on the goal square"; card 003's toggle and pickup rules for the rest),
  make its unmet conditions true first, most persistent first (persistence
  measured in the data: how often a condition that holds still holds a step
  later), then act. Walking uses an exact shortest-path walker. Run from the
  start of 500 new layouts.

## 4. Dependencies

Card 003 (done, keep): the world, the symbolic step (tested against
MiniGrid), the condition finder and its rules. Nothing learned.

## 5. Data check

Card 003's data, every 4th state: 460,487 states to walk to the goal square
from (43,776 can reach it), 450,841 to the door (all can), 258,447 to the
matching key (all can), 248,988 to the distractor (247,164 can). Walking to
the matching key never fails here, because the door can only be unlocked
while holding that key, so the agent is never cut off from it; a world with
a key behind another door is needed to test that part properly.

## 6. Success criteria and prediction

1. **Walking to the goal square** has "door open" as a condition with
   removal score ≥ 0.9, in the rule covering the most successes; walking to
   the door and to the matching key have no conditions (a rule with none,
   achieving ≥ 0.9) in the rule covering the most successes.
2. **Chain.** Backward chaining over the found rules from "on the goal
   square", from a start state, gives: walk to the matching key, pick it up,
   walk to the door, toggle, walk to the goal square, with no place named
   and the distractor key never used.
3. **Acting.** Executing the chain reaches the goal square in ≥ 95% of 500
   new layouts; report steps against the exact shortest solution, and random
   play's success in the same step budget.

Prediction: 1–3 pass. Walking to the goal square may get a second rule for
being already on the far side of a closed door, written in raw positions
because "side" is removed; it should cover few successes. Runtime: minutes.

## 7. Result

Code: `src/worldmodel/reach.py`; numbers: [results.json](results.json).
Runtime 55 s.

| Walk to | Conditions found (removal score) | Reaches | Share of successes | Without conditions |
|---|---|---|---|---|
| Goal square | door open (0.996) | 1.0 | 0.958 | 0.095 |
| Door | none | 1.0 | 1.0 | 1.0 |
| Matching key | none | 1.0 | 1.0 | 1.0 |
| Distractor key | none | 0.993 | 1.0 | 0.993 |

The remaining 4.2% of walks to the goal square start already on the far
side of a closed door; they fall under the 5% floor for a second rule.
Keeping "side" in the vocabulary gives identical rules.

**Chain and acting.** Persistence under random play ordered the conditions:
holding the key 1.0, door open 0.994, empty hands 0.99, facing the door
0.57, facing the matching key 0.56, so the key is fetched before walking to
the door. On 500 new layouts:

| Plan (from the found rules only) | Layouts |
|---|---|
| walk to key → pickup → walk to door → toggle → walk to goal | 465 |
| pickup → walk to door → toggle → walk to goal (started facing the key) | 35 |

Success 500/500; distractor never picked; 15.5 steps on average against
15.4 for the exact shortest solution (ratio 1.006); random play reaches the
goal within the same number of steps with chance below 0.0001.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Walk to goal: door open ≥ 0.9; door and key: no conditions | 0.996; none, none | pass |
| 2. Chain key → door → goal, no place named, no distractor | as above, in every layout | pass |
| 3. Acting ≥ 95% of 500 new layouts | 100%, 1.006 × shortest | pass |

Prediction check: all three passed as predicted; the second rule for the
far side did not form (below the 5% floor), rather than forming in raw
positions.
## 8. Decision

**Keep.** Treating walking as one more action with conditions gave the
whole chain, and a near-shortest plan, without naming any place: rooms and
doorways were not needed. What made it work was supplied: the variables
(including "matches the door"), an exact competent walker, and exact
reachability labels. The theory is now consistent end to end in the
smallest world; the next test is whether it can be learned. Candidates for
the next card: (1) learn the achievement and walk-reach probabilities from
frames and compare with these exact values on new layouts; (2) a harder
exact world first (a key behind another door, as in chained rooms) to test
chains of walk conditions. The user chooses.
