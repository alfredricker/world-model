---
id: "009"
title: learned reachability
rung: 0
serves: [P12, P19, P9, P4]
status: abandoned   # draft | approved | gated | running | done | abandoned
verdict:
arch_version: 0
date: 2026-09-27
---

# 009: learned reachability

## 1. Question

Cards 007–008 define a condition as "the states where this goal's achieving
action works are within walking reach". Two parts of that are designed by
hand: which actions count as walking (left, right, forward), and "within
reach" read as a discounted walk value above 0.1, which confused far with
unreachable and produced a spurious condition in card 008. Can both be
replaced by things the agent learns from its own experience, and does
discovery then find the chain cleanly and stop by itself? Serves P12
(conditions), P19 (an event matters when its effect lasts), P9 (the base
skill is learned, not listed) and P4.

## 2. The idea

**Walking was a stand-in for "moves you can undo".** Turning and stepping
can be taken back; picking up a key (there is no drop) and unlocking the
door cannot. A condition is then something you cannot get to by changes
you can undo: it takes a lasting change. "Within reach" becomes: the
states where the achieving action works can be reached, and left again,
without any lasting change. This is general: in Minecraft, walking and
opening a chest can be undone; crafting a pickaxe and mining ore cannot.
Proximity is still a condition, as the user argued, but it is the part
handled by reaching, not a lasting change.

**What changes in the conditions.** With the door's open/closed toggle
undoable, the goal square's condition becomes "door unlocked" rather than
"door open": the lasting change is the unlock; opening an unlocked door is
reaching. The expected chain is door unlocked → holding the matching key →
empty hands (with the key still on the floor) → stop.

**How the agent learns which changes can be undone, from pixels.** In its
own play, a step from frame f to f′ was undone if frame f appears again
later in the episode (the frames are identical; this is an observation,
not simulator state). A small head U(frame, action) learns to predict that
for new frames and layouts. Steps too near an episode's end to tell are
left out.

**How it learns "reachable at all", separately from "how soon".** A second
walk-like head R is trained by Q-learning over undoable steps (U > 0.5, any
action) toward the ready states, but with no discount: its target is 1 if
the next frame is ready, else the best R of the next frame. A reachable
state's value is then about 1 however far it is, an unreachable one about
0, so the detector R > 0.5 does not depend on distance. The discounted
value stays, for choosing moves and judging how soon.

## 3. What changes

From card 008 arm C (continual, starting from encoder G, `runs/008_encG`):

| | Card 008 | Card 009 |
|---|---|---|
| Steps used to reach | left, right, forward (listed by us) | any step the agent predicts it can undo (U > 0.5) |
| "Within reach" | discounted walk value > 0.1 | undiscounted reach value R > 0.5 |
| Choosing moves when acting | discounted walk value over left, right, forward | discounted value over steps with U > 0.5 |

Unchanged: the level procedure, fine-tuning the encoder on each discovered
goal, thresholds A > 0.5, the stopping rule (fewer than 10 successes), at
most 4 levels, the speed settings, the test layouts. New: the final
network and every level's heads are saved. Still designed by hand and
declared: the 0.5 thresholds, the level recursion, "identical frame seen
again" as the sign of undoing.

## 4. Dependencies

Card 008 arm C (the continual procedure; found the chain but failed on the
threshold) and encoder G (gate passed). Q-learning without discount on a
deterministic world is standard; its known risk is values leaking upward
through the max, which criterion 2 checks.

## 5. Data check and feasibility gate

Card 005's data (10k training episodes, 500 new test layouts). Undo
labels: every step with at least 200 steps left in its episode.

- **Upper bound (exact):** the same procedure with exact undoability (a
  step is undoable if its start state can be reached again from its end
  state) and exact reachability (same group of mutually reachable states
  as some ready state). Must give door unlocked → holding the matching key
  → empty hands → stop. Run first; if not, the card stops.
- **Undo head check:** U against exact undoability on test steps, area
  under the curve ≥ 0.95; moves predicted undoable ≥ 95%; pickups and
  unlocks predicted lasting ≥ 95%. If not, the card stops (the rest would
  rest on a wrong notion of reach).
- **Trivial baseline:** card 008 arm C's result (75.8% acting, spurious
  level 4).

## 6. Success criteria and prediction

On the 500 new layouts:

1. **The chain, stopping by itself:** the first two detectors match door
   unlocked and holding the matching key with area ≥ 0.95, and the
   procedure stops because no action achieves the next goal (not because
   of the 4-level cap).
2. **They behave as conditions, and do not flicker:** from test frames
   with the detector on, reaching (exact, by undoable steps) a ready state
   and taking the achieving action achieves the level's goal ≥ 90%; with
   it off ≤ 10%. In training data, a detector turns on during undoable
   steps in ≤ 1% of its switch-ons (card 008: 1837 such flips at level 3).
3. **Acting:** ≥ 90% of new layouts reach the goal square (card 008 C:
   75.8%).

Prediction: gate and undo check pass (moves are revisited constantly in
random play; keys and locks never). Criteria 1–3 pass. The main risk is R
leaking upward (unreachable states creeping above 0.5 through the max):
criterion 2's "off ≤ 10%" catches it. Runtime about 30 minutes (four
fine-tunes, undo head, reach and walk heads per level).

## 7. Result

## 8. Decision

**Stop** (not run; 2026-09-27, with the user). Undoability is the wrong
definition: many conditions can be undone (a switch, an equipped tool, a
held key once there is a drop action) and some lasting changes matter to
no goal. It passes in this room only because keys cannot be dropped.
Reversibility may still help decide how a condition is kept or scheduled.
Replaced by [card 010](../010-condition-logic/card.md), which tests a
threshold-free definition on worlds that separate these cases. The
undiscounted "reachable at all" value remains a candidate fix for card
008's flicker.
