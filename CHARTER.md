# Charter

Changes to this file need the user's explicit agreement.

## Objective

See [GOAL.md](GOAL.md): the constraints (C1–C6) and properties (P1–P21) the
finished system must satisfy. This file governs how we move toward them.

## Current world

**Chained rooms** (decided 2026-09-24), ported from the old repo's Gamma
trajectory: a MiniGrid world of 3–4 connected rooms with keys, locked doors
(opened by the same-coloured key), switch doors (opened by a grey switch that
may be out of view), timed doors, boxes with hidden contents, and lava.

- **The learner receives:** an egocentric RGB frame (7×7 tiles at 6 pixels,
  42×42×3) and its own previous action, one of 5 opaque IDs (turn left, turn
  right, forward, pickup, toggle). No drop action.
- **The learner never receives:** simulator state, object positions or
  identities, event labels, rewards as supervision unless a card declares
  it, or the evaluator's probe labels.
- **Splits:** train and development use red/green/blue; `transfer_combo`
  holds out colour/door-row combinations; `transfer_colour` uses purple and
  yellow, never seen in training.
- **Declared curriculum:** half of collected episodes are "play starts" (any
  room, some doors open, half holding a locked door's key). Screens report
  ordinary and play starts separately.

The next world (Crafter) is entered only when the ladder rungs that chained
rooms can test have passed.

## Capability ladder

Each rung is one capability with one test. A rung is interpreted only after
every rung below it has passed.

| Rung | Property (GOAL.md) | Capability | Test that shows it | Status |
|---|---|---|---|---|
| 1 | P12, P6, P19 | Tell how close a goal is from any state, and how each action changes that, including through rare interactions | Goal-conditioned fork: at a development probe step, the learner is shown a goal condition (holding a key of a given colour, a given door open, a given box open) as a set of example frames from other episodes in which it holds. For each of the 5 actions, the evaluator computes the true fewest steps to the goal after that action by searching a copy of the simulator (unreachable counts as infinite; the learner never sees these numbers). The learner must rank the actions by it. Scored on interaction-decisive forks (the best action is a pickup or toggle) and movement-decisive forks; a pickup or toggle that changes nothing must not be ranked best. Two pass criteria, each at a threshold fixed by the card's feasibility gate: interaction-decisive top-1 above a goal-swapped control (the same model shown another goal), and the rank correlation between predicted and true steps-to-goal over fork states. Also report the score against the number of distinct interaction events in training (10, 30, 100, 300) | open |
| 2 | P12, P16 | Infer the conditions a goal needs | Subgoal choice: in a state where a condition of the goal is unmet, the learner ranks candidate conditions, each shown as example frames like a goal, by how much achieving them brings the goal closer; the evaluator's search gives the truth. Chains up to box → key → door. Scored above a goal-swapped control. The candidates are a test device; the learner never trains on them | sketch |
| 3 | P12 | Act to reach an unrewarded subgoal that enables a goal | Defined when rung 2 passes | sketch |
| 4 | P21 | Deliberate over chains of conditions | Defined when rung 3 passes | sketch |
| 5 | P2 | Remember an unseen event | Toggle at a switch door whose switch was pressed out of view: history model above a trained current-frame model by ≥ 0.1, n ≥ 30 | sketch |
| 6 | P3 | Apply a relation to new participants | Locked door with matching key, on `transfer_colour` (colours never seen) | sketch |
| 7 | P8 | Compose known consequences into longer chains | Held-out key → door → switch sequences | sketch |
| 8 | P9 | Learn a reusable skill | Defined when rung 7 passes | sketch |

"Sketch" rungs get their exact test and thresholds in a card when they are
reached, not before.

The order follows GOAL.md's main insight (decided 2026-09-26): how close a
goal is, then which conditions it needs, then acting on those conditions as
subgoals, then deliberating over them.

**Architecture selection** (decided 2026-09-25). Before rung 1, while there
is no model (arch_version 0), the first architecture is chosen by one
bounded screen: at most five whole candidate architectures, one design card,
the same data and the same rung-1 feasibility screen for all, runs under 10
minutes each. The winner becomes arch_version 1 and then takes rung 1
properly; the losers and their scores stay in the card so the choice can be
revisited with evidence. The screen may have one follow-up round of at most
three combinations of the first round's parts, each justified by first-round
diagnostics. This is the only exception to "one component per experiment",
and it applies once.

## Rules

Agreed with the user 2026-09-24.

0. **Every proposal names its property.** A card, a component change or a
   change of direction states which GOAL.md property it serves and why the
   current approach cannot reach it.
1. **Switching direction.** Allowed only when a rung fails after its
   feasibility gate passed (an upper-bound fit showed the test is passable and
   the learner still failed), with a one-page post-mortem card and the user's
   sign-off. The session after a switch is thinking only: no code, no runs.
2. **Replacing a component.** Allowed only when an ablation or a component
   test places the failure in that component. Only that component changes.
3. **Components before integration** (decided 2026-09-24). Each component
   gets its own small test before it is combined with others: in isolation
   where that is informative, otherwise attached to parts that have already
   passed (rule 7).
4. **Documents.** Only those listed in AGENTS.md. Cards have a length cap.
5. **Runs.** No long run without a card the user has read and approved.
6. **Success criteria declared first.** Before a run, the card states the
   metrics, the threshold for each, what it is compared against, and why
   those numbers would show the capability. They are not changed after the
   run starts; a second change of criteria means a new card.
7. **Build from primitives; no assumed dependencies.** A component test is
   valid only if everything it relies on is either (a) already built here and
   passed its own test, or (b) an established method with evidence in a
   setting like ours, cited in LITERATURE.md. If a dependency is neither (for
   example object-bound latents with no standard method for this
   architecture), that dependency becomes the next card first; the test that
   needs it waits.
   If a component's behaviour only means something inside the larger system,
   test it attached to the parts that have already passed, not in isolation
   and not with a stand-in. Oracle or label-driven stand-ins may show a test
   is passable (a feasibility gate) but never show the component works.
