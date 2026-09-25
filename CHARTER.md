# Charter

Status: **draft**, to be settled with the user. Changes to this file need the
user's explicit agreement.

## Objective

See [GOAL.md](GOAL.md): the constraints (C1–C5) and properties (P1–P19) the
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
| 1 | P6, P19 | Predict what each action causes, including rare interactions | Fork top-1 (the learner's imagined result of each action must pick the real next frame from the other actions' frames and the unchanged frame) on the development probes: pickup and toggle ≥ 0.7, above an action-blind control. Also report the score against the number of distinct pickup/toggle events in training (10, 30, 100, 300); the pass threshold applies at 300, the curve below it is reported and tightened by later cards. Feasibility: a label-driven fine-tune in the old repo reached 0.87 / 0.79 | open |
| 2 | P2 | Remember an unseen event | Toggle at a switch door whose switch was pressed out of view: history model above a trained current-frame model by ≥ 0.1, n ≥ 30 | sketch |
| 3 | P3 | Apply a relation to new participants | Locked door with matching key, on `transfer_colour` (colours never seen) | sketch |
| 4 | P8 | Compose known consequences into longer chains | Held-out key → door → switch sequences | sketch |
| 5 | P9 | Learn a reusable skill | Defined when rung 4 passes | sketch |
| 6 | P12 | Reach an unrewarded subgoal that enables a goal | Defined when rung 5 passes | sketch |

"Sketch" rungs get their exact test and thresholds in a card when they are
reached, not before.

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
