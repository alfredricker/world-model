# Charter

Status: **draft**, to be settled with the user. Changes to this file need the
user's explicit agreement.

## Objective

See [GOAL.md](GOAL.md): the constraints (C1–C5) and properties (P1–P18) the
finished system must satisfy. This file governs how we move toward them.

## Current world

TODO (user): the environment experiments run in now, what the learner
receives (observations, actions), and what it must never be given.

## Capability ladder

Each rung is one capability with one test. A rung is interpreted only after
every rung below it has passed.

| Rung | Property (GOAL.md) | Capability | Test that shows it | Status |
|---|---|---|---|---|
| 1 | TODO | TODO | TODO | open |

## Rules

Proposed; confirm or edit.

0. **Every proposal names its property.** A card, a component change or a
   change of direction states which GOAL.md property it serves and why the
   current approach cannot reach it.
1. **Switching direction.** Allowed only when a rung fails after its
   feasibility gate passed (an upper-bound fit showed the test is passable and
   the learner still failed), with a one-page post-mortem card and the user's
   sign-off. The session after a switch is thinking only: no code, no runs.
2. **Replacing a component.** Allowed only when an ablation or a component
   test places the failure in that component. Only that component changes.
3. **Components before integration.** Each component gets its own small test
   before it is combined with others.
4. **Documents.** Only those listed in AGENTS.md. Cards have a length cap.
5. **Runs.** No long run without a card the user has read and approved.
