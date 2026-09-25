---
id: "000"
title: short name
rung: 0
serves: [P6]    # GOAL.md properties
status: draft   # draft | approved | gated | running | done | abandoned
verdict:        # pass | fail | uninterpretable, when done
arch_version: 0
date: 2026-09-24
---

# 000: title

Keep sections 1–6 under about 120 lines. Anything longer goes in an appendix
at the end, which the reader may skip.

## 1. Question

One sentence, which rung of the ladder it tests, and which GOAL.md property
it moves toward.

## 2. What changes

The difference from ARCHITECTURE.md at this arch_version, with a small
before/after sketch. At most one component changes.

## 3. Dependencies

Every part this experiment relies on, and why it can be trusted: a passed
card here, or an established method with a LITERATURE.md entry
(CHARTER rule 7). Anything else is not allowed; make it its own card first.

## 4. Data check

How often the events this test depends on occur in the training data, and
in the evaluation data.

## 5. Feasibility gate

- **Upper bound:** a cheap fit (oracle sampling, labels, a smaller world)
  showing the test can be passed.
- **Trivial baseline:** what a model without the capability scores.

Result of the gate, before the main run:

## 6. Success criteria and prediction

At most three criteria (CHARTER rule 6). For each: the metric, the
threshold, the comparison, and why meeting it would show the capability.
Then what we expect and why, the budget and expected runtime.

## 7. Result

Appended after the run. One table; a verdict per criterion.

## 8. Decision

Exactly one: keep, revise or stop, with one paragraph of reasoning.
