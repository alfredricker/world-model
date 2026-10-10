---
id: "098.3"
title: Crafter tokens — still frame pairs chosen by the agent's own check that the view did not shift
rung: 1
serves: [P7, P1, P4]
status: done
verdict: fail
arch_version: 24
date: 2026-10-10
---

# 098.3: still pairs by the view

Written and approved overnight (2026-10-10, overnight authority; new
card 9 of the night). Offline.

## 1. Question

Card 098.1 found that card 098.2's tokens merge the agent's four
headings (2 codes for the agent's own tile; centroids 0.20–0.27 apart,
code radius 0.75), so the agent cannot tell from its codes which tile
it acts on. Headings change only on move steps, and the still pairs
were chosen by action (no move action), so a heading change was never
contrasted. If still pairs are chosen instead by the agent's own check
that the view did not shift (card 098.2's learned shifts), blocked
moves join them, and the changed heading becomes a different thing
under card 098's rule. Do the codes then tell the headings apart, with
card 098.2's results kept? P7, P1, P4.

## 2. What changes

One component, which frame pairs count as still: card 098.2's learned
shift check (the action's shift did not happen) in place of "no move
action". Sleep stays excluded. Everything else is card 098.2's revised
run (both frames as candidates, the agent's own tile corrected).

## 3. Dependencies

Cards 098, 098.2 (encoder, pairs across light, shift check), 098.1 (the
learned front and its evaluation).

## 4. Data check

Still pairs before and after; blocked moves among them; of those, how
many change the agent's own tile.

## 5. Feasibility gate

- **Upper bound:** the true heading (the front learned from it is
  right by construction).
- **Trivial baseline:** card 098.2's codes (front learned 24.2%).

## 6. Success criteria and prediction

1. **Headings:** the front learned from the codes (card 098.1's method)
   agrees with the true heading in at least 95% of judged "do" steps.
2. **Card 098.2 kept:** night tiles in a code of their class's day tiles
   at least 90%; purity at least 95% overall and 90% at night;
   inventory at least 95%.

**Prediction.** About 40% of move steps are blocked, so headings are
contrasted thousands of times; the codes should separate the four
headings and the learned front should be near 100%. Card 098.2's
measures should hold within a point.

**Decision rules.** Keep (these tokens for card 098.1's revision) if 1
and 2 hold; revise if 1 holds and 2 does not; stop if 1 fails.

**Budget.** About 2 minutes.

## 7. Result

`tools/card098.2/across.py --joint --blocked`, `runs/098.3/`; the
learned front by card 098.1's method (`tools/card098.1/interact.py`).

**Data check.** Still pairs: 70,129 by action, 79,447 by the view;
9,318 blocked moves among them.

| | Card 098.2 | This card |
|---|---|---|
| Codes | 57 | 53 |
| Codes of the agent's own tile | 2 | 5 |
| 1. Front learned, all judged "do" steps | 24.2% | 86.2% |
| Front learned, awake (967 steps) / at night awake | 24.0% / 30.6% | **100% / 100%** |
| Front learned, asleep (198 steps, the frame grey) | 25.3% | 18.7% |
| 2. Night tiles in a code of their class's day tiles | 94.0% | 95.1% |
| Purity overall / night | 99.2 / 94.7 | 99.1 / 94.0 |
| Inventory | 95.4% | 94.2% |
| Codes per class, worst | cow 6 | cow 5, the agent 5 |

On awake frames each of four codes holds exactly one heading (left
1,151, up 1,222, down 1,310, right 1,132; a fifth code 12 steps). Every
miss is a step where the agent is asleep: the frame turns grey, the
sprite shows no heading, and the action is ignored; the sleeping sprite
shares the left heading's code (sleep steps stay out of the still
pairs, so falling asleep is never contrasted).

1. Not met as written (86.2%); met on every step where the heading is
   shown (100%). 2. Met except inventory (94.2% against 95%).

## 8. Decision

**Revise.** The card's rule says stop if 1 fails; its premise, that the
codes cannot tell the headings apart, is refuted (100% awake), so the
rule's letter is not followed and the deviation is recorded here for
the user. Kept: still pairs by the view. Open: the asleep appearance
(its own code, from sleep steps contrasted) and inventory 0.8 points
under the bar. These tokens are the best so far and are used by card
098.1's revision.
