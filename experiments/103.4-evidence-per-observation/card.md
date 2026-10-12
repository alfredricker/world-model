---
id: "103.4"
title: evidence counted per independent observation — the placing likelihood tempered, its temperature chosen by self-consistency
rung: 1
serves: [P7, P2]
status: done
verdict: pass
arch_version: 27
date: 2026-10-12
---

# 103.4: evidence counted per independent observation

Drafted and run on 2026-10-12, in series
[103](../103-tracking-with-general-priors/card.md), after cards 103.1
and 103.3.

## 1. Question

Card 103.1 sums a log-likelihood over pixels as if each were independent.
The 44 pixels of one stepping cow are not independent, yet they count 44
times: about 19 nats each against an action prior of 3–5 nats. So suppose
the sum is divided by τ, the number of pixels per independent observation,
with τ chosen without labels. Is the view's motion then read right on 98% of
Crafter's steps? Rung 1. Serves P7 and P2.

## 2. What changes

One component: the weight of the evidence against the prior in card
103.1's placing (camera layer learned, everything else unchanged).

```
103.1:      score(s) = log p(s|a) + sum_q log(...)
this card:  score(s) = log p(s|a) + (1/tau) sum_q log(...)
            tau chosen from {1, 30, 100, 300, 1000, 3000} on training play, without labels: the value whose
            placements leave the agent's world most self-consistent, i.e. the largest share of confirmed world
            content that, seen again from a new view position, is found as it was written
```

- **Why tempering:** with correlated pixels, an independent-pixel
  likelihood is overconfident. Dividing the log-likelihood by the effective
  number of pixels per observation is the standard correction (a tempered
  or composite likelihood).
- **Why self-consistency:** a wrong placement writes the world where it is
  not. When that content is seen again from elsewhere, it is not found as
  written. The agent can measure this on its own experience.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| Neighbouring pixels are correlated; one τ for the frame | yes | yes | yes | general; τ learned per environment |
| Re-seen world content can be compared | yes | yes | yes (with a 3-D canvas) | general |
| The world moves by one 2-D shift | yes | yes | no | declared, as in 103.1 |

## 3. Dependencies

- Card 103.1's placing and camera pixels (criterion 2 passed).
- Card 098.5's Crafter play: workers 0–1 for training, 8 for testing.
- Tempered and composite likelihoods: a standard correction for dependent
  observations.

## 4. Data check

As card 103.1: 24,204 test steps, 22% moves, 7% at night.

## 5. Feasibility gate

- **Upper bound:** 098.5's declared placing reads 98.1–98.5%.
- **Baselines:**
  - 103.1 with the camera layer: 90.6%;
  - 103.1's robust arm: 94.7%;
  - the prior alone, with the agent's own move rates: about 92%, since
    blocked moves are the only steps it gets wrong.
- **Smoke test (8,000 training steps, 6,418 test steps; chosen by the
  evaluator, so a gate only):**

  | τ | 1 | 10 | 40 | 100 | 300 | 1,000 |
  |---|---|---|---|---|---|---|
  | Right | 86.4% | 87.5% | 90.2% | 93.6% | 99.1% | 98.2% |

## 6. Success criteria and prediction

1. **View's motion, τ chosen without labels.** At least 98% of worker 8's
   steps right, and at least 95% at night.
2. **The label-free choice is the right one.** The τ chosen by
   self-consistency is within one step of the grid of the τ that is most
   accurate on training (evaluator only).

**Prediction:** self-consistency picks 300 or 1,000. Criterion 1 then holds
(smoke test: 99.1% at 300).

**Decision rules:** keep if both hold. Revise if criterion 1 holds only at
a τ the label-free rule did not choose.

## 7. Result

Run 2026-10-12 (`tools/card103.4/tau.py select|test`, capped). Selection on
workers 0–1 (48,325 steps), about 12 minutes per τ with six running at
once. The test on worker 8 took 7 minutes.

**Choosing τ on training play.** Consistency is the share of re-seen
confirmed content found as written. Accuracy is the evaluator's; the
learner did not see it.

| τ | 1 | 30 | 100 | **300** | 1,000 | 3,000 |
|---|---|---|---|---|---|---|
| Consistency | 0.9792 | 0.9810 | 0.9828 | **0.9853** | 0.9821 | 0.9705 |
| Right | 83.1% | 86.5% | 92.6% | **99.4%** | 98.0% | 86.2% |
| Moves / still steps | 82 / 83 | 91 / 85 | 98 / 91 | **99.7 / 99.3** | 99.96 / 97.4 | 74 / 90 |
| Night | 49% | 53% | 61% | **97.1%** | 94.8% | 83% |

**Test, worker 8, τ = 300:**

| | Right | Moves / still steps | Night |
|---|---|---|---|
| 098.5, the agent's tile declared (upper bound) | 98.1–98.5% | | |
| 103.1, camera layer | 90.6% | 89.8% / 90.8% | 72.4% |
| 103.1, robust alone | 94.7% | 77.8% / 99.4% | 94.0% |
| **This card** | **99.27%** | **99.57% / 99.19%** | **96.88%** |

- **Criterion 1 holds:** 99.3% overall, 96.9% at night. It is above the
  declared upper bound, with no tile, no agent and no grid declared.
- **Criterion 2 holds:** self-consistency peaks at τ = 300, which is also
  the most accurate τ on training. It falls on both sides: too little
  tempering lets movers outvote the prior; too much leaves blocked moves
  to the prior alone.
- **What τ = 300 means:** one independent observation per ~300 pixels,
  about 6 of Crafter's tiles. A stepping cow's 44 pixels then weigh about
  3 nats, as much as the action's prior.
- **Breaking world 3** (a view that follows the agent, 30,000 steps): 100%
  of 29,900 test steps at τ = 1 and at τ = 300. The camera's pixels are 49,
  the agent's whole sprite, and none elsewhere.
- **Files on these origins** (card 103.2's tracker, worker 8): IDF1 0.411,
  against 0.303 on 103.1's origins and 0.522 on the evaluator's. The 0.7% of
  misplaced steps still give birth to false files (38,781 detections
  against 25,849).

## 8. Decision

**Keep** (2026-10-12). Crafter's view motion is read on 99.3% of held-out
steps by one general rule:
- the world and the camera as layers, the camera's pixels learned;
- a robust likelihood with an exposure offset;
- evidence counted per independent observation, with the count chosen by
  the agent's own self-consistency.

It replaces the Crafter adapter's declared agent tile (098.5). It is not
yet an architecture version: adoption needs MiniGrid's tiers and the
breaking worlds (CHARTER rule 10), and MiniGrid's view turns, which this
2-D shift does not cover (declared). The next card puts card 103.2's
revision (sprite templates under 098.6's files level) on these origins.
