---
id: "072"
title: try the likeliest untried way, and learn the relation from each try
rung: 6
serves: [P19, P11, P12, P3, C5]
status: approved
verdict:
arch_version: 15
date: 2026-10-05
---

# 072: try the likeliest untried way, and learn the relation from each try

Asked for by the user on 2026-10-05: "It's fine if our model thinks that
blue key on red door is the right answer initially, but when it gets it
wrong it should be able to reevaluate its beliefs and try other likely
options before resorting to random explorations", and "as it tries new
things it updates the relational weights from the projections regarding
success or failures".

## 1. Question

In card 070's decoy folds recall gives every key about the same small
chance of opening the door whose key it never saw open it (2.8–3.3% with
card 071's fit; ~10⁻⁵⁴ with card 070's), so the planner, which takes a
way only when recall predicts it works, finds no plan and acts at random
(7% success). If, when no likely way exists, the agent tries the
likeliest untried way, stores the outcome, and refits the relation's
weights from it, does it open the door within few tries, and do the
weights move toward the relation? P19 (few tries), P11 (memory and
weights updated, not erased), P12 (a door stays a condition), P3, C5
(CHARTER: transfer is judged by tries).

## 2. What changes

One component: what the agent does with ways recall thinks unlikely, and
what it learns from trying them. Compared with card 070's configuration
(version 15 with card 070's encoder and `rel:P`, recall fitted per try as
version 15 is; tier 1 100%, decoy folds 7%).

| | Card 070's configuration | This card |
|---|---|---|
| A way whose needed effect recall predicts below 1/2 (a key for a door) | absent from the condition search | **a try**: when the search finds no plan, the ways to the blocked condition are ranked by recall's probability of the needed effect divided by the steps to carry them out, and the best untried one is planned and carried out |
| Order of fallbacks | plan, explore the unseen, random actions | plan, **try the likeliest way**, explore the unseen, random actions; exploration first when no way at all is known (the needed things not yet seen) |
| After a try | memory stores it (card 050's own tries first: a failed key and door predict failure from then on) | the same |
| The relation's online refit (card 069) | after a pick up or toggle recall predicted wrong; P alone | **after every try made on purpose, success or failure, and every prediction recall got wrong**; P and the weight on `rel:P` together, on the admission's likelihood over all of memory (50 Adam steps) |
| Each episode | starts from memory and P as stored (World.reset) | the same |

A try is chosen only among ways recall gives a probability, so a wall
(whose own tries all failed) ranks last; "untried" means no stored try of
that action, front and held combination in this episode.

## 3. Dependencies

Card 070's encoder, `rel:P`, decoy folds (amended: every hue in memory,
less the fold hue's opening tries) and runner; card 069's refit
(`tools/card069/relation.py`); card 047's openable test and situations;
card 045's planner; card 066's fallbacks (`PV.fallback`).

## 4. Data check

None new: card 070's memories and folds (55–88 opening tries removed per
fold).

## 5. Feasibility gate

- **Upper bound:** a known key: card 070's configuration on tier 1,
  100% in 20.8 steps.
- **Trivial baseline:** card 070's folds, 7% (random actions).
- **Smoke test:** fold red, 4 episodes, with a trace of the tries.

## 6. Success criteria and prediction

Card 070's six folds, 100 episodes each; then CHARTER's tiers.

1. **Judged by tries:** success ≥ 99% in every fold, the decoy tried at
   most once per episode on average (two keys: one wrong try at most is
   needed).
2. **No needless tries:** tier 1 ≥ 99%, steps when successful within 5%
   of 20.8 (where a plan exists, no try is made).
3. **CHARTER's tiers:** tier 2 not worse than the best version (0%);
   tier 3 cannot be built.

Reported: tries per episode and what was tried (action, front, held);
random-action share; refits; how P and the weight on `rel:P` moved within
an episode (the relation's gap, same-colour against different-colour
pairs, on MiniGrid's tiles, before and after an episode's refits); tier 2
traces, where trying may also reach the hand's conditions.

**Prediction.** Criterion 1 holds: with recall's ranking flat, the agent
takes one key or the other; a failure is remembered and the other is
taken. The refits move P little (one try among thousands of stored
tries), so the relation's weight rises only slightly within an episode.
Tier 1 unchanged. Tier 2 may move off 0% if trying reaches "drop the ball
first".

**Decision rules.** Keep if 1–3 hold (version 16: version 15, card 070's
encoder, `rel:P` and this). Revise once if 1 fails for a reason in how
tries are ranked or chosen. Stop if tier 1 falls below 99%.

**Budget.** Folds about 12 minutes; tier 1 a minute; tier 2 about 15
minutes.

## 7. Result

`runs/072/` (`run.sh`; card 070's encoder and decoy folds; recall fitted
per try, as version 15). Code: `tools/card072/trying.py`; the refit of
the weight on `rel:P` beside P in `tools/card069/relation.py`.

**Built as declared, with three findings from the smoke test** (fold
red; without them no hypothesis gave a plan, 1 of 4 episodes):
- A hypothesis must carry the effect the search needs from it. For a door
  never seen opened, recall's after-tile is a fresh token 0.24 from the
  open door, and walking recall calls it unwalkable (card 068's finding).
  While a hypothesis lasts, that one tile counts as walkable; recall's
  prediction is unchanged elsewhere.
- The goal is behind the door and unseen, so the plan that uses a try is
  version 15's exploration (open a door next to unseen places), not a
  plan to the goal. The order is: continue the hypothesis; version 15's
  exploration; the candidates, each asked for a plan to the goal or for
  exploration; random actions.
- "Steps" in the ranking is the walk to the plan's first need. The
  planner holds no plan length (card 067: walking knows its next step).
- Also fixed: after a refit, every kind's predictions are now cleared
  (P is shared; before, only the kind that learned was).

| Fold | Success | Decoy tries / episode | Tries made / episode | Steps when successful | Random-action share |
|---|---|---|---|---|---|
| red | 97% | 0.12 | 1.49 | 28.8 | 32% |
| green | 98% | 0.08 | 1.51 | 30.5 | 29% |
| blue | 98% | 0.09 | 1.51 | 31.4 | 31% |
| purple | 96% | 0.08 | 1.55 | 27.4 | 34% |
| yellow | 97% | 0.07 | 1.46 | 32.2 | 31% |
| grey | 96% | 0.07 | 1.47 | 27.1 | 35% |

Card 070's configuration on the same folds: 7% (random actions).

- **Criterion 1: not met as written** (96–98% against 99%); decoy tries
  0.07–0.12 per episode (at most 1: met). In 564 of 600 episodes the
  first try made was the right key, and it opened the door.
- **Every failure is a layout where the key is not the problem.** A
  control (`runs/072/control_red_known.json`, report only): fold red's
  100 episodes with every opening in memory and no trying, which is card
  070's configuration knowing the key: **94%**, 21.4 steps. Its six
  failed seeds contain every failed seed of all six folds. The trace
  (seed 1001029): the decoy key lies on the tile in front of the door;
  holding the right key, the plan drops it to pick up the decoy; with the
  hand empty, it picks the right key up again; for 640 steps. This is the
  hand's conditions (STATUS; tier 2's cause), not trying. Fold red against
  the control: 3 episodes solved only with trying, 0 only without
  (McNemar p = 0.25, no different). The random-action share is these
  episodes' steps.
- **Criterion 2: met.** Tier 1 100% (200), 20.8 steps, no random action:
  no try is made where a plan exists.
- **Criterion 3: met.** Tier 2 0% (100) against 0%: no different
  (McNemar p = 1). Tier 3 cannot be built.
- **The refits after each try do not learn the relation.** The relation's
  gap on MiniGrid's tiles (smallest different-colour relation less the
  largest same-colour one) is 1.14 at setup. After an episode's refits it
  averages 0.30–2.05 by fold, and at worst −3.0 to −4.6 (a different
  colour closer than the door's own key). The weight on `rel:P` ends near
  where it began (fold red 2.97 to 2.33; the others within 0.15). Each
  refit restarts Adam, whose first steps move every entry of P by about
  the learning rate whatever the gradient's size, so one try moves P as
  far as a thousand would (P moved 29–35 in L1 per episode). Within an
  episode this did no harm: after the door opens nothing else needs the
  relation. What found the right key was recall's ranking (the right key
  0.014, the decoy lower) and memory's record of each try.
- **The feasibility gate was wrong.** The upper bound (100%) was tier 1's,
  a world with one key. Measured in the decoy world itself, it is 94%, so
  criterion 1's 99% could not be met by any version (the same holds for
  cards 070 and 071's criteria on these folds).

## 8. Decision

Not taken: the declared rules do not cover this outcome (criterion 1 fails
for neither declared reason). For the user: amend criterion 1 to the
measured bound (no fold worse than the known-key control; met) and keep
(version 16: version 15 with card 070's encoder, `rel:P` and trying; the
refits after a try kept, or dropped as noise); or revise behind a card on
the hand's conditions.
