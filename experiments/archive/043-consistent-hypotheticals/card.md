---
id: "043"
title: conditions checked in situations the learned effects produce
rung: 0
serves: [P12, P2, P6]
status: done
verdict: pass
arch_version: 5
date: 2026-10-01
---

# 043: conditions checked in situations the learned effects produce

## 1. Question

Card 042's planner, with codes in recall, failed only the either world:
87.8% and 90.6% of layouts in two of three seeds. The cause is in how the
planner checks a condition, not in recall.

To find what it must hold, the planner splices parts of different
situations. It takes the view of a stored try where the door opened, swaps
in another held thing, and asks recall about the result. That assumes the
parts are independent, and here they are not: "holding the green key while
the green key also lies in view" cannot occur. Recall has no tries for it,
its vector level guesses 55% "opens", and the agent fetches the wrong key
until its budget runs out.

Principle: a condition is checked only in a situation the world model can
produce from the present, as in backward chaining through learned effects
(regression planning). Does checking conditions that way pass card 042's
criteria? Serves:
- P12, subgoals from conditions;
- P2, belief about a changing world: a thing is in one place;
- P6, learned effects building the situations that are checked.

## 2. What changes

One component: how the planner checks a part need (card 038's
`conditions`).

```
card 042:  need "the hand must change": met when doing a works with the hand as in the checked
           situation and the view fixed to a stored try's view v (spliced)
card 043:  the same need: met when doing a works in the checked situation itself, its own hand and
           view; that situation comes from imagining an achiever (pick up the green key) with its
           learned effect on the present, which changes the hand and the view together
```

- **What memory still does.** Stored successes still say which part has
  to change and order the alternatives. They no longer supply the
  situation that is checked.
- **No special rule.** Nothing about held things or views is written in.
  Pick up's learned effect moves the key from the floor to the hand.
- **Declared exception** (CHARTER rule 8). A success that needs both parts
  changed, such as holding the matching key with the switch on in the both
  world, keeps card 038's spliced check. Checking a conjunction in produced
  situations means imagining both achievers in turn, which a later card
  does. The both world passed in every seed of card 042.
- **Unchanged:**
  - card 042's recall (two levels, codes for the same thing);
  - walking;
  - arm A's encoders.
- **Replaced:** an earlier draft of this card removed a held thing from
  spliced views. The user judged it a patch for this environment, not a
  principle (2026-10-01).

## 3. Dependencies

- Card 042's recall: effects 1.0 for every action in every seed and world,
  and key, switch and both at card 029's steps in 3 of 3 seeds.
- Card 039's planner. Card 038's upper bound passed: 100% in the four
  familiar worlds on one-hot vectors.
- Backward chaining with preconditions checked in the states actions
  produce: regression planning (STRIPS; Fikes and Nilsson 1971, not in
  papi).

## 4. Data check

On seeds 399 and 401, each familiar key has 14,268–15,444 weighted
pick-up tries that moved it into the hand, in every world. So pick up's
imagined effect on a key is learned from many tries.

## 5. Feasibility gate

- **Upper bound:** card 038's arm 1 (one-hot vectors), 100% in all four
  worlds.
- **Trivial baseline:** card 042 itself. Either world: 100%, 87.8% and
  90.6% in seeds 400–402; the other worlds 100%.
- **Mechanism** (seed 401, either world, the first 60 layouts played one
  by one): card 042 failed 8; this change fails none. Seed 401 was used to
  find the cause, but nothing here is fitted to it.
- **Shakedown** (spare seed 399, 30 layouts per world,
  `runs/043_shakedown_399.json`): unchanged from card 042.
  - Familiar worlds: 30/30 in all four, effects 1.0, steps 0.94–0.98 of
    card 029's.
  - New colours: key world 30/30 for both; switch world with a new-colour
    door 1/30 and 5/30.

## 6. Success criteria and prediction

Card 042's criteria, unchanged. Arm A, seeds 400–404, 500 layouts per
familiar world.

1. **Familiar worlds.** Card 038's criterion 1 in all four worlds, in at
   least 4 of 5 seeds:
   - held-out effects ≥ 0.999 for every action;
   - goal in ≥ 99% of layouts;
   - mean steps within 5% of card 029's.

   Card 042 met it in 1 of 3 seeds; card 038 arm A in 0 of 3.
2. **Nothing lost on new colours.** Key world with yellow and with purple:
   goal in ≥ 99% of layouts, in at least 4 of 5 seeds. Reported without a
   bar: first-sight effects, and the switch world with a new-colour door.

**Prediction.**
- Both criteria pass in 5 of 5 seeds.
- The either world reaches ≥ 99.5% in every seed.
- The other worlds are unchanged from card 042, at card 029's steps.

**Budget.** About 20 minutes for five seeds. Approved and started by the
user's request (2026-10-01):

```
bin/prun python tools/card043/consistent.py --arm A --seeds 400-404 --layouts-b 100 --out runs/043_armA.json
```

## 7. Result

Main run, arm A, seeds 400–404 (`runs/043_armA.json`, 17 minutes).

| Seed | key | switch | either | both | Criterion 1 | Key world, yellow / purple |
|---|---|---|---|---|---|---|
| 400 | 100% | 100% | 100% | 100% | pass | 100% / 100% |
| 401 | 100% | 100% | 100% | 100% | pass | 100% / 100% |
| 402 | 100% | 100% | 100% | 100% | pass | 100% / 100% |
| 403 | 100% | 100% | 100% | 100% | pass | 100% / 100% |
| 404 | 100% | 100% | 100% | 100% | pass | 100% / 100% |

- **Criterion 1: pass, 5 of 5 seeds** (card 042: 1 of 3; card 038 arm A:
  0 of 3).
  - Held-out effects were 1.0 for every action in every seed and world.
  - Mean steps were 16.38, 17.27, 15.42 and 21.77 in every seed, card
    029's to within 0.3%.
  - Planning took 0.26–0.92 s per layout.
- **Criterion 2: pass, 5 of 5.**
- **Reported:**
  - first-sight effects of the new colours: right in seeds 402 and 404,
    wrong in 400, 401 and 403. The pass/fail summary in the run file
    labels this card 038's criterion 2: 2 of 5;
  - the switch world with a new-colour door: 8–74% by seed and colour,
    against 6–7% in card 038.
- **Prediction met:** either world ≥ 99.5% in every seed (100%); the
  other worlds unchanged from card 042.

## 8. Decision

**Keep** (the user, 2026-10-01). Codes in recall (card
042) together with conditions checked in produced situations reach the
counted planner's results (card 029) in all four familiar worlds, on
encoder vectors, in every seed. Open, in order:
- it is now the architecture of record (version 6);
- card 041 (movement through conditions) on it;
- the declared conjunction exception;
- per-rule codebooks and relations when colour transfer resumes.
