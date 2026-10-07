---
id: "081"
title: the relation readout as System 1 for toggle, taking over where it agrees with memory
rung: 6
serves: [P3, P10, P21, P6]
status: abandoned
verdict:
arch_version: 18
date: 2026-10-06
---

# 081: the relation readout as System 1 for toggle

Drafted at the user's request (2026-10-06), after card 080. **Abandoned
before running** (the user, 2026-10-06): its agreement check groups
tries by a combination of two hand-picked roles, a grouping designed for
the colour test. CHARTER's "weights take over from recall" rule, which it
applied, was replaced the same day by "recall and networks, each where it
predicts better". The next card is a network that reasons over recalled
tries.

## 1. Question

In card 080, a readout linear in card 070's relation, one line per
combination of the two tiles' roles, predicted a key never seen opening
its door in 6 of 6 colour folds at two of three seeds. At one of those
seeds it made no other error. It failed in two ways:

- **It was unstable.** Seed 81 opened no matching pair anywhere, and
  seed 79 confused the blue door with the purple key.
- **It was overruled.** As the vote's prior, it was outweighed wherever
  the vote was confidently wrong.

Can both be fixed with the following, so that the left-out pair opens in
every fold with no other error?

- **A stable fit.** The readout is a multinomial logistic regression,
  which is convex. Different seeds giving different answers means the
  fit had not converged: on data that a line separates exactly, the
  weights grow without bound, and where they stop depends on the path.
  An L2 penalty gives one unique answer, and full-batch L-BFGS reaches
  it.
- **CHARTER's System 1 rule:** "Weights take over from recall only for
  rules where they agree with recall on every case, rare ones included.
  Recall stays the fallback."

P3 (a known relation applied to new participants), P10, P21 (a weight
carrying a rule), P6.

## 2. What changes

One component: how toggle recall answers a query that has no identical
stored try.

| | Version 18 | This card |
|---|---|---|
| Identical stored try | its counts (card 050) | the same |
| Otherwise | recall's weighted neighbours (cards 049–050, 070) | the **readout**, where its rule has passed the agreement check; otherwise version 18's recall |

**The readout.**
- **Inputs:** the front and held tiles' roles under the other actions
  (whether forward moves onto them, whether a pick-up changes them, as
  the agent's own recall predicts; card 077). Then r = ‖P(z_front −
  z_held)‖₁, card 070's relation, and r multiplied by each role.
- **Output:** a softmax over toggle's outcome categories.
- **No tile's own vector enters,** so the readout cannot learn a door's
  identity.
- **Fit:** on the stored toggle tries, weighted by their counts. L2
  penalty λ, chosen from {10⁻⁴, 10⁻³, 10⁻², 10⁻¹} by the
  leave-combination-out likelihood (card 071). Full-batch L-BFGS, to
  convergence.

**The agreement check.**
- **A rule** is one combination of the front's and the held tile's
  roles, for example "front cannot be walked onto or picked up; held
  can be picked up".
- **Taking over:** the readout takes over a rule when, for every stored
  (front, held) combination under it, its prediction with that
  combination left out matches the stored outcome.
- **Rejected rules** stay with version 18's recall.
- **Unseen rules:** a query whose rule has no stored try is novel, and
  goes to recall.

## 3. Dependencies

Card 079's fold setup and 56-cell table (`tools/card079/effects.py`),
card 077's roles, card 070's encoder and P. Literature: MacKay 1992 (the
penalty chosen by held-out evidence); CHARTER's System 1 rule.

## 4. Data check

Per fold: how many rules (role combinations) memory holds, how many
stored combinations fall under each, and which rules pass the check.

## 5. Feasibility gate

Run before the folds:

- **Upper bound:** with nothing removed from memory, all 56 cells right.
- **Stability:** the fit from three starting points (seeds 79, 80, 81)
  gives the same predictions on all 56 cells.

## 6. Success criteria and prediction

Card 079's table, per fold:

1. **Transfer:** the left-out pair predicted to open in 6 of 6 folds.
   (Version 18: 0 of 6.)
2. **No new errors:** every other cell right in every fold.

Also reported:
- which rules take over in each fold, and λ;
- the readout's slope on r for a locked door and a key;
- the readout alone, without the agreement check.

**Prediction.** With the penalty, the fit lands near card 080's seed 80
(no errors). The door-and-key rule passes the check, and the left-out
pair opens in every fold.

**Risks.**
- If any stored door-and-key combination disagrees, for example the blue
  door with the purple key as at seed 79, the whole rule falls back to
  recall. The fold pair then fails as in version 18.
- The wall and the floor share roles with some other tiles, so their
  rules may fail the check. That only sends them back to recall, which
  gets them right.

**Decision rules.**
- **Keep** if 1 and 2 hold. The next card puts this inside the agent's
  toggle recall, on the decoy folds with trying and the three tiers.
- **Revise** if only 2 fails.
- **Stop** if 1 fails.

**Budget.** Seven fold setups of about 70 s each; fits of under a second.
About 10 minutes.
