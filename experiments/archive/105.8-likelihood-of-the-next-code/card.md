---
id: "105.8"
title: the likelihood of the next code — one vector out, scored as a distribution over perception's codes
rung: 1
serves: [P6, P3, P7]
status: done
verdict: revise
arch_version: 27
date: 2026-10-13
---

# 105.8: the likelihood of the next code

Series [105](../105-one-transition-model/card.md), written in the overnight
session of 2026-10-13 under the user's approval of the series. Offline; no
agent. It follows [105.7](../105.7-plain-loss/card.md) (stop).

## 1. Question

Squared error failed Crafter both ways:
- **Weighted toward change** (105.2), it predicted random changes, such as
  a cow stepping in, as if they were certain.
- **Plain** (105.7), it never learned rare certain ones. Working a tree
  gives wood, and the model predicted no change on all 138 held-out steps.

Suppose the predicted vector is scored by the likelihood of the next code
under a softmax of negative distances to perception's codes. That is a
proper scoring rule: no outcome is favoured, and a confident miss costs
without bound. Does the gate then pass, and do the encoder's tokens meet
105.2's criteria? Rung 1; P6, P3, P7.

## 2. What changes

One component, the loss:
- **Target:** for each token with a target, its next code, the prototype
  nearest its next vector (perception's own name for it; 098.3's codes, or
  the distinct one-hot vectors for the gate).
- **Loss:** −log p(next code), where p(c) ∝ exp(−|v̂ − μ_c|² / τ). Here v̂
  is the model's predicted vector, μ_c the code's prototype, and τ a
  learned temperature (0.1 at the start). Every token with a target weighs
  the same; presence is scored as before.

The output is still one vector per token in the encoder's space, with no
second head. The most likely next code is the nearest one, as before.
Everything else is 105.2's: arm Rb, tokens, places, data, 30,000 steps and
scoring.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| Perception has codes (identity names) | yes (098.3) | yes (colours) | yes, from a pretrained encoder's codebooks | declared (perception) |
| One distribution per token, not joint | yes | yes | yes | declared; joint outcomes come from recall (105) |

## 3. Dependencies

105.2's code and data (`tools/card105.2/run.py --loss code`, outputs in
`runs/105.8`); 105.7's diagnosis. The loss is the standard categorical
log-likelihood with distance logits, as in prototypical networks (Snell,
Swersky and Zemel 2017).

## 4. Data check

105.2's held-out transitions: 236 makes that made a tool, 16,698 move steps
(1,896 blocked) and 196 role tests. Do has 630 scored changes. Of these,
275 are saplings, which Crafter gives with probability 0.1 when grass is
worked, so no model can predict them. This card scores do's changes
without them, a change from 105.2 stated before the run.

## 5. Feasibility gate

- **Upper bound:** the evaluator's kinds as one-hot tokens, with this loss.
  To pass, it meets criteria 1–3 below itself.
- **Trivial baseline:** "nothing changes", as in 105.7.

## 6. Success criteria and prediction

On the held-out worlds, for arm Rb with seeds 1021 and 7:
1. **Every action in one model.**
   - **Make:** of the tries that made a tool, the whole scored next state
     is right in at least 90%.
   - **Do and place:** the whole scored next state is right in at least as
     many steps as under "nothing changes". At least 80% of the changed
     tokens are right; for do, saplings are left out.
2. **Moves in the same model.** The agent's next world place is right on at
   least 98% of move steps, blocked ones included.
3. **A table at new places.** The role test passes in at least 95% of its
   tests.

**Prediction:** the gate passes 1 and 2, and criterion 3 stays close
(93–94%). With tokens, makes and moves pass. The role test is the most
likely failure, as in 102.1.

**Decision rules:**
- **Keep** (this loss becomes the series' loss) if the gate passes and the
  tokens pass 1–3 in both seeds.
- **Revise** if one criterion fails by less than its margin.
- **Stop** if the gate fails 1 or 2 by more than its margin.

**Budget:** three runs of about 5 minutes, one after another, capped.

## 7. Result

Run 2026-10-13, 30,000 steps per run ([results.json](results.json)). The
learned temperature ended at 0.10 (gate) and 0.15 (tokens).

**Read-out for criterion 2, set after the gate's first scoring.** The
likelihood depends only on how distances to the codes compare, not on their
size. So "the tile nearest an agent code" stopped meaning anything across
tiles (17%). The agent's next place is read instead as the tile where an
agent code is most probable, the model's own distribution. Both read-outs
are in results.json.

| Measure | Gate | Gate, evaluator's origins | Tokens, 1021 | Tokens, 7 | Bar |
|---|---|---|---|---|---|
| Makes that made a tool: whole scored state right | 95.8% | 97.0% | 97.0% | 95.3% | ≥ 90% |
| Do: whole scored state right ("nothing changes" 88.4%) | 89.0% | 88.3% | 85.8% | 85.6% | ≥ 88.4% |
| Do: changed tokens right, saplings left out | 91.7% | 95.1% | 82.7% | 86.7% | ≥ 80% |
| Place: whole scored state right ("nothing changes" 92.2%) | 96.2% | 96.1% | 93.3% | 93.0% | ≥ 92.2% |
| Place: changed tokens right | 98.0% | 98.2% | 92.9% | 93.4% | ≥ 80% |
| The agent's next place: moved / blocked | 99.8% / 78.4% | 99.99% / 100% | 99.9% / 76.9% | 99.8% / 79.4% | |
| The agent's next place: all moves | 97.3% | **99.99%** | 97.25% | 97.5% | ≥ 98% |
| Role test (196 tests) | 87.8% | 93.9% | 94.4% | **96.4%** | ≥ 95% |

- **What the loss fixed.** Rare deterministic changes are learned: working
  a tree, 82.7–86.7% of do's changes are right with tokens, against 0%
  under 105.7's plain loss. Random ones are no longer forecast: no-op
  steps are 94–96% right, against 45% under 105.2's weighted loss. Makes
  pass in every run.
- **Moves fail, and the cause is perception.** With the evaluator's view
  motion, the model gets the agent's next place right on 100% of blocked
  moves. With 103.4's learned placing it gets 77–79%. That placing is
  right on 99.6% of moves that succeed but only 76% of blocked ones in the
  test (82% in training). It is worst when a creature blocks the way
  (zombie 47%, arrow 56%, cow 64%). On a blocked move it misplaces, the
  training target shows the agent moving. The model learns exactly what
  perception tells it.
- **Do's whole state fails** with tokens: 85.6–85.8% against 88.4%. The
  changes themselves pass. So the model predicts some change on do steps
  where none happens; this was not diagnosed tonight.
- **The role test** passes in seed 7 (96.4%) and misses in seed 1021
  (94.4%). It is the same 94% ceiling as 102.1's 93.9%, now crossed once.

## 8. Decision

**Revise** (2026-10-13). Keep requires every criterion. The stop rule
(the gate failing 1 or 2 by more than its margin) does not apply. The gate
misses criterion 2 by 0.7 points, and with correct correspondence it
passes at 99.99%. The loss itself did what it was changed for: neither
random nor rare certain changes are mistaken any longer. It becomes the
series' loss for 105.4 and 105.5. Two things remain for the next card:
- **Blocked moves are perception's error,** made where the action leads
  the placing to expect motion. The model already predicts blocking
  exactly when its correspondence is right, so the integrated fix is for
  the placing to take the model's own predicted next place as its
  expectation instead of the action's usual effect.
- **Do's whole state** needs a diagnosis.

*Diagnosed the same night (seed 1021, no new run).*
- **Where:** 70% of the do steps with an unchanged token predicted to
  change are at night, against 15% of all do steps.
- **Which tokens:** ordinary tiles away from the agent's front (grass,
  stone, tree). The model predicts another of perception's night codes
  for them.
- **Not a scoring artifact:** judged by kind instead of by exact code, the
  whole state is right less often (73%, against 85.8%), because night
  codes also read as the wrong kind.
- **The limit is perception's codes at night,** not the model's account of
  "do".
