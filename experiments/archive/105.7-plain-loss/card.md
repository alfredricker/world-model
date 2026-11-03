---
id: "105.7"
title: a plain loss — the expected next vector when the world's changes are random
rung: 1
serves: [P6, P3, P7]
status: done
verdict: stop
arch_version: 27
date: 2026-10-13
---

# 105.7: a plain loss

Series [105](../105-one-transition-model/card.md), written in the overnight
session of 2026-10-13 under the user's approval of the series. Offline; no
agent. It follows [105.2](../105.2-one-model-crafter-places/card.md)
(revise).

## 1. Question

In 105.2's gate, fed the evaluator's kinds, the model got Crafter's changes
right: 97–99.5% of changed tokens at make, do and place. But on half of the
no-op steps it predicted a grass tile turning into a cow. The loss, from
105.1, gives the tokens that change half the total weight, so the rare
chance of a random change outweighs the likely outcome. Suppose every token
weighs the same, so that the model predicts the expected next vector. Does
the gate then pass, and do the encoder's tokens meet 105.2's criteria?
Rung 1; P6, P3, P7.

## 2. What changes

One component, the loss's weights:
- **Before (105.1–105.6):** tokens whose vector changed carry half the
  total weight.
- **Now:** every token with a target carries the same weight. Presence is
  scored as before.

The minimiser of squared error is then the expected next vector given the
state and the action. The most likely outcome is read as the code nearest
to it, and the spread of outcomes stays with recall (105, section 2).
Everything else is 105.2's: arm Rb, tokens, places, data, 30,000 steps and
scoring.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| The expected next vector is a useful point prediction | yes, where one outcome dominates | yes (mostly deterministic) | yes, with recall's spread for the rest | general |
| No weighting of rare changes | yes | yes | yes | general (nothing declared) |

## 3. Dependencies

105.2's code, data and gate (`tools/card105.2/run.py --plain`, outputs in
`runs/105.7`).

## 4. Data check

105.2's: 132,785 training and 69,685 test transitions. The test holds 236
makes that made a tool, 16,698 move steps (1,896 blocked) and 196 role
tests.

## 5. Feasibility gate

- **Upper bound:** the evaluator's kinds as one-hot tokens, plain loss. To
  pass, it meets criteria 1–3 below itself.
- **Trivial baseline:** "nothing changes" (whole scored state right on
  99.8% of no-op, 98.8% of make, 88.4% of do, 92.2% of place and 97.1% of
  move steps).

## 6. Success criteria and prediction

105.2's criteria, on the held-out worlds, for arm Rb with seeds 1021 and 7:
1. **Every action in one model.**
   - **Make:** of the tries that made a tool, the whole scored next state
     is right in at least 90% (102.1: 96.2%).
   - **Do and place:** the whole scored next state is right in at least as
     many steps as under "nothing changes", and at least 80% of the changed
     tokens are right.
2. **Moves in the same model.** The agent's next world place is right on at
   least 98% of move steps, blocked ones included.
3. **A table at new places.** The role test passes in at least 95% of its
   tests (102.1: 93.9%).

**Prediction:**
- The gate passes 1 and 2. Criterion 3 is close: 93.4% under the weighted
  loss.
- With tokens, 1 and 2 pass. Do's changed tokens are the risk: 102.1 got
  6%, because a tree gives wood but stays a tree, so only the carried slot
  changes.

**Decision rules:**
- **Keep** (the plain loss becomes the series' loss) if the gate passes
  and the tokens pass 1–3 in both seeds.
- **Revise** if one criterion fails by less than its margin.
- **Stop** if the gate fails: a point prediction then cannot carry
  Crafter, and the output must become a distribution.

**Budget:** three runs of about 5 minutes, one after another, capped.

## 7. Result

Run 2026-10-13, the gate with 30,000 steps ([results.json](results.json)).

| Measure | Plain loss (gate) | 105.2's weighted loss (gate) | "Nothing changes" | Bar |
|---|---|---|---|---|
| Whole scored state right: no-op | 96.9% | 45% | 99.8% | |
| Whole scored state right: make, do, place | 97.0%, 86.5%, 97.1% | 47%, 7%, 43% | 98.8%, 88.4%, 92.2% | at least "nothing changes" (do, place) |
| Changed tokens right: make, do, place | 84.8%, **0%**, 98.2% | 96.7%, 97.5%, 99.5% | 0% | ≥ 80% |
| Makes that made a tool: whole scored state right | **95.8%** | 57.6% | 0% | ≥ 90% |
| The agent's next place: moved / blocked / all | 99.98% / 80.1% / **97.7%** | 100% / 67.3% / 96.3% | | ≥ 98% |
| Role test | **92.9%** | 93.4% | | ≥ 95% |

- **Gate: fail** on do (changed tokens 0%, whole state below "nothing
  changes"), on moves (97.7%, blocked 80%) and on the role test (92.9%).
- **The plain loss fixed the random changes:** no-op steps went from 45% to
  96.9% right, and makes from 57.6% to 95.8%.
- **It lost the rare deterministic ones.**
  - On all 138 held-out steps where the agent works a tree, it predicts
    the tree stays: distance 0.018 to the tree code, 1.40 to grass. It
    predicts no wood.
  - The weighted loss got 134 of these right.
  - Under squared distance a rare change can cost at most the distance
    between two codes. The steady churn of creatures elsewhere swamps it:
    the loss sat at 0.013–0.016 from step 8,000 on.
- **Some of do cannot be predicted by any point:** 275 of do's 630 changes
  are saplings, which Crafter gives with probability 0.1 when the agent
  works grass.
- The token runs went ahead in the chain; they carry the same loss and are
  not scored here.

## 8. Decision

**Stop** (2026-10-13), squared error as the series' loss, by this card's
rule: the gate failed, so a point prediction under squared error cannot
carry Crafter. Weighted, it predicts random changes as if they were
certain. Plain, it never learns rare certain ones. The output must carry a
distribution over what each token becomes. Card
[105.8](../105.8-likelihood-of-the-next-code/card.md) keeps the output a
single vector in the same space, and scores it by the likelihood of the
next code under a softmax of negative distances to perception's codes.
That is a proper scoring rule, so the predicted distribution is not tilted
toward change or toward staying. And a confident miss on a rare
deterministic change costs without bound.
