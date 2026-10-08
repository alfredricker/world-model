---
id: "085.1"
title: recall's weight fits on sampled pairs, with the learning rate falling to zero over the last half
rung: 1
serves: [P17, P18]
status: done
verdict: fail
arch_version: 18
date: 2026-10-07
---

# 085.1: sampled fits, the end made precise

Card 085's revision, approved by the user (2026-10-07: "alright try it
out").

## 1. Question

In card 085, recall's weights fitted on sampled pairs predicted the same
category as the exact fit for at least 99.6% of keys, but scored worse
on the leave-one-out likelihood both fits maximise, in all six fits, by
up to 0.006 nats per key; three of six missed the 2% bound. The
likeliest cause: Adam with a constant learning rate on minibatch
estimates ends at a noisy point near the optimum. If the rate falls to
zero over the last half of the steps, do the sampled weights score
within 2% of the exact ones, so that tier 3 can be built? P17, P18.

## 2. What changes

One thing against card 085: the learning rate of the two sampled fits.

| | Card 085 | This card |
|---|---|---|
| Learning rate (card 039: 0.02 for 1,500 steps; card 042: 0.03 for 2,000) | constant | constant for the first half, then falling linearly to zero |

The sampling, the objectives, the steps and α's grid are card 085's.
`WM_SAMPLE_DECAY=1` with `WM_SAMPLED=1` (`tools/card085/sampled.py`).
The exact path (2,048 keys or fewer) is untouched, so card 085's gate
part A (tiers 1 and 2 identical) still holds and is not rerun.

## 3. Dependencies

Card 085 (its exact arm, `runs/085/gate_tier3cut_exact_eval.npz`, is the
reference); step-size decay for stochastic gradient methods
(Robbins–Monro conditions; the usual practice for minibatch fits).

## 4. Data check

As card 085: tier 3's memory cut to 1.5% (800–839 keys per action).

## 5. Feasibility gate

Card 085's part B, as run there (1.5% cut, threshold 400 keys, 128 × 128
pairs per step): the sampled weights, evaluated exactly, predict the
same category as the exact fit's for at least 99% of keys, and their
leave-one-out log-likelihood per key is within 2% of the exact
weights', in all six fits.

## 6. Success criteria and prediction

Card 085's criteria, unchanged, run only if the gate passes:
1. **Tier 3 builds:** setup within 20 minutes, under 24 GB of RAM and
   16 GB of GPU memory.
2. **Linear:** setup time and peak RAM on 1/8, 1/4, 1/2 and all of tier
   3's memory grow at most linearly (log–log slope at most 1.2).
3. **No change where it worked:** tiers 1 and 2 identical (card 085).

**Prediction.** The gaps shrink to well within 2% for card 039's λ (a
smooth objective). Risk: card 042's pick-up and drop likelihoods are
near zero (−0.0030 and −0.0005), so 2% of them is 0.0001 nats and
remaining sampling noise may still miss it; or the gap is a bias of
the estimates (a ratio of scaled sums inside a logarithm), which decay
does not remove. If so the gate fails and the result says which.

**Decision rules.** Keep if the gate and 1–3 hold (next: the trial on
all three tiers with card 084.3's conditions); revise if the gate and 1
hold and 2 does not; stop if the gate fails.

**Budget.** Gate: one sampled setup at the cut, about 5 minutes.
Criteria 1 and 2: four tier 3 setups one at a time, each under 20
minutes, about 50 minutes; every run under a 24 GB memory cap.

## 7. Result

`runs/085.1/gate_tier3cut_sampled.{json,log}` and `_eval.npz`
(`WM_SAMPLED=1 WM_SAMPLE_DECAY=1`, as card 085's gate), compared with
card 085's exact arm by `runs/085/compare_eval.py`; about 5 minutes.

**Gate: failed.** The gaps grew:

| Action (keys) | Category agrees | Card 039's λ, LOO per key: exact / 085 / this card | Card 042's: exact / 085 / this card |
|---|---|---|---|
| pick up (839) | 99.6% | −0.1165 / −0.1172 / −0.1202 (**3.2%**) | −0.0030 / −0.0039 / −0.0041 (**34%**) |
| drop (834) | 100% | −0.0800 / −0.0813 / −0.0861 (**7.6%**) | −0.0005 / −0.0006 / −0.0007 (**29%**) |
| toggle (800) | 99.9% | −0.0779 / −0.0840 / −0.0856 (**9.8%**) | −0.1204 / −0.1226 / −0.1243 (**3.2%**) |

**Why** (a diagnosis after the gate, card 039's fit; the exact fit's
weights and this card's, each scored exactly and by the sampled
estimate averaged over 400 draws of 128 × 128):

| Action | Exact objective: exact w / sampled w | Sampled estimate: exact w / sampled w |
|---|---|---|
| pick up | −0.1165 / −0.1202 | −0.1328 / −0.1328 |
| drop | −0.0800 / −0.0861 | −0.0870 / −0.0923 |
| toggle | −0.0779 / −0.0856 | −0.1218 / **−0.1199** |

- The estimate is **biased**: it is lower than the objective at any
  weights, by up to 0.044 nats per key (toggle). The neighbours' vote is
  a sum inside a logarithm, and a sum over 128 sampled keys, scaled, is
  noisy, so its logarithm is low on average (Jensen's inequality).
- For toggle the bias **moves the optimum**: the estimate prefers the
  sampled weights, the objective the exact ones. For pick up the
  estimate cannot tell them apart; for drop it prefers the exact
  weights, so there the fit also stopped short.
- A smaller learning rate at the end only fits the biased objective more
  closely, the likeliest reason every gap grew.
- The kernel's weight is spread: the 10 nearest keys carry a median
  10–17% of a key's weight, so no small set of near neighbours could be
  computed exactly and the rest sampled.

## 8. Decision

**Stop** (gate). Sampling the keys a query is compared with biases the
objective; with 512 of 54,119 keys at full tier 3 the bias would be
larger. Sampling only the query keys does not: the objective is a mean
over query keys, so a sample of them gives an unbiased estimate and
gradient. Each query compared with every stored key costs a step
linear in memory (against the distinct view sets, 26,050 at tier 3,
not all pairs), so the whole fit is linear. The revision, for the user:
sample only the query keys (a few per step), compare each with all
keys, keep this card's decaying rate (which an unbiased stochastic fit
needs to converge), and rerun the same gate.

† Correction (card 085.2): the decaying rate is also a cause. Version
18's fit has not converged after its 1,500 steps, and with every key at
every step (no sampling) the decaying rate alone ends 3–6% below it on
card 039's λ. Bias explains toggle's preference for the sampled weights;
the gaps that grew here grew partly because the rate fell to zero.
