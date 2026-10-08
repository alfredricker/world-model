---
id: "085.2"
title: recall's weight fits on sampled query keys, each compared with every stored key
rung: 1
serves: [P17, P18]
status: done
verdict: fail
arch_version: 18
date: 2026-10-08
---

# 085.2: sampled query keys, each compared with every key

Card 085.1's revision, approved by the user (2026-10-08: "085.2 first
then v19 card"). The trial of version 19 on all three tiers waits for
tier 3's memory to build.

## 1. Question

Recall's weight fits (card 039's λ and α, card 042's λ1, λ2 and β)
compare every pair of stored keys at every step, so tier 3's memory
(54,119 keys for one action) cannot be built. Cards 085 and 085.1
sampled both sides of each pair: the query keys and the keys each is
compared with. That biases the objective. The neighbours' vote is a sum
inside a logarithm, and the logarithm of a noisy sum is low on average,
by up to 0.044 nats per key (card 085.1). The objective is a mean over
query keys, so sampling only the query keys, and comparing each with
every stored key, gives an unbiased estimate and gradient, at a cost
per step linear in memory. Does that reach the exact fit's weights at
the gate, and build tier 3's memory in linear time? P17, P18.

## 2. What changes

One thing against card 085.1: what is sampled.

| | Card 085.1 | This card |
|---|---|---|
| Per step | 512 query keys, each against 512 keys drawn uniformly (and 8 from its own code group) | **64 query keys, each against every stored key** (and every key of its code group) |
| Set distances | for the pairs drawn | each query's view set against every distinct stored set, a slice of sets at a time; the chamfer matching found without gradient, then the distance written as the matched differences times the weights, so its gradient is exact (as card 039's `chamfer_a2b`) |
| α's grid | 4,096 sampled keys, each against 512 others | up to 4,096 sampled keys, each against every key (at the gate, every key: exact) |
| Card 042's report (each key's prediction) | against sampled others | every key against every key, in slices |
| Learning rate | constant, then falling to zero over the last half (085.1) | the same; an unbiased stochastic fit needs it to converge |

The objectives, steps, starting scales and grids are unchanged.
`WM_SAMPLE_QUERIES=1` with `WM_SAMPLED=1 WM_SAMPLE_DECAY=1`
(`tools/card085/sampled.py`). At 2,048 keys or fewer the exact fits run,
so tiers 1 and 2 stay identical (card 085's gate part A) and are not
rerun.

**Why 64 queries.** The gradient's noise depends on how many queries a
step averages over, not on the size of memory. The gate uses the same
64 as tier 3 will, so it measures the noise tier 3 will have (card
085's gate saw 2.4% of pairs per step, tier 3 would have seen 0.01%).

## 3. Dependencies

Card 085 (its exact arm `runs/085/gate_tier3cut_exact_eval.npz` is the
reference; `exact_eval`, `measure.py`); card 085.1 (the decaying rate;
the diagnosis of bias); stochastic gradient descent on a mean over
examples, unbiased by construction (Robbins–Monro; `1604_02354` fits
the same kind of leave-one-out objective by minibatches); card 039's
gradient through a minimum at the nearest thing.

## 4. Data check

As card 085: tier 3's memory cut to 1.5% (pick up 839 keys in 422 sets,
drop 834 in 479, toggle 800 in 463). Full tier 3: 54,119 keys and
26,050 distinct sets for the largest action.

## 5. Feasibility gate

Card 085's part B at the same cut (threshold 400 keys, so the sampled
path runs), evaluated exactly against card 085's exact weights:

- every key's predicted category agrees on at least 99%;
- each of the six leave-one-out log-likelihoods per key is within 2%
  of the exact weights', **or within 0.001 nats per key where 2% is
  less than that**.

The absolute floor is new in this card, declared before the run. Card
042's pick up and drop likelihoods are −0.0030 and −0.0005 nats per
key, where 2% is 0.00006 and 0.00001 nats: the average probability of
the stored outcome differs in the fifth decimal. The category agreement
is the measure of behaviour; the likelihood bound is there to catch a
fit that stopped short (card 085.1's toggle λ, 0.008 nats, still fails
it).

## 6. Success criteria and prediction

Card 085's criteria, unchanged, run only if the gate passes:
1. **Tier 3 builds:** setup within 20 minutes, under 24 GB of RAM and
   16 GB of GPU memory.
2. **Linear:** setup time and peak RAM on 1/8, 1/4, 1/2 and all of tier
   3's memory grow at most linearly (log–log slope at most 1.2).
3. **No change where it worked:** tiers 1 and 2 identical (card 085).

**Prediction.** The gate passes: with no bias left, 1,500–2,000 steps
of 64 queries and a decaying rate land within the bound. Risk for
criterion 1: each step compares 64 sets with 26,050, about 40 ms, so
the three actions' fits take about 10 minutes; another part of setup
not reached before may grow with the square of memory. If so it is
reported with its sizes and becomes its own card.

**Decision rules.** Keep if the gate and 1–3 hold (next: version 19,
card 084.3's conditions, on all three tiers); revise if the gate and 1
hold and 2 does not; stop if the gate fails.

**Budget.** Gate: one sampled setup at the cut, about 5 minutes.
Criteria 1 and 2: four tier 3 setups one at a time, each under a
20-minute limit and a 24 GB memory cap, about 50 minutes.

## 7. Result

`runs/085.2/` (`gate.sh`; diagnoses `diag_fullbatch.sh`,
`diag_fullbatch_const.sh`, `diag_q64_const.sh`), each compared with card
085's exact arm by `runs/085/compare_eval.py` (`*_compare.json`). Every
run is the 1.5% cut; about 1 minute per sampled setup, 8 minutes per
full-batch one; peak 2.8 GB RAM, 0.4 GB GPU.

**Check before the gate.** On random sets, the matched differences
times the weights give card 085's chamfer distance to 5 × 10⁻⁵ (values
near 180), and the same gradient to float rounding.

**Gate: failed** (card 039's λ in all three actions).

| Leave-one-out log-likelihood per key (exact weights: pick up / drop / toggle) | Card 039's λ (−0.1165 / −0.0800 / −0.0780) | Card 042's (−0.0030 / −0.0006 / −0.1204) | Category agrees |
|---|---|---|---|
| **Gate:** 64 queries, decaying rate | −0.1215 / −0.0870 / −0.0834 (**4.3%, 8.7%, 7.0%**) | −0.0038 / −0.0006 / −0.1220 (0.0008, 0.0001 nats; 1.3%) | 99.6–100% |
| Diagnosis: every key, decaying rate | −0.1200 / −0.0840 / −0.0824 (**3.0%, 4.9%, 5.7%**) | −0.0032 / −0.0006 / −0.1215 | 100% |
| Diagnosis: every key, constant rate | −0.1165 / −0.0800 / −0.0780 (0.0%) | −0.0030 / −0.0006 / −0.1204 (≤ 1.0%) | 100% |
| Diagnosis: 64 queries, constant rate | −0.1176 / −0.0814 / −0.0791 (0.9%, 1.7%, 1.5%) | −0.0037 / −0.0006 / −0.1210 (0.0006, 0.00001 nats; 0.5%) | 99.8–100% |

- **The new path is exact.** With every key a query at every step and
  version 18's constant rate, it reproduces version 18's fit to four
  decimals in all six fits.
- **The decaying rate is what fails.** With every key at every step
  (no sampling at all), the decaying rate alone ends 3–6% below the
  exact fit. Version 18's fit has not converged after 1,500 steps at a
  constant 0.02; it is still climbing, and a rate that falls to zero
  over the last half travels three quarters as far. This also accounts
  for part of card 085.1's gaps, which that card attributed to bias
  alone.
- **Sampling only the queries is within the bound.** With version 18's
  constant rate, 64 queries per step land within 2% (or the 0.001-nat
  floor) in all six fits, and the predicted category of 99.8–100% of
  keys agrees. Card 042's fits pass the gate in the declared arm too.

## 8. Decision

**Stop** (gate, declared arm). Sampling only the query keys removes the
bias of cards 085 and 085.1, and the new path is exact when nothing is
sampled; the decaying rate, carried over from card 085.1 on the belief
that the fit was near its optimum, stops version 18's unconverged fit
short. The revision, for the user: this card with version 18's constant
rate (one change; the diagnosis above is that configuration and passes
the gate), then criteria 1–3 on tier 3.
