---
id: "085"
title: recall's weight fits on sampled pairs, so that building memory grows linearly
rung: 1
serves: [P17, P18]
status: done
verdict: fail
arch_version: 18
date: 2026-10-07
---

# 085: recall's fits on sampled pairs

Drafted at the user's request (2026-10-07: "work on making memory scale
more linearly ... before we run a trial on all tiers"). Cards 084–084.4
found general conditions; the trial on all three tiers waits for tier 3
to be buildable.

## 1. Question

Tier 3 (ObstructedMaze-Full) has never run: building its memory fails.
Measured on version 18 (2026-10-07):
- tier 3 holds 54,119 stored keys for one action (tier 2: 1,480) and
  26,050 distinct view sets (tier 2: 561);
- card 039's weight fit makes every pair's per-number differences, a
  698 GiB tensor, and stops there;
- with that fit skipped, card 039's α (an n × n table of distances,
  about 23 GB) exhausted a 24 GB memory cap; card 042's fit and its
  n × n "same codes" table come next.

Each of these three fits compares every pair of stored keys, thousands
of gradient steps over n² pairs. Card 051 made each step's cost flat in
memory, and card 068 fused one kernel; neither touched this. If each
fit compares a sample of pairs at each step, does memory build for tier
3, with cost that grows linearly, while tiers 1 and 2 stay exactly as
they are? P17 (memory, retrieval and planning usable as experience
grows), P18 (tier 3 at all).

## 2. What changes

One component: how recall's weights are fitted (card 039's λ and α,
card 042's λ1, λ2 and β). What they are, and every use of them, stay
the same.

| | Version 18 | This card |
|---|---|---|
| Up to 2,048 keys per action | every pair, full batch | **unchanged** (the same code path) |
| Above 2,048 keys | every pair (n² per step) | each step: 512 query keys, each compared with 512 keys drawn uniformly (card 039's level and card 042's similar-tile level) and with 8 keys drawn from its own code group (card 042's same-tile level); each sum over others scaled by its sampling fraction; set distances computed only for the pairs drawn |
| α (card 039) | grid search over the n × n distances | the same grid, on 4,096 sampled keys, each against 512 others |
| Starting scale (median distance) | over all pairs | over the sampled pairs |

Steps, learning rates and objectives are unchanged. The fit is
stochastic (minibatch estimates of the same leave-one-out likelihood,
as for neighbourhood components analysis, `1604_02354`), seeded.
Installed by `WM_SAMPLED=1` (`tools/card085/sampled.py`; card 069's
runner reads the flag, and card 042's kind calls the sampled fit when it
is installed), so version 18 without the flag is untouched. Measured by
`tools/card085/measure.py`: one tier's setup, its time and peak memory,
on memory cut to its first fraction of stored rows (episodes are stored
in order).

**CHARTER.** Recall, its candidates and its stored tries are unchanged;
only the cost of fitting changes.

## 3. Dependencies

Version 18 (cards 039, 042, 051, 068); card 066's tiers and tier 3's
stored memory (`runs/066/memory_tier3.npz`); stochastic fitting of a
leave-one-out objective (`1604_02354`).

## 4. Data check

Per tier and action: stored keys, distinct view sets, code groups, and
view set sizes (tier 3, first action: 54,119 keys, 26,050 sets, 237
groups, 12 tiles per set on average, 19 at most).

## 5. Feasibility gate

- **Exactness below 2,048 keys:** with `WM_SAMPLED=1`, tiers 1 and 2
  give every fitted weight, α and β identical to version 18's.
- **Sampled against exact, where both fit:** on tier 3's memory cut so
  that the largest action has 3,000–5,000 keys (where an exact fit
  still fits on the GPU), both fits' weights are evaluated exactly over
  every pair of keys (`exact_eval`). The sampled weights' predicted
  category of every stored key (the action changes nothing, or which
  part it changes) agrees with the exact weights' on at least 99%, and
  their leave-one-out log-likelihood per key is within 2% of the exact
  weights'.

## 6. Success criteria and prediction

1. **Tier 3 builds:** setup completes within 20 minutes, under 24 GB of
   RAM and 16 GB of GPU memory (version 18: fails at 698 GiB).
2. **Linear:** setup time and peak RAM on tier 3's memory cut to 1/8,
   1/4, 1/2 and all of its episodes grow at most linearly (log–log slope
   at most 1.2).
3. **No change where it worked:** tiers 1 and 2 identical to version 18
   (the gate's exactness; no tier rerun needed).

**Prediction.** The fits become a fixed cost per step, and setup is then
dominated by parts linear in memory (indexing, admission over 237
groups). Risk: another part, not seen yet because setup never got that
far, grows with the square of memory; if so it is reported with its
sizes, and becomes its own card.

**Decision rules.** Keep if 1–3 hold (next: the trial on all three
tiers with card 084.3's general conditions); revise if 1 holds and 2
does not; stop if the gate fails.

**Budget.** Every run is one setup under a 24 GB memory cap (an
uncapped tier 3 setup was killed twice on 2026-10-07, taking the
session with it). Gate:
tiers 1 and 2 with and without the flag, and the cut tier 3 memory both
ways, about 15 minutes. Criterion 1: one tier 3 setup, at most 20
minutes (a step of the sampled fits takes 0.02–0.07 s at tier 3's set
sizes, about 10 minutes for the three actions' fits). Criterion 2: the
1/8, 1/4 and 1/2 cuts, about 30 minutes. About 65 minutes in all, no
single run over 20.

## 7. Result

`runs/085/` (`gate_a.sh`, `gate_b.sh`, `compare_fits.py`,
`compare_eval.py`; `tools/card085/measure.py`).

**Deviation.** The gate's second part was declared at 3,000–5,000 keys
per action (5.5% of tier 3's memory). The exact fit there had not
finished its first action after 10 minutes at full use of the GPU: its
set distances grow with the square of the sets' tiles (about 1,700 sets
of up to 19), about 30 times tier 2's. It was stopped and the gate run
at 1.5% (800–839 keys per action), where the exact fit takes 70 s. To
keep sampling active there, the threshold was lowered to 400 keys and
the samples to 128 × 128 per step (2.4% of all pairs per step; full tier
3 at 512 × 512 sees about 0.01%, so this gate is the easier case).

**Gate, part A (exactness): passed.** With `WM_SAMPLED=1`, tiers 1 and 2
give all 23 fitted arrays (λ, λ1, λ2, α, β of every action) identical to
version 18's (largest difference 0). Setup times are the same (tier 1:
30 s; tier 2: 283 s).

**Gate, part B (sampled against exact): failed on the likelihood.**
Both fits' weights evaluated exactly over every pair of keys:

| Action (keys) | Category agrees (≥ 99%) | Card 039's λ: LOO per key, exact / sampled (within 2%) | Card 042's: LOO per key, exact / sampled (within 2%) | α, β |
|---|---|---|---|---|
| pick up (839) | **99.6%** | −0.1165 / −0.1172 (0.6%) | −0.0030 / −0.0039 (**29%**) | the same |
| drop (834) | **100%** | −0.0800 / −0.0813 (1.6%) | −0.0005 / −0.0006 (**13%**) | the same |
| toggle (800) | **99.9%** | −0.0779 / −0.0840 (**7.8%**) | −0.1204 / −0.1226 (1.8%) | the same |

- Every key's predicted category agrees on at least 99.6%, and α and β
  are the same on the grid. The sampled weights score worse on the
  leave-one-out likelihood in all six fits; three of the six miss the
  2% bound. Two of the three are near zero (card 042's level predicts
  pick up and drop almost perfectly), where 2% is 0.0001 nats; the
  largest absolute gap is toggle's λ, 0.006 nats per key.
- The sampled fit is noisier where it is easiest: here each step sees
  2.4% of the pairs, and on full tier 3 it would see 0.01%.
- The cause is not established. The likeliest is where the fit stops:
  Adam with a constant learning rate on minibatch estimates ends at a
  noisy point near the optimum, not on it. The fits' own estimates at
  their end (one sample of 128 keys) differ from the exact values by up
  to 0.06 nats per key (toggle, card 042: −0.186 against −0.123), too
  noisy to tell that from a bias in the estimates.

Criteria 1–3 were not run (the gate comes first).

## 8. Decision

**Stop** (gate, part B). Sampling keeps every prediction of category
(≥ 99.6%) and makes tiers 1 and 2 exactly version 18, but the weights
it ends on are measurably worse on the objective they fit, by up to
0.006 nats per key, in a case easier than tier 3's. The revision this
points to, for the user: the same sampling, with the end of the fit
made precise (the likeliest cause, not established). Either the
learning rate decays over the last steps, or the iterates are averaged
(Polyak–Juditsky), the standard ways a minibatch fit reaches the
optimum and not a noisy point near it. The
gate is then rerun as declared.
