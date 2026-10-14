---
id: "063"
title: replay what the transition model misses
rung: 1
serves: [P19, P6, P3, C1]
status: done
verdict: fail
arch_version: 14
date: 2026-10-05
---

# 063: replay what the transition model misses

Drafted and started overnight on 2026-10-05 (AGENTS.md, "Overnight
sessions"), returning to the night's first objective: card 054's
encoder left recall's colour cases short. The numbers below were fixed
before the main run.

## 1. Question

Card 054's encoder (no codebook; transition model, EMA target,
visibility margin, margin 0.5 between visibly different tiles) keeps the
planner and its conditions, but on three of four seeds it does not
reliably tell that a key of another colour leaves a locked door shut:
the transition model is right in 70–85% of such tries, recall in
62.5–70%. Such tries are rare: a sliver of each update's 64 no-change
toggles. If the transition model replays the tries it gets wrong more
often, does it learn this rare consequence, and does recall follow,
without costing the planner? P19 (important consequences from few
examples), P6, P3 (the key–door relation), C1 (no labels: the priority
is the model's own error).

## 2. What changes

One component: how the encoder's training draws its transition tries.

| | Card 054 | This card |
|---|---|---|
| Per update and action: 128 tries from the last 20,000 | half with a change, half without; uniform within each half | the same halves; within each, half uniform and half in proportion to (the try's last transition error + 0.001)^0.6; a new try enters at the largest priority seen; after each update each drawn try's priority is its error in that update |

Prioritized experience replay (Schaul et al. 2016), without importance
weights: within a half every try has the same kind of outcome, so the
draw changes which situations are seen, not the outcomes' balance.

## 3. Dependencies

Card 054's recipe (`runs/054/b.sh`, m = 0.5) and evaluation
(`runs/054/eval.sh`: recall on the generator's tries with identity up to
noise; the planner in the familiar worlds). Literature: Schaul et al.
2016, "Prioritized experience replay" (not in papi).

## 4. Data check

The generator's stream as card 054 (20,000 updates, about 200,000 steps
of play). The probe holds 40 "locked door, other key" tries and 40
"locked door, matching key" tries per seed.

## 5. Feasibility gate

- **Upper bound:** card 054's seed 399 (recall 100% / 100%).
- **Trivial baseline:** card 054 on each seed (transition model's other
  key 72.5 / 85 / 80 / 70%; recall's 100 / 62.5 / 70 / 62.5%).

## 6. Success criteria and prediction

Seeds 399–402, each criterion in at least 3 of 4:
1. **Transition model:** locked door, other key ≥ 90%.
2. **Recall:** locked door, other key ≥ 90% and matching key ≥ 90%.
3. **Planner:** familiar worlds 100% with card 029's steps
   (`planner_check` criterion 1), on all four seeds.

**Prediction.** Criterion 1 holds; criterion 2 improves but may stay
short on one seed, since recall also depends on noisy copies splitting.

**Decision rules.** Keep (the encoder's recipe gains prioritized replay)
if all three hold; one declared revision if one fails; stop if criterion
1 does not improve on average over card 054.

**Budget.** About 40 minutes (four trainings in parallel, then the
evaluations).

## 7. Result

`runs/063/{b,recall,planner}_{399..402}.json`; card 054's numbers from
`runs/054/{recall,planner}_m05*.json`.

| Seed | 399 | 400 | 401 | 402 | Mean |
|---|---|---|---|---|---|
| 1. Transition model, locked door, other key | 40% (054: 72.5) | 95% (85) | 70% (80) | 47.5% (70) | 63% (77) |
| 2. Recall, other key | 17.5% (100) | 82.5% (62.5) | 65% (70) | 72.5% (62.5) | 59% (73) |
| 2. Recall, matching key | 100% (100) | 100% (100) | 100% (100) | 100% (100) | 100% |
| Transition model, matching key | 100% (100) | 100% (100) | 87.5% (100) | 80% (97.5) | 92% (99) |
| 3. Planner, familiar worlds (× card 029's steps) | 100% (0.93–0.97) | 100% | 100% | 100% | |

Criterion 1 fails (one seed of four at 90%) and got worse on average;
criterion 2 fails; criterion 3 holds, and the planner still admits the
same conditions (key world: the relation or the hand; switch world: the
switch).

Over training, the replay learned the rare case sooner (checkpoints 1–8
above card 054 on three seeds) but did not settle: the other-key
accuracy swung by up to 35 points between checkpoints (seed 399: 75% at
checkpoint 9, 40% at 12). Averaged over the last four checkpoints it is
70% against card 054's 66%, within that noise. What the rows also show,
in both cards alike: colour can be read from the whole vector (90–100%)
but from any single part only 44–81% (chance 11%). Recall compares the
tile ahead and the hand by a distance per part (card 049's relation
conditions), so its "relation" mixes colour with kind. The transition
model sees both vectors whole as well as those distances, so it could
express "same colour", but a small network learns such an equality over
concatenated inputs slowly, and here it did not settle with rare tries
drawn more often.

## 8. Decision

**Stop** (the declared rule: criterion 1 did not improve on average).
Prioritized replay is not the missing piece. The evidence points at how
the key and the door are compared: a distance per part cannot single out
colour while no part carries colour apart from kind. The next encoder
card should change that comparison, for recall and the transition model
alike (a learned comparison of the two vectors in which every number can
meet every number, as the user's relations direction asks), not the
sampling; that is a direction for the user.
