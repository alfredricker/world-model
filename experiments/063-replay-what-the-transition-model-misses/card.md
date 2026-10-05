---
id: "063"
title: replay what the transition model misses
rung: 1
serves: [P19, P6, P3, C1]
status: approved
verdict:
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

## 8. Decision
