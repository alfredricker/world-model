# Overnight 2026-10-13

The user's objective for the week: align the model with GOAL by integrated
learning. One transition model over vector states, for every action (motion
included) and the world's own changes, with no event types and no supplied
condition forms. The planner's conditions come from what that model learns.
Perception parts (SAM 2.1, later depth) feed it. A general form may cost a
stated loss against version 27's special cases (AGENTS.md margin).

Approved for tonight: card 104 (SAM 2.1 for correspondence) and series 105.
Every run is capped (CHARTER rule 5).

## Summary

- **104 stop.** SAM 2.1 cannot tell look-alike cows apart (oracle IDF1
  0.78 against 0.85).
- **One transition model carries every action.** In world 1 it carries
  walking, blocking, pushing and the lamp at unseen columns, in 3 of 3
  seeds (105.6, keep: a bounded, log-spaced position bias).
- **In Crafter**, with the next code's likelihood as the loss (105.8),
  makes, do and place pass. Moves fall just short (97.6% against 98%)
  because perception misplaces blocked moves. Feeding perception the
  model's prediction helped, but not enough (105.9, stop).
- **Conditions read by intervention** are exact for world 1's pushes and
  moves and close for Crafter's makes (105.4). A planner over them lights
  world 1's lamps as often as the same planner on the true rule: 65.5%,
  where a quarter of worlds are unsolvable (105.5).
- **Ten cards used:** 105 and 105.1–105.9.
- **Your decisions:** listed in STATUS.

## Before the night

- **GOAL:** C1 amended; P12, P21 and "Not assumed" updated.
- **CHARTER's "Current direction"** has four new bullets:
  - one transition model over vector states;
  - a general form may cost some loss;
  - perception may have several parts, prediction has one;
  - conditions come from the effect model, searched from both ends.
- **AGENTS.md:** the margin for general forms.
- **LESSONS:** "no loss in this world" narrows the model into the world's
  shape.
- **ARCHITECTURE:** version 27's undeclared priors recorded (the five
  condition forms, forward's categories, arm C's four result relations, and
  card 087's simulator labels).
- **103.5 stop; series 103 closed.**
- **Cards written:** 105 (plan) and 105.1–105.5.

## 104: SAM 2.1 for correspondence — stop

- **Play collected** at 64 and 16 px in one process: workers 10–12, 30,000
  steps each, 2.5 minutes per worker.
- **Baselines:** 103.4's placing and 103.5's files on the new workers.
- **Tracker:** SAM 2.1's own track step per object, online, with memory
  pruned. Tuned in oracle mode on worker 10's first 3,000 frames:
  - re-prompting lost creatures and ending duplicates helped (0.62 →
    0.72);
  - bicubic upscaling, as SAM's own loader does, helped further (0.78);
  - box prompts, the small model, splitting overlaps by logit and
    SAMURAI's memory selection did not.
- **Gate** (frames 3,000–29,999, held out): oracle IDF1 **0.784**
  against 0.85, so it fails. Speed passes at 37 frames per second, and
  2.2% of detections are on no creature. The upper bound is generous: the
  oracle re-prompted lost creatures under their true identity 2,574 times.
  Prompted once, SAM scores 0.60–0.62, as 103.5 does (0.623 and 0.625 on
  workers 11–12).
- **Why:** night hides creatures, tracks swap to look-alike cows, and
  masks shrink to part of a sprite. The view's motion does not matter.
- **Decision: stop** SAM's propagation as the source of correspondence.
  Keep SAM for outlines on prompt. Telling one cow from another needs a
  prediction of where each will be next, which is the transition model's
  job (series 105). 105.3's input must change.

## 105.1: one transition model in breaking world 1 — stop (arm R)

- **What ran:** the series' model (a transformer over the 81 tile tokens
  and an action token, predicting every token's next vector) in breaking
  world 1. Arm R reads positions only as differences; arm A gets raw
  positions. Two seeds each, plus the gate on the evaluator's kinds.
- **Gate:** pass, just (whole next state right 99.54%, bar 99.5%).
- **R, seed 1021:** pass on all three criteria. Lamps changing at new
  columns 99.7%, blocked moves 100%, whole state 99.95%.
- **R, seed 7:** fail on all three (blocked moves 88%, whole state 97.7%).
  It never fit its own training data: its final loss was 25 times the
  others'.
- **A:** criteria 2 and 3 pass in both seeds. Criterion 1 misses by 0.1
  point in seed 7, on lamps that don't change.
- **Decision:** stop for arm R as written, by the card's rule. One model
  with no part for moves did carry walking, blocking and pushing in three
  of four runs. What failed is R's unbounded position bias. Next: card
  105.6 bounds it and feeds it log-spaced differences, as in Swin V2's
  released code, and runs three seeds.
- **Speed fix:** the first launch ran at 1.5 s a step, because a gather's
  backward pass used atomic adds. Rewritten as a product: 21 ms a step.

## 105.6: a bounded, log-spaced position bias — keep

- **What changed:** only the position term. The network reads
  sign(d)·log2(1 + |d|)/log2(8) and its output is bounded, 16·sigmoid, as
  in Swin V2's released code. Three seeds.
- **Result:** all four criteria pass in all three seeds:
  - lamps changing at new columns: 100%;
  - lamps not changing: 96.7–98.9%;
  - moves and blocked moves: 100%;
  - whole state: 99.89–99.97%;
  - final training loss: 0.00001–0.00002.
- **Speed:** a run takes 2.5 minutes instead of 8–10, because the bias
  network runs once per batch when the position differences are the same
  (an exact shortcut).
- **Decision: keep.** Arm Rb is the series' form.

## 105.2: one model on Crafter, tiles at world places — revise

- **Setup:** the 81 tiles as tokens, the view's tiles at world places from
  103.4's learned view motion (99.3–99.5% right on the curriculum). Every
  awake step is used, moves included; 132,785 training transitions.
- **Gate (evaluator's kinds): fail.** Changes were right (make 97%, do
  98%, place 99.5%), but on half of no-op steps the model predicted grass
  turning into a cow.
- **Cause:** the loss, inherited from 105.1, gives changing tokens half the
  weight. Fine when changes follow from the state (world 1), but in Crafter
  it makes a random cow step look certain.
- **Decision: revise** → 105.7 (the loss alone).

## 105.7: a plain loss — stop

- **What changed:** every token weighs the same, so the model predicts the
  expected next vector.
- **Gate: fail.**
  - Fixed: no-op steps 45% → 97% right; makes that made a tool 58% →
    96%.
  - Broke: rare deterministic changes are never learned. Working a tree
    gives wood, and the model predicted "tree stays" on all 138 held-out
    steps.
  - Also short: moves 97.7% against 98%, and the role test 92.9% against
    95%.
- **Decision: stop** squared error. Weighted, it over-predicts random
  changes; plain, it ignores rare certain ones. Next: 105.8 scores the same
  output vector by the likelihood of the next code (a proper scoring rule).
- **Also found:** 105.2's do criterion was infeasible. 275 of do's 630
  changes are saplings, which Crafter gives with probability 0.1. 105.8
  scores do without them, stated before its run.

## 105.8: the likelihood of the next code — revise

- **What changed:** the same output vector, scored by the log-likelihood
  of the next code under a softmax of negative distances to perception's
  codes (a proper scoring rule, learned temperature). Do's criterion left
  out the random saplings (an infeasible bar otherwise), stated before the
  run.
- **Result (tokens, seeds 1021 / 7):**
  - makes that made a tool: 97.0% / 95.3% (bar 90%);
  - place passes;
  - do's changes are right (82.7% / 86.7%), but do's whole state is 2–3
    points under "nothing changes";
  - role test: 94.4% / 96.4% (bar 95%);
  - moves: 97.25% / 97.5% (bar 98%).
- **Moves are perception's fault:** with the evaluator's view motion the
  same model gets 100% of blocked moves. 103.4's learned placing is right
  on only 76% of blocked moves (zombies 47%), so training targets show the
  agent moving.
- **Decision: revise.** Keep the loss as the series' loss. Next: let the
  placing expect the model's own predicted next place, not the action's
  usual effect, and diagnose do.

## 105.4: conditions by intervention — revise

- **Method:** replace one tile at a time by another kind's vector from the
  same state. A tile is a condition if that changes the model's predicted
  outcome. The truth applies the same replacement to the world's own rule
  (scoring only).
- **World 1 (105.6's models):** exact for pushes, moves and blocked moves
  (precision and recall 1.0; baseline precision 0.09–0.14), and for lamps
  lighting in two seeds of three.
  - Lamps going out fail (precision 0.66, recall 0.40). The model doesn't
    know a second box would keep the lamp lit (rare in training).
  - 493 of the 501 extra conditions come from impossible states (a lit
    lamp with no box beside it).
- **Crafter (105.8's models):** makes are exact in 94–96.5% of events
  (precision 0.69 / 0.93, recall about 1.0).
  - Moves fail: blocked-move precision 0.23–0.31, recall on successful
    moves 0.70–0.76.
  - Traced to 103.4's placing (76% right on blocked moves, worst near
    creatures). The model learned a spurious dependence on distant
    creatures.
- **Decision: revise.** Fix the placing with the model's own prediction,
  intervene only in producible states, and leave rare cases to the
  agent's experiments.

## 105.9: perception expects what the model predicts — stop (10th card)

- **What changed:** 103.4's placing took its prior from 105.8's model (the
  probability that the agent is at each tile next) instead of the action's
  habit (a move action moves the view about 90% of the time).
- **Placing on blocked moves, held out:** 76% → **85.0%** (bar 95%; stop
  line 85%):
  - curriculum test worlds: 86–87% → 93–95%;
  - random play: 63–69% → 74–79%. Half of these blocked moves are at
    night, and zombie blockers are still only 54–57%. The model giving the
    prior was itself trained on the old placing, so it learned zombie
    blocks badly.
- **Successful moves:** unchanged on the curriculum, 0.2–0.3 points worse
  on random play.
- **Retrained on the new places (seeds 1021 / 7):**
  - moves: 97.6% / 97.6% (bar 98%; blocked 83%, was 77–79%);
  - makes: 96.2%;
  - do's changes improved;
  - role test: 91.8% / 89.3%, down from 94.4% / 96.4% (this test varies
    by up to 9 points between runs).
- **105.4's Crafter conditions on these models:** blocked-move precision
  0.40–0.46 (was 0.23–0.31); 82–83% of events exactly right (was 74–76%).
- **Decision: stop** this form, by the rule (85.0% is the stop line).
  - The direction held: every move and condition measure improved, and
    the curriculum reached the bar.
  - One step from a model trained on the old errors is not enough for
    random play's night and zombies.
  - Next, for the user: iterate the loop, or train the model's next-place
    prediction on the frames as well.

## 105.5: planning both ways (world 1 only) — revise

- **How it plans:** statements are learned from training events by 105.4's
  interventions on the model. The search goes backward over them by
  means-ends (depth ≤ 6), walks by closeness propagated over the model's
  predicted moves, and checks forward one imagined effect at a time.
- **The statements it learned** are the world's rules, none written by
  hand: enter from the neighbouring cell; push with the agent behind and
  floor beyond; a lamp lights when a box is pushed beside it.
- **Result, 200 test worlds:** the lamp is lit within 50 steps in 65.5%
  (random play 13%). The upper bound, the same planner on the world's own
  rule, also gets 65.5%. 45 worlds cannot be
  solved at all (breadth-first search on the true world), so 90% was
  infeasible.
  - On solvable worlds: 131 of 155 (84.5%).
  - The misses need lamp-lighting variants training never showed: 6 of 12
    learned.
- **Crafter:** not run.
- **Decision: revise.** Score only solvable worlds, and find unseen
  variants by intervening on imagined states around the goal.

## Diagnosis: 105.8's do shortfall (no new card)

- **Night:** 70% of the do steps with an unchanged token predicted to
  change are at night (15% of all do steps). The tokens are ordinary tiles
  away from the agent, given another night code.
- **Not a scoring artifact:** judged by kind it is worse (73%). The limit
  is perception's night codes, not the model's account of "do". Added to
  105.8's card.

## Tools written tonight

- `tools/card104/track.py`: SAM 2.1 tracker, online.
- `tools/card105/model.py`: the transition model; position bias R / Rb;
  losses weighted, plain, code.
- `tools/card105.1/run.py`: world 1 runner.
- `tools/card105.2/run.py`: Crafter runner. Options: `--loss
  weighted|plain|code`, `--origins learned|model|true`, `--eval NET --tau
  T`.
- `tools/card105.2/origins_curr.py`: placing on the curriculum.
- `tools/card105.4/world1.py`, `tools/card105.4/crafter.py`: conditions by
  intervention.
- `tools/card105.9/place_model.py`: placing with the model's prior.
- `tools/card105.5/world1.py`: the planner in world 1.
