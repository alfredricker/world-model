---
id: "171"
title: a transition head trained on its own chains, with a pull's end as a range
rung: 6
serves: [P6, P12, P21]
status: done
verdict: fail
arch_version: og5
date: 2026-11-27
---

# 171: a transition head trained on its own chains, with a pull's end as a range

OGBench `visual-scene`, following [card 170](../170-effects-imagined-by-the-scene-model/card.md). Drafted at the user's request, 2026-11-27 ("ok, draft it"). Approved by the user the same day ("approved").

Card 170 let the search take every imagined next state from the scene model (`SceneWNet`, [card 168](../168-consequence-trained-scene-world-model/card.md)), with no written effects and no ordering guards. Chains of 2–5 touches lost nothing against og5 (87% against 83% on 100 problems). Chains that unlock, open, put the cube in, close and re-lock lost 23 points (58% against 81%). Two of the three faults found are the scene model's:

- It was trained on one touch at a time, from a state read off a real frame. In a chain, the second and later touches are predicted from a state it imagined, which it never saw in training.
- It predicts one value for where a pull leaves a slide, and the agent tests "same reading" with one fixed threshold. Where a pull ends varies by more than that threshold.

The third fault (a partial chain that flips between decisions) is the planner's and is not touched here.

## 1. Question

If the transition head is trained on chains of three recorded touches, each later step predicted from the state it imagined itself, and predicts the range a slide's reading ends in rather than one value, does card 170's search come within 10 points of og5 on long chains? Rung 6; Roadmap step 7 (planning forward one effect at a time, side effects included). Serves P6 (consequences of "each available action, or sequence of actions"), P12 and P21.

## 2. What changes

One component: the training of `SceneWNet`'s transition head, and one output of it (`tools/card171/model171.py`, card 168's training with two additions). The encoder's form, the data, the training targets (`discover_slots`), the skills and the own-tries counts are unchanged. The planner is card 170's; the only edit is that it reads the new output where it compares slide readings.

```
168:  frame t --read--> state --touch t--> predicted state   vs   reading of frame t+1
171:  frame t --read--> state --touch t--> imagined state --touch t+1--> imagined state --touch t+2--> ...
                                           each step's prediction vs the reading of the real frame after it
168:  a pull's end = one value, "same" within 0.18
171:  a pull's end = low, middle, high (10th, 50th, 90th percentile); "same" if both readings lie in that range
```

- **Chains.** A training example becomes three consecutive touches of one episode. The first is predicted as now. The state after it is then built exactly as the search builds it. A slot predicted changed takes the predicted value, and its vector is recalled: the mean of the 16 remembered vectors nearest that value, from the training frames re-read by the current encoder every 250 steps. A slot predicted unchanged keeps its vector. The cube takes its predicted position and whether it is in view. The second and third touches are predicted from that imagined state, and each is compared with the reading of the real frame after it. No gradient passes through the imagined state, since recall is a lookup; the head learns to be right from the states it will be handed. Chain terms start at step 1,000 of 3,500, once one-step predictions are usable. Card 168's balancing of batches (locked and unlocked pulls, touches that hide the cube) is applied to the last touch of the chain as well.
- **A range for a pull's end.** For a slide predicted to move, the head gives three values, trained with the quantile loss on the reading after the touch. The middle one is the imagined next value. The range is used in two places, replacing card 170's rule (`tol_imag`). In the search, an imagined pull reaches a goal reading that lies inside its range. After the agent's own pull, the reading counts as the goal's if both lie inside the range predicted before that pull; a reading outside it is a pull that fell short, and the agent plans again. Everywhere else the threshold stands. The range is widened by the threshold at each end.
- **Each step is one touch, not one tick.** The chain runs over events that end in a stable condition, with states recalled from memory. No frame-by-frame latent state is rolled forward (CHARTER, Roadmap note of 2026-11-01).

In the brain's terms: sequences are replayed through the same forward model that predicts single events, an imagined value completes to a stored pattern, and the motor system knows how variable its own movement's outcome is and judges "arrived" against that (Körding and Wolpert 2004). These are analogies, not evidence.

Declared exceptions, unchanged from card 170: the drawer-reading range over remembered put-ins as the put-in's condition; the progress score for partial chains; the thresholds. New declared constants: the chain length (3) and the percentiles (10 and 90).

**Loss accepted before the run** (AGENTS, exception for general forms): up to 10 points on the five-task mean and on each task, as in card 170.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Memory holds consecutive events of one episode | holds | holds | holds | holds | general |
| An imagined state is completed by recalling remembered vectors | holds | holds | holds | holds | general; the recall key is card 168's one value per slot, declared there |
| The end of a touch is a range of one value per slot | a block's state is more than one value | same | holds | holds for joints | declared with card 168's read-out; a later card needs recall by nearest vector |
| Nothing but the arm changes the scene between touches of a chain | mobs, day and night | mobs | holds | holds for a still scene | declared, as in card 170 |

## 3. Dependencies

- [Card 168](../168-consequence-trained-scene-world-model/card.md): the model and its training (kept at 2 of 3). [Card 169](../169-zero-heuristic-event-bfs-on-ogbench/card.md): the agent, skills, practice counts and benchmark protocol (pass).
- [Card 170](../170-effects-imagined-by-the-scene-model/card.md) failed its gate. It is used here as the measuring instrument, not as a passed part: its planner with card 168's weights is the baseline in every comparison.
- Established methods, entered in [LITERATURE.md](../../LITERATURE.md) on approval: training a sequence model on its own predictions (scheduled sampling, Bengio et al. 2015; DAgger, Ross et al. 2011; multi-step latent consistency, TD-MPC, Hansen et al. 2022), and quantile regression (Koenker and Bassett 1978).
- LESSONS: "Query a network only in the form it was trained on" (43% against 67%, card 170) and "a loss must not forecast random changes" (cards 105.2–105.8).

## 4. Data check

Measured 2026-11-27 on the 400 training episodes with the frozen card 168 model (`tools/card171/datacheck.py`; no test episode read; simulator positions read for this table only).

- **Chains are plentiful.** 8,230 frames give 7,830 touches with a frame after them, 7,030 chains of three consecutive touches and 5,030 of eight.
- **Where a pull ends varies, and what is known before the pull does not say where.** The reading runs over about 2.5 units per slide, and the threshold is 0.18.

| Pulls that moved the slide | Count | End reading: shortest tenth / middle / fullest tenth of pulls | Within one threshold of the middle end | True position after (cm open), same order | End explained by start reading and contact point (R²) |
|---|---:|---|---:|---|---:|
| Window, opening | 628 | −0.87 / −1.29 / −1.35 | 69% | 15.4 / 18.7 / 20.0 | 0.10 |
| Window, closing | 604 | 1.11 / 1.17 / 1.18 | 93% | 3.4 / 0.0 / 0.0 | 0.03 |
| Drawer, opening | 472 | −1.00 / −1.24 / −1.26 | 89% | 14.6 / 16.0 / 16.0 | 0.08 |
| Drawer, closing | 521 | 0.68 / 1.16 / 1.25 | 77% | 3.5 / 0.5 / 0.0 | 0.16 |

  Over all four, 82% of ends lie within one threshold of the middle end, which is why card 170's written "goes to the end" rule scored 79.9%. The spread is partly real: readings follow the true position (correlation 0.5 to 0.85).
- **The agent's test is about three times stricter than the benchmark's.** The benchmark accepts a slide within 4 cm of its goal (`scene_env.py`), a quarter of the drawer's travel. The threshold of 0.18 is about 7% of the reading's range, near 1.3 cm. A drawer closed to 3.5 cm passes the benchmark and fails the agent's own test, so the agent goes on pulling. The agent is not told the benchmark's figure; the range has to come from how its own pulls end.
- **The range can also accept too much.** One in ten window-opening pulls ends under 15.4 cm, where the benchmark needs 16. Gate (b) counts how often an accepted slide is outside 4 cm.
- **The compound that tasks 4 and 5 need is rare in recorded chains.** 127 put-ins are read; 38 are followed at once by a drawer touch, and in 12 of those the cube is read as hidden afterwards. In all, 161 touches hide the cube and 100 show it again. Chains ending in these are oversampled; this is the card's weakest point.

## 5. Feasibility gate

Two parts, both before any benchmark task or test episode.

**(a) Recorded chains** (`tools/card171/effects171.py gate <arm>`: fitted on 320 training episodes, scored on the 80 held out). Four fits, so the two additions can be told apart: card 168's training (the baseline), chains only, range only, and both (this card's model).

| Measure | Upper bound | Baseline | Bar for this card's model |
|---|---|---|---|
| 1. One touch, as card 170: presses, table moves, unchanged slots kept | — | card 170: 98.6, 94.2, 99.3% | each within 2 points |
| 2. A pull's read end lies in the predicted range | the 10th–90th percentile range of remembered ends per slide and direction | one value and the threshold (card 170: 69.0%) | at least 85%, with a mean range at most 1.25 times the remembered one |
| 3. Whole state after three imagined touches matches the reading after the third | the third touch predicted from the real reading before it | card 168's training, same scoring | closes at least half the gap from baseline to upper bound |
| 4. Cube in view or not after a drawer touch, the cube having been put in by an earlier imagined touch | the same touch predicted from the real reading | card 168's training, same scoring | at least 80% balanced (one touch, card 170: 80.9%) |

If the gap in measure 3 is under 4 points, compounding is not the fault. The card then stops with that as its finding, since a bar on a gap that small cannot be passed or failed. Measure 4 is judged only if the held-out episodes hold at least 20 such cases with the cube hidden; otherwise it is reported.

**(b) Self-made problems, live.** Developed on problem seeds 101 and 102, then run once on fresh seeds 401 and 402, 100 problems per set (card 170's 30 were too few: 80% on 30, 87% on the next 100). og5 and card 170 run on the same problems.
- **Upper bound:** og5's planner. **Baseline:** card 170 with card 168's weights (58% on locked put-away). *Written before the run as "on the same problems"; they were not the same, see section 7.*
- **Bars:** locked put-away within 10 points of og5 and at least 10 points above card 170; mixed problems at most 5 points under og5.
- **Reported:** of the slides the agent accepted as at the goal, the share truly within 4 cm.

Result of the gate, before the main run: **both parts failed, each narrowly. The benchmark and the 300 test episodes were not touched.**

Part (a), 80 held-out training episodes, one fit per column (`runs/171/gate_a.json`):

| Measure | Card 168's training | Chains only | Range only | Both (this card's model) | Bar | Verdict |
|---|---:|---:|---:|---:|---|---|
| 1. Presses / table moves / unchanged kept | 98.6 / 94.2 / 99.3 | 98.4 / 92.5 / 99.2 | 97.9 / 91.4 / 99.3 | 98.2 / 92.4 / 99.2 | within 2 points of the first column | pass (table moves by 0.2) |
| 2. A pull's read end in the predicted range (472 pulls) | 69.0% | 65.6% | 89.1% | **83.5%**, range 0.87 times the remembered one | ≥ 85%, ≤ 1.25 times | **fail** by 1.5 points |
| 3. Whole state after three imagined touches (1,384 chains) | 62.6% | 60.5% | 69.0% | **69.9%** | ≥ 73.2% (half of the gap from 62.6 to 83.8) | **fail** |
| 4. Cube in view or not after a drawer touch, put in by an imagined touch (19–22 cases, 6–7 hidden) | 68.1% | 78.0% | 63.5% | 84.0% | ≥ 80% balanced, judged only with 20 hidden cases | reported, not judged |

The remembered range (measure 2's upper bound) holds 95.6–96.2% of ends. Part (b) is in section 7.

## 6. Success criteria and prediction

Frozen benchmark as card 169: tasks 1–5, 50 episodes each at evaluation seeds 0, 1 and 2 (750 episodes); three full fits; files hashed in `runs/171/frozen.txt` first.

1. **Chains hold on unseen episodes.** Measures 2, 3 and 4 of gate (a), once on the 300 test episodes with the frozen checkpoints, mean of three seeds, against the same bars.
2. **Success within the stated loss.** Five-task mean at least **80.5%** (og5 90.5%), and tasks 1–5 at least 88.7, 81.3, 87.3, 77.3 and 68.0% (each within 10 points of og5).
3. **Reading the scene is not damaged.** The encoder is trained together with the head, so card 168's measures are repeated on the same 300 episodes: mean pair accuracy at least **92.9%** (card 168: 94.9%) and "does a pull move the slide" at least **89.5%** (91.5%).

Search time per decision is reported (card 170: 9–15 ms; og5: 2 ms).

Prediction. Measure 2 should pass almost by construction, near 88%. The open questions are measures 3 and 4. I expect the three-touch measure to gain 10 points or more, and hidden-after-closing to land between 75 and 85%, on 12 direct examples. Live, locked put-away should rise from 58% to 70–75% (og5 81%) with mixed problems unchanged, which would pass the gate narrowly. On the benchmark I expect tasks 1–3 within 5 points of og5, task 4 near 78% and task 5 near 65%, a five-task mean of about 85%. If the failures left are mostly partial chains that flip, the next card is the planner's: keep a chain while its predictions keep coming true.

Budget (as drafted; fits took 3–5.5 minutes when run three at once): a fit takes 2–3 minutes (card 168's took 88 seconds; chains add two passes). Gate (a) is four fits in separate runs; gate (b) about 5 minutes; three full fits; the test episodes 1 minute; the benchmark about 5 minutes (card 169's took 4). No run is over 10 minutes, and all are capped (CHARTER rule 5).

If the card is kept, replacing og5's written effects and guards is a new architecture version and needs the user's sign-off.

## 7. Result

**Fail at the gate.** No criterion was run: three full fits, the test episodes and the benchmark are untouched, and og5 is unchanged. Numbers are in [results.json](results.json).

Part (b), live. Development problems (30 per set, seeds 102 and 101): mixed 83% and locked put-away 67% with this card's model, against 87% and 67% for card 170. Then once on fresh seeds 402 and 401, 100 problems per set, four agents (`runs/171/gate_b_*.json`):

| Self-made problems, 100 per set | og5 | Card 170 (card 168's weights) | Range only | Both (this card's model) | Bar | Verdict |
|---|---:|---:|---:|---:|---|---|
| Mixed locks, slides and table cube | 89% | 86% | 88% | **87%** | at most 5 under og5 | pass |
| Locked put-away | 75% | 57% | 55% | **64%** | within 10 of og5 (65%) and 10 above card 170 (67%) | **fail**, by 1 and by 3 points |
| Touches per put-away episode | 5.3 | 8.6 | 8.8 | 8.9 | — | — |
| Search time per decision, put-away | 3.7 ms | 16.8 ms | 21.9 ms | 14.6 ms | — | — |
| Slides the agent took to be at the goal that were truly within 4 cm | — | — | 333 of 334 | 309 of 310 | reported | — |
| Touches on the window or its lock while both were already at the goal | 1.8% | 11.8% | 11.2% | 15.4% | — | — |

| Criterion | Verdict |
|---|---|
| 1. Chains hold on unseen episodes | **Not run** (gate failed) |
| 2. Benchmark within 10 points of og5 per task | **Not run** |
| 3. Reading the scene not damaged | **Not run** |

**Two faults in how this card was set up, both mine.**
- *The agents did not get the same problems.* The problem generator and the agent share one random stream, so after the first episode each agent draws different problems from the same distribution (1 to 10 of 100 identical). This was also true of card 170, which said "the same problems"; a correction is added there. Each figure above is one sample of 100, good to about ±5 points, and a difference between two agents to about ±7. The 64% against 57% is inside that.
- *Measure 3's upper bound was not a bound for three touches.* One touch from a real reading (83.8%) cannot be matched by three touches even with no fault at all, since each touch can be wrong. The right ceiling is each touch predicted from the real reading before it, with the predicted changes carried along. Measured afterwards (`runs/171/diag_*.json`): 65.6% for card 168's training, against 62.6% when all three touches are imagined. The criteria were not changed.

What the runs show.

- **Predicting from an imagined state was never the main loss.** It costs 3.0 points on recorded three-touch chains (65.6% against 62.6%). Chain training cuts that to 1.6 points (71.5% against 69.9%), and there is little left to gain. The fall from 80–84% after one touch to 61–70% after three is one-touch errors adding up. The reason I gave for this card in section 1 was mostly wrong.
- **The range does what it was built for, and that is not what limits the agent.** A pull's read end falls in the predicted range for 89% of pulls (69% for one value), with a range slightly narrower than the remembered one. It does not accept too much: all but one of 310–334 accepted slides were truly within the benchmark's 4 cm. Yet the range alone changed nothing live (55% against 57% on put-away).
- **Chain training helps the chain the search needs, on a measure the offline gate did not have.** A probe imagines the route every put-away needs (unlock, open, put in, shut, lock again) from the start of 60 problems (`tools/card171/probe.py`). The whole route reaches the goal in imagination for 19 of 60 with card 168's weights, 11 with the range only, and 38 with both. "Cube still in view after shutting" falls from 21 to 7, and "opening does nothing after an imagined unlock" from 19 (range only) to 1. Recorded chains rarely contain these pairs, which is why measure 3 barely moved.
- **The live gain is small and not established:** 64% against 57% on fresh problems and 67% against 67% on development problems, 84 of 130 against 77 of 130 together.
- **The fault that now dominates is needless touches.** With effects taken from the model, 11–15% of touches in put-away episodes go to the window or its lock while both are already at the goal; og5 makes 1.8%. Of this card's 36 failures, 20 have two or more such touches; 6 gave up, against 22 of og5's 25. The chains show a press on the window's lock placed before a drawer pull or a put-in, then the same press again at the next decision.
- **Why it wanders is not known.** My guess was that a prediction differs between a vector read from a frame and its recalled twin, so the search "cleans" the state through an unrelated touch. On recorded touches that is not so: replacing every untouched slot by its recalled twin flips 0.3–0.4% of predictions (`runs/171/twin.json`). The cause may lie in live frames, or in how equal-length chains and partial chains are chosen.
- **Part of every agent's put-away failures is outside the planner.** Of og5's 22 episodes that stopped with a failure, 9 had every fixture at the goal and the cube hidden (it lay over 4 cm from the goal's spot in the drawer), and 11 had the cube still on the table while the agent believed it put away.

## 8. Decision

**Revise.** The card fails at its gate and og5 stays as it is. Three things are learned. A pull's end is better held as a range than as a value, and the range learned from the agent's data agrees with the benchmark's tolerance without being told it. Training on chains of imagined states roughly doubles how often the put-away route is imagined correctly from the start, at 2.5 times the fitting time. Neither moves the live result by more than the noise, because the search that takes its effects from the model spends a seventh of its touches on parts that are already where the goal wants them. That is the next fault to find, and it is not yet known whether it belongs to the model or to the search. The next card should start by locating it: log, for each needless touch, which later touch the model predicts would fail without it, on live frames. Both additions and the probe are kept as tools for that card (`tools/card171/`).
