# Lessons

Durable lessons, each backed by evidence. At most three pages: merge or delete
rather than append. Pruned 2026-11-17 from 371 lines; the longer evidence is
in the cards named and in git history.

`old:` is `~/Projects/demilabs/experiments/research/trajectories/`; "phwm
record" is `old:phwm/docs/05-experimental-record.md`.

## Designing a test

- **Check that the events a test depends on are common in training, and
  that the answer can be known from what the model sees.** Unlocks were
  0.018% of transitions; one egocentric view matched the best action
  0.30–0.33 of the time (chance 0.27). Count labels per situation before
  blaming the features (card 143: 2 frames of the cube in the drawer).
  *Evidence:* old:gamma; cards 001, 143.
- **Measure the upper bound and the trivial baseline in the test world,
  under the scoring rule itself.** Card 072 took 100% as the bound where
  knowing the key gives 94%; card 105.2 set a bar chance made unreachable.
  Random answers scored F1 0.33 on dense events matched in time alone
  (0.05 in time and place, card 115), and "found" 56% of changes with one
  patch of slack (card 120). A bar on one top answer fails an event with
  two changed places (card 122).
- **Run a baseline given every input except the cue under test, and test
  the cue counterfactually.** Actions alone were right on 95–98.5% of
  locked pulls; only changing the lock with the actions kept showed the
  model used it (97–100% against 0–5%). *Evidence:* cards 114, 114.1.
- **Score against the simulator, never against the model's own codes**
  (card 052: 40 of 40 against 0 of 40). Offline gains and hand-built
  passes need checking in live behaviour (old:grl reset note).
- **Split a calibration test by what the input shows, never by the
  outcome** (card 110). Goal examples count only if drawn as the agent's
  own experience was (card 056).
- **Verify that held-out cases are held out, and report each case apart.**
  PHWM's first held run left 98% of queries in support. *Evidence:* phwm
  §5.2–5.6; card 027.
- **Check a truth label's timing against the sensor.** 24% of lock flips
  fall 1–5 frames before contact is sensed; the sensor also misses whole
  pulls, and stops reporting 7–25 frames before a slide stops. Recall only
  touches across which the change was seen. *Evidence:* cards 125, 125.1,
  127, 143.
- **Build a memory test with an exact best possible answer, and score
  apart the touches only earlier episodes can help.** Scored together,
  long-term factors hid behind short-term ones. A longer context is not
  memory; the past must be necessary. *Evidence:* cards 135–136.1; RLM-003.
- **Check an explanation before building on it.** Three wrong ones each
  cost a card (138–140); so did my untested choices in 142 and 144.

## Reading objects from frozen features (OGBench)

- **A position read by soft arg-max over patches needs a per-patch
  correction, and a free one does not extrapolate; bound it and let a
  low-order map do the placing.** Weighted patch positions alone 2.1 cm,
  with a free offset 0.49 (card 128). In a third of the table never
  labelled: free offset 3.32 cm, smooth map alone 1.84, map plus a shift
  held to one patch 1.21 (inside: 0.45, 1.09, 0.61). Fit in stages: map
  and score trained together were unsteady (8.1, 1.5, 1.5 cm over seeds).
  *Evidence:* cards 128, 141, 142, 144.
- **Where the hand let go is as good a label as where it closed.** Play
  put the cube in the drawer 119 times and took it out once (27.7 cm
  error there); release points, right to 0.3–0.4 cm, gave 1.7 cm and
  improved the table (0.60 → 0.45). Only if the object was in the hand
  (carried by the drawer: 9.6 cm). *Evidence:* card 143.
- **Where the hand meets an object is its variable only if the object is
  what moves under the hand.** True for the cube (0.65 cm), not for a
  slide, whose state must be read from appearance. Targets sharing one
  softmax compete (cube found in 40% of frames); ask one group at a time.
  *Evidence:* card 129.
- **Places joined by remembered moves both ways are one variable, but
  merge only what the group shares.** One set of appearance patches per
  group made half-moved slides read as done (task 2: 96% → 54%).
  *Evidence:* cards 129, 129.1.
- **Decide by recall, not a threshold on raw distance; steady is not
  unoccluded.** "Nearer a remembered change than a remembered non-change"
  took live planning from 77% to 93%; resting the arm farthest from every
  place in use, 67% to 77%. *Evidence:* card 127.
- **Counting own tries is only as good as the reading of "worked".** A
  window left 4–10 cm open reads as closed, so short touches were kept
  (42–52% against 61–65%). *Evidence:* card 130.
- **A frozen ViT loses where a change is with depth; change alone gives a
  seed, not an object.** Lasting change found condition changes at F1
  0.63 on DINOv2's block 2 and 0.30 on block 12. Slots trained only to
  reconstruct patches keep the background and skip small objects.
  *Evidence:* cards 116.1, 119.
- **A cost on how many tokens change merges the agent with what it
  moves** (65% of the cube in the arm's token). Change one loss term per
  arm. *Evidence:* card 124.

## Generalisation

- **In one fixed world a lookup table and a general rule fit equally
  well; only the task distribution or the model's form can prefer the
  rule.** PHWM's tier rule stayed at 0.16–0.23 until new fillers per task
  made it the only solution (0.98); a threshold per colour vanished with a
  fresh hue per try; a transformer reading positions only as bounded
  differences carried in 3 of 3 seeds. *Evidence:* phwm §5.4–5.5; cards
  069, 070, 102.1, 105.6.
- **"No loss in this world" narrows the model into the world's shape.**
  Version 27's planner ended with five hand-written condition forms that
  do not cover Crafter. Test a closure that cites "no loss" against the
  next world. *Evidence:* cards 087, 095.3, 095.4.
- **Nothing in a reconstruction or effect objective lines an attribute up
  across kinds** (hue: R² 0.96–0.99 within a kind, 33–53 of 66 pairs
  across). *Evidence:* cards 031, 032, 070, 088–090.
- **Novelty depends on the action:** per-action recall weights got "like a
  key but not a known key" where one radius per code could not (cards
  034–037).
- **Trust a learned distance only within the reach its input shows** (card
  097.1), and a declared front hides a relation the model must learn
  elsewhere (card 102.1).

## Conditions and acting

- **Contrast successes with failures; one smooth distance cannot mark
  conditions.** Memory's contrasts found Crafter's conditions (0.97–1.0).
  Compute significance in log space (p underflowed with 59,000 cases).
  What never varied in memory is experiment's to find, and an answer
  stays open while a fact it needs went untested. *Evidence:* cards 001–
  010, 111–112.
- **Check a condition only in a situation the learned effects produce.**
  493 of 501 false conditions came from states the world cannot make, and
  the model cannot flag them itself (0 of 39,744). Filtering unfamiliar
  states also drops true conditions: doubt should lead to a try.
  *Evidence:* cards 042, 043, 105.4, 106, 110.
- **Conditions read from a model inherit the errors of its training
  targets** (blocked-move precision 0.23–0.31 with misplaced views,
  0.95–0.97 placed by the agent's pose). *Evidence:* cards 105.4, 105.10.
- **Choose what to act on by the action's effect on its parent condition,
  and keep a choice while it still gives a plan** (71–73% → 100%).
  *Evidence:* cards 027, 029, 058, 060, 074.
- **Softmax attention must spend its weight, so it learns from tokens that
  explain nothing** (42% on walls); closing gates fixes it but pruning
  loses rare relations. *Evidence:* card 102.1.

## Networks and objectives

- **A loss must neither drop rare certain changes nor forecast random
  ones; score the next code by its likelihood.** Squared error predicted
  "the tree stays" on all 138 steps that gave wood; weighting toward
  change forecast a cow's step on half of no-ops; the likelihood over
  perception's codes fixed both (83–87% and 94–96%). *Evidence:* cards
  005, 006, 105.2, 105.7, 105.8.
- **Train each place's source where the content changed** (32% → 92%,
  card 102.4).
- **Query a network only in the form it was trained on, and check that
  copies agree;** copies still share blind spots. *Evidence:* cards
  096.1, 096.2, 106.
- **Bound a learned bias.** An unbounded position network failed to fit
  in one seed of two; bounded, 3 of 3. *Evidence:* cards 105.1, 105.6.
- **Latent self-prediction erases what actions change; test action
  sensitivity before training.** Distance to a goal state is not distance
  to a condition. *Evidence:* GRL-105–108; cards 001, 052, 053.

## Perception in moving-view worlds (Crafter, Craftium)

- **Appearance cannot tell look-alikes apart; correspondence needs a
  prediction of where each will be.** Trackers swap to the next cow or
  lose it once out of view (13% kept); a belief in the world's frame with
  the pose brings every cow back. *Evidence:* cards 098.6–098.8, 103.5,
  104.
- **Place a view by the agent's own motion, not by matching alone.**
  Learned and pretrained view motion misplaced 15–24% of blocked moves;
  the pose now places the view. *Evidence:* cards 102.5–105.9.
- **Pretrained depth sees terrain, not creatures** (a small cow 31% too
  near, card 107). Common fate groups tiles into objects once ground is
  learned (card 102.2). Put both conditions' candidates in one softmax to
  learn invariance to a frame-wide change (card 098.2).

## Memory

- **Recall needs a space where shared properties are near and new things
  far, and crisp identity at a second level, mixed with it, not switched.**
  A failed try rules out its situation, not the object. *Evidence:* cards
  033, 042–047, 091.
- **One flat key over everything reads memory badly; compare the touched
  object's own record first and add the records stored outcomes show to
  matter.** A 33-item key read 57.9%; keyed per object, 90.3% and 95.3%.
  Store "nothing happened" as an event (70.0 → 90.3%). *Evidence:* cards
  132–134.1, 136.
- **Do not empty an object's record when the episode ends; learn from the
  store how much an old item counts.** One learned weight per key item
  beat a network correction, which was unsteady between worlds.
  *Evidence:* cards 137, 138.
- **Memory cannot recover what changed unseen; it can say when to look
  again** (74% of the loss recovered for 0.38 looks per pull).
  *Evidence:* card 135.
- **Recall that compares by a network's context cannot see a condition the
  network has not learned,** and a prediction of what a place will hold
  is not identity. Targets from looking arrive only where the agent
  looks: train also on memory as it stood earlier. *Evidence:* cards
  102.1, 108, 109, 110.

## Process

- **Change pass/fail criteria at most once; then open a new experiment.**
- **Log every loss term separately, time a change under the run's own
  load, and profile before a long run** (cards 074.2, 092, 105.1).
- **Wait on a file a run writes, not on a process name.**
