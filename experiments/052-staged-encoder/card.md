---
id: "052"
title: an encoder learned from interaction, in stages, over diverse worlds
rung: 0
serves: [P4, P7, P1, P3, P12, P10, P19, P17]
status: approved
verdict:
arch_version: 9
date: 2026-10-02
---

# 052: an encoder learned from interaction, in stages, over diverse worlds

Drafted on 2026-10-02 from a discussion with the user. It is not
approved, and its numbers are proposals. Renumbered twice: from 049 to
050 on 2026-10-03, when the user put card 049 (recall through the
conditions that matter) first, and from 050 to 052 on 2026-10-04, so
that the cards run in the order they depend on each other: 049 and 050
(recall, kept as architecture version 9), 051 (the planner, near-linear
cost), then this card. This card uses version 9's recall, and step 4
takes conditions from it instead of card 010's rule lists. The user
agreed to one card in four steps, each with its own gate, because the change is large (section 2). Appendix A
records the theory; appendix B the longer plan.

## 1. Question

Version 8's encoder is trained once on about 20 distinct tiles per world,
mainly to rebuild their pixels, and then frozen. Three things follow:
- it places a new colour by appearance, so a new-colour door is often too
  far from the known doors for recall to carry anything over (card 047:
  8–93%);
- nothing it learns comes from what actions do or which conditions they
  meet;
- in a new world (BabyAI, Crafter) it must be retrained, and everything
  keyed by its codes is lost.

Suppose the encoder is instead learned online from the agent's own
experience across many varied worlds (BabyAI with an extended palette),
from what actions do to things and which conditions they meet. Suppose
also that long-term memory waits until the encoder has settled. Does
version 8's planner then predict and act on things of colours it has
never seen, while losing nothing it does now? Before rung 1. Serves:
- P4 and P7, distinctions kept because they change outcomes;
- P1 and P3, colour as a part of its own that a relation can read;
- P12, conditions met at every level of the hierarchy shape the encoder;
- P10 and P19, memory whose keys stay valid; new colours from few tries;
- P17, the cost of re-encoding and of training (reported).

Version 8 cannot get there: its encoder never sees a colour vary within a
kind, so nothing tells it that a key is a shape and red is something else
(LESSONS, "In one fixed world a lookup table and a general rule fit
equally well").

## Revised plan (approved by the user, 2026-10-04)

Steps 2a–2h tuned the encoder against a stand-in for recall, with random
play and no planner. Card 052's central idea, that the needs of acting
and planning shape the encoder, has not entered yet (the user,
2026-10-04). This plan brings it in, keeps what those steps settled, and
supersedes section 2's steps 2–4 where they differ (kept below for the
record).

**Settled by steps 2a–2h** (details in "Overnight work" and log.md):
- Settling is judged by recall's predictions, not raw codes: most code
  changes were relabelling (2a, 2c).
- Test tries are stratified by kind, with the colour cases from play
  starts (a held key at a locked door, its colour or another); the play
  starts' held key had been lost to minigrid's reset (2d).
- The pixel anchor with a learned codebook merges tiles and loses
  colour, even on three clean colours (2e, 2f). Uniformity (Wang and
  Isola 2020) with a fixed codebook replaces it (2g); VISReg was tried
  and dropped (its shape term killed the network on this data).
- The relation term is removed (2e: it fragments the codes).
- The effect term and error-driven differentiation both separate key
  colours on clean tiles (2g); differentiation fails where recall's
  attention, not the tiles, is wrong (pick up with a full hand).
- Under tint, uniformity splits every tile's copies apart (2h). The
  stream never shows one tile under two tints, since the tint is fixed
  per episode, so no label-free method could learn that invariance from
  it (Wood and Wood 2018: invariant object recognition in newborn chicks
  needs temporally smooth experience of one object).

**Steps, each with its own gate and declared before its run:**

- **R1, nuisance that changes over time.** The generator's floor tint
  drifts slowly within an episode and differs between rooms, as light
  does; noise stays per render. The invariance term (two renderings of
  one cell a step apart) then sees one tile under neighbouring tints.
  Encoder: uniformity, fixed codebook, invariance; arms with the effect
  term and with differentiation; the stand-in measurement as in 2h.
  **Gate R1** (main setting, seed 399): at most 8 of 84 identities split
  and at most 8 tuples shared (2h's criterion 1). If it fails: a
  declared colour-constancy prior (the view's floor colour discounted),
  with the user's agreement.
- **R2, the agent's recall as the source of errors.** Version 10's
  recall (card 049's admitted conditions, card 050's own tries first,
  card 051's index), refitted on the buffer at every checkpoint, replaces
  the stand-in in the encoder's error signal and in the measurements. An
  error recall makes is answered in two ways, the loop Drescher's schema
  mechanism describes: if an existing part explains it, admission takes
  that part in (attention: the 2g failure); if none does, the tiles are
  pushed apart (perception). Arms: effect term against differentiation,
  so the user's choice of term is made on the agent's own recall.
  **Gate R2** (main setting): the colour tries at least 90% each and
  prediction flips below 1%, by the agent's recall; R1's gate still met.
- **R3, conditions from goals** (section 2's step 4, unchanged in
  substance): goals as example frames; conditions found by contrast
  through card 049's admission; an event that meets or breaks a found
  condition pulls the tiles involved into, or out of, the region the
  condition reads. **Gate 4** as declared (the contrast recovers "holding
  the X key" for "open the X door" from at most 30 successes per task
  type; P19 curve at 3, 10, 30).
- **R4, the encoder drives the agent.** Version 10's agent, with the
  infant-stage encoder at checkpoints, in the familiar worlds rendered by
  the generator in training colours. The hand-off: prediction flips by
  the agent's recall below 1% for K checkpoints (buffer N and K set on
  seed 399). **Gate 2's acting half:** an always-plastic arm (memory from
  the first step, never re-encoded) loses familiar-world success before
  the hand-off point.
- **R5, consolidation and adult,** as in section 2, with one new rule:
  with a fixed codebook every vector falls in some code's cell, so a
  vector farther than a set multiple of its cell's radius from the
  code's centre reads "new" and gets a fresh code (cards 035 and 036,
  adapted).
- **Main runs:** section 6's criteria on seeds 400–404 with step 3's
  ablation arms (no diversity, no interaction terms, no staging), handed
  to the user as commands.

**Budget.** R1: about 45 minutes of runs (two arms in parallel). R2: a
bridge from the encoder's buffer to card 049–051's recall (the larger
build), then about 45 minutes. R3 and R4: estimated when R2 passes.
Runs over 30 minutes have been run by Claude in this card with the
user's approval; the user may say otherwise.

## 2. What changes

*Steps 2–4 below are the plan as first approved in outline; the revised
plan above supersedes them where they differ.*


One component, how the encoder is learned: its data, its objective and
its schedule. They are one component because none can be tested without
the others. Diverse data with a pixel objective repeats card 031's
failure, and interaction terms on three colours repeat cards 032–034's.
The step-3 ablation arms give each part its credit.

```
version 8:
  5,000 random episodes in one world -> encoder: pixels + codebooks + pairs
  + 0.01 recall -> freeze -> memory -> recall weights -> act
card 052:
  infant:        random play over the step-1 generator; encoder learns
                 online from a buffer of the last N tries, mostly
                 interaction terms (step 3) and conditions (step 4);
                 no long-term memory
  consolidation: when drift < threshold for K checkpoints (step 2),
                 re-encode the kept pixels, build long-term memory, fit
                 recall weights
  adult:         encoder fixed except fresh codes for new appearances
                 (card 036); act as version 8
```

- **Unchanged:** recall as card 049 leaves it, tokens, walking and the
  planner; the observation (card 016's 13 × 13 egocentric view, 8-pixel
  tiles); goals in acting tests (the episode's end).
- **Not in this card:** goals given to the planner as example frames (the
  next card; here example frames shape the encoder only), relations in
  the planner (the relation term gives them a part to read; reading it is
  later), ordered relations (the tiered lock, left out by the user), maps
  and partial view (card 048's line).
- **Declared exceptions:** card 045's (the route procedure; placing the
  view and recall's appearance set).

### Step 1: the generator

BabyAI's level generator (`minigrid.envs.babyai`, installed with minigrid
3.1.0; run through `bin/prun`), changed so that:
- **Colours:** about 12 hues instead of MiniGrid's 6. 3–4 are held out of
  training and used only in tests.
- **Every kind in every training colour:** keys, doors, boxes, balls, and
  a switch door from our worlds. Pairings of keys, doors and switches are
  drawn per episode.
- **Varied success conditions:** go to, pick up, open, put next to,
  unlock. Each ends the episode when met.
- **Rooms:** 1–4 rooms, sizes 5–10, varied layouts.
- **Obstacles:** walls, lava, boxes, closed doors.
- **Nuisance:** per-episode floor tint and pixel noise (σ = 4/255), and
  decorative objects that no action or goal involves.

Acting tests use single fully visible rooms of at most 8 × 8, which
version 8 can plan in. Multi-room worlds are used for training data only,
until card 048's line provides a map.

**Gate 1:** event counts per kind and colour in 5,000 episodes of random
play: pick-ups, unlocks, switch presses, each success condition met. Any
below 30 per training colour gets play starts (CHARTER's declared
curriculum) until it reaches 30. Rendering checks: every hue is distinct
from its neighbours at 8 pixels, and the nuisance never makes two kinds
look alike.

### Step 2: the instrument and the schedule

- **Drift,** on a fixed probe set of stored tries, between checkpoints t
  and t + Δ:
  - the code flip rate: the share of probe tiles whose code tuple changes;
  - the prediction flip rate: the share of probe tries whose recall
    prediction changes, with memory and weights rebuilt from the same
    tries under each checkpoint.
- **Pixels are kept for every try** (CHARTER, "Recall replaces counting"),
  so long-term memory is built by re-encoding, never by carrying old
  vectors.
- **Buffer N and patience K** are set on spare seed 399 only.

**Gate 2,** on version 8's objective and the step-1 data: both flip rates
fall below 1% and stay there. An "always plastic" arm (long-term memory
from the first step, never re-encoded) loses familiar-world success
before that point. If the rates never settle, the hand-off cannot be
defined, and the card stops here.

### Step 3: the interaction terms

Added to the encoder's objective, with pixel reconstruction kept as a
weak anchor (its weight set at gate 3):
- **Effect:** recall's leave-one-out likelihood of each action's outcome
  (card 033's term). Things an action treats alike move together, in the
  parts that action's recall reads.
- **Persistence:** card 037's pair term, generalised. When an action
  changes a thing in place, the parts that survive are identity and the
  part that changes is state.
- **Relation (equality only):** when a toggle's success depends on the
  held thing, success should coincide with the held thing and the thing
  in front sharing a code in some part. Failures with a different colour
  are the negatives.
- **Group sparsity** on recall's weights per part (Yuan and Lin 2006,
  card 035's first draft), so each action reads few parts.
- **Batches:** half interactions (LESSONS, confirmed). Events are weighed
  by surprise and persistence, not pixel area (P19).

Ablation arms, on the same data and schedule:
- **(a) all terms;**
- **(b) no diversity:** three colours, fixed pairings, the step-1
  generator otherwise;
- **(c) no interaction terms:** version 8's objective on diverse data;
- **(d) no staging:** always plastic.

**Gate 3:** test action sensitivity before training (LESSONS, GRL-106):
the terms' value with actions shuffled must differ from their value
unshuffled. Arm (a) must keep familiar tiles on their codes, the failure
of cards 033–034, in 4 of 5 seeds.

### Step 4: conditions from the hierarchy

- **Goals as example frames:** each success yields positive frames (the
  goal holds) and free negatives (frames from the same episode before it
  held).
- **Conditions by contrast:** card 049's admitted conditions, found by
  recall: what separates positives from negatives. Then the same is done
  for each achieving action's own successes and failures, down to
  actions. (First drafted with card 010's rule lists; changed 2026-10-03
  so that the planner and the encoder share one notion of a condition.) Never a smooth distance to the example frames (card 001: rank
  correlation 0.01–0.05).
- **The encoder term:** an event that meets or breaks a found condition
  pulls the things involved into, or out of, the region that condition
  reads. Conditions are recomputed from the buffer in the infant stage;
  nothing long-lived is keyed to them until hand-off.

**Gate 4,** on the training colours: the contrast recovers the conditions
the evaluator knows (for example "holding the X key" for "open the X
door") from at most 30 successes per task type. The P19 curve is
reported at 3, 10 and 30 successes.

## Overnight work (2026-10-04)

The user delegated cards 051 and 052 overnight (2026-10-04: "proceed
with reasonable keep / throwaway / revisions"). Steps here are recorded
as Claude's, with their criteria written before they run; the running
record is [log.md](log.md).

### Step 1, as built (`tools/card052/generator.py`), declared before gate 1

- BabyAI's room grids (`minigrid.core.roomgrid`) with BabyAI's verifier
  for success conditions (go to, pick up, open, put next to, unlock).
- 12 hues: MiniGrid's 6 and orange, cyan, pink, brown, white, teal.
  **Held out:** pink, brown and teal (tests only). Training colours: 9.
- Kinds: keys, balls, boxes in every training colour; doors closed,
  locked (key of the same colour reachable) or switch-operated (a switch
  of the same colour reachable; a switch door looks like a locked door);
  switches (a plate with a lever, on and off). Lava as an obstacle;
  coloured floor tiles as decoration nothing involves.
- Rooms: 1 × 1 (sizes 5–10; split by a wall with a door 60% of the time
  at size 6 or more), 1 × 2 and 2 × 1 (5–8), 2 × 2 (5–7).
- Nuisance in rendered 8-pixel tiles: a per-episode floor tint (up to
  24/255 per channel on the background) and pixel noise σ = 4/255.
- Not yet: our 13 × 13 egocentric view of these worlds (step 2 needs
  it).

**Gate 1 passes when:**
1. After 5,000 episodes of random play (100 steps each) plus play starts
   for any cell below 30, every training colour has at least 30 of each:
   pick-ups of key, ball and box; door opens; key unlocks; switch-door
   unlocks; switch presses; and each of the five success conditions at
   least 30 times. Play starts are the agent starting in front of the
   event's object (holding the key for an unlock; the switch on for a
   switch door; holding an object for a put-next).
2. Rendering: 200 draws of random tint and noise per tile (each kind,
   state and hue, 101 tiles); every draw is nearest (L1) to its own clean
   tile.

### Gate 1 result: pass, with one revision of the rendering check (Claude, overnight)

Numbers: `runs/052/gate1.json` (not in git; summarised here).

- **Event counts: pass.** 5,000 random episodes (430,547 steps, 37 s).
  Random play alone left 22 cells below 30 (mostly key unlocks and
  switch-door unlocks per colour, and put-next successes); 13,050
  play-start episodes over 21 cells brought every cell to 30 or more.
  Per training colour after play starts: key pick-ups 1,066–1,364, ball
  pick-ups 499–645, box pick-ups 227–298, door opens 46–85, key unlocks
  30–31, switch-door unlocks 33–51, switch presses 139–202. Successes:
  go to 1,589, pick up 583, open 208, put next to 66, unlock 130 (random
  play alone: 540, 176, 70, 13, 13).
- **Rendering: pass under the revised comparison.**
  - First run: 6 of 20,200 noisy draws of an empty tile read as a grey or
    brown box, because MiniGrid's grid lines are drawn in the grey of a
    grey box's outline. Grid lines carry no information and are no longer
    drawn.
  - Second run: still 11 of 20,200 (empty to dark box), because the
    comparison was against untinted clean tiles and the tint alone moves
    an empty tile toward a dark box's thin outline. **Revision:** two
    kinds "look alike" when they give the same image under the same
    nuisance, so the comparison is now against clean tiles with the
    episode's tint (every tile of an episode shares it). Under it: 0 of
    20,200 misread. The tint-unknown figure is kept in the results.
  - Closest pair: a switch that is off, in yellow against orange (mean
    0.0077 per channel, under the noise's 0.0125); it still separates
    over a tile's pixels, but it is the first place codes may merge.

### Step 2a (the encoder online, and its code drift): gate 2 not met; revise (Claude, overnight)

Built: `tools/card052/drift.py`. Version 8's objective (rebuild,
codebook and commitment, adaptive pair term on tiles an action changed in
place), trained online on the generator's tiles: one update of 256
tiles from a buffer of the last 20,000 every 10 steps of random play.
Drift is the share of a fixed probe set's tiles (stratified, up to 40 per
identity, 84 identities, rendered once with nuisance) whose code tuple
changes between checkpoints of 2,000 updates. Seed 399 only; 40
checkpoints (80,000 updates, 800,000 steps of play) each.

| Configuration | Flip rate, last 5 checkpoints | Tuples for 84 identities | Identities split / tuples shared |
|---|---|---|---|
| Plain, constant rate | 24–33% | 23 (collapsed: parts 0 and 2 use one code) | 29 / 17 |
| Codebooks as moving averages, restarts every 2,000 | 33–46% | 55 | 34 / 10 |
| Moving averages, rate × 0.9 per checkpoint | 42–82% | 51 | 25 / 17 |
| Plain, rate × 0.9 per checkpoint (last rate 2e-5) | 10–16% | 13 (collapsed) | 46 / 13 |

Also tried and dropped: no re-seeding at first (6–9 tuples); restarts
every 250 updates (nearly every tile flipped every checkpoint).

What it shows:
- With version 8's objective the codes do not settle online. Most flips
  are in one part, whose code boundaries cut through dense clusters of
  noisy, tinted tiles, so even tiny updates move many tiles across them.
- Where flips fall (plain, decaying rate), the codes have collapsed, and
  a third to a half of the identities are split by nuisance. Low drift
  there means nothing.
- The card's rule says the card stops if the rates never settle. That
  rule was written for the hand-off; this result is about version 8's
  pixel objective under nuisance, which the card already expected to be
  a weak anchor. So the decision proposed is **revise**, not stop.

Options for the user (none taken overnight):
1. Run gate 2 with step 3's objective (interaction terms, pixels as a
   weak anchor) instead of version 8's: what must settle is the encoder
   that will be used.
2. Measure drift by recall's predictions (the card's second rate), not
   raw code tuples: a flip that changes no prediction does not hurt
   memory.
3. Remove the tint before encoding (subtract the floor's colour, which
   the agent sees every step): a declared prior about lighting.
4. Fewer codes per part, or a hysteresis on reading codes (a tile keeps
   its code until clearly past the boundary).

**The user's choice (2026-10-04, morning):** option 1 first, then
option 2 if needed. The user approved the runs below; the overnight
"sixth" option (codes mapped by overlap) is set aside. It was the fifth;
the numbering was wrong.

### Step 2b (gate 2 with the interaction terms): declared before the run

This reverses the card's order (gate 2 was to pass before step 3) with
the user's approval. The terms are added one at a time, so that each
one's effect on drift is seen; this step adds the first.

Built: `tools/card052/effect.py`, on step 2a's harness (same stream,
buffer, probe set, checkpoints of 2,000 updates = 20,000 steps of play,
seed 399, 40 checkpoints).
- **Effect term** (card 033's): for pick up, drop and toggle, events
  (front tile and held tile before the action, rendered with the
  episode's nuisance; an empty hand is the floor tile) and their outcome
  (nothing, the front tile changes, the hand changes, both). Per update
  and per action, 128 events from a buffer of the last 20,000: half with
  a change, half without (the card's interaction-dense batches). The key
  of an event is the two tiles' pieces and their distance in each part
  (card 033's "rel" keys); the loss is minus recall's leave-one-out
  log-likelihood within the batch, with attention weights λ per action
  learned alongside. Weight μ.
- **Persistence:** version 8's pair term, as in step 2a (the
  generalisation waits for the next term).
- **Pixels:** rebuilding kept at weight 1; μ sets the relative weight.

Arms (4 in parallel): μ = 0.1 (card 033's choice) and μ = 1 (pixels as
the weaker term), each with plain codebooks and with moving-average
codebooks (step 2a's first two rows are the baselines).

Criteria, per arm, over the last 10 checkpoints:
- **Gate 2 met:** code flip rate below 1%, and it stays there.
- **Progress (revise on to the next term, else drop this term):** flip
  rate at most half the matching step-2a arm's (plain 22–33%, moving
  averages 25–51%), with no more collapse (distinct tuples on the probe
  set at least step 2a's: 23 plain, 55 moving averages) and the
  label-free pair change per part reported.
- **Action sensitivity (gate 3's check, on this term):** the
  leave-one-out log-likelihood with outcomes shuffled across events
  differs from the unshuffled one; reported at every checkpoint.

### Step 2b result: gate 2 not met; progress criterion narrowly not met

Seed 399, 40 checkpoints per arm (`runs/052/effect_*_399.json`,
`runs/052/effect.sh`); 32 minutes with four arms in parallel. Flip rate
is the share of probe tiles whose code tuple changed since the previous
checkpoint, over the last 10 checkpoints; the step-2a rows are the
baselines.

| Arm | Flip rate, last 10 | Tuples (step 2a) | Identities split (step 2a) | Codes used per part |
|---|---|---|---|---|
| Plain, μ = 0.1 | 9–19% (mean 15; step 2a 22–33, mean 28) | 37 (23) | 14 (29) | 1, 8, 2, 8 |
| Plain, μ = 1 | 10–19% (mean 16) | 49 (23) | 19 (29) | 7, 3, 7, 4 |
| Moving averages, μ = 0.1 | 16–42% (mean 29; step 2a 25–46, mean 34) | 38 (55) | 16 (34) | 7, 8, 8, 8 |
| Moving averages, μ = 1 | 31–58% (mean 47) | 89 (55) | 35 (34) | 8, 8, 7, 8 |

- **Gate 2:** not met in any arm (1% needed).
- **Progress:** with plain codebooks the term roughly halves drift (mean
  15–16% against 28%; the criterion asked for at most half, so 0.54–0.57
  of it narrowly fails) and collapses less: 37–49 tuples against 23, and
  half as many identities split by nuisance (14–19 against 29). With
  moving-average codebooks it does not help.
- **Action sensitivity:** passes clearly. Recall's leave-one-out
  log-likelihood is −0.02 to −0.04 nats per try, against −0.72 with
  outcomes shuffled, in every arm and action.
- **Why it cannot settle the codes on its own:** recall here compares
  the encoder's vectors before quantisation, and predicts outcomes almost
  perfectly from them. Its loss is near its floor, so it gives little
  gradient, and nothing in it depends on which code a vector falls in.
  The codes still flip at the codebook's boundaries. Whether those flips
  change recall's predictions is not measured yet (option 2).
- **Decision (the declared rule):** the progress criterion is not met,
  so this term alone is not the fix for drift. The result points to
  option 2, which the user named next: measure drift by recall's
  predictions, with memory keyed by each checkpoint's codes. The user
  decides.

### Step 2c (option 2: drift measured by recall's predictions): declared before the run

The user approved (2026-10-04). Step 2b's plain arm at μ = 0.1, rerun
the same way (seed 399, 40 checkpoints), with the card's second rate
measured at every checkpoint (`tools/card052/predflip.py`).

- **Tries, rendered once with nuisance, from separate episodes:** for
  pick up, drop and toggle, a memory set (up to 2,000 tries with a change
  and 2,000 without, per action) and a probe set (up to 200 and 200 per
  action). Each try: the front and held tiles before, and the outcome
  (nothing, front changes, hand changes, both).
- **Recall by codes** (as the agent's memory is keyed): at each
  checkpoint, memory tries are keyed by (action, front code tuple, held
  code tuple) under that checkpoint's codes; a probe try is predicted as
  its key's most frequent outcome, or "unknown" if no memory try has its
  key. Also reported: recall by vectors (step 2b's term, the memory as
  voters, λ as learned).
- **Prediction flip rate:** the share of probe tries whose prediction
  (an outcome or "unknown") differs from the previous checkpoint's.
- **Accuracy:** the share of probe tries predicted correctly ("unknown"
  is wrong). Upper bound: the same recall keyed by the evaluator's
  identities (minigrid's object encoding), computed once; trivial
  baseline: each action's most frequent outcome in memory.

Criteria, over the last 10 checkpoints:
- **Gate 2 by predictions met:** prediction flip rate below 1% and
  staying there, with accuracy within 5 points of the upper bound (a
  stable but wrong recall does not count).
- Otherwise reported as not met, with how much of the code flips change
  predictions.

### Step 2c result: met as declared, but the probe does not test colour

Seed 399, 40 checkpoints, 24 minutes (`runs/052/predflip_399.json`,
`runs/052/predflip.sh`). Over the last 10 checkpoints:

| Measure | Recall by codes | Recall by vectors |
|---|---|---|
| Probe tiles whose code tuple changed | 8.7–17.1% (mean 12.6) | – |
| Probe tries whose prediction changed | 0.0–0.3% (mean 0.17) | 0.0–0.2% (mean 0.07) |
| Accuracy on the 1,200 probe tries | 98.8–99.0% | 99.8–99.9% |
| Probe tries with no memory try on their key | 0.0–0.2% | – |

References: the trivial baseline (each action's most frequent outcome)
scores 50%. The "upper bound" (memory keyed by the evaluator's exact
identities) scores 97.5%: it is not an upper bound, since codes group
tiles and so answer tries whose exact pair of tiles memory never saw.

- **By the declared rule, gate 2 by predictions is met.** Prediction
  flips stay below 1% from the third checkpoint on (1.5% at checkpoint
  3, at most 0.8% after), and accuracy is within 5 points of the
  reference.
- **But the test is too easy to count.** Accuracy was 99.5% at the first
  checkpoint, and the codes here are collapsed: 19–22 tuples for 84
  identities, part 0 uses 1 code, part 2 uses 2 (step 2b's run of the
  same arm had 37; GPU non-determinism). Counting the probe tries by what
  they hold explains it. Almost every outcome depends only on the kind of
  tile (a key, ball or box in front is picked up; a switch, box or door
  toggles; a wall does nothing). Only 2 of 1,200 probe tries are a held
  key at a locked door of its colour, and no try at all contrasts a
  matching key with a non-matching one at a locked door. Random play
  almost never produces those tries, and they are the ones whose
  outcome depends on colour.
- So this shows that code flips rarely change kind-level predictions. It
  does not show that colour-level predictions are stable, which is what
  the hand-off needs. **Proposed (not run): revise** the probe with step
  1's play starts (the agent facing a locked door or switch door,
  holding a key of the same or another colour), stratified per kind and
  colour, and rerun this measurement. The user decides.

### Step 2d (play starts in training and in the test tries): declared before the run

The user approved (2026-10-04), pointing out that the encoder must train
on the situations random play rarely reaches, as step 1's play starts
provide (states the agent could reach itself). Steps 2a–2c used random
play only; that was a gap in the harness.

- **Stream:** each new episode is a play start with probability 0.5,
  otherwise random play as before. A play start is drawn uniformly from,
  per training colour: facing a locked door holding its key (gate 1's
  unlock start); **facing a locked door holding a key of another
  colour** (new: the agent can reach it, and it is where the toggle does
  nothing); a switch door with its switch on; a switch; a closed door; a
  key, ball or box in front. Play-start episodes last 10 steps (the
  start's event comes first; the rest is random play), random episodes
  100 as before.
- **A bug found while building this (fixed before the run):** a play
  start's held key was lost. Minigrid's `reset()` empties the hand after
  building the grid, so gate 1's unlock starts faced the locked door
  with an empty hand, not holding its key as step 1 says. Gate 1's
  counts stand (they count unlocks that happened), but the agent had to
  find the key first, which is why 13,050 play-start episodes were
  needed. `Generator.episode()` now hands the key over after the reset;
  checked on 200 starts each: the matching key 192 times, another colour
  192 times (the rest are starts where the door could not be faced).
- **Training:** step 2b's plain arm at μ = 0.1, seed 399, 40 checkpoints,
  on this stream. The code flip rate uses step 2a's probe tiles
  (unchanged, comparable).
- **Tries for recall's predictions** (step 2c's measurement), from this
  stream in separate episodes, stratified by kind of try (action; the
  front tile's kind and door state; the held tile's kind, and whether
  its colour matches the front's): memory up to 300 tries per kind of
  try, probe up to 40. Reported per kind of try as well as overall; the
  colour tries are a held key at a locked door, matching and not.

Criteria, over the last 10 checkpoints:
- **Gate 2 by predictions met:** prediction flip rate below 1% overall
  and on the colour tries, with accuracy on the colour tries at least
  90% for both matching and non-matching keys (chance: a recall that
  ignores colour predicts the same for both, so at most about 50% of the
  two together).
- Also reported: whether the play starts change code collapse (tuples on
  the probe tiles; step 2c had 19–22).

### Step 2d result: gate 2 not met; the encoder does not learn colour matching

Seed 399, 40 checkpoints, 25 minutes plus 7 to collect the tries
(`runs/052/predflip_starts_399.json`, `runs/052/predflip_starts.sh`).
Colour tries: 40 + 40 in the probe, 193 (matching key) and 300 (other
colour) in memory. Over the last 10 checkpoints:

| Measure | Recall by codes | Recall by vectors |
|---|---|---|
| Probe tiles whose code tuple changed | 10–36% (mean 22) | – |
| Prediction flips, all tries | 3.6–16.7% (mean 8.5) | 1.3–1.9% (mean 1.6) |
| Accuracy, all tries (trivial baseline 84%) | 83–90% (mean 86) | 97–98% (mean 97) |
| Accuracy, locked door + matching key | 18–78% (mean 51) | 48–98% (mean 75) |
| Accuracy, locked door + other key | 58–95% (mean 75) | 15–48% (mean 32) |
| Prediction flips, the colour tries | 28–32% per checkpoint (means) | – |

- **Gate 2 not met.** Predictions are no longer stable once the tries
  include the cases that need colour: 8.5% flip overall by codes, about
  30% on the colour tries.
- **Colour matching is not learned, by codes or by vectors.** The two
  colour accuracies add to about 125% by codes and 106% by vectors; a
  recall that ignores colour predicts the same for both and adds to
  100%. The tries are there (193 and 300 in memory), so it is not for
  lack of data now that the play starts work.
- **The play starts did not stop collapse:** 17–26 tuples on the probe
  tiles (step 2c 19–22); parts use 3, 8, 3 and 2 codes.
- **Overall accuracy by codes, 86%, is hardly above the trivial 84%** on
  this stratified probe; recall by vectors, 97%, does better on
  everything except colour.
- **Reading:** the effect term asks recall to predict outcomes from the
  two tiles' vectors and their distance in each part. A "same colour"
  relation needs a part where a key and a door of one colour are close
  and of two colours far apart. Nothing builds that part: the encoder's
  parts mix kind and colour (an open limit since card 042). Step 3's
  relation term (success with the held tile and the tile in front
  sharing a code in some part; other colours the negatives) targets
  exactly this.
- **Decision proposed: revise.** Next, step 3's relation term on this
  stream. A cheap check before it: whether colour can be read from the
  vectors at all (a linear probe on the evaluator's colour, report
  only); if it cannot, the pixel anchor or collapse is losing it and the
  relation term would have nothing to align. The user decides.

### Step 2e (is colour in the vectors; the relation term): declared before the run

The user approved (2026-10-04) both: the check, then the relation term
if colour is there. No encoder was saved by steps 2a–2d, so the check
runs inside training. Three runs in parallel, all on step 2d's stream
and measurements (seed 399, 40 checkpoints); the relation arms are read
only if the check finds colour in the vectors.

- **Colour check (report only, evaluator labels):** at every
  checkpoint, on step 2a's probe tiles that have a colour (keys, balls,
  boxes, doors, switches), a linear classifier (multinomial logistic
  regression, 300 steps, half the tiles to fit and half to test) reads
  the evaluator's colour from (i) the encoder's whole vector, (ii) each
  part's vector, (iii) each part's code. Chance is the most frequent
  colour's share. The same for kind (key, ball, box, door, switch).
- **Arm A:** step 2d unchanged, with the check.
- **Relation term (arms B and C, weight ρ = 0.1 and 1, μ = 0.1):** per
  update, 512 toggle tries; tries are grouped by the front tile's code
  tuple, and only groups with both outcomes are used (there, whether the
  toggle works depends on something other than the front tile: the
  label-free form of "depends on the held tile"). For each try, the
  smallest distance over parts between the front and held tiles' vectors
  gives P(same) = sigmoid(a − b·d) with a and b learned; the loss is the
  cross-entropy of P(same) against "the front tile changed". Successes
  pull one part of the two tiles together; failures push the closest
  part apart.

Criteria:
- **Colour is in the vectors** if the whole-vector classifier is at
  least 90% right on test tiles at the last checkpoint (a design floor:
  9 training colours). **Amended before the run:** the smoke test shows
  an untrained encoder already scores 99.1% (chance 11.6%; per part
  49–63%, per part's code 14–24%), so the floor says nothing. What is
  read instead: whether training lowers the whole-vector score from the
  untrained 99.1% (colour lost), and whether any part's vector or code
  reads colour better than the untrained parts (colour gathered into a
  part, which the relation term needs).
- **The relation term works** (arms B, C) if, over the last 10
  checkpoints, accuracy on the colour tries reaches 90% for both the
  matching and the other key, by codes or by vectors; gate 2 by
  predictions as declared in step 2d.

### Step 2f (the harness on the old setting, an upper bound): declared before the run

The user approved (2026-10-04). Steps 2a–2e had no upper bound for the
encoder. This runs step 2d's configuration with step 2e's colour check
(effect term μ = 0.1, plain codebooks, play starts at 0.5, tries
stratified by kind) on the setting the earlier encoders learned: three
colours (red, green, blue), no tint, no noise. The generator's kinds and
rooms are unchanged, so it is closer to the old setting, not identical.
Shorter, because the setting is smaller: 20 checkpoints of 1,000 updates
(200,000 steps of play), tries collected over 150,000 steps, seed 399.

Reading, over the last 5 checkpoints:
- **The harness works (scale is the problem)** if, by codes, both colour
  tries are at least 90% right and prediction flips are below 1%.
- **The harness or objective is broken** if colour tries stay near
  ignoring colour (the two accuracies adding to about 100%).
- In between: reported as it is.

### Step 2e result: training loses colour; the relation term trades it for fragmentation

Seed 399, 40 checkpoints each (`runs/052/s2e_*_399.json`; the ρ = 1 arm
ran after the others, out of GPU memory with three at once). Over the
last 10 checkpoints (colour check at the last):

| | Step 2d (check arm) | Relation ρ = 0.1 | Relation ρ = 1 |
|---|---|---|---|
| Colour from the whole vector (untrained 99%) | 79% (51% after 2,000 updates) | 78% | 91% |
| Colour from the best part's code (chance 12%) | 25% | 25% | 32% |
| Codes: locked door, matching key | 5% | 81% | 91% |
| Codes: locked door, other key | 99% | 69% | 67% |
| Codes: all tries (trivial 84%) | 88% | 81% | 58% |
| Codes: tries with no memory on their key | 10% | 18% | 40% |
| Codes: prediction flips | 1.5% | 7.5% | 23% |
| Vectors: matching / other key | 62% / 37% | 37% / 76% | 11% / 79% |
| Distinct tuples | 27 | 64 | 76–132 |

- **Colour is lost early in training** (99% untrained, 51% after the
  first 2,000 updates, about 80% at the end), while kind is read
  perfectly; the codes carry almost none of it.
- **The relation term** separates key colours by codes (the two colour
  accuracies add to 150% and 158%, against 104%), but it fragments the
  codes under nuisance, so many tries find no memory, overall accuracy
  falls below the trivial baseline and predictions flip more. It used
  7–20 tries per batch once the codes spread (about 200 at first).
  **Not met; removed** from later steps.

### Step 2f result: the harness fails on the old setting too

Seed 399, 20 checkpoints of 1,000 updates, three colours, no tint or
noise, 13 minutes (`runs/052/s2f_old_399.json`). Over the last 5:
- codes collapse to 9–11 tuples for 30 identities; colour from three
  parts' codes is at chance (33%), the fourth 46%;
- the colour tries: matching key 2.5%, other key 100%, **by codes and by
  vectors alike**: recall ignores colour entirely, although a linear
  classifier reads colour from the vectors 79–92% of the time;
- overall 84% by codes against a trivial 83%, 99% by vectors.

**By the declared reading, the harness or objective is broken, not
only the scale.** The earlier encoders (cards 031–037) kept the
training tiles apart in this setting because they trained offline on
the 19–20 distinct clean tiles, each once, and re-seeded codes whenever
two distinct tiles shared one (knowledge of which tiles are distinct).
Steps 2a–2f have neither: an online stream weighted by frequency and
label-free restarts. Recall by vectors also ignores colour here even
where the vectors hold it: its attention weights are fitted on batches
dominated by tries where colour does not matter.

### Step 2g (replace, not add: VISReg for collapse, error-driven differentiation): declared before the run

The user's direction (2026-10-04): change the encoder by replacing
pieces that do not work, not by adding terms; try error-driven
differentiation; the pixel anchor may not be helping, and VISReg
(Wu, Balestriero and Levine 2026, `visreg-variance-invariance-sketching-regularization-for-jepa`)
may prevent collapse better, though not preserve colour alone.

Two replacements, crossed, on step 2f's old setting (three clean
colours; seed 399; 20 checkpoints of 1,000 updates), so that each
piece's effect is seen; step 2f is the fourth cell:

| | Effect term (card 033) | Error-driven differentiation |
|---|---|---|
| **Pixel reconstruction** | step 2f (done) | arm C |
| **Uniformity, fixed codebook, no decoder** (VISReg dropped, below) | arm A | arm B |

Removed from every arm: the relation term (step 2e), dead-code restarts
(as in the plain arms). Kept: the codebook and commitment terms (codes
are read from the vectors) and version 8's pair term.

- **VISReg** (replaces the decoder and pixel loss): the paper's
  centring, scale ((σ* − σ_j)² per dimension) and shape terms (sliced
  Wasserstein distance to an isotropic Gaussian over 64 random
  projections, scale stopped), weights 1 as the paper's default, on the
  32 numbers of the normalised parts, where an even spread has
  σ* = 1/√8. **Amended before the run:** declared first on the raw
  numbers before normalisation, but the smoke test (200 updates) left
  every part with one code: the scale term was met by the vectors'
  lengths while their directions, which the codes read, collapsed. Its
  invariance term uses natural views, not augmentations (colour jitter
  would teach colour blindness): the same cell, other than the front
  cell, rendered before and after one step, 64 pairs per update, MSE
  between their vectors, weight 1. On the clean old setting the two
  views are identical, so this term is zero there. Correspondence of a
  cell across one step comes from the simulator here; in the
  architecture the agent's placements give it (card 044). The MSE is
  taken on the normalised parts as well.
- **Error-driven differentiation** (replaces the effect term): per
  update and per action, step 2b's batch of 128 tries. Recall by vectors
  predicts each try from the others (leave-one-out); its attention
  weights λ are fitted to that likelihood on stopped vectors (they train
  recall, not the encoder). A try is an **error** if recall gives its
  true outcome less than 0.5. For each error, the most similar try (by
  recall's kernel) with another outcome is its partner. Of the two
  slots (front tile, held tile), the one whose tiles differ more is
  taken (the other is likely the same tile), and within it the part
  where they differ most; that part's two vectors are pushed apart by a
  hinge, up to the median distance between that part's codebook
  entries, weighted by the error's surprise (−log of the probability
  recall gave the true outcome). Nothing pulls tiles together. A pair
  whose two tiles differ by no more than two views of one cell do (the
  95th percentile of the invariance pairs' part distances) is skipped:
  the cause lies outside the two tiles. Weight 1.
- **Measurement added:** recall by codes, then vectors (a try whose code
  key memory holds is predicted from those tries, otherwise by vectors),
  as version 9's recall takes the own situation first and neighbours
  otherwise. Distinct codes leave more keys unseen, so codes alone would
  punish keeping tiles apart.

**Amended before the run: VISReg dropped, uniformity instead.** Checked
on the old setting's 30 distinct tiles before any arm ran (the
collapse-prevention term alone, plus commitment; 1,500 updates; scripts
in the log):
- VISReg as published (on the raw or the normalised numbers) kills the
  network within 150–200 updates: every number's spread falls to 0 and
  the ReLUs die. Its shape term alone does this. Matching standardised
  projections to a Gaussian, with the spread held fixed within a step,
  shrinks the vectors at every step when the data is a few repeated
  tiles (most of every batch is walls and floor).
- Its scale and centring terms alone, or VICReg's variance and
  covariance terms, spread the vectors but keep only 7–16 code tuples
  for the 30 tiles: per-number statistics are dominated by the frequent
  tiles, so rare tiles merge cheaply.
- **Uniformity** (Wang and Isola 2020: the log of the mean Gaussian
  potential, t = 2, over all pairs of the batch's whole normalised
  vectors) keeps every pair of tiles apart (closest pair 0.65–1.03) and
  gives 25 tuples for the 30 tiles with a **fixed codebook** (each
  part's 8 axes, as FSQ: no codebook loss and no restarts; a learned
  codebook gave 20). The pixel anchor gave 9–11 (step 2f).
So arms A and B use uniformity (weight 1) with the fixed codebook and
commitment 0.25, plus the invariance term on natural views (Wang and
Isola's alignment; zero on the clean setting). Arm C keeps step 2f's
pixel anchor and learned codebook.

Criteria, over the last 5 checkpoints (an arm that meets all three goes
on to the main setting; a piece whose replacement meets them is removed
for good):
1. **Codes keep the tiles apart:** at least 27 distinct tuples for the 30
   identities, and colour read from the best part's code at least 90%
   (chance 33%). The earlier encoders kept all 20 tiles apart here.
2. **Colour tries:** at least 90% right for both the matching and the
   other key, by codes then vectors.
3. **Stable:** prediction flips by codes then vectors below 1%.

### Step 2g result: every replacement fixes the colour tries; none keeps all tiles apart

Seed 399, old setting, 20 checkpoints of 1,000 updates, three arms in
parallel, 17 minutes (`runs/052/s2g_*_399.json`, `runs/052/s2g.sh`).
Over the last 5 checkpoints ("codes then vectors" is the measure the
criteria name; step 2f did not have it):

| | 2f: pixels + effect | A: uniformity + effect | B: uniformity + differentiation | C: pixels + differentiation |
|---|---|---|---|---|
| 1. Tuples for 30 identities (≥ 27) | 9–11 | 15 | 22–24 | 25–26 |
| 1. Colour from the best part's code (≥ 90%; chance 33%) | 46% | 46% | 75% | 67% |
| Codes used per part | 1, 3, 2, 4 | 4, 4, 7, 5 | 7, 7, 7, 6 | 8, 7, 2, 1 |
| 2. Matching key / other key (≥ 90% each) | 2.5% / 100% (codes) | **97.5% / 100%** | **97.5% / 100%** | **97.5% / 100%** |
| 3. Prediction flips (< 1%) | 2.4% (codes) | **0.2%** | **0.5%** | **0.0%** |
| All tries, codes then vectors | – (codes 84%) | **99.5%** | 94.3% | 94.6% |
| All tries, vectors alone | 98.6% | 99% | 92% | 93% |
| Probe tiles changing code (step 2a's rate) | 0–20%, mean 8.7 | 0–3%, mean 0.7 | 0–13%, mean 6.0 | 3–13%, mean 6.7 |

- **Criterion 1 is not met by any arm**, so none goes on as declared, and
  no piece is removed for good by the declared rule.
- **Criteria 2 and 3 are met by all three.** Recall now separates a key
  of the door's colour from another (39 of 40 and 40 of 40; the one miss
  is the same try in every arm). Step 2f, the old recipe, got 1 of 40.
  With differentiation (B, C) the key colours are separated from the
  first checkpoint; with uniformity and the effect term (A), from the
  twelfth.
- **The differentiation arms lose elsewhere:** picking up while the hand
  is full (the outcome is "nothing") is predicted wrong in every such
  kind of try (0%), by vectors. Differentiation pushes apart tiles that
  already differ (a held key and an empty hand), but recall's attention
  weights, fitted on stopped vectors, do not weigh the held slot for
  pick-ups, so pushing does not help. In A, where the effect term trains
  the vectors and the weights together, those tries are right.
- **Uniformity uses all four parts** (B: 6–7 codes in each); the pixel
  anchor leaves two parts nearly unused (2f, C). A's codes are the most
  stable (0.7% of probe tiles change code, spikes to 13% and 20% at two
  checkpoints), but A merges the most tiles (15 tuples): the effect term
  pulls tiles that behave alike together.
- **Colour still does not gather into one part's code** in any arm (at
  most 75%): the open limit "parts mix kind and colour" remains.
- **Decision proposed: revise** (the user decides which arm goes to the
  main setting): uniformity with a fixed codebook replaces the pixel
  anchor on this evidence (better than pixels on criteria 2 and 3 with
  either interaction term, and all parts used); between the effect term
  (A: best predictions and stability, more tiles merged) and
  differentiation (B: more tiles kept apart, pick-ups with a full hand
  wrong), neither is clearly better.

### Step 2h (arms A and B on the main setting): declared before the run

The user approved (2026-10-04): the pixel anchor is dropped; arms A
(uniformity with a fixed codebook, effect term) and B (the same, error-
driven differentiation) run on the main setting: nine training colours,
tint and noise, play starts at 0.5, tries stratified by kind, seed 399,
40 checkpoints of 2,000 updates, as steps 2d–2e. Here the invariance
term is active (two noisy renderings of one cell), and B's skip
threshold comes from it. Baseline: step 2e's check arm (pixels and the
effect term on the same stream; 5% / 99% on the colour tries by codes).

Criteria, over the last 10 checkpoints:
1. **Codes keep identities:** at most 8 of the 84 identities split over
   several tuples by nuisance, and at most 8 tuples shared by several
   identities.
2. **Colour tries:** at least 90% for both the matching and the other
   key, by codes then vectors.
3. **Stable:** prediction flips by codes then vectors below 1%.
Also reported: probe tiles changing code, accuracy on all tries, the
worst kinds of try.

### Step 2h result: under tint and noise the codes fragment; gate 2 not met

Seed 399, 40 checkpoints, 39 and 43 minutes (`runs/052/s2h_*_399.json`).
Over the last 10 checkpoints; baseline step 2e's check arm (pixels and
the effect term, same stream):

| | 2e check (pixels + effect) | A: uniformity + effect | B: uniformity + differentiation |
|---|---|---|---|
| 1. Identities split by nuisance (≤ 8 of 84) | 11 (2d) | 55–61 | 49–58 |
| 1. Tuples shared by identities (≤ 8) | – | 78–89 | 82–105 |
| Distinct tuples on the probe tiles | 24–28 | 324–357 | 363–421 |
| Test tries with no memory under their code key | 10% | 86–90% | 89–94% |
| 2. Matching / other key, codes then vectors (≥ 90% each) | – (codes 5% / 99%) | 72–85% / 18–40% | 20–57% / 78–95% |
| 3. Prediction flips, codes then vectors (< 1%) | – (codes 1.5%) | 3–4% | 2–5% |
| All tries, codes then vectors (trivial 84%) | – (codes 88%) | 89–92% | 88–90% |
| Probe tiles changing code | 7–18% | 26–35% | 31–38% |

- **All three criteria fail in both arms.** Step 2g's result on clean
  tiles does not carry over to tint and noise.
- **Why:** uniformity pushes every pair of different vectors apart,
  tinted and noisy copies of one tile included, and its push is
  strongest for the closest pairs. The invariance term pulls together
  only two renderings of one cell within an episode, which share the
  tint, so nothing says that two tints of a tile are the same tile. The
  codes split each identity over many tuples; nearly every test try
  then finds no memory under its own code key and falls back to recall
  by vectors, which ignores colour (as in steps 2d–2f).
- **Decision proposed: revise.** The collapse prevention needs a notion
  of "the same tile" across tint that does not come from labels.
  Candidates: discount the floor colour the agent sees in every view (a
  declared prior, as colour constancy discounts the illuminant), or let
  what tiles do in the agent's tries say which differences matter (the
  planner's conditions, step 4). The user decides, with the revised
  plan.

### Step R1 (nuisance that changes over time): declared before the run

The revised plan's first step (approved 2026-10-04).
- **Tint drift:** in the training stream the floor tint takes a random
  walk within the episode: after every step each channel moves by a
  normal step of σ = TINT/20 (1.2/255), reflected at 0 and TINT
  (24/255), from the episode's random start. Over a 100-step episode
  it wanders across about half the range. Noise stays per render.
  **Amended before the run:** the plan also named a tint per room; it
  is left out, because a cell never changes room, so a room's tint never
  shows one tile under two tints.
- **The invariance term** then pairs one cell's renderings a step apart
  under neighbouring tints; through overlapping pairs (slowness,
  Földiák 1991) it may reach the whole range.
- **Unchanged from step 2h:** uniformity with the fixed codebook,
  commitment, version 8's pair term; seed 399, 40 checkpoints of 2,000
  updates; the probe tiles and the test tries are step 2h's (collected
  without drift, tint fixed per episode), so invariance is measured
  across episodes' tints.
- **Arms:** with the effect term (μ = 0.1) and with error-driven
  differentiation, as A and B in step 2h.

**Gate R1,** per arm, over the last 10 checkpoints: at most 8 of the 84
identities split over several tuples and at most 8 tuples shared by
several identities (step 2h: 49–61 and 78–105). Also reported: the
colour tries, prediction flips, test tries with no memory under their
code key. If neither arm meets it, the next proposal is the declared
colour-constancy prior, with the user's agreement.

### Step R1 result: gate not met, but the invariance term was too weak to test the drift

Seed 399, 40 checkpoints, two arms (`runs/052/r1_*_399.json`). Over the
last 10 checkpoints, against step 2h:

| | 2h A | R1 A (effect) | 2h B | R1 B (differentiation) |
|---|---|---|---|---|
| Identities split (≤ 8) | 55–61 | 61–69 | 49–58 | 52–59 |
| Tuples shared (≤ 8) | 76–89 | 89–106 | 82–110 | 73–106 |
| Matching / other key | 79% / 29% | 86% / 16% | 44% / 84% | 14% / 93% |
| Prediction flips | 3.4% | 3.3% | 3.3% | 2.2% |

The drift changed nothing. Measured afterwards on a batch at the start
of training: the gradient of the uniformity term on the encoder is 7.0,
that of the invariance term as built 0.011, about 600 times weaker. The
card declared Wang and Isola's alignment; as built it averaged over the
32 numbers instead of summing, on differently normalised vectors, and
on 64 pairs while uniformity acted on 256 other tiles. Wang and Isola
apply both terms to the same batch of positive pairs, so every vector
pushed apart is also pulled to its other view. **Revise:** R1b below,
the published recipe, before the colour-constancy fallback.

### Step R1b (alignment and uniformity as published): declared before the run

- **Alignment and uniformity on one batch** (Wang and Isola 2020): per
  update, 256 natural view pairs (one cell a step apart, the tint
  drifting as in R1). Alignment: the mean over pairs of the squared
  distance between the two views' normalised whole vectors (summed over
  the numbers). Uniformity: over the same batch's vectors (both views),
  as before. Weights 1 and 1, t = 2, as in their paper. The stream's
  tiles still feed the commitment, pair and interaction terms.
- Measured at the start of training before the run: both terms' pull on
  the encoder, reported.
- **A short diagnostic first:** arm A (effect term), 10 checkpoints of
  2,000 updates, compared with R1 A at checkpoint 10 (60 identities
  split). The full two-arm run follows only if identities split falls
  clearly (below 30 at checkpoint 10).
- The encoder's weights are saved at the end of every run from here on,
  so diagnostics no longer need a rerun.
- Gate R1 as declared.

## 3. Dependencies

- Version 9's recall (cards 049 and 050), planner and tokens.
- **Before this card's BabyAI runs:** recall and the planner's reasoning
  over conditions near-linear in cost per step, independent of memory
  size (the user, 2026-10-03; ARCHITECTURE.md, "Known limits").
- Card 036's fresh codes; card 037's encoder objective.
- Card 049's admitted conditions for conditions by contrast (gate 4
  checks them on codes over the new generator).
- Card 033's recall term (it broke codes on three colours; gate 3 checks
  it on diverse data).
- New and untested: the generator (gate 1), drift and the hand-off (gate
  2), the relation and sparsity terms (gate 3), the condition term (gate
  4). Each step waits for the gate before it (CHARTER rule 7).
- Literature, to add to LITERATURE.md when the card becomes the focus:
  - Chevalier-Boisvert et al. 2019 (BabyAI);
  - Konidaris, Kaelbling and Lozano-Pérez 2018 ("From Skills to
    Symbols"; appendix A);
  - Lachapelle et al. 2022 (`2107_10098`);
  - Peters, Bühlmann and Meinshausen 2016; Arjovsky et al. 2019
    (invariance across environments);
  - Feinman and Lake 2018 (`1802_02745`); Smith et al. 2002;
  - Akers et al. 2014; Guskjolen et al. 2018; Held and Hein 1963;
  - `1703_01988` (Neural Episodic Control).

## 4. Data check

Gate 1 (step 1). The rarest events are unlocks and switch presses per
training colour; LESSONS records 0.018% unlocks in chained rooms
under random play, which is why play starts are declared.

## 5. Feasibility gate

- **Upper bound:**
  - version 8 with its own offline encoder in the familiar worlds (100%);
  - for new colours, version 8's planner with label codes (cards 031 and
    033's label arm: new keys carried fully, the matching door opened at
    the first try).
- **Trivial baselines:**
  - a random-initialised encoder, frozen at step 0;
  - arm (c), version 8's objective on diverse data;
  - card 047's new-colour switch tests (mean 39.6%).
- **Shakedown:** spare seed 399 through all four steps, 30 layouts per
  test.

Result of the gates, before the main run: (not yet run)

## 6. Success criteria and prediction

Seeds 400–404, the full encoder (arm a with step 4); each criterion in at
least 4 of 5 seeds.

1. **Nothing lost.** Card 045's three criteria in version 8's own worlds,
   rendered by the new generator in training colours: familiar worlds ≥
   99% with steps within 5% of card 029's; unseen rooms ≥ 99% at most 1.15
   times the shortest route; how-soon rank correlation ≥ 0.95. Also,
   single-room BabyAI tasks in training colours at least as high as arm
   (c), minus 2 points.
2. **A new colour acts like its kind.** On held-out colours:
   - card 047's switch-world test with a new-colour door ≥ 80% (card
     047: 8–93%, mean 39.6%);
   - first-sight effects right (pick up a new key, toggle a new door with
     the switch on, walk through it once open) in at least 9 of 10 cases;
   - both above arm (c) by at least 20 points.

   This is the behavioural test (C4) that colour is a part separate from
   kind.
3. **The hand-off holds.** After consolidation, code and prediction flip
   rates stay ≤ 1% to the end of the run, and arm (d) scores at least 10
   points below arm (a) in some familiar world. If (d) is as good, staging
   is unnecessary here, and the card says so.

Reported:
- **Ablations:** arms (b)–(d) on criteria 1–2, and arm (a) without step 4.
- **P19:** each criterion-2 measure after 0, 1, 3 and 10 tries on the new
  colour.
- **Matching keys:** a new-colour key and door, matched or not. Whether
  recall alone tells a match from a non-match, before any relation reader
  exists.
- **Nuisance:** the share of tinted or noisy copies of a tile that keep
  its codes.
- **Cost:** hand-off step per seed, re-encoding time, training time,
  planning time against version 8 (P17).

**Prediction.**
- Without nuisance, the random encoder already passes criterion 1.
- Arm (c) fails criterion 2, as card 031 did: new colours come out "new"
  everywhere.
- Arm (b) fails it too: three colours cannot separate kind from colour.
- Arm (a) passes criterion 2's first-sight effects. The switch test is
  the doubtful one, near 70–90%.
- Hand-off comes within the first 10–20% of training.
- Step 4 adds most to the switch test: "switch on" is a condition met
  that changes nothing in view of the door.

**Budget.** Not estimated until gate 1. Encoder training is online over
many episodes, so the main runs will go to the user as commands.

## 7. Result

## 8. Decision

## Appendix A: the theory (2026-10-02)

**What the encoder should keep.** A distinction matters if changing it
changes what happens or what can be reached (P4). "From Skills to Symbols"
(Konidaris et al. 2018) proves that the abstract state a planner needs is
exactly what decides where each skill can start (its conditions) and
where it leaves you (its effects). So two things are equivalent for the
encoder when they meet the same conditions and actions on them have the
same effects on conditions. This is one step deep, and the hierarchy
supplies the depth. It was chosen over bisimulation, whose equivalence
runs through future states, not through conditions.

**Why diversity is the missing ingredient.** Factors are identifiable
when each mechanism reads few of them and the data vary enough
(Lachapelle et al.). Features whose link to an outcome holds across
environments are the causal ones (Peters et al.; Arjovsky et al.). Cards
031–035 had three colours, each tied to the same objects, so kind and
colour were confounded; that, not only the objective, may be why
similarity never formed. Each axis of variation breaks one confound:

| Axis | Confound broken | Learned |
|---|---|---|
| Every kind in many colours | Colour ↔ kind | Shape carries the effect; colour its own part |
| Random pairings | This key ↔ this door | Colour matters through matching |
| Many obstacles | Blocking ↔ one appearance | Walkability from what moves do |
| Varied success conditions | Goal ↔ one route | The union of distinctions any goal needs |
| Rooms and layouts | Layout ↔ position | Relative where |
| Nuisance | Appearance ↔ effect | Invariance (P4; Crafter's lighting) |

**Why varied goals and a pixel anchor.** Any equivalence learned from
outcomes keeps only what the outcomes it saw needed. Varied goals keep
the union. Reconstruction, weakly weighted, keeps what has not mattered
yet until something needs it.

**Goals as example frames, success as a source of them.** A success bit
cannot be inspected, so a goal never reached cannot be recognised or
pursued (P12's test). Example frames are in the agent's own terms. They
vary in the irrelevant and agree on the condition, which is the
strongest signal the encoder gets. Demonstrations (P16) and self-set
goals (P20) share the format. A success signal gives example frames for
free (the frames at success, and the frames before it as negatives).
Subconditions are never signalled. The agent knows one is met when its
state falls in the region the contrast found, and credits it when
meeting it made the parent reachable.

**Relations, each in its place.**
- **Equality** in a part (same colour) is a relation on top of codes:
  symmetric and transitive by construction.
- **Order** (tool tiers, counts) is a comparison along an axis, left out
  for now.
- **AND and OR** are the hierarchy's structure (version 8's "both" and
  "either" worlds), not the encoder's. What the encoder gets from OR:
  things that achieve the same condition are pulled together.

**Order, for later.** An architecture supports order when some reader
compares with a direction (a ≥ b when w·a ≥ w·b). An encoder learns order
when outcomes push things along that direction. Both are needed. Today
recall reads only unsigned distances and codes only equality, so tiers
would be learned pair by pair. "Where" is already ordered, and moves are
shifts along it (card 044). A count or tier part treated the same way,
with "collect" as a shift, is the natural form. This card keeps it open:
parts stay continuous vectors under the codes, and the relation term is a
constraint between two things in one part, never a learned pair table
(LESSONS, card 040).

**Sameness from three sources (added 2026-10-04, with the user, after
steps 2a–2h).** Tiles that must share a code, or lie close, need a
reason to. The revised plan rests on three sources, each answering a
different question:
- **Continuity says what is the same object.** One cell's renderings a
  moment apart are the same thing, whatever their effects (Földiák 1991;
  Wiskott and Sejnowski 2002; Wood and Wood 2018: newborn chicks form
  invariant object recognition only from temporally smooth experience).
  Its copies under nuisance should share a code. This needs nuisance
  that changes while the agent watches (R1); a per-episode tint never
  shows one tile under two tints, so nothing could learn it (step 2h).
- **Effects and conditions say which differences matter.** Tiles that
  meet the same conditions and give the same effects are equivalent
  (above; acquired equivalence, Honey and Hall 1989); tiles whose
  outcomes differ must be told apart (acquired distinctiveness,
  Goldstone 1994). This acts in the parts each action reads (card 049's
  admitted conditions), not everywhere: a red and a blue ball become
  close where pick up looks and stay distinct elsewhere, so codes remain
  names and kinds form on top. Step 2g's effect term pulled everywhere
  and merged tiles (15 tuples for 30).
- **Errors say when a new distinction is needed.** An error recall makes
  is answered by admitting an existing part that explains it
  (attention) or, when none does, by pushing the tiles apart
  (perception), as in Drescher's schema mechanism (1991), whose
  synthetic items are made when no existing item explains a result.
The gap these close: effects alone cannot tell nuisance from a rarely
relevant difference until a try reveals it; a tint change and a key's
colour change leave pick up, drop and walking alike, and only a locked
door separates them. The pixel anchor above was replaced by uniformity
with a fixed codebook (step 2g); reconstruction is no longer the term
that keeps what has not mattered yet.

**Rare effects: starts the agent could reach, as contrasts (added
2026-10-04, with the user).** The events that separate relevant from
irrelevant differences are rare in random play (0.018% unlocks in
chained rooms, LESSONS). Play starts begin episodes in states the agent
could reach itself but rarely does (CHARTER's declared curriculum):
- **Now:** each rare event with its near miss, so that two tries differ
  only in the condition that matters: the door's key and another key at
  a locked door; the switch on and off at a switch door; a door locked
  and unlocked with the key in hand; an object held next to the one to
  place it beside; states just before each mission succeeds (also R3's
  goal frames).
- **Later (BabyAI, Crafter):** checkpoints from the agent's own play.
  As in Go-Explore (Ecoffet et al. 2021), states the agent reached are
  archived in cells (here the code tuples of front, held and the
  admitted view conditions) and episodes restart from rare cells and
  from cells where recall errs, so every start is reachable by
  construction and the curriculum follows the same errors the encoder
  learns from. Starts near goals, moving outward, cover goal-directed
  cases (reverse curricula, Florensa et al. 2017).

## Appendix B: the longer plan

**The interface stays fixed.** Tokens, each a what and a where (card 044),
are the contract between the encoder and the rest. Only what produces
them changes across worlds:

| World | Units | Encoder must learn | New demand elsewhere |
|---|---|---|---|
| MiniGrid, one room | The grid | Appearance | None |
| BabyAI, extended (this card) | The grid | Kind, colour, state, invariance | Goals as frames; relations; a map |
| Crafter | A grid; lighting, movers, counts | Invariance to lighting; counts and tiers as ordered parts | Threats, counts in goals |
| Minecraft | Not given; 3D | Units from continuity and interaction (P7) | A where in 3D |

**Three stages, re-entered locally in a new world.**

| Stage | Encoder | Memory | Ends when |
|---|---|---|---|
| Infant | Plastic | Short buffer | Drift below threshold |
| Consolidation | Slowing | Kept pixels re-encoded into long-term memory | Recall's predictions stop changing |
| Adult | Fixed; fresh codes for new appearances | Long-term | A new world brings appearances it cannot place |

**Why staging.** The user's hypothesis (2026-10-02): infants form few
lasting memories because their encoder is still changing, so stored
memories stop matching what would retrieve them.
- In mice, added hippocampal neurons cause infant forgetting (Akers et
  al. 2014).
- "Forgotten" infant memories return when their original cells are
  reactivated (Guskjolen et al. 2018).
- Neural Episodic Control works because its embedding changes slowly.
- Card 033 is the same failure here: replay into the encoder moved
  familiar tiles off their codes in 9 of 10 seeds, and acting broke.

Re-encoding kept pixels is something a brain cannot do, and its cost
grows with memory (P17).

**Why interaction.** Of two kittens with the same visual input, only the
one moving itself developed visually guided behaviour (Held and Hein
1963).

**Risks.**
- Three objectives fight in one vector (card 034): reconstruction,
  recall and quantisation. Diversity and per-part sparsity are this card's
  answer; if they fail, the conflict needs the latent space organised into
  parts some other way (CHARTER, "One latent space").
- Interaction objectives can collapse what has not mattered yet. The
  pixel anchor and varied goals guard against it.
- Diversity thins rare events per kind; play starts and interaction-dense
  batches compensate.

**Overnight diagnostics after step 2a (Claude):** the drift remains
without tint or noise, so the online objective keeps moving the codes
and the nuisance does not cause it. Measured without labels, most of it
is relabelling: by the last checkpoints only 0.1–13% of probe-tile pairs
change between the same code and different codes, against 32–51% of
tiles changing code number. One part (part 1) keeps regrouping. A sixth
revise option follows: keep code names by mapping each checkpoint's codes
to the last checkpoint's by overlap, and slow or freeze the part that
regroups. Details are in log.md.
