---
id: "052"
title: an encoder learned from interaction, in stages, over diverse worlds
rung: 0
serves: [P4, P7, P1, P3, P12, P10, P19, P17]
status: draft
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

## 2. What changes

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
