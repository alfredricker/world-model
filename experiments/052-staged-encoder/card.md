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
