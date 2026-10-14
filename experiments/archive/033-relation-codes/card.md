---
id: "033"
title: relation codes
rung: 0
serves: [P3, P10, P19, P5, C5]
status: done
verdict: fail
arch_version: 5
date: 2026-09-29
---

# 033: relation codes

## 1. Question

Can recall predict what happens with things the agent has never seen,
including a relation such as "this key fits this door"? Recall here is a
vote of past events, each weighted by how alike it is. And does "fits"
become a code where it explains the tries more cheaply than one rule per
colour? The test is a yellow door, a colour never seen on a door, opened
within two tries. Serves:
- P3, GOAL.md's example "key opens door of the same colour";
- P10, recall for a new question;
- P19, a few tries;
- P5, a confidence that drops for new things;
- C5, nothing already known lost.

This follows [card 032](../032-fewest-codes/card.md) (stopped). It was
revised with the user and approved on 2026-09-29. The design rests on
four points agreed with the user that day:
- counting → recall → weights;
- a relation gets a code once it explains the cases more cheaply than the
  separate rules it replaces;
- one learner, one representation: recall compares things in card 031's
  encoder;
- recall must be principled and carry to other environments: no
  thresholds, and a confidence in every answer.

## 2. What changes

One component: the prediction of an event's outcome gains recall.

```
counting (card 029): outcome of (toggle, closed red door, key red) = its own counts
recall (this card):  its own counts + a vote of the other past toggles, each
                     weighted by how alike it is; a new event has only the vote
```

- **Memory.** Every pick up, drop and toggle is kept as three things: the
  action, the two things it involves (the tile in front and the held
  tile), and what changed. What changed is one of five kinds, and none
  names an object:
  - nothing changed;
  - the thing in front went to the hand;
  - the held thing went to the front;
  - the thing in front changed in place;
  - something else.

  Each distinct (action, front, held) is one memory with its counts.
- **What recall compares.** A thing is its 32 numbers in card 031's
  encoder, before the codebooks. An event is described by the front's
  numbers, the held thing's numbers, and how far apart the two are within
  each of the encoder's four pieces (as built; the approved plan said
  signed differences, see appendix A):
  - the things' own numbers let a new key recall keys, the way a new kind
    of tree would recall trees;
  - the distances let a relation be recalled: "same colour" here. A
    distance puts a new value on the familiar scale, since "same" is 0
    whatever the colour. An ordering, such as tool tier against block
    hardness, needs a signed comparison, which is left for later.
- **The vote.** A memory's weight is k = exp(−Σ_j λ_j |x_j − x′_j|). This
  is Shepard's generalisation law, with the per-number weights (attention)
  of Nosofsky's exemplar model. The chance of outcome kind c for event q
  is:

  P(c | q) = (n_q,c + Σ_i k_i r_i,c + π_c) / (n_q + Σ_i k_i + 1)

  Here n_q,c are the event's own counts, and i runs over the other
  memories of the same action. Each memory counts once, through its
  outcome rates r_i. The prior π is uniform. As exact experience grows it
  dominates, so counting is the special case. A new event is predicted
  from similar memories alone.
- **Confidence, with no threshold.** n_q + Σ_i k_i measures how much
  relevant memory backs a prediction, and a new thing has less of it.
  Nothing switches recall on or off. Unlike neighbourhood components
  analysis, the weights are not normalised, so distance shows as doubt.
- **What "alike" means is learned by one objective:** predict each memory
  from the others (leave-one-out, as neighbourhood components analysis
  does, `1604_02354`). A memory never votes for itself, so the weights
  must generalise across memories.
- **Replay trains the encoder.** Card 031's encoder objective gains this
  leave-one-out term, at a weight μ. The weights λ, one set per world and
  action, are learned with it. μ is chosen on familiar tiles only, by card
  032's rule (section 5).
- **The code.** For toggles at doors, card 010's evidence chooses among
  three descriptions:
  - rules per door;
  - rules pooled over the doors;
  - pooled rules that may use "fits". fits holds for a try when recall
    from the other memories (leave-one-out) predicts the door changes.

  The relation becomes a code only if the last description wins. It is
  then written out per known door as "held = X" for each known X that
  fits, so card 029 reads it unchanged.
- **Declared for this world:** an event's things are fixed places (in
  front, held) until segmentation gives objects. The five outcome kinds
  are read from exact tiles.
- **Left out:** acting among new things, and replacing the counted
  model's entries by recall. Both come later.
- **Test colour: yellow.** Yellow keys and doors never occur in
  experience. Yellow does appear on the switch when it is on, and the
  yellow key shares four pixel values with that tile. Purple is reported
  only, because the drafting probes looked at it.

**Arms:**
1. Upper bound: things described by one-hot labels from the renderer
   (object with state, colour), with recall on those.
2. **Main: the encoder trained with replay, and recall; ten seeds.**
3. Trivial baseline: counting only (card 029). Yellow is unknown, so keys
   are tried in random order.
4. Ablation, no replay: card 031's encoder unchanged; λ learned by the
   same objective.
5. Ablation, no relation: recall without the differences (the things' own
   numbers only), trained the same way.

## 3. Dependencies

Passed cards:
- card 010: rule-list evidence;
- cards 028–029: the counted model and subgoals;
- card 031: the encoder, and codes that lose nothing (10 of 10 seeds);
- card 032: the rule for choosing a loss weight.

Methods, in LITERATURE.md:
- the exemplar model and the generalisation law (Nosofsky 1986; Shepard
  1987);
- neighbourhood components analysis (`1604_02354` restates it);
- episodic control (`1703_01988`, `1606_04460`): a slow encoder with fast
  nearest-neighbour memory learns far faster than a network alone;
- the relational bottleneck (`2012_14601` and others): comparisons carry
  to new objects;
- Plotkin, description length and contrasts, from card 030's reading.

## 4. Data check

Toggles at closed doors in the key world, from 5,000 random episodes (card
028's seed). Counts are weighted, with stored tries in brackets.

| Door | Own key (opens) | Wrong key (stays shut) | Nothing held |
|---|---|---|---|
| red | 172 (172) | blue 96 (12), green 136 (17) | 2,776 (347) |
| green | 178 (178) | blue 120 (15), red 144 (18) | 2,912 (364) |
| blue | 213 (213) | red 184 (23), green 232 (29) | 3,088 (386) |

All six wrong-key pairings occur. They are the contrasts that separate
"fits" from "any key opens any door". Memory sizes per world and action are
counted in the gate. The yellow tries' outcomes come from the world's rule
(the door opens exactly when the held key has its colour), applied by the
evaluator.

## 5. Feasibility gate

- **Replay weight μ,** on familiar tiles only. The grid is 0.001, 0.003,
  0.01, 0.03, 0.1, 0.3 and 1; it spans three orders of magnitude because
  card 032's grid, set without checking the loss scale, failed. Take the
  largest weight that keeps all 20 training tiles on distinct tuples, with
  none "new", in all 10 seeds (seeds 100–109). If none qualifies, stop.
- **Upper bound:** arm 1 passes all three criteria.
- **Trivial baseline:** arm 3 opens the yellow door by the second try in
  half the seeds.
- **Drafting probes** (appendix B), on purple:
  - card 031's encoder, unchanged, did not carry the relation;
  - training it with a replay term made the purple key the first choice
    in 8 of 10 seeds, but at weight 1 it merged tiles.

Result of the gate, before the main run (2026-09-29):

- **Upper bound: passed** (after the as-built changes in appendix A). In
  arm 1, "fits" is admitted in the key world (−19.8 nats against −48.1
  pooled and −52.2 per door) and in the either world ("switch on in view"
  or "fits"). It is not admitted in the switch or both worlds. The yellow
  and purple doors open at the first try with all 8 pairs right, the other
  yellow and purple events are right, and no familiar memory is
  overridden.
- **Replay weight: no weight qualifies, so by the declared rule the card
  stops here.** Numbers are in `runs/033_sweep_*.json`, seeds 100–109.
  Without replay, card 031's encoder keeps all 20 tiles distinct, with
  none "new", in 10 of 10 seeds. With replay:

  | Weight | Tiles distinct | No training tile "new" |
  |---|---|---|
  | 0.001 | 10 | 5 |
  | 0.003 | 10 | 4 |
  | 0.01 | 10 | 5 |
  | 0.03 | 10 | 5 |
  | 0.1 | 10 | 1 |
  | 0.3 | 1 | 3 |
  | 1 | 0 | 2 |

  Replay keeps tiles apart up to 0.1. What fails is that it moves some
  tiles' numbers away from every code of a codebook, so they come out
  "new".
- **Diagnostic, not the declared gate:** re-seeding an unused code
  wherever a training tile is "new" (like card 031's re-seeding for
  collisions) did not fix it. It qualified in 3–7 of 10 seeds per weight
  from 0.001 to 0.1, and 0.03 and 0.1 now merged tiles in some seeds
  (`runs/033_sweep_reseed_new_*.json`). Replay pulls the numbers to where
  comparisons work, which is between the codebook's codes. So the encoder's
  two jobs do conflict. Recall itself was not run on learned numbers.
- **Rule changed with the user (2026-09-29), the only change:** the weight
  must keep all 20 training tiles distinct in all 10 seeds. Whether "new"
  tiles lose anything is judged by criterion 2, card 031's full check,
  which is what the "none new" condition was protecting. **Chosen: μ =
  0.1**, the largest weight with distinct tiles in 10 of 10 seeds. At that
  weight 9 of 10 seeds leave some training tile "new". The re-seeding
  diagnostic is off.

## 6. Success criteria and prediction

Each criterion must hold in at least 8 of arm 2's 10 seeds.

1. **The code is created where it pays, and not where it does not.** In
   the key world, "fits" is in the chosen description. In the switch
   world, where the held key does not matter, it is not.
2. **Nothing lost.**
   - The codes of the replay-trained encoder pass card 031's criterion 1 in
     the four familiar worlds: held-out effects exact; the goal reached in
     ≥ 99% of 500 layouts; mean steps within 5% of card 029's.
   - Recall never overrides exact experience: every familiar memory whose
     own tries all had one outcome is predicted as that outcome.
3. **A new colour, mapped quickly.** At the closed yellow door, with four
   keys (yellow, red, green, blue), the agent tries keys in order of
   recall's chance. Each failure is stored at once.
   - Pass: the door opens by the second try. Afterwards, all 8 yellow
     pairs are predicted right: the yellow door with each key or with
     nothing, and the yellow key at each familiar door.
   - Chance order passes a seed half the time, so 8 of 10 has a
     probability of about 0.05.

Also reported:
- tries per seed;
- confidence at the first yellow try against familiar doors (P5);
- the other yellow events: picking up the yellow key, picking it up while
  holding something, dropping it, toggling the open yellow door;
- purple;
- the either and both worlds;
- arms 4 and 5;
- seconds.

**Prediction.**
- Arm 1 passes.
- Criterion 2 is likely: the weight rule protects the codes, and exact
  counts outweigh a vote bounded by the number of memories.
- Criteria 1 and 3 are uncertain. The likeliest failure is that no weight
  both keeps tiles apart and shapes comparisons. That would point to
  giving recall its own part of the encoder's output.
- Arm 5 cannot express "same colour", so it should do no better than
  chance at the yellow door. It should still carry the yellow key's pick
  up.

**Budget.**
- Collecting experience: about 3 minutes.
- The weight sweep: 70 encoders in seven processes, about 3 minutes.
- The main run in three processes: about 12 minutes, dominated by card
  031's check on 10 encoders.
- About 20 minutes in all.

## 7. Result

Run 2026-09-29 with `tools/card033/recall.py` at μ = 0.1. Arm 2 ran in two
processes of five seeds, about 9 minutes each, with 33–35 seconds to train
each encoder. Arms 1, 3, 4 and 5 ran in a third process. Numbers are in
`runs/033_main_a.json` and `runs/033_main_b.json` (arm 2), and in
`runs/033_others.json`. Arm 1 has no seeds. The other columns count seeds
out of 10.

| Criterion | Arm 1, labels | **Arm 2, main** | Arm 3, counting | Arm 4, no replay | Arm 5, no relation |
|---|---|---|---|---|---|
| 1. "fits" admitted in the key world, not in the switch world | pass | **2** | | 0 | 0 |
| 2a. Codes lose nothing (card 031's check) | | **1** | | 10 (card 031) | |
| 2b. No familiar memory overridden | pass | **10** | | 10 | 10 |
| 3. Yellow door opened by the 2nd try | 1st try | **9** | 4 | 3 | 6 |
| 3. ... and then all 8 yellow pairs right | pass | **2** | 0 | 0 | 0 |

Verdict: **fail** on all three criteria. They held in 2, 1 and 2 seeds,
where 8 were needed.

**Criterion 2: replay broke the codes.** Card 031's check passed only in
seed 103, the one seed with no training tile "new". The counted model has
no entry for a tile with a "new" code, so the agent cannot use that tile:
- the goal square was "new" in seeds 100 and 107, and the goal was reached
  in 0.2–1% of layouts, against 100% with card 031's encoder;
- the open doors were "new" in seed 101, which reached the goal in 0.4–4.6%
  of layouts;
- the floor was "new" in seeds 102 and 104, and held-out effects were right
  51% of the time.

Seed 109 left five tiles "new" and still reached the goal in every layout,
in card 029's number of steps. It failed only on exactness: its held-out
effects were right 99.86% of the time. Recall itself never overrode a
familiar memory.

**Criterion 1: the relation was found in 2 seeds.** In seeds 104 and 109,
"fits" held for exactly the three own-key pairings. It was admitted at
−19.8 nats against −48.1 for the pooled rules, as in the label arm. That
happened in the key and either worlds, not in the switch or both worlds.
In the other 8 seeds "fits" held for no pairing, so the evidence chose the
pooled rules (−48.1 against −48.5 with "fits").

**Criterion 3: the door opened, but one try did not correct the model.**
- **Order of tries.** The yellow door opened at the first try in 6 seeds
  and by the second in 9. A chance order opens it by the second try half
  the time, so 9 of 10 has a probability of about 0.01.
- **Low chance even when first.** Recall's chance that the yellow key
  opens the door was low even when that key came first: 0.02–0.37.
- **Still wrong after the success.** Once the success was stored, the
  yellow door with the yellow key was still predicted not to open in 8
  seeds.
- **Why.** At the first yellow try, the similar memories' votes summed to
  a weight of 1.3–13. That is about as much as a familiar door gets with
  its own counts set aside (2.6–11.1). It was at most halved in 6 seeds,
  and larger in 4. A stored success has weight 1 and cannot outweigh it.
- **With labels** the weight was 0.15, against 2.0 for a familiar door, so
  the one success decided the prediction.

So with learned numbers, a new thing does not look new to recall. Seeds
104 and 109, where the relation was found, got all 8 pairs right.

**What similarity carried.** The other yellow events were:
- picking up the yellow key;
- picking it up while holding a red key;
- dropping it;
- closing the open yellow door.

With replay, all four were right in all 10 seeds, both with the relation
(arm 2) and without it (arm 5). Purple gave the same result.

Without replay (arm 4), picking up the yellow key was wrong in every seed.
Card 031's encoder puts yellow things far from everything known. The
weight of the vote at the yellow door was 0–0.7, against 3.1–4.3 for a
familiar door, so the uniform prior decided. Arm 4 tried the yellow key
last in 7 seeds, worse than chance.

Arm 5 cannot express "same colour", as predicted. The yellow door with the
yellow key was wrong in all 10 seeds, and the door opened by the second
try in 6, about chance. Arm 3, counting only, opened it by the second try
in 4.

**Reading.**
- **Recall itself works.** On label codes it:
  - finds the relation;
  - admits it only where it pays;
  - maps the new colour at the first try;
  - doubts the new thing.
- **The learned encoder is what fails.** At the weight that keeps tiles
  apart, replay shaped a comparison that carries "same colour" in only 2
  of 10 seeds. In 9 of 10 it moved familiar tiles off their codes, and
  that broke acting. As predicted, and as the gate's diagnostic showed,
  one output cannot serve both jobs.
- **Replay is what makes similarity carry** what a new thing shares with
  known things. Picking up and dropping the yellow key were right with
  replay and wrong without it.
- **A new thing must also look new to recall.** With learned numbers, the
  yellow door's vote weighed as much as a familiar door's, so one try
  could not correct a wrong answer. With labels it weighed 13 times less.
  The leave-one-out objective sees only familiar memories, so nothing in
  it rewards doubt about new things.

## 8. Decision

**Stop** (2026-09-29, with the user). The user judged that making the
encoder's numbers carry similarity and relations is too delicate a
problem. Here, replay broke the codes and found the relation in 2 of 10
seeds. The codes stay as names of appearances, which card 031 showed
works. Relations are to be found by a mechanism on top of the codes,
either as codes or in a network's weights. What makes two things similar
is to be judged by what they do (GOAL.md P1, P3, P4).

**The direction, as agreed with the user:**
1. **Kinds by what things do.** Two things are the same kind if they play
   the same part in the same events.
   - Exact interchangeability is not enough. The red key and the blue key
     do not do the same thing: each opens a different door. They play the
     same role, though: each is "the thing that opens its matching door".
     So kinds come from the pattern of relations, not from identical
     outcomes.
   - Part of this exists already. Card 027 found kinds by counting what
     actions do, and the counted model already treats all keys alike for
     picking up.
2. **Learn what varies within a kind.** Once three things are known to be
   keys, compare them: the outline is shared and the colour differs.
   Generalise widely along what varies inside a kind, and narrowly along
   what does not (`tenenbaum-griffiths-generalization`, `lake-bpl`; the
   same process teaches children that shape tells what an object is).
   - A new appearance is judged by the features its kind shares. A yellow
     thing with a key's outline is probably a key. Its new colour is no
     reason for doubt, since keys vary in colour anyway.
   - If it matches no kind on those shared features, it is a new kind.
     That gives the principled doubt this card lacked.
3. **Relations between roles.** Once there are keys and doors, ask what
   decides which key opens which door. Compare the pairs that opened: in
   each, what varies within keys (colour) equals what varies within doors.
   That is "fits", found by comparison rather than hoped for in the
   encoder's numbers. The search is small, because only features that vary
   within kinds are candidates.

Refined with the user the same day (CHARTER.md, "Current direction"):
kinds are not merged into hard groups, because that loses information that
later structure may need. A kind is distributed across the codebooks: the
tiles that share codes in the codebooks a rule reads. Recall, not
counting, supplies the similarity.

Recall is essential at every layer. Each layer compares stored
experiences: which things did the same things, what the members of a kind
share, and what the pairs that opened have in common. Counting keeps only
tallies under fixed names, so it cannot redo these comparisons when a new
kind or feature appears.

**The known risk is layer 2.** It needs appearance features in which the
members of a kind agree. A network trained on counted kinds placed
withheld keys and doors by colour and brightness (card 027: 4 of 33
right), and card 031's codebooks did not separate outline from colour.

**Probe of that risk (2026-09-29; fail, as the user expected).**
`tools/card033/kinds_probe.py` (numbers in `runs/033_kinds_probe_*.json`)
retrained card 031's encoder for seeds 100–109, about 10 seconds each, and
kept all 20 tiles distinct and none "new". The kinds were taken from the
evaluator's labels. It asked whether any part of the encoder's output
holds a kind together:
- the four pieces;
- all 32 numbers;
- each codebook's codes.

A part holds a kind together if the kind's members are nearer to each
other than any member is to another tile.
- **Keys and closed doors: never,** in any part or seed. Over all 32
  numbers, the keys lie 1.9–2.6 apart, while the nearest other tile to
  some key lies at 0.87–1.9. The three keys shared a code in one codebook
  of one seed, and nine other tiles shared that code; the yellow key's code
  there was "new".
- **Raw pixels do no better:** the keys lie 4.3 apart, against 2.9 to the
  nearest other tile.
- **What is there is weaker: a nearest neighbour.** Over all 32 numbers,
  the yellow key's nearest familiar tile was a key in 7 of 10 seeds (the
  others: the switch), and the purple key's in 8 (the others: floor). For
  the yellow closed door it was a closed door in 3 of 10 (otherwise wall,
  the switch, and others).
- The only kind held together was "agent in doorway": in one piece of one
  seed, with the yellow version falling in with it.

So the encoder gives no features in which a kind's members agree. Layer 2
needs a source for them. That is the subject of the conversation and
literature review with the user.

## Addendum: the next step if this passes

**Proposed by the user (2026-09-29).** Train a System 1 network on recalled
experience, so that recall runs only on uncertain experiences (P21). This
is counting → recall → weights, with recall as the network's teacher. I
agree, with two conditions:

1. **The trigger for recall must not be the network's confidence alone.**
   Networks are often confidently wrong on inputs unlike their training.
   LESSONS: a network gave the goal square the green door's code. Recall
   should also run when something outside the network signals novelty:
   - a code that is "new" in some codebook;
   - or recall's own confidence: little similar memory.
2. **The network takes over an event type only after it agrees with recall
   on every case, rare ones included** (STATUS; LESSONS: networks drop rare
   events). Recall stays the fallback.

Its test: recall calls saved, with no loss in predictions, and every new
thing still routed to recall.

## Appendix A: the procedure

**Outcome kinds,** from exact tiles at the front and held places, before
and after (as built, four kinds):
- nothing: both unchanged;
- swapped: the front and the held tile traded places (picking up and
  dropping; an empty hand shows as floor);
- in place: the front changed, the held tile did not;
- else: other.

**Memory.** Per world and action (pick up, drop, toggle), each distinct
(front, held) appearance pair keeps weighted counts per kind (storage
weights as card 028).

**Keys.**
- z: card 031's four unit-length pieces of 8, before quantisation.
- An event's key: x = [z(front), z(held), ‖z_k(front) − z_k(held)‖ for
  each piece k] (as built).
- Arm 5: [z(front), z(held)].
- Arm 1: one-hot labels (card 031's arm 1: object with state and agent;
  colour) in place of z, with its two parts (object, colour) as the pieces.
- λ_j = softplus(θ_j), one set per world and action. As built, θ starts
  so that the median distance between two memories of a group is 1.

**Replay term.** Minus the mean over memories (each once) of
Σ_c r_q,c log P₋q(c | q), summed over worlds and actions, times μ. P₋q
leaves out q's own counts and its own vote. Otherwise training is card 031
as built: 5,000 updates, Adam at 0.001, pairs at 0.1 with the adaptive
rule, re-seeding, and "new". In arm 4 the encoder is card 031's (same
seed); θ alone is learned, with Adam at 0.01 for 2,000 steps.

**Evidence.** Card 010's rule-list search, with its costs, runs over the
toggles at doors whose entry has several outcome kinds. It scores three
descriptions:
- per door, over "held = X" and "X in view";
- pooled, with the same atoms;
- pooled, with "fits" added. fits for a try is P₋q(in place | q) ≥ 1/2,
  from the other memories only.

The highest score in nats is chosen. A tie goes to the description
without "fits".

**Criterion 3.** Recall runs over the key world's toggle memories, plus
each yellow try as it happens: either a new memory, or its counts
increased. The order of tries:
- arms 1, 2, 4 and 5: highest P(in place) first, ties in seeded random
  order;
- arm 3: random.

After the door opens, the 8 pairs are predicted, and the most probable kind
must be the true one. The renderer's object list gains yellow keys and
doors at import (`src/` unchanged).

## Appendix B: drafting probes

These ran on 2026-09-29 as scratch runs. Outcomes were built from colour
names; the comparison was trained on red, green and blue and tested on
purple, over 10 seeds.

| Comparison read from | All 8 purple pairs right | Purple key tried first |
|---|---|---|
| Card 031's encoder, learned weights on differences | 0 of 10 | 2 of 10 |
| Card 031's encoder, nearest comparison, unweighted | 1 of 10 | — |
| Encoder trained with a fit term, weight 1 | 0 of 10 | 8 of 10 |
| A separate network (breaks C2; not used) | 8 of 10 | — |

At weight 1, the fit term merged training tiles, or left one "new", in 7
of 10 seeds. A withheld pairing failed in every variant: trained on red
and green, with blue seen only in failures, the comparison learned "blue
never opens". That is why the test is a new colour.

### As built

The code is `tools/card033/recall.py`, on card 031's code (unchanged) and
card 028–029's. Where it differs from the approved plan, all found by the
gate before any learned-code result was seen:

- **The relation is a distance per piece, not a signed difference.** As
  planned, arm 1 (the upper bound, label codes) failed. The yellow door
  opened first, but 4 of the 8 yellow pairs were wrong, and "fits" was not
  admitted in the key world (−48.5 nats against −48.1 pooled). A signed
  difference puts a new colour into dimensions no memory uses. So "yellow
  minus red" sits as far from "red minus red" (same) as from "green minus
  red" (different). A distance within a part is 0 for "same" whatever the
  value. This is the relational bottleneck's own form, a similarity between
  two things. The cost is that orderings are not expressible yet.
- **Four outcome kinds, not five.** Picking up and dropping both swap the
  tile in front with the held tile, so the planned "to the hand" and "to
  the front" matched both, and drops were mislabelled. They are one kind,
  "swapped".
- **θ starts at a data scale.** Starting at θ = 0, every vote is near zero
  and no gradient reaches λ.
- **The replay term is computed for all groups at once** (padded).
  Training an encoder takes about 30 seconds, not 10.

With these, arm 1 passes all three criteria:
- key world: "fits" admitted, with one rule, "opens if fits(held,
  front)", at −19.8 nats against −48.1 pooled and −52.2 per door;
- switch world: not admitted;
- the yellow and purple doors open at the first try, with all 8 pairs
  right;
- no familiar memory is overridden.
