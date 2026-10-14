---
id: "040"
title: relations from attention over the things present
rung: 0
serves: [P3, P12, P4, C1, C2]
status: abandoned
verdict:
arch_version: 5
date: 2026-09-30
---

# 040: relations from attention over the things present

## 1. Question

Whether a key fits a door is a relation between two things. Recall's key
has no relation in it, so the fit gives the held thing no weight (0.0–0.1,
card 038) and imagined pickups get the world's base rate (card 039's
check: 20 of 40 in the either world). Suppose recall's key gains relations
computed as a transformer computes attention: every thing present scored
against every other, in several heads, with projections learned from
outcomes. Does the planner then act where the held thing must fit the
door? Does it open a door of a colour never seen with the key of that
colour? And which score carries to new things: a transformer's dot
product, or a distance? Before rung 1, toward rung 6. Serves:
- P3, a relation applied to new participants;
- P12, conditions read from memory;
- P4, learning which respects matter;
- C1, nothing named;
- C2, the same encoder.

## 2. What changes

One component: a relation part in recall's key for pick up, toggle and drop
(the user's proposal, 2026-09-30: attention over the slots, learned).

```
card 039: key = (front, held, view as a set of things)
card 040: key = (front, held, view as a set of things, relations)
  tokens    a thing is one token (its encoder vector) or 64 (its parts: the encoder's first-layer
            features at each pixel)
  score     per head h, token p of thing a against token q of thing b:
              dot   (Wq_h p)·(Wk_h q) / √r          a transformer's score
              dist  −|W_h p − W_h q|² / √r          the same with one projection and the norms
  relation  r_h(a, b) = over a's tokens, the mean log-mean-exp of their scores with b's (attention
            pooled to one number), averaged with r_h(b, a)
  key part  per head: front–held; front–in view, held–in view (best match); in view–in view (best pair)
  distance  recall's distance + Σ λ_r |R − R′|
```

- **Attention over every thing present,** two heads, rank 32. Only the
  scores pass on, never one thing's features mixed into another's (the
  relational bottleneck). Not slot attention (Locatello et al.): things
  are tiles; forming things in richer worlds comes later (the user,
  2026-09-30).
- **Learned, not named.** The projections and λ_r are fitted per action
  with recall's other weights, by card 039's objective (leave out each
  (front, held) pair, predict its outcomes). A relation gains weight only
  by predicting pairs never tried together. No table: relations are
  computed for any two things, new ones included.
- **The projections are recall's metric, not a second predictor** (C2,
  CHARTER rule 8): their output is a part of recall's key, never a
  prediction. Parts are read from the existing encoder's first layer; a
  thing no tile shows gets the decoder's picture of its vector.
- **For acting (built after the check):** the planner uses stored
  successes with their relations, so a condition reads "hold something
  related to this door as that key was to its door". Otherwise a
  new-colour door has no condition, since every stored held thing fails
  with it.
- **Unchanged:** card 039's set view, the Dirichlet prior, conditions from
  memory, walking, arm B's encoders (seeds 400–404).
- **Declared:** the roles (front, held, in view) as view places, after
  Pasula's deictic references; the pooling; two heads; rank 32.

**Arms** (same planner; the score and the tokens vary): oracle vectors
(arm 1; parts are its four one-hot pieces) and arm B's encoders, each with
dot or dist, vector or part tokens. The gate picks the score and tokens
for the main run; arm 1 is gate only.

## 3. Dependencies

- Card 039's set view (gate check 40 of 40 in the switch world; approved,
  main run not done). Card 038's recall, planner and arm B's encoders.
- Methods (LITERATURE.md): comparisons only, transfer to new objects
  (`the-relational-bottleneck-as-an-inductive-bias-for-efficient`,
  `2012_14601`); roles relative to the action (`1110_2211`); relations as
  alignments of parts (`gentner-structure-mapping`); set matching (chamfer,
  Barrow et al. 1977, not in papi; `1703_06114`); a metric fitted by
  leaving out (`1604_02354`).

## 4. Data check

Stored toggles of a closed door per world (card 028's data): holding the
matching key 519–775 tries, a wrong key 678–1,080, nothing 7,291–9,416.
The wrong key is the contrast that identifies the respect (LITERATURE.md,
2026-09-28). World (c), card 031's key world with the new-colour door
closed, has no test data yet; it is built as card 034 built (a) and (b).

## 5. Feasibility gate

- **Upper bound:** arm 1 passes criteria 1–3.
- **Trivial baseline:** card 039 without relations, 20 of 40 in its
  either-world check; card 038's arm B, goal in 33–66% of layouts in the
  switch, either and both worlds.
- **Mechanism check** (before acting, seed 400): criterion 3's measure and
  criterion 2's predictions, on arms 1, B and P.
- **Weights:** toggle's relation weight rises from its start. The held
  thing's identity weight is reported.
- **Shakedown:** spare seed 399, 100 familiar layouts per world.

**Mechanism check, round 1** (2026-09-30, seed 400, 10 layouts per case,
`attention_planner.py --check`, `runs/040_check_*.json`). Toggling the
closed door before acting. The new colours are card 031's world (c):

| Vectors, tokens, score | Familiar (key, either, both) | Yellow: own key, wrong key | Purple: own key, wrong key |
|---|---|---|---|
| oracle, vector, dot | all right | 0, 10 of 10 | 0, 0 |
| oracle, vector, dist | all right | 10, 6 | 0 (pickup not predicted), 10 |
| oracle, parts, dist | all right | 0, 0 (opens whatever is held) | 0, 10 |
| B, vector, dot | all right | 10, 4 | 10, 4 |
| B, vector, dist | all right | 0, 10 | 10, 8 |
| B, parts, dot | all right | 0 (pickup), 8 | 0 (pickup), 10 |
| B, parts, dist | all right | 0 (pickup), 10 | 0, 10 |

- **Relations fix the familiar worlds in every setting.** Card 039's
  either world went from 20 of 40 to all right; the both world is right
  too. On arm B the held thing's identity weight falls to 0.01–0.04, and
  the front–held relation carries the weight.
- **No setting carries to both new colours.** The dot product gives "any
  key opens" or nothing, even on the oracle. Relations of a thing never
  seen take values never seen: unbounded scores push it away from memory,
  so even picking up a new key goes unpredicted. And leaving out one
  (front, held) pair never asks the weights to predict a thing never
  tried. Round 2 tests both: a bounded relation (exp of the pooled
  distance, sigmoid of the dot product), and leaving out every key with
  the same thing in front.

**Round 2–3** (key world, `runs/040_check2_*`, `040_check3_*`). Familiar
cases stay all right in every setting. Best for new colours: arm B,
vector tokens, distance, bounded, leaving out every key with the same
thing in front. Own key opens 20 of 20, but a wrong key also opens in 12
of 20 (purple 10 of 10). Parts tokens fail completely once bounded or held
out by thing. Dropping the held thing's identity (a strict bottleneck)
changes little.

**Why:** the learned front–held relation is not "same colour". On the
oracle all nine familiar key–door pairs score 5.97–6.23 ("a key is
held"); on arm B the matching pair is often not the highest. The data let
other explanations predict left-out tries equally well. One is the
layouts: whenever a wrong key is held, the matching key still lies in
view. Nothing in the objective prefers the one that carries (LESSONS.md:
table and rule fit equally). Next candidates:
- a cost per relation used (card 010);
- data that breaks the confound;
- more training colours.

**Round 4: four training colours, roles as tokens** (agreed with the user,
2026-09-30; `tools/card040/four_colours.py`, data `runs/040_data.pkl`, with
grey keys and doors added and yellow and purple still unseen). Every thing
is a token: its vector plus its place (ahead, hand, elsewhere in view).
All pairs are scored (bounded distance), and learned queries pick the
pairs. Nothing declares which pairs matter; all four queries chose
ahead–hand.
- Familiar cases: all right on both arms, in all three worlds.
- **Oracle, yellow: all right** (own key opens 10 of 10; wrong key and
  nothing stay shut). This is the first new colour fully carried.
- Oracle, purple: the pickup of the purple key is not predicted (10 of
  10).
- **Arm B: no prediction for a new door with its key.** The nearest stored
  try is 5.3–5.7 away, almost all of it from the relations part (identity
  parts 0.01–0.07). Arm B's vectors put a new matching pair where no
  familiar pair lies.

**Round 5: the same with first-layer parts** (`runs/040_check5_*`, key
world). Familiar cases are all right.
- Oracle: the queries chose ahead–view ("a key of the door's colour still
  lies in view") with the held identity (λ 6.65). New doors with their own
  key: 0 of 10.
- Arm B: new-colour doors are predicted to open whatever is held,
  including nothing.

So parts do not help here, and the confound still shows: in every layout
the matching key lies somewhere.

**Where this leaves the card.** Every form of comparison tried on arm B's
encoder fails to make a new matching pair look like a familiar one. The
comparisons were tile vectors (per-dimension weights, offsets, learned
projections, dot or distance) and first-layer parts. Raw pixels do (5 of
5 colours, drafting check), and so do oracle vectors with four colours
(yellow fully). By CHARTER rule 2 the failure lies in the encoder's
representation: trained to rebuild pixels from three or four colours, it
does not keep colour as something keys and doors share.

**Paused by the user (2026-09-30):** transfer to new colours is set aside,
to revisit in a richer world. The focus is familiar worlds, then movement
through conditions.

**Acting shakedown** (`runs/040_dev_acting.json`, three-colour data, arm
B, spare seed 399, 30 layouts per world, tokens with vector, distance,
bounded, holdout by front):

| World | Goal | Mean steps | Seconds per layout |
|---|---|---|---|
| key | 30 of 30 | 16.1 | 1.4 |
| either | 30 of 30 | 16.1 | 2.0 |
| both | 12 of 30 | 80.2 | 19.2 |
| switch | 0 of 30 | none | 57.0 |

On the same seed, the check is right in every switch-world and both-world
case. So recall predicts correctly and the fault lies in how the planner
reads conditions. With relations, toggle's front weight falls from about
38 (card 039) to 0.1–0.2. The planner still picks "tries on things like
this one" by the front part alone (`templates`, k ≥ 0.01), so nearly
every stored try qualifies.

## 6. Success criteria and prediction

Arm B, seeds 400–404; each criterion in at least 4 of 5 seeds.

1. **Nothing lost, familiar worlds acted.** In the key, switch, either
   and both worlds, the goal in ≥ 99% of 500 layouts with mean steps
   within 5% of card 029's; held-out effects at most 0.1 points below card
   038's arm B on the same seed; in world (a), card 038's criteria 2 and 3.
2. **A relation applied to new participants (P3).** In world (c), yellow
   and purple, from memory before acting: toggling the closed new door
   holding its key predicted to open, and holding a wrong key or nothing
   predicted to stay shut (each ≥ 99% of occurrences). Acting: the goal in
   ≥ 98% of 500 layouts at ≤ 1.15 times the shortest route. Card 031
   excluded this case: rules per appearance cannot reach it.
3. **The comparison does what is claimed.** Toggling a closed door holding
   nothing, in the view as it is and after each imagined change (each key
   picked up, the switch turned on), predicted as the world's rule gives
   it (the evaluator's) in ≥ 99% of cases. 50 layouts per world, in the
   four familiar worlds.

**Prediction.** Arm 1 passes all three. Arm B passes criteria 1 and 3
(with fitted weights the matching pair was nearest in 9 of 9 left-out
familiar cases, Appendix A). Criterion 2 is the risk: purple likely (3 of
3), yellow likely to fail in at least 2 of 5 seeds (1 of 3), because
weights fitted on three colours tune to the differences those three show.
If B fails yellow and P passes, the failure lies in the encoder's parts,
and the next card trains the parts to keep what pixels share rather than
adding to the comparison.

**Budget.** Building first: relations, stored successes with their
relations, world (c)'s data. The fit's cost is measured in the shakedown.
I guess about 15 minutes per seed, so the main run (about 1.5 hours) goes
to the user.

## 7. Result

No main run; the card was stopped after its checks and acting shakedown
(section 5). What the checks showed:
- relations in recall's key made every familiar prediction right, in
  every setting tried;
- acting with them reached the goal in 30 of 30 layouts in the key and
  either worlds, but 0 of 30 in switch and 12 of 30 in both. The planner
  picks a condition's stored tries by the weight on the thing in front,
  and relations shrank that weight to 0.1–0.2. Two planner patches tried
  on 2026-09-30 did not fix the switch world and were removed;
- dot-product scores never carried to a new colour, even on one-hot
  vectors (appendix A). Distance scores carried on oracle vectors, but
  arm B's encoder does not keep colour shared across shapes, so a new
  door's relation values were unlike any stored one.

## 8. Decision

**Stop** (the user, 2026-09-30). Relations were meant for transfer to new
colours, which the user paused until a richer world; in the familiar
worlds they needed planner patches that still failed the switch world.
What the vector planner lacked was identity, not relations: with card
039's planner and no relations, recall predicted a held key at a door
right in 34% of key-world cases, because one weighting per action cannot
make colour count at doors alone. Card 042 gives recall that identity
from the codes. Kept for later: relations must be computed for any two
things; dot-product scores are tables; relations go on top of the codes
(CHARTER, "Current direction"). The code stays in `tools/card040/` as the
record.

## Appendix A: why the parts (drafting checks, 2026-09-30)

Evaluator probes on tile vectors, before any design choice. Each number
is the matching pair's distance over the nearest wrong pair that shares
its key or its door (below 1 is right). Familiar colours are red, green
and blue; new ones are yellow and purple. Scripts are in the session
scratchpad; nothing here is the agent's method.

| Comparison | Oracle | Arm B, seeds 400–404 |
|---|---|---|
| Tile vectors, per-dimension weights fitted on familiar colours (labels) | 0.66 on new colours | in-sample familiar: wrong for 1–2 of 3 colours per seed; new: right 2 of 10 |
| Tile vectors, offset key → door shared across colours (analogy) | 0.0, right 5 of 5 | offsets differ by 1.5–3.4 across colours (keys by 0.9–2.0); analogy right 0 of 10 new |
| Parts: raw pixels, nothing fitted | | 0.46–0.74, right 5 of 5 colours |
| Parts: first layer (64 × 32), nothing fitted (seeds 400, 401) | | right 7 of 10 (0.78–1.06) |
| Parts: second layer (16 × 32), nothing fitted (seeds 400, 401) | | right 4 of 10 |
| Parts: first layer, channel weights fitted on familiar colours (seeds 400–402) | | one familiar colour left out: right 9 of 9 (0.70–0.92); new: purple 3 of 3 (0.90–0.91), yellow 1 of 3 (0.95, 1.00, 1.20) |
| Tile vectors, transformer-style scores (Wq z)·(Wk z), shared or separate, rank 4 or 32, fitted on familiar colours | familiar fitted 100%; one familiar colour left out 0 of 9; new 0 of 6 | the same: left out 0 of 9 in every seed; new 0–3 of 6 |

A dot-product score learns which values go with which: a table over
feature pairs. It fits the three familiar colours and carries to none,
even on the oracle vectors, where a value never seen gets no learned
entry. A distance along learned respects ("the same here") does not
depend on the value, so it carries (oracle, first row).

The relation is in the pixels and is lost as the encoder mixes. A key and
its door share no exact pixel value (LITERATURE.md, 2026-09-28), so the
match must be graded, as chamfer matching is.

## Appendix B: how this scales

Each extension reuses the same three things (roles, the matching of
parts, learned weights) and none is built here:
- several respects per rule: more than one weighting per action, admitted
  by evidence (card 010);
- order (tier above, more than): signed comparisons;
- space (left of, adjacent): the same matching on where things are, the
  movement cards' vectors;
- relations between relations: comparing relation values across
  situations (key : door :: pickaxe : ore);
- quantities: a set's size, now that the view is a set.

Pairs grow with the square of the number of things. Roles from events and
the evidence for each relation limit which pairs are formed.
