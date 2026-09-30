---
id: "034"
title: recall teaches the encoder
rung: 0
serves: [P3, P1, P10, C5]
status: abandoned
verdict:
arch_version: 5
date: 2026-09-29
---

# 034: recall teaches the encoder

## 1. Question

If card 031's encoder is also trained so that recall can predict each
past event from the other events, do keys and open doors of a colour never
seen share codes with the things that act like them? Then card 029's model
would predict and act on them, and the codes would lose nothing. Serves:
- P3, a known shape in a new colour;
- P1, a thing means what actions do to it;
- P10, retrieving the relevant experience for a new question;
- C5, nothing lost.

This follows [card 033](../033-relation-codes/card.md), which was stopped.
The direction is set out in [CHARTER.md](../../CHARTER.md), under "Current
direction":
- codes are names, and kinds are distributed across the codebooks;
- recall replaces counting, and nothing is merged;
- similarity comes from what things do.

Revised with the user on 2026-09-29. An earlier draft took kinds from
counting and merged them into one codebook, and the user rejected it:
merging loses information that later structure may need.

## 2. What changes

One component: card 031's encoder objective gains card 033's recall term,
with room in the codebooks.

| | Card 031 | Card 034 |
|---|---|---|
| Encoder objective | Rebuild pixels, codebook terms, pairs | The same, plus μ × the recall term |
| Codebooks | 4 of 8 codes | 4 of 16 codes |

- **The recall term** is card 033's, on the things' own numbers only, as in
  card 033's arm 5, with no relation part:
  - each stored event (action, tile in front, held tile, what changed) is
    predicted by a similarity-weighted vote of the other stored events of
    that action and world;
  - the prediction is scored by its log-likelihood and trained through the
    encoder;
  - attention weights per number are learned per world and action, so
    picking up can rely on different numbers than toggling.

  No kind is named and nothing is merged. Tiles come to share numbers, and
  so codes, to the extent that they predict each other's outcomes.
- **Memory:** card 033's pick-ups, drops and toggles, plus forward moves
  (moved, blocked, or reached the goal) (appendix A).
- **Why 16 codes.** Card 033's arm 5 used this term with 8 codes, and
  training tiles came out "new" in 9 of 10 seeds. In the drafting probe,
  "new" tiles came from a codebook with fewer codes than groups of alike
  tiles (appendix B); the same may hold here. Arm 3 checks that 16 codes
  alone change nothing, and arm 4 shows what they change with the term.
- **What stays:** the rest of card 031 as built. Its rules read the fewest
  codebooks that lose no evidence, so kinds stay distributed and specific
  to each action.
- **Declared exceptions** (CHARTER rule 8):
  - the planner and its predictions still run on the counted model over
    codes (architecture version 5). Recall shapes the encoder here, and its
    own predictions are reported. Replacing the counted model with recall
    in the planner is its own card, next if this one passes;
  - what changed is read from exact tiles, as declared in card 033.
- **Left out:** relations such as "fits".
- **Closed doors are not a criterion.** Even in the best case, a
  new-colour closed door did not group with the closed doors on sight
  (0–3 of 10 seeds, appendix B). They are reported, including how many
  tries recall needs to predict them right.

**Test colours:** yellow and purple keys and doors, never in training. The
seeds are 200–209, new for this card.

**Arms:**
1. Upper bound: card 031's label codes (object with state, colour).
2. **Main: the recall term, 16 codes; ten seeds.**
3. Baseline: 16 codes, no recall term.
4. The recall term with 8 codes, the main arm's μ: what the codebook size
   changes.

## 3. Dependencies

- Card 031: the encoder, rules keyed per group, the checks and the test
  worlds.
- Card 033: recall (the vote, leave-one-out attention, memory), which
  passed on labels; its arm 5 is this term with 8 codes.
- Card 032: the rule for choosing a weight on familiar tiles only.
- Cards 028–029: the counted model and subgoals.
- Methods in LITERATURE.md:
  - neighbourhood components analysis (`1604_02354`);
  - Shepard's generalisation law and Nosofsky's exemplar model;
  - episodic control (`1703_01988`).

## 4. Data check

Training is card 031's: four worlds, 5,000 random episodes each. Card 033's
memory holds about 50–55 distinct (tile in front, held tile) pairs per
world for each of pick up, drop and toggle. The door tries are in its
section 4. Forward moves are counted in the gate.

Test: card 031's world (a), the key world with the new-colour door open at
the start, in yellow and in purple, 1,000 random episodes each. Purple
counts, from card 031's section 4:
- picking up the key: 4,035;
- dropping it: 3,959;
- forward onto the open door: 523, the rarest criterion case.

Yellow is counted in the gate.

## 5. Feasibility gate

- **Codebooks:** arm 3 keeps all 20 training tiles on distinct tuples,
  with none "new", in 10 of 10 seeds.
- **Weight μ,** on familiar tiles only. The grid is card 033's: 0.001,
  0.003, 0.01, 0.03, 0.1, 0.3 and 1. Take the largest weight that keeps
  all 20 tiles distinct, with none "new", in all 10 seeds. If none
  qualifies, stop. This is card 033's rule before it was relaxed; the
  relaxed rule let broken codes through.
- **Upper bound:** arm 1 passes criteria 1–3.
- **Trivial baseline:** arm 3 fails criterion 2. Card 031 had the purple
  cases right in 2, 6 and 3 of 10 seeds, with worlds (a) and (b) pooled.

Result of the gate, before the main run (2026-09-29): **failed, so by the
declared rule the card stops here.** Numbers are in `runs/034_collect.json`
and `runs/034_sweep_M16_*.json`, seeds 200–209.

- **Data.** The exact-names reference reproduces card 031: 16.38, 17.27,
  15.47 and 21.77 steps, and 18.4% and 2.8% on worlds (a) and (b). Yellow
  and purple test episodes share layouts. In world (a):
  - picking up the key: 4,060;
  - dropping it: 3,985;
  - forward onto the open door: 503.

  Card 034's memory adds 52–57 forward pairs per world.
- **Codebooks: failed.** With 16 codes and no recall term (arm 3), all 20
  tiles stayed distinct, but in seed 203 the red key was "new" in one
  codebook (9 of 10; 10 needed).
- **Weight: none qualifies.** Seeds out of 10:

  | μ | Tiles distinct | None "new" | Both | Rebuild error |
  |---|---|---|---|---|
  | 0 (arm 3) | 10 | 9 | 9 | 0.0001 |
  | 0.001 | 10 | 7 | 7 | 0.0001 |
  | 0.003 | 9 | 7 | 7 | 0.0001 |
  | 0.01 | 10 | 7 | 7 | 0.0001 |
  | 0.03 | 9 | 4 | 4 | 0.0003 |
  | 0.1 | 1 | 6 | 0 | 0.016 |
  | 0.3 | 0 | 8 | 0 | 0.022 |
  | 1 | 0 | 8 | 0 | 0.023 |

  Up to 0.03, recall pushes some training tiles off their codes. From 0.1
  on, it outweighs rebuilding (the error rises about 150 times) and
  merges tiles.
- **The reason given for 16 codes was wrong.** More codes did not stop
  "new" tiles. Even without recall, 16 codes gave one, where card 031's 8
  gave none. More codes sit closer together, and "new" is judged against
  half the gap to the nearest other code.
- Not run: the main run, arm 1, and the diagnostic.

## 6. Success criteria and prediction

For arm 2, each criterion must hold in at least 8 of 10 seeds, for yellow
and for purple.

1. **Nothing lost.** Card 031's criterion 1 in the four familiar worlds:
   - held-out effects exact;
   - the goal reached in ≥ 99% of 500 layouts;
   - mean steps within 5% of card 029's.
2. **New keys and open doors, predicted.** In world (a), card 029's model
   over the codes predicts each of three cases exactly in ≥ 99% of its
   occurrences (card 031's measure):
   - picking up the new key;
   - dropping it;
   - forward onto the open new-colour door.

   This shows the codes carried the rules to a new appearance.
3. **Acted.** In world (a), the goal is reached in ≥ 98% of 500 layouts,
   with mean steps at most 1.15 times the shortest route's.

Also reported:
- recall's own predictions of these cases, without codes;
- which codebooks the new tiles share with familiar ones;
- every other case of world (a), including closing the open door;
- world (b), the switch world with the door closed;
- arms 1, 3 and 4;
- mutual information of each codebook with the labels;
- seconds.

For the new-colour closed door, reported:
- the tries curve (P19): after 0, 1, 2, 4 and 8 stored tries of each
  action, is recall's prediction right?
- the weight of the vote against the weight of its own tries;
- a diagnostic: arm 2 with 4 times the updates (20,000). Does the door
  then share codes with the closed doors?

**Prediction.**
- Arm 1 passes.
- Arm 3 keeps the codes but fails criterion 2.
- Arm 2: in card 033's arm 5 this term let recall carry the yellow key's
  pick-up and drop in 10 of 10 seeds. Whether the codes carry them too
  depends on keys sharing codes in the codebooks the pick-up rules read.
  I expect criterion 1 to pass with 16 codes, and criteria 2 and 3 in some
  seeds, perhaps not 8.
- For closed doors, I expect them wrong on sight, and right within a few
  tries unless the vote is as heavy as in card 033.

**Budget.**
- Memory and data: about 4 minutes.
- The weight sweep: 70 encoders at about 30 seconds each, in seven
  processes, about 6 minutes.
- The main run: 30 encoders plus the label codes, each with card 031's
  full check, in four processes, about 15 minutes.
- The diagnostic: alongside.
- About 25–30 minutes in all. Past 30, it goes to the user as commands.

## 7. Result

The feasibility gate failed (section 5), so by the card's own rule the
main run, arm 1 and the diagnostic were not made.

| Criterion | Result | Verdict |
|---|---|---|
| Gate: 16 codes keep the tiles (arm 3) | 9 of 10 seeds, against 10 | Fail |
| Gate: a recall weight μ | None keeps all tiles distinct with none "new" in 10 of 10 seeds; the best is 7 of 10 | Fail |
| 1. Nothing lost | Not run | Not assessed |
| 2. New keys and open doors predicted | Not run | Not assessed |
| 3. Acted | Not run | Not assessed |

## 8. Decision

**Stop** (2026-09-29, with the user). Every attempt to push similarity
into the encoder has broken the codes: cards 032, 033 and 034 and the
kinds probe. Three objectives fight inside one vector: the decoder wants
every tile distinct, recall wants tiles that act alike close together,
and the quantiser wants every tile on a code. A split was considered and
rejected: recall on its own output head, or a second set of "functional"
codebooks, with the planner on the appearance codes. That would be a
second representation. The user set a further point of direction instead
(CHARTER.md, "One latent space"). Codes, recall, planner conditions and
relations all read the same vectors, so the remedy is to organise that
space into parts. The three objectives conflict only when the same
dimensions must serve all of them. If what things share (a key's outline)
and what varies among them (colour) sit in different parts, all three can
hold. Card 031's encoder has no such part (appendix B). In this card
recall's per-number weights started equal, and nothing pushed them toward
few numbers, so its pull moved whole vectors. Next is card 035: the same
recall term, with a sparsity penalty on its weights per part, so that each
action's outcome is predicted from as few parts as possible.

## Appendix A: the procedure

**Memory.** Card 033's `build_memory`, per world and action: each distinct
(tile in front, held tile) with weighted counts of what changed (nothing,
swapped, in place, other). Added here, forward: each distinct (tile in
front, held tile) with counts of moved, blocked and reached the goal, read
from the egocentric view as in card 027.

**The recall term.** Card 033's `loo_loglik` in "own" mode:
- an event's key is the front tile's 32 numbers and the held tile's 32;
- the weight is exp(−Σ_j λ_j |x_j − x′_j|);
- λ is learned per world and action, from card 033's starting scale;
- μ × the negative sum is added to card 031's objective in `train_encoder`.

**The encoder.** `tools/card033/recall.py`'s `train_encoder`, with the
codebook size M = 16 (arms 2 and 3) or 8 (arm 4).

**The test worlds.** Card 031's worlds (a) and (b) are made
colour-general: the door and its matching key take the test colour.
Purple stays as built.

**Recall's own predictions.** Card 033's `Recall.predict` with the
trained numbers and λ, for the same cases. For the tries curve, k true
outcomes of the new closed door are added (`Recall.add`) before
predicting.

**Seeds.** Encoders 200–209. Experience and test episodes as in card 031.

## Appendix B: drafting evidence

From 2026-09-29, before this card's seeds.

- **Card 033's arm 5 (this term, 8 codes, μ = 0.1):** recall got the yellow
  key's pick-up, pick-up while holding a key, drop, and the closing of the
  open yellow door right in 10 of 10 seeds. Card 031's encoder alone did
  so in 0, 10, 2 and 6. But tiles stayed distinct in only 7 of 10 seeds,
  with none "new" in 1.
- **The kinds probe** (`tools/card033/kinds_probe.py`, seeds 100–109,
  numbers in `runs/033_kinds_*.json`):
  - card 031's encoder has no part in which the keys, or the closed doors,
    lie nearer each other than to other tiles;
  - with label kinds pulled together in one part (8 codes), new-colour
    keys and open doors joined their kind in 10 of 10 seeds, and closed
    doors in 0–3;
  - the tiles that went "new" were all tiles alone in their kind, in the
    one codebook with fewer codes (8) than kinds (12);
  - adding yellow to training left the purple closed door at 3 of 10.
  This was a best case with labels, used only to find where things break.
