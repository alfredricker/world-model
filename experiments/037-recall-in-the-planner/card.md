---
id: "037"
title: recall in the planner
rung: 0
serves: [P3, P10, P5, P1, C5]
status: done
verdict: fail
arch_version: 5
date: 2026-09-29
---

# 037: recall in the planner

## 1. Question

Suppose the planner's predicted effects come from recall instead of the
counted lookup over codes. Recall is a similarity-weighted vote of stored
tries on the same encoder vectors, and its weight also judges novelty per
action. Does the planner then predict and act on keys and open doors of a
colour never seen, with nothing lost? Serves:
- P3, a known shape in a new colour;
- P10, retrieving the relevant experience for a new question;
- P5, knowing when a prediction applies;
- P1, a thing means what actions do to it;
- C5, nothing lost.

This removes the declared exception of cards 034–035. Card 035 found the
counted lookup cannot use a thing that is like a key but not one of the
known keys, while recall on the same vectors predicted every new case in
10 of 10 seeds. It depends on [card 036](../036-fresh-codes/card.md),
which gives every new tile a name of its own.

## 2. What changes

One component: which stored tries a planner entry is built from.

| | Cards 031–036 | Card 037 |
|---|---|---|
| Tries behind the entry for tile q | Those whose tiles share q's codes in the codebooks chosen by evidence | Every stored try of that action, weighted by recall |
| Weight of a try of tile t | 1 if the codes match, else 0 | 1 if t = q; else k_t / n_t |
| New tile | No entry if it is "new" where the rule looks | An entry if the vote is heavy enough (below) |

- **Recall's weight.** k_t = exp(−Σ_j λ_j |x_q,j − x_t,j|), where x is the
  tile's 32 encoder numbers (64 with the held tile, for drop), and n_t is
  t's own weighted count of tries. So each other tile contributes at most
  k_t in all (card 033's vote).
- **λ is learned per world and per kind of entry** by leave-one-out on the
  stored tiles, with the encoder fixed (card 033's `fit_theta`). The kinds
  are forward, turning, the agent drawn onto a tile and off it, pick up,
  toggle and drop.
- **The entry itself is built as in card 031** from the weighted tries:
  outcome categories stated per codebook (unchanged, taken from the other
  changed place, set to a code), the default, and card 010's rules where
  there are several outcomes. For a familiar tile with many tries its own
  tries dominate, and the entry is its counted one.
- **Novelty, per action.** With the prior of card 033, an outcome's
  probability is (own + Σ k r + π) / (own + Σ k + 1). The entry holds an
  outcome only if its probability is at least one half, the planner's
  declared threshold. With no tries of its own, that needs a vote weight
  Σ k of at least 0.5. Below that, the tile is new for that action and the
  entry is empty. No extra threshold is tuned.
- **Tiles an outcome creates** (an opened new door, the agent drawn onto
  a new open door) get vectors piece by piece. Unchanged pieces keep the
  tile's own piece, taken pieces come from the other place, and set pieces
  take the code's vector (for a fresh code, the mean of its pieces).
- **Encoders and codes:** card 035's recipe on new seeds 400–409 (recall
  term, μ = 0.01, 8 codes). The novelty rule has α re-chosen by
  leave-one-out on familiar tiles. Card 036's fresh codes.
- **Declared exceptions** (CHARTER rule 8):
  - card 010's rules over held and in-view tiles still decide between
    several outcomes. Relations such as "fits" are later cards;
  - the planner does not add its own tries while acting. The tries curve
    is reported from recall's memory, as in cards 034–035;
  - outcomes are read from exact tiles, as declared in card 033.

**Test colours:** yellow and purple, never in training.

**Arms:**
1. Upper bound: card 031's label codes.
2. **Main: recall entries.**
3. Baseline: card 031's counted entries on the same codes.
4. Recall with λ fixed at its starting value, equal on every number: does
   the per-action metric matter?

## 3. Dependencies

- Cards 031 and 035: the encoder, the codes, the novelty rule and the
  entries. Card 036: fresh codes (must pass first).
- Card 033: recall, its prior and leave-one-out λ. Cards 028–029: the
  planner.
- Methods in LITERATURE.md: episodic control (`1703_01988`), the
  exemplar model with learned attention (Nosofsky; Kruschke's ALCOVE),
  neighbourhood components analysis (`1604_02354`).

## 4. Data check

As in card 035: in world (a), each new colour, picking up the key 4,060
times, dropping it 3,985 and forward onto the open door 503.

## 5. Feasibility gate

- **Names:** card 036's criterion 1 holds on seeds 400–409.
- **λ:** the leave-one-out fit gives, on familiar tiles, a higher
  log-likelihood than λ at its starting value, for every world and kind.
- **Upper bound:** arm 1 passes criteria 1–3.
- **Trivial baseline:** arm 3 fails criterion 2 (card 035: 0 of 10).

Names (2026-09-29, `runs/037_gate.json`): passed. Leave-one-out on
familiar tiles chose α = 4 (796 right of 800 tile-codebook readings; α = 6
gave 794). All 28 tiles were named apart in 9 of 10 seeds. In seed 407
the purple key took the red key's codes.

## 6. Success criteria and prediction

For arm 2, each criterion in at least 8 of 10 seeds, yellow and purple.

1. **Nothing lost.** Card 031's criterion 1 in the four familiar worlds:
   held-out effects exact, the goal reached in ≥ 99% of 500 layouts, mean
   steps within 5% of card 029's.
2. **New keys and open doors predicted.** In world (a), picking up the
   new key, dropping it and forward onto the open new door are each
   predicted exactly in ≥ 99% of their occurrences.
3. **Acted.** In world (a), the goal is reached in ≥ 98% of 500 layouts,
   with mean steps at most 1.15 times the shortest route's.

Also reported: each new tile's vote weight per action, and which tiles
carry it; the new closed door toggled while holding its key (not a
criterion, since "fits" is a later card) and recall's tries curve for it;
every other case of world (a); world (b); arms 1, 3 and 4; seconds.

**Prediction.**
- Criterion 1 passes. Familiar tiles have thousands of their own tries.
- Criterion 2: pick up and drop pass (card 035: recall right in 10 of
  10). Forward onto the open door also needs the drawn tile: the agent on
  the new open door, built per codebook, must equal the tile actually
  seen. That is the riskiest part.
- Criterion 3 follows if forward onto the open door is predicted.

**Budget.** Encoders: 10 at about 45 seconds, 2 minutes in five
processes. Entries: seconds per world. Checks: 31 at about 80 seconds,
about 11 minutes in four processes. About 15 minutes.

## 7. Result

Numbers are in `results.json`, built by `tools/card037/summarise.py`
from `runs/037_*`. Run on 2026-09-29, after the changes listed at the
end of Appendix A.

**Gate.** Passed.
- Names: 9 of 10 seeds (section 5).
- λ: the fitted metric beat its starting value in every world and every
  kind with more than one outcome, in all 9 seeds whose model was built.
- Upper bound: arm 1 passed criteria 1–3.
- Trivial baseline: arm 3 failed criterion 2 in 10 of 10.

| Criterion (seeds of 10) | Arm 2: recall | Arm 3: counted | Arm 4: λ at start | Verdict |
|---|---|---|---|---|
| 1. Nothing lost | 9 | 6 | 0 of 4 run | Pass |
| 2. New key and open door predicted | 2 | 0 | 0 of 4 run | **Fail** |
| 3. Acted | 8 | 2 | 0 of 4 run | Pass |

Seed 405 counts as a failure on every criterion of arm 2: its model
needed more than the planner's 254 tuples.

**What transferred.** The counts below are over the 9 seeds whose model
was built, per colour. Arm 3's counts are over 10 seeds.
- Picking up and dropping the new key were exact in 8 of 9 seeds for
  each colour. Arm 3 managed 0 to 2 of 10.
- Walking into the new key or the closed new door (blocked) was exact in
  9 of 9, as it was for arm 3.
- In the familiar worlds nothing was lost in any built model. Arm 3 lost
  familiar predictions in 3 seeds (400, 401 and 405): the agent standing
  in a red or blue doorway came out wrong. In seed 407 its predictions
  were exact, but it reached the goal in only 64–73% of the layouts in
  three worlds.

**What did not: forward onto the open new door**, exact in 3 of 9 seeds
for each colour (arm 3: 2 of 10). Recall's own prediction on the same
vectors (card 033's report) said the agent moves onto it in 9 of 9. So
the vote was right, and the loss is in turning it into the planner's
tile. The 12 failing cases (colour and seed):
- **5: the vote was too light.** The "draw" kind (the agent drawn onto
  the tile in front) has only 5 stored tiles to fit its metric on. Its
  vote for the new open door was 0.02 to 0.41, below the 0.5 needed, so
  there was no entry.
- **7: the vote was enough, but the tile built was not the tile seen.**
  Five were made-up tuples and two left the tile unchanged. What decides
  it is the codebook where the agent in the new doorway differs from the
  new open door. An outcome stated from familiar tries can keep a piece,
  take one from the other place, or set a code that a familiar doorway
  has. Where the new doorway's code there was also a familiar doorway's
  code, the tile was exact in 5 of 6 cases with enough vote. Where it was
  a code of its own (card 036's fresh codes), it was exact in 1 of 7.

**Acting.** 100% of 500 layouts in all 9 built seeds. Mean steps were
1.05 times the shortest route in 8 seeds and 1.23 in seed 401. It passed
even where forward onto the open door was mispredicted. The door starts
open in world (a), and after each step the planner replans from what it
sees.

**Also reported.**
- Toggling the open new door (closing it) was never predicted (0 of 18
  colour-seed cases). Toggling the closed new door while holding its key
  was never predicted either (not a criterion; arm 1 also misses it).
  - Closing the open new door was predicted as no change in 14 of the 18
    cases and as a made-up tuple in 4. The cause is the same statement
    problem as above. Each familiar door closes to its own colour's code,
    so the doors' tries have different statements and do not pool.
  - World (b) needs toggling the closed new door with the switch on. The
    goal was reached in 5.6% of layouts in every seed (arm 3: 2.8–5.6%;
    arm 1: 100%).
- Recall's tries curve on the new closed door, toggled holding its key,
  was right after 0 to 8 tries (median 4). Cards 034–035 needed 1 to 4.
- Tuples per seed: 56 to 199 in the 9 built seeds, 405 over 254. Seconds
  to build a world's model: 3 to 137, mean 15.
- The first launch of arm 2 (Appendix A) gave the same verdicts on seeds
  400 and 401 as the rerun.
- **Arm 4 (λ at its start, equal on every number) was stopped at 25
  minutes**, under the 30-minute rule.
  - Seeds 405–408 all needed more than 254 tuples.
  - Seed 400 had not built its first world's model after 25 minutes.
  - Seed 409 was still building, and 401–404 were not reached.
  - So the fitted metric matters. With it, 9 of 10 seeds fit, in 3 to 137
    seconds per world. Without it, none of the 4 finished seeds fit, and
    seed 400's first world did not finish in 25 minutes. The likely reason,
    not measured here: the fit sharpens λ, so votes reach fewer stored
    tiles, and entries carry fewer outcomes and create fewer tuples.
  - To finish arm 4, the command (well over 30 minutes) is
    `bin/prun python tools/card037/recall_planner.py --main --arm 4 --seeds 400-404 --out runs/037_arm4_a.json`,
    and the same with `--seeds 409-409`.

## 8. Decision

**Stop** (agreed with the user, 2026-09-30). Criterion 2 failed: 2 of 10
seeds, against 8 needed. Criteria 1 and 3 passed.

What the card settles:
- **Recall is the better source for the planner's entries.** On the same
  codes it lost nothing familiar in 9 of 10 seeds, against 6 of 10 for
  counting. It carried picking up and dropping to new keys in 8 of 9
  seeds per colour, against 0 to 2 of 10.
- **The fitted metric is needed** (arm 4).
- **The failure is not in recall but in the planner's tile names.**
  Recall's vote said the agent can step onto the new open door in 9 of 9
  seeds. But the planner must name the exact tile the step makes: the
  agent in the new doorway. Usually that tile has a code of its own in
  the codebook where it differs from the door, and an outcome stated from
  familiar tries cannot produce one. With enough vote, the step was exact
  in 5 of 6 cases where that code was a familiar doorway's, and in 1 of 7
  where it was not. The same kind of statement, where each door colour closes
  to its own code, is why closing the new door was never predicted. It is
  also why world (b) failed.

What that points to, in the one latent space:
- **(a) Parts in the encoder.** If the agent's part of a tile did not
  carry the door's colour, the agent in a new doorway would be the new
  door's colour pieces plus the familiar agent piece. The statement
  "set the agent's part" would then give the right tile. This is
  CHARTER's "organise the space into parts".
- **(b) Outcomes as vectors.** The planner would predict a tile's pieces
  as vectors, from the vectors of the things involved. It would compare
  them with what it sees by distance, not by exact codes. This is a
  larger change to the planner.

The user chose (b), tested in
[card 038](../038-planning-on-vectors/card.md). The architecture stays at
version 5, and recall-weighted entries are not adopted.

## Appendix A: the procedure

`tools/card037/recall_planner.py` replaces `entry_for` in card 031's
`pooled_model` and keeps the rest.
- For a key q (a tuple id, or a pair for drop), it computes k_t for every
  stored key t of the kind from the vectors of the tuples. Real tiles use
  their encoder pieces; created tuples use the piecewise vectors of
  section 2.
- It builds the weighted set of tries: q's own tries at weight 1, t's
  tries scaled by k_t / n_t.
- It builds card 031's categories and rules over that set, with the
  weights passed to card 010's evidence finder, and adds the prior when
  choosing the entry.

λ is fitted by leave-one-out on each kind's distinct keys, as card 033's
`fit_theta` does. The outcome of a key is its per-codebook statement
(below). λ starts equal on every number, at one over the median distance
between keys. The fit's prior is uniform over the kind's outcomes. The
λ gate is read from arm 2's own model reports. The turning kinds have a
single outcome, so their log-likelihood is 0 before and after; they are
left out of that check.

Details fixed before the main run (2026-09-29):
- **Outcomes are stated per stored tile.** Each tile's tries are stated
  as card 031 states them for that tile alone: per changed place and
  codebook, unchanged, taken from the other place, or set to a code.
  Tries of different tiles that have the same statement are the same
  outcome and pool.
  - Why: my first version stated the pooled tries all together. Where
    tiles disagreed in a codebook, it fell back to each try's own
    reading. A switch's tries where the held tile happened to share the
    new code then read "taken from the held tile", and that produced a
    second, wrong outcome. It showed on a spare seed (399) in the familiar
    worlds only: toggle exact in about 98.3% (a switch toggled with a key
    in hand). After the change all four familiar worlds were exact. The
    new-colour test worlds were not run on that seed.
- **Prior** π = 1/4 per outcome for entries: with no own tries, an
  outcome needs a vote of 0.5 (section 2).
- **Small votes and outcomes.** Tiles whose weight k_t is below 0.01 are
  left out of an entry, and so are outcomes with under 1% of the entry's
  weight. These are fixed, not tuned. With thousands of its own tries, a
  familiar tile's entry is its own.
- **One set of vectors per seed.** The four worlds of a seed share one
  set of tuples, so a tuple created in one world keeps its vector in the
  next.

Changed during the run (2026-09-29), after the first launch of arm 2
stopped on seed 405. The planner holds at most 254 distinct code tuples
(card 028's arrays are 8-bit), and recall's entries created more:
- **What happened.** Recall gives an entry wherever the vote is heavy
  enough, including to tuples that outcomes create. Their outcomes create
  further tuples. Most were hybrids:
  - the familiar data cannot tell "unchanged" from "taken from the other
    place" in a codebook where the floor and the keys share a code;
  - card 031's statement then reads "unchanged";
  - on a tile with a fresh code there, the outcome keeps the fresh code
    and changes the rest.
- **The first launch** ran arm 2 on seeds 400 and 401 before it was
  stopped. Seed 400: criteria 1 and 3 passed, and criterion 2 passed for
  yellow but not purple. Seed 401: criterion 1 passed, 2 and 3 failed.
  Those runs are set aside in `runs/037_arm2_first_try*`, and every
  seed was rerun with the rules below.
- **Rules added, the same for arms 2 and 4:**
  - tile codes the view never shows get no entries;
  - drop entries are made only for things that can be in the hand: seen
    held in a stored try, or put in the hand by an outcome;
  - a tuple created from real tiles gets entries as section 2 says, but a
    tuple created from a created tuple gets none.
- **What they did.** In the familiar worlds only, these rules left all
  four worlds exact in 9 of 10 seeds, with 56–199 tuples. Seed 405 still
  needs more than 254 tuples. A seed whose model cannot be held counts
  as failing all three criteria.
