---
id: "035"
title: novelty from a code's own tiles
rung: 0
serves: [P5, P3, P1, C5]
status: done
verdict: fail
arch_version: 5
date: 2026-09-29
---

# 035: novelty from a code's own tiles

## 1. Question

Card 034's recall-trained codes are read with a new novelty rule: a tile
is "new" in a codebook only when it lies farther from its nearest code
than that code's own training tiles do. Then do the codes stay intact,
carry the rules for keys and open doors to a colour never seen, and still
mark that colour as new? Serves:
- P5, a sense of when a prediction applies;
- P3, a known shape in a new colour;
- P1, a thing means what actions do to it;
- C5, nothing lost.

The user approved this card on 2026-09-29, with the instruction to get
the novelty signal working. It was first drafted as a sparsity penalty
on recall's weights. The drafting probe (appendix B1) placed the failure
of cards 034 and 035 elsewhere, in card 031's novelty rule (CHARTER rule
2). With recall's pull, training tiles that share a code spread wider
than half the gap to the next code, and the rule flagged them "new",
often a whole colour in one codebook. So the one component became that
rule. Direction: CHARTER.md's "Current direction", including "One latent
space". A code is still a region of the same vectors; only where the
region ends changes.

## 2. What changes

One component: how a piece of a tile's vector is read as a code.

| | Cards 031–034 | Card 035 |
|---|---|---|
| The code | Nearest used code | Nearest used code |
| "New" when farther than | Half the distance from that code to the next used code | α × the code's radius |
| The code's radius | None | Its farthest training piece, at least the codebook's median distance of training pieces to their codes |

- A used code is one that some training piece is nearest to. The median
  floor serves codes with one or two training tiles, whose own spread
  says little.
- **α is set on familiar tiles by leave-one-out.** Each training piece is
  left out in turn, and radii are taken from the others. Where another
  training tile shares its code, it should get that code. Where none
  does, its value was never seen, so it should be "new". α is the value
  in {1, 1.5, 2, 3, 4, 6, 8} that judges most of these cases right, over
  the gate's ten seeds; ties go to the smaller.
- With α ≥ 1 no training piece is "new". So cards 032–034's check that
  none is "new" now holds by construction; the gate checks that the 20
  tiles stay distinct.
- **The encoder** is card 034's: its recall term, 8 codes per codebook
  (card 031's size), weight μ from the gate. There is no sparsity
  penalty: recall alone already grouped new keys with keys (appendix B1).
- **Declared exceptions** (CHARTER rule 8), as in card 034: the planner
  runs on the counted model over codes, and outcomes are read from exact
  tiles. Recall replaces the counted model in the planner in its own
  card.
- **Left out:** "fits"; the sparsity penalty; judging novelty by recall's
  vote weight.

**Test colours:** yellow and purple keys and doors, never in training.
The seeds are 300–309, new for this card.

**Arms:**
1. Upper bound: card 031's label codes (object with state, colour).
2. **Main: the recall term, the new rule; ten seeds.**
3. Baseline: no recall term, the new rule and arm 2's α.
4. Arm 2's encoders read with card 031's rule: what the rule changes.

## 3. Dependencies

- Card 031: the encoder, rules keyed per group, the checks and the test
  worlds. Card 033: recall. Card 034: its memory with forward moves, its
  priors and the colour-general test worlds.
- Cards 028–029: the counted model and subgoals.
- Methods in LITERATURE.md: prototypical networks (`1703_05175`), where
  distance to a class's prototype is the doubt; leave-one-out selection
  (`1604_02354`).

## 4. Data check

As in card 034's gate. Memory holds 50–57 distinct (tile in front, held
tile) pairs per world for each of forward, pick up, drop and toggle. In
the test (world (a), each colour):
- picking up the key: 4,060;
- dropping it: 3,985;
- forward onto the open door: 503.

## 5. Feasibility gate

- **Codebooks:** arm 3 keeps all 20 tiles distinct in 10 of 10 seeds.
- **Weight μ,** on familiar tiles: 0.01, 0.03, 0.1 and 0.3. Take the
  largest that keeps all 20 tiles distinct in 10 of 10 seeds. If none
  does, stop.
- **α,** at that μ: the leave-one-out choice must judge at least 90% of
  the cases right, and more than both trivial readings, "never new" and
  "always new". If not, stop.
- **Upper bound:** arm 1 passes criteria 1–3.
- **Trivial baseline:** arm 3 fails criterion 2.

Result of the gate, before the main run (2026-09-29): **passed.**
Numbers are in `runs/035_gate_*.json` and `runs/035_alpha.json`.
- **Codebooks:** arm 3 kept all 20 tiles distinct in 10 of 10 seeds.
- **Weight:** μ = 0.01. Seeds out of 10 with all 20 tiles distinct:
  - 0.01: 10;
  - 0.03: 9 (the green and red open doors shared codes in one seed);
  - 0.1 and 0.3: 0 (tiles merged, and the rebuild error rose from at most
    0.0003 to 0.004–0.024).

  Card 031's rule, reading the same μ = 0.01 encoders, kept the tiles
  distinct with none "new" in only 6 of 10.
- **α = 6.** It judged 98.75% of the 800 leave-one-out cases right,
  against 93.25% for "never new" and 6.75% for "always new".
- **Upper bound:** arm 1 passed criteria 1–3 for both colours.
- **Trivial baseline:** arm 3 failed criterion 2 in 10 of 10 seeds.

## 6. Success criteria and prediction

For arm 2, each criterion must hold in at least 8 of 10 seeds, for yellow
and for purple.

1. **Nothing lost.** Card 031's criterion 1 in the four familiar worlds:
   held-out effects exact; the goal reached in ≥ 99% of 500 layouts; mean
   steps within 5% of card 029's.
2. **New keys and open doors predicted, and still new.** Both must hold:
   - in world (a), card 029's model over the codes predicts each of three
     cases exactly in ≥ 99% of its occurrences: picking up the new key,
     dropping it, and forward onto the open new-colour door;
   - each of the colour's four new tiles (key, closed door, open door,
     agent in the doorway) has codes that differ from every familiar
     tile's.

   The first part shows the codes carry rules to a new appearance. The
   second is the novelty signal. Without it, the new key would be taken
   for one familiar key, and that key's door would be predicted to open
   for it.
3. **Acted.** In world (a), the goal is reached in ≥ 98% of 500 layouts,
   with mean steps at most 1.15 times the shortest route's.

Also reported: α's leave-one-out table; which codebooks the new tiles
share with familiar ones and where they are "new"; recall's own
predictions; the closed-door tries curve (P19); every other case of world
(a); world (b); arms 1, 3 and 4; seconds.

**Prediction.**
- The gate passes at μ = 0.01 or 0.03 (appendix B2), with α about 3–4.
- Arm 1 passes. Arm 4 fails criterion 2 through training tiles going
  "new" or new tiles taking a familiar tile's codes.
- Arm 2: the second part of criterion 2 holds (appendix B2). The first
  part is open. Each rule reads the fewest codebooks that lose no
  evidence. If picking up reads a codebook where the new key is "new",
  it gets no prediction. I expect criterion 2 in some seeds, perhaps not
  8.

**Budget.** The gate: 50 encoders, about 8 minutes in five processes.
The main run: 30 full checks plus the label codes, about 15 minutes in
four processes. About 25 minutes in all; past 30, it goes to the user as
commands.

## 7. Result

Arm 2 (the main arm), seeds out of 10, 8 needed for each colour. Numbers
are in `results.json` and `runs/035_arm*.json`.

| Criterion | Yellow | Purple | Verdict |
|---|---|---|---|
| 1. Nothing lost | 10 | 10 | Pass |
| 2. Predicted and still new | 0 | 0 | Fail |
| – told apart | 10 | 8 | |
| – pick up / drop / forward onto the open door, ≥ 99% | 0 / 1 / 0 | 0 / 1 / 3 | |
| 3. Acted | 0 | 3 | Fail |

The other arms, seeds out of 10 (arm 1 is a single run), yellow / purple:

| Arm | Nothing lost | Told apart | New key picked up | Acted |
|---|---|---|---|---|
| 1. Label codes | Pass | Pass | Pass | Pass |
| 2. Recall term, new rule | 10 | 10 / 8 | 0 / 0 | 0 / 3 |
| 3. No recall term, new rule | 10 | 10 / 10 | 0 / 0 | 1 / 2 |
| 4. Arm 2's encoders, card 031's rule | 7 | 1 / 2 | 9 / 9 | 7 / 8 |

- **The novelty rule did what it was built for.** On the same encoders
  it kept the training tiles intact in 10 of 10 seeds, against 6 with
  card 031's rule, and lost nothing in 10, against 7. The new tiles were
  told apart in 10 and 8 seeds, against 1 and 2.
- **The codes stopped carrying rules.** The yellow key was "new" in 2–4
  of its 4 codebooks, the purple key in 1–4. Only 2 seeds had a codebook
  with all three familiar keys on one code. Yet in every seed some part
  put the new key nearer every familiar key than any other tile. The
  keys group in the vectors, each on its own code. The counted model
  matches codes exactly, so a key that is "new" where its rule looks gets
  no prediction. Without "forward onto the open door", the goal was
  reached in 18.4% of layouts in most seeds.
- **Card 031's rule carried the key but not the novelty** (arm 4). It
  carried picking up the new key in 9 of 10 seeds. The key was "new" in
  0–3 codebooks and shared codes in the rest. It took one familiar key's
  full codes in only 2 (yellow) and 3 (purple) seeds. The new open doors
  took a familiar open door's full codes in 7 and 5 seeds, and training
  tiles broke in 4.
- **Recall on the same vectors got both.** It predicted picking up the
  new key, dropping it, forward onto the open door and closing that door
  right in 10 of 10 seeds for both colours. Toggling the new closed door
  while holding its key was right after 1–4 stored tries in every seed.
- **Where the failure sits** (CHARTER rule 2). Both rules give each code
  one radius, the same for every action. Card 031's rule is loose for
  new tiles and tight for training tiles; this card's rule is the
  reverse. But what counts as new depends on the action. A new colour
  does not matter for picking a key up; it does matter for which door the
  key opens. A radius per code cannot depend on the action, and the
  counted model then matches codes exactly. Recall's weights are learned
  per action, and recall got these cases right. So the failure is in the
  declared exception, the counted lookup over codes in the planner, and
  in judging novelty per code rather than per action.
- **Seconds:** the gate took about 6 minutes, the main run about 12.

## 8. Decision

**Stop** (2026-09-29, with the user). Criteria 2 and 3 failed. The
novelty rule kept the codes intact and told every new tile apart, so it
is kept as how codes are read as names. What failed is the declared
exception: the counted lookup over codes in the planner cannot use a
thing that is like a key but is not one of the known keys, and one radius
per code cannot say what is new for each action. Recall on the same
vectors predicted the new cases in 10 of 10 seeds. The next card removes
the exception. Recall replaces the counted model for predictions in the
planner, and novelty is judged per action by the weight of recall's vote.

## Appendix A: the procedure

`tools/card035/novelty.py`, on card 034's `tools/card034/teach.py`
(memory, priors, recall reports, test worlds) and card 033's
`train_encoder` (which now keeps its codebook vectors). Encoders are
saved in `runs/035_enc/` and read again by every arm that uses them.
The first draft's probe is `tools/card035/sparse.py`.

**Reading the codes.** For codebook k, with each training piece's
nearest code among all 8:
- the used codes are those some training piece is nearest to;
- m_k is the median distance of the training pieces to their codes;
- the radius of used code c is max(largest distance of c's training
  pieces to c, m_k);
- every tile's piece takes its nearest used code c, or "new" if its
  distance to c exceeds α × c's radius.

**Leave-one-out.** For each codebook and each training piece t: the
radii are recomputed from the other 19 pieces (the code vectors stay).
If some other piece shares t's code, t is right when it gets that code
within α × the radius. If none does, t is right when it is "new".
"Never new" and "always new" are scored the same way.

**The first draft's penalty** (appendix B1): β Σ_groups Σ_parts ‖λ_part‖₂
on recall's weights λ = softplus(θ), reshaped to (groups, 8 parts, 8),
added as μ × β × that sum. A part counts as read when it holds at least
10% of its group's summed norms.

**Everything else** is card 034's appendix A, with M = 8.

## Appendix B: drafting evidence

From 2026-09-29, seeds 100–102, not this card's. The encoder, the gate's
grids and the rule for α were settled on familiar tiles. The new colours
were looked at only to see where things break.

**B1. The first draft: a sparsity penalty on recall's weights.** This
card was first drafted as "recall reads few parts": a group-lasso penalty
β on recall's per-action weights, one group per codebook part, so that
each action reads few parts. Codes were read with card 031's rule. Numbers
are in `runs/035_probe_*.json`. Seeds out of 3:

| μ | β | Distinct | None "new" | Rebuild error | Parts read (fwd / pick / drop / toggle) | Recall right on new key (pick / drop) |
|---|---|---|---|---|---|---|
| 0.03 | 0 | 3 | 2 | 0.0002–0.0003 | 3–4 / 3–4 / 2–3 / 3–4 | 3 / 3 |
| 0.03 | 0.003 | 3 | 2 | 0.0001–0.002 | 2–3 / 2 / 1 / 3 | 3 / 3 |
| 0.03 | 0.01 | 3 | 1 | 0.0002–0.004 | 2–3 / 2–3 / 1 / 3 | 3 / 3 |
| 0.03 | 0.03 | 3 | 1 | 0.0001–0.0003 | 1–2 / 2–3 / 1 / 2–3 | 2 / 0 |
| 0.1 | 0 | 0 | 1 | 0.022–0.023 | 4 / 3–5 / 2–4 / 3–4 | 3 / 3 |
| 0.1 | 0.003 | 0 | 1 | 0.010–0.022 | 4 / 2–3 / 2–3 / 2–3 | 3 / 3 |
| 0.1 | 0.01 | 1 | 0 | 0.0007–0.011 | 2 / 2–3 / 1 / 2–3 | 3 / 3 |
| 0.1 | 0.03 | 3 | 0 | 0.0004–0.0006 | 1–2 / 2–3 / 1 / 2 | 3 / 0 |

- "Parts read" is for the key world; the other worlds agree. Yellow and
  purple gave the same recall results. Recall predicted forward onto the
  new open door, and toggling it closed, right in every seed and setting.
- **Dropping the new key** was lost when drop read a single part at
  β = 0.03.
- **Kinds.** Across the 24 runs (8 settings × 3 seeds), a new tile lay
  nearer every familiar member of its kind than any other tile, in at
  least one part:
  - the new key in 23 (both colours);
  - the new open door in 21 (yellow) and 19 (purple);
  - the new closed door in 7 (both colours).

  Card 031's encoder alone had no part in which even the familiar keys
  grouped (card 034, appendix B). The recall term does this by itself,
  and the penalty did not change it much.
- **Closed doors.** Toggling one while holding the matching key
  was first right after 0–8 stored tries, and not within 8 in one seed at
  μ = 0.1 and β = 0.03.

**B2. The novelty rule on the same kind of encoder** (β = 0, 8 codes;
`runs/035_draft_*.json`):
- **Training tiles:** under nearest-code reading, all 20 were distinct in
  3 of 3 seeds, at μ = 0.01 and at μ = 0.03. Card 031's rule flagged
  training tiles "new" in 3 of those 6 runs.
- **Leave-one-out on familiar tiles**, summed over the three seeds (240
  cases, of which 14–21 have no other tile sharing their code):

  | μ | α = 1 | 1.5 | 2 | 3 | 4 | 6 | 8 | 12 | Never new | Always new |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0.01 | 196 | 230 | 234 | 237 | 239 | 239 | 237 | 237 | 226 | 14 |
  | 0.03 | 205 | 228 | 231 | 235 | 234 | 233 | 234 | 230 | 219 | 21 |

  An earlier grid stopped at 3, where the count was still rising. It was
  widened to 8 for that reason alone.
- **Told apart** (evaluator): at every α from 1 to 3 and in all 6 runs,
  each of the 8 new tiles had codes different from every familiar tile's.
  With card 031's rule, all 6 runs gave some new tile exactly a familiar
  tile's codes. The yellow and purple keys took the blue or red key's
  codes in all three runs at μ = 0.01, and the yellow open door took a
  familiar open door's codes in all three at μ = 0.03.
