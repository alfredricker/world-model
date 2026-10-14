---
id: "031"
title: codes from pixels
rung: 0
serves: [P4, P3, P5, C5]
status: done
verdict: fail
arch_version: 5
date: 2026-09-29
---

# 031: codes from pixels

## 1. Question

Can the counted model of what actions do (cards 028–029) work on discrete
codes that a learned encoder gives each tile, in place of each tile's
exact pixels, without losing anything? And does it then carry what it
knows to keys and doors in a colour it has never seen them in? Serves P4
(keep the distinctions that change outcomes), P3 in its simplest form (a
known shape in another colour), P5 (a code far from every known one says
so) and C5. Follows [card 029](../029-subgoals-from-the-model/card.md)
(keep).

It is the first step of the direction set with the user on 2026-09-29,
when [card 030](../030-rules-over-found-attributes/card.md) was stopped
(appendix B).

## 2. What changes

One component: what the model's entries are stated over. Placing views,
the move maps, the rule finder, subgoals and walking (cards 028–029) stay.

| Part | Version 5 | Card 031 |
|---|---|---|
| A tile | An integer, by exact pixel identity | A tuple of 4 codes. An encoder maps the tile's pixels to 4 pieces; each piece becomes the nearest of its codebook's 8 codes, or "new" when far from all of them |
| Entries: forward by the tile in front, drawing the agent, pick up, drop, toggle | One per appearance | Each keyed by the fewest codebooks that lose no evidence, so tiles that behave alike share one (mechanism sparsity, `2107_10098`, done by counting). Outcomes per codebook: unchanged, taken from the other changed place, or set to a code |
| A tile never seen | Blocked and unchanged by every action (declared) | Gets each entry whose codebooks it matches; otherwise blocked and unchanged, as before |

In the key world (codes as the agent has them; our names in brackets):

```
seen:   pick up (c1=3, c2=5 [red key])    -> front: floor's codes; held: front's codes
        pick up (c1=3, c2=1 [green key]), (c1=3, c2=6 [blue key]) -> the same
pooled: pick up (c1=3, any c2)            -> front: floor's codes; held: front's codes
test:   purple key (c1=3, c2=new)         -> can be picked up
```

Entries are written out for every code tuple the agent meets, so card
029's code reads the table unchanged.

**The encoder's objective** names no attribute (appendix A). A decoder
rebuilds each tile's pixels from its codes (a training aid; the agent
imagines in codes). When an action changes a tile in place (a door opens;
the agent steps into a doorway), an absolute-value penalty per codebook
makes few codebooks change (`2007_10930`, pairs as `2002_02886`).
Codebooks of 8 codes against 19 tiles cannot name every tile alone. Each
distinct tile and pair counts once, so the goal square counts as much as
floor. All of it is fixed before the run and never tuned on purple
(`1811_12359`); ten seeds, all reported.

**The test colour.** Purple keys and doors never occur in training;
purple appears only on the vase, in darker shades (appendix A). What must
carry over is the shape. No change in this world separates a thing's
colour from its shape, so that rests on the encoder's structure and the
small codebooks.

**Worlds and arms.** Training: card 029's four worlds (key, switch,
either, both) in red, green and blue, 5,000 random episodes each. Test,
500 layouts each, 200-step budget: (a) the key world, purple door open at
the start; (b) the switch world, purple door closed. Reported only: (c)
the key world, purple door closed, which needs "the held key's colour
equals the door's" (the next card).

1. Upper bound: codes from the simulator's labels (the object with its
   state and whether the agent is on it; its colour), and the shortest
   route.
2. **Main: learned codes, 4 codebooks of 8, ten seeds.**
3. Trivial baseline: exact tile names (card 029 unchanged).
4. Ablation: one codebook of 64, a plain VQ-VAE that can name every tile.
5. Ablation: 4 codebooks of 8, without the pairs.

## 3. Dependencies

Cards 028–029 (the counted model, subgoals, walking; keep), 027
(Dirichlet-multinomial evidence for grouping tiles by what actions do; and
its warning that a network placed withheld tiles by colour, 4 of 33
right) and 010 (the rule finder). Methods: vector quantisation
(`1711_00937`) with several codebooks for one vector (`1803_03382`),
sparse changes (`2007_10930`, `2002_02886`, `2107_10098`), and why none of
these identifies factors without a declared bias (`1811_12359`). New
here: entries pooled over codes, which arm 1 tests with ideal codes
before learned codes are read (CHARTER rules 3 and 7).

## 4. Data check

Training is as card 028. Key world: 172–213 door openings per colour, all
holding its key, and about 15,000 pick ups per key colour. Switch world:
617–676 openings per colour, all with the switch on. The encoder sees the
19 distinct tiles and about 17 distinct pairs (door toggles both ways, the
switch, the vase, the agent drawn and undrawn).

Test: 1,000 random episodes per purple world (seed 31, scratch count).

| Event | (a) key, door open | (b) switch, door closed |
|---|---|---|
| Purple key picked up; dropped | 4,035; 3,959 | 4,450; 4,369 |
| Forward into the purple key (blocked) | 4,473 | 4,995 |
| Forward into the closed purple door (blocked) | 1,180 | 1,788 |
| Forward onto the open purple door | 523 | 231 |
| Open purple door toggled (closes) | 495 | 213 |
| Closed purple door toggled: switch on (opens); off (stays) | — | 388; 1,272 |
| Closed purple door toggled without the purple key (stays) | 999 | — |
| Closed purple door toggled holding the purple key (excluded) | 57 | — |

The rarest case in a criterion: forward onto the open purple door in the
switch world, 231.

## 5. Feasibility gate

- **The codes first** (every seed, reported): codes used per codebook;
  any two of the 19 tiles sharing a tuple; the purple tiles' codes.
- **Upper bound:** arm 1 matches card 029 in the familiar worlds and
  passes criteria 2 and 3; the shortest route solves every test layout.
- **Trivial baseline:** arm 3, predicted 0% in (a) and (b): the open
  purple door is unseen, so it blocks, and the closed one cannot open.

Result of the gate, before the main run: **passed** (full data, one
process, 2 minutes; `runs/031_codes_main.json` repeats it). Arm 3 (card
029 unchanged) reproduces card 029 in the four familiar worlds (100%,
16.38 / 17.27 / 15.47 / 21.77 steps) and fails the purple cases (0% on
the worst case; acting 18% in (a) and 3% in (b), from random steps). Arm 1
passes criteria 1–3. The shortest route solves every test layout. As
declared (one set of codebooks per action), arm 1 failed criterion 2:
drop was keyed on "held = key red / green / blue", which gives purple
nothing (0% of purple drops). Entries are now keyed per group (as built,
appendix A). The encoder's pair term and re-seeding also changed (as
built); neither was chosen by looking at purple.

## 6. Success criteria and prediction

For the main arm, each criterion must hold in at least 8 of the 10
encoder seeds. Every seed of arms 2, 4 and 5 is reported.

1. **Nothing lost.** In the four familiar worlds, held-out effects are
   predicted exactly (card 028's check, about 100,000 transitions per
   world). Acting reaches the goal in at least 99% of 500 layouts, with
   mean steps within 5% of card 029's (100%, 1.03–1.11 times the shortest
   route). This shows the codes keep every distinction that changes an
   outcome.
2. **Purple, predicted.** On all transitions of the test episodes, each
   case of section 4 except the excluded one is predicted exactly in at
   least 99% of its occurrences: blocked or moved; what changes, and into
   which codes; for the switch-world door, opening exactly when the switch
   is on. Arm 3 gets pick up, drop, moving onto the open door and every
   toggle that changes the door wrong. This shows entries carried over by shape, not by exact
   appearance.
3. **Purple, acted.** The goal is reached in at least 98% of 500 layouts
   in each of (a) and (b), with mean steps at most 1.15 times the shortest
   route's. Arm 3 is predicted at 0%.

Reported with them: for each codebook, which simulator label (object,
state, agent on it, colour) its codes best predict, with no bar, since
meanings are not dictated; the purple tiles' codes; world (c); arms 4 and
5; the chains worked out in (b); seconds.

**Prediction.** Arm 1 passes all three; the main arm passes criterion 1
(rebuilding pixels keeps the 19 tiles apart). Criteria 2 and 3 are
uncertain: only the encoder's structure and the small codebooks separate
a key's shape from its colour, and card 027's network grouped unseen
tiles by colour. I expect a pass in some seeds, perhaps not in 8 of 10.
If arm 1 passes and arm 2 does not, the codes are at fault, and arms 4
and 5 say whether several codebooks and the pairs helped. Arms 3 and 4
fail: a purple tile is new, or snaps to one known tile.

**Budget.** Collecting the experience (four worlds, as card 029) and the
test episodes dominates. The encoder trains on 19 tiles in seconds per
seed, 30 runs in all; acting takes about 0.02 seconds per layout.
Estimate 15–25 minutes; if over 30, it is handed to the user as commands.

## 7. Result

Run 2026-09-29, `tools/card031/codes.py`, three processes in parallel
(17 minutes); numbers in `runs/031_codes_main.json` (arms 3, 1, 2),
`runs/031_codes_arm4.json` and `runs/031_codes_arm5.json`. Seeds are
encoder seeds; each criterion needed 8 of 10.

| | Arm 3, exact names | Arm 1, labels | **Arm 2, main** | Arm 4, one codebook of 64 | Arm 5, no pairs |
|---|---|---|---|---|---|
| 1. Nothing lost (effects exact, acting, steps) | 100%, 100%, card 029's steps | pass | **10 of 10 seeds** | 10 of 10 | 10 of 10 |
| 2. Purple predicted (worst case) | 0% | pass (100%) | **0 of 10** (0% in every seed) | 0 of 10 | 0 of 10 |
| 3. Purple acted, (a) / (b) | 18% / 3% | 100% / 100% (1.05 / 1.07 × shortest) | **0 of 10**: (a) 18–100%, (b) 3–6% | 0 of 10 | 0 of 10 |

Verdict: criterion 1 **pass**; criteria 2 and 3 **fail**.

Seeds (of 10) in which each purple case was predicted right in ≥ 99% of
its occurrences, (a) and (b) together:

| Case | Arm 2 | Arm 4 | Arm 5 |
|---|---|---|---|
| Pick up the purple key | 2 | 5 | 4 |
| Drop it | 6 | 5 | 10 |
| Forward onto the open purple door | 3 | 3 | 6 |
| Toggle the open purple door (closes) | 0 | 1 | 0 |
| Closed purple door, switch on (opens) | 0 | 1 | 0 |
| Blocked or unchanged: into the key, into the closed door, toggling it without the key or with the switch off | 10 | 10 | 10 |

The last row is right by default: a tile the model cannot place is
blocked and unchanged, which these cases happen to be.

**What the codes are.** Rebuilding keeps all 20 training tiles apart in
every seed of every arm (no two share a tuple), so the counted model loses
nothing on learned codes. The pairs work as intended: every in-place
change touches one codebook. But no codebook is a colour or a shape
(normalised mutual information with colour 0.63–0.75 and with the object
0.61–0.77 for the best codebook, the others mixed). The three keys share
no set of codes that sets them apart. So a purple tile is "new" in one to
four codebooks (the closed purple door in every seed), or takes a blue
tile's exact codes (the open purple door in 5 seeds, the key in 2). A
purple door that becomes a blue one can be walked through, which is why
(a) reached 100% in 5 seeds, but toggling it predicts a blue door. Arm 4
behaves the same way (the closed purple door took the wall's code in 8
seeds). Arm 5, without the pairs, is no worse on any case, so the pairs
did not help transfer.

The chains in (b) with label codes read as intended: "episode ended ←
facing the goal ← (blocked) the door tile shows the open purple door ←
switch on in view ← facing the switch".

**Reading.** Arm 1 passes and arm 2 fails, so, as section 6 said, the
codes are at fault, not the model over them. Neither several codebooks
nor the pairs made the codes carry a known shape to a new colour. That
fits `1811_12359`: nothing in this objective prefers a colour/shape split.
Three colours across four door and key tiles already make the factorised
code the shortest description, but codebooks of 8 are big enough to name
the tiles any way. Two new parts do carry over: entries keyed per group,
and scoring rules and groups on one footing.

## 8. Decision

**Revise** (2026-09-29, with the user). Keep the model over codes: arm 1
carries it to purple, and learned codes lose nothing in the familiar
worlds (the user's takeaway: replacing the integer map by learned
codebooks cost nothing end to end). Revise the encoder so that the
shortest description is preferred, with a penalty on the number of codes
it uses, which names no attribute:
[card 032](../032-fewest-codes/card.md). Not chosen: more colours in
training, or a declared colour × shade prior (card 030, appendix B).

## Appendix A: the procedure

**Encoder.** Input: one tile, 8 × 8 × 3 pixels scaled to [0, 1]. Two
3 × 3 convolutions with 32 channels and ReLU (the second with stride 2),
then a linear map to 4 pieces of 8 numbers. Each piece is replaced by the
nearest of its codebook's 8 vectors, with straight-through gradients and
`1711_00937`'s codebook and commitment terms (commitment weight 0.25). A
decoder (linear, then a transposed convolution) rebuilds the pixels from
the 4 chosen vectors; its loss is mean squared error.

**Pairs.** From the stored transitions:
- at a place a pick up, drop or toggle changed, the tile before and after;
- for forward, the tile in front before and the centre after (drawing the
  agent), and the centre before and the tile it left after (undrawing).

A pair is left out when its "before" tile shows after at the other changed
place, or its "after" tile showed there before: the thing moved (pick up,
drop). Each distinct pair counts once. One kept pair shares nothing (the
vase breaking leaves floor), and the penalty below tolerates that.

**Changes touch few codebooks.** For a pair, the distance between the two
tiles' pieces is taken codebook by codebook, before quantisation, and the
penalty is the sum of the four distances. Unlike a squared penalty, it
prefers one large change in one codebook to small changes spread over all
four (`2007_10930`'s Laplace prior, per codebook).

**New.** A piece is new when its distance to the nearest code is more
than half the distance from that code to the nearest other code in the
same codebook (declared).

**Training.** Adam, learning rate 0.001, 5,000 updates, each on all 19
tiles and all pairs. Loss weights: rebuild 1, pairs 0.1 (declared). Ten
seeds. Arm 4: one codebook of 64 vectors of 32 numbers. Arm 5: pairs
weight 0.

**Entries over codes.**
- **Tuples.** A tile is its tuple of codes, interned as an integer for
  card 029's arrays.
- **Grouping.** For each action, and for each set S of codebooks (16 sets;
  for drop, one set for the tile in front and one for the held tile), the
  seen tiles are grouped by their codes in S.
- **Outcomes** are stated per changed place and codebook, as the first of
  these that holds for all tries: unchanged; taken from the other changed
  place before; set to a code.
- **Choosing S.** Each S is scored by the Dirichlet-multinomial evidence
  of the grouped outcome counts (card 027). Where an entry has several
  outcomes, card 010's rule-list evidence is scored on the grouped tries.
  The best S is kept; on ties, the smallest.
- **Rule atoms** stay as card 028's ("held = X", "X in view", over code
  tuples).
- Forward's outcome in front, the draw and undraw tables, pick up, drop
  and toggle are all keyed this way.

**Write-back.** When a tuple is first seen, at learning or while acting,
each action's entry for it is the entry of its group under the kept S.
That holds only if the group exists and none of the tile's codes in S is
new; otherwise the tile is blocked and unchanged (card 028's default).

**Arm 1's codes,** read from the renderer's object list, for arm 1 only.
Codebook 1: the object, with its state and whether the agent is drawn on
it (floor, wall, goal, key, closed door, open door, agent on floor, agent
in a doorway, ball, box). Codebook 2: its colour as the renderer names it.

**Why purple.** Keys and doors in purple never occur in training. Purple
appears only on the vase, in darker shades: the purple key shares one of
its six non-grey pixel values with the vase, and the purple doors share
none. The pairs should separate a door's state and the agent drawn on it
from the rest. No change in this world separates a thing's colour from
its shape: pick up and drop change both together, and both papers'
conditions for identifying a factor (`2002_02886`, `2107_10098`) fail for
colour.

**Which transitions form pairs** is a declared data policy, a form of
implicit supervision (`1811_12359` asks for it to be stated).

**Evaluation.** Predicted codes are compared with the encoder's codes of
the real next view at the changed places, and the predicted outcome
(moved, blocked, ended) with the simulator's. Codebooks against labels:
for each codebook, the normalised mutual information between its codes
and each label, over the 19 training tiles and the 4 purple ones (key,
closed door, open door, agent in the doorway).

### As built

The code is `tools/card031/codes.py`, on cards 028–029's code
(unchanged). The renderer's object list gains purple keys and doors at
import (`src/` unchanged). Where it differs from, or fills in, the plan:

- **Pair penalty.** As declared (the sum over codebooks, weight 0.1) it
  gave each door colour's closed, open and agent-in-doorway tiles one tuple
  in 3 of 3 smoke seeds (familiar tiles only), which fails criterion 1 by
  construction. As built: `2002_02886`'s adaptive rule. Per pair, the
  codebooks whose distance is below the midpoint of the pair's smallest
  and largest distance are pulled together; the others are left free.
- **Re-seeding.** Every 250 updates, while two training tiles share every
  code, an unused code of each codebook is moved onto one such tile's piece
  (a standard dead-code restart). Without it, 5 of 5 smoke seeds left
  collisions.
- Pieces and codes are unit length (a distance penalty could otherwise be
  met by shrinking the scale). cuDNN is deterministic: a seed gives one
  set of codes. With one codebook (arm 4), the adaptive rule never pulls,
  so arm 4 has no pair term in effect.
- **Entries keyed per group** (a tree over codebooks). A group is split by
  the set of one key tile's codebooks that raises the evidence most, if
  any does, and each part is treated the same way. For drop, the key tiles
  are the tile in front and the held tile. As declared (one set per
  action), arm 1 keyed drop on "held = key red / green / blue", because
  keying the held tile would split every other tile in front too, and
  purple drops were 0% right (reported as arm "1g", criteria 1 and 3
  passed).
- **Scoring.** Each set of tries that a group or a rule singles out is
  scored by the Dirichlet evidence over the kind's categories: every
  outcome the action shows, each try described on its own. Rules are
  found by card 010's search and pay its cost. As declared, rules were
  scored with card 010's two-outcome prior against groups with the
  many-outcome one, which favoured rules naming each tile.
- **Categories.** Within a group, tries are split by which places changed.
  Then per place and codebook, the first description that holds for all
  of them is taken; where none holds, they are split. At a place that is
  not a key (the held tile for pick up and toggle, the agent's tile when
  drawing), "unchanged" and "taken from the other place" use the
  category's most common tile there.
- Left and right are keyed the same way (the card named forward); they
  come out keyed on nothing. "Agent on goal" (seen only in the last view
  of an episode) is the 20th training tile.
- **Test data.** Every transition of 1,000 random episodes per purple
  world (seed 31, through the card's code, so counts differ from section
  4's scratch count by up to 15%). Cases are found from the simulator's
  states; "other transitions" is a sample of 20,000. Test layouts are card
  029's 500 (seed 777), recoloured purple. Criterion 1's steps are compared
  with arm 3 in the same run, which reproduces card 029's.
- Arm 1's colour codebook includes purple from the vase (the renderer's
  name for it); no purple door was seen, so it gives a purple door no
  colour entry.

## Appendix B: the direction

Set with the user on 2026-09-29, when card 030 was stopped: tile
(later, a segment) → vector → several discrete codebooks → rules and goals
over the codes. Next come rules with a shared variable over codes (card
030's tests); goals as any condition over codes and positions (holding
something of a given shape, being closer to a thing); then recall of past
episodes in place of counting. **Counting is a simplified memory** (the
user's idea): it keeps tallies and forgets the experiences. Recall keeps
them, so old experience can be re-sorted under new codes, and replaying it
trains network weights until a rule is known without recall. The weights
take over a rule only once they agree with recall on every case, rare ones
included (LESSONS).
