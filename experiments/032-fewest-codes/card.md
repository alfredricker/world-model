---
id: "032"
title: fewest codes
rung: 0
serves: [P3, P4, P5, C5]
status: done
verdict: fail
arch_version: 5
date: 2026-09-29
---

# 032: fewest codes

## 1. Question

Does an encoder that must describe the tiles with as few codes as it can
give codes that carry card 029's model to keys and doors in a new colour?
Card 031's codes kept every tile apart and lost nothing, but named each
tile arbitrarily, so purple keys and doors came out "new" or as blue ones
(0 of 10 seeds). Serves P3 (a known shape in a new colour), P4, P5 and C5.
Follows [card 031](../031-codes-from-pixels/card.md) (revise, with the
user). Approved by the user on 2026-09-29 ("try the number of codes
penalty; it's a revision, so go ahead and run it").

## 2. What changes

One component: the encoder's objective gains a penalty on the number of
codes it uses. Everything else is card 031 as built: the pieces, the
codebooks, the pair penalty, re-seeding, "new", the entries keyed per
group, the tests and the arms.

| | Card 031 | Card 032 |
|---|---|---|
| Encoder objective | rebuild pixels + codebook terms + pairs | the same + w × Σ over codebooks of the entropy of its code use |

**The penalty.** For each codebook, the share of the training tiles
(each distinct tile once) that uses each code, from soft assignments
(softmax of minus the squared distance to each code, temperature 0.1).
Its entropy is the log of the codebook's effective number of codes, so
the sum is the code length of the tile set when each codebook is
described on its own. Rebuilding forces the tiles apart. With the tiles
distinct, their joint entropy is fixed (log 20), so the sum is smallest
when the codebooks share no information: a factorial code (Barlow 1989;
the total correlation of `1802_04942`, `1802_05983`). Our tiles offer
one: three colours crossed with key, closed door, open door and agent in
the doorway. Nothing names colour or shape.

The weight w is chosen before the run, on familiar tiles only, by a
declared rule: the largest of 0.003, 0.01, 0.03, 0.1 and 0.3 that keeps
all 20 training tiles on distinct tuples in all 10 seeds. Neither purple
nor any label is read. If none qualifies, the card stops there.

## 3. Dependencies

Card 031 (the model over codes, keyed per group; its arm 1 carried to
purple; revise). Cards 028–029 (keep). Methods: minimum entropy (factorial)
codes and total correlation (`1802_04942`, `1802_05983`); why such a
penalty alone does not identify factors (`1811_12359`).

## 4. Data check

Unchanged from card 031: the same experience, held-out transitions, test
layouts and purple test episodes, with the same seeds. The encoder sees
the same 20 tiles and 17 pairs.

## 5. Feasibility gate

- **Upper bound:** arm 1 (codes from the labels), as card 031 (passed
  there: 100% of every purple case, 100% of layouts in (a) and (b)).
- **Trivial baseline:** card 031's main arm, the same encoder without the
  penalty (0 of 10 seeds on criteria 2 and 3), and exact tile names.
- **Codes first,** every seed: codes used per codebook, the code length
  (sum of the entropies of hard code use), collisions, the purple tiles'
  codes.

Result of the gate, before the main run:

- **The declared weights failed.** No weight in the declared grid kept the
  20 training tiles apart in all 10 seeds. At 0.003, 5 seeds merged tiles
  and 8 left a training tile "new" in some codebook. At 0.03 and above, the
  codes collapsed to 4–11 in use (code length 0–1.7 nats, against 7.5 for
  card 031). The grid was set without checking the loss scale: rebuilding
  is a mean over pixels, so merging two tiles costs it little.
- **Grid extended once** (the only change, before any purple result;
  2026-09-29). The weights 0.0001, 0.0003 and 0.001 were added. The rule
  also requires that no training tile is "new" in any codebook, since
  such a tile cannot key an entry. Still familiar tiles only, and label
  free.
- **Chosen: w = 0.0003**, the only weight that qualifies (0.0001 left a
  training tile "new" in 2 seeds; 0.001 failed in 4). There the code
  length is 6.3–7.4 nats (23–30 codes in use), barely below card 031's
  7.5. For comparison, a colour × kind code of these 20 tiles is about
  3.7 nats and their joint entropy is 3.0. Between 0.0003 and 0.003,
  training merges tiles before it finds a shorter code. Numbers in
  `runs/032_sweep_*.json`.

## 6. Success criteria and prediction

Card 031's three criteria, unchanged, each in at least 8 of 10 encoder
seeds: 1. nothing lost in the four familiar worlds; 2. each purple case
of card 031's section 4 predicted exactly in ≥ 99% of its occurrences;
3. the goal reached in ≥ 98% of 500 layouts in each of (a) and (b), with
mean steps ≤ 1.15 × the shortest route's.

Arms: 1 labels; 2 **main**, card 031's encoder plus the penalty, ten seeds;
3 exact tile names (card 029); 4 the penalty without the pairs, ten seeds.
Card 031's main arm (same seeds, deterministic) is the comparison without
the penalty.

Reported with them: the weight sweep; per codebook, the label its codes
best predict (normalised mutual information); the code length against
card 031's; per case, the seeds that pass.

**Prediction.** Fewer codes in use, and some seeds with a codebook close
to colour. Criteria 2 and 3 are uncertain. The penalty is blind to how
the grid is labelled: any arrangement of the 12 coloured tiles into three
rows of four has the same code length as colour × kind. Only the
encoder's structure prefers colour × kind (tiles of one colour share
pixel values; tiles of one kind share a mask), as `1811_12359` warns. I
expect criterion 1 to hold, and 2 and 3 to pass in some seeds, perhaps
not 8.

**Budget.** The sweep trains 50 encoders on familiar tiles (about 10
seconds each, five processes): 2–3 minutes. The main run is card 031's
with 20 encoders: about 15 minutes in two processes.

## 7. Result

Run 2026-09-29, `tools/card032/fewest.py` at w = 0.0003, two processes
(20 minutes); numbers in `runs/032_fewest_main.json` (arms 3, 1, 2) and
`runs/032_fewest_arm4.json`. Arms 1 and 3 repeat card 031's exactly.

| | Arm 3, exact names | Arm 1, labels | **Arm 2, main** | Arm 4, no pairs | Card 031 main (no penalty) |
|---|---|---|---|---|---|
| 1. Nothing lost | card 029's numbers | pass | **10 of 10** | 9 of 10 | 10 of 10 |
| 2. Purple predicted | 0% | pass | **0 of 10** | 0 of 10 | 0 of 10 |
| 3. Purple acted, (a) and (b) | 18% / 3% | 100% / 100% | **0 of 10** | 2 of 10 | 0 of 10 |
| Code length (nats) | | | 6.3–7.4 | 5.4–6.6 | about 7.5 |

Verdict: criterion 1 **pass**; criteria 2 and 3 **fail**.

Seeds (of 10) in which a purple case was right in ≥ 99% of occurrences in
both (a) and (b):

| Case | Card 031 main | Arm 2 | Arm 4 |
|---|---|---|---|
| Pick up the purple key | 2 | 6 | 9 |
| Drop it | 6 | 9 | 9 |
| Forward onto the open purple door | 3 | 7 | 6 |
| Toggle the open purple door (closes) | 0 | 0 | 1 |
| Closed purple door, switch on (opens) | 0 | 0 | 2 |

**What the penalty did.** At the one weight that keeps the tiles apart,
it shortened the code a little and made no codebook more a colour or a
shape (best mutual information with colour 0.56–0.76 in arm 2, 0.39–0.57
in arm 4). What it did do, most clearly without the pairs, is leave fewer
codes. A purple tile then falls on the nearest known one: the purple key
took the blue key's exact codes in 8 of 10 seeds of arm 4 (1 of 10 in arm
2). That carries whatever the blue key does, which is right for picking
up and dropping (9 of 10 seeds). It is wrong wherever colour matters: in
the key world a purple door taken for a blue one "opens" with the blue key
(seed 8 of arm 4, 97.7% on that case, the rest right). The two arm 4 seeds
that act at 100% in both worlds do so this way. Toggling the purple door
stays wrong because the closed purple door is "new" in some codebook in
most seeds.

**Reading.** A penalty on the number of codes does not make codes that
separate colour from shape here. Stronger weights merge tiles before any
shorter code appears. Its useful effect is a crude version of
similarity: a new thing treated as the nearest known thing. That is right
for what the two share and wrong for what they do not, with no way to
tell which.

## 8. Decision

**Stop** (2026-09-29, with the user): no penalty on the number of codes.
The user's direction: new codes are acceptable if the agent can relate
them to what it knows. Something like recall should do the work: find
similar known things, infer the new thing's effects from them, and let
experience confirm or correct the inference. This card's arm 4 shows the
first half (the nearest known thing carries the shared effects) and
why the second half is needed (it also carries colour rules that do not
apply). The next card is to be drafted with the user.
