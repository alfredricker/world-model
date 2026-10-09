---
id: "089"
title: an encoder whose leftover means the same in every kind, for card 088's relation
rung: 6
serves: [P3, P4, C5, C2]
status: done
verdict: fail
arch_version: 18
date: 2026-10-08
---

# 089: an encoder with a "made of" part

Drafted at the user's request (2026-10-08: "draft 089"), as card 088's
revision ("keep 088").

Approved by the user (2026-10-08: "yes").

## 1. Question

Card 088 found that what card 087's properties ignore is mostly hue
(78–85%), but that the encoder codes it in different directions for
each kind: keys' and doors' leftovers are 70° and 88° apart, so "equal
along the leftover" told matching pairs apart only 53 of 66 times. If two
of the encoder's four parts can see only what the tile is made of (each
pixel's values, pooled over the tile, their arrangement dropped), the
same material is coded the same way in every kind. Do the properties
then ignore those parts, and does card 088's relation, found as before
from what the properties ignore, tell matching key and door pairs apart
for colours left out and hues never seen, with no objective that names
colour? P3, P4, C5, C2 (one encoder).

## 2. What changes

One component: the encoder's architecture, trained from scratch with
card 054's recipe and no relation term (card 070's colour-matching term
is dropped).

```
card 054 / 070:  tile (8×8×3) → conv → conv → linear → 4 parts of 8
this card:       tile → conv → conv → linear              → parts 0, 1  (how the tile is arranged)
                 tile → per-pixel MLP (1×1) → mean and max over the 64 pixels → linear → parts 2, 3  (what it is made of)
```

- **The prior** (C3, declared): a tile has an arrangement and a
  material; the material part sees each pixel alone and cannot see where
  it is. Nothing says which part matters for what, nor that material
  means colour. In vision this is the split between style (pooled
  feature statistics) and content (arrangement).
- **Training:** card 054's recipe and stream unchanged (transition model,
  visibility margin, pair margin 0.5; 12 checkpoints, seed 399).
- **After training,** as cards 087 and 088: the property ensemble is
  retrained on the new vectors (property play), and P is the leftover
  (C v = μ (G + εI) v) at the rank admission chooses. Reported: how much
  of what the properties read (G) falls on each part.
- **In the agent:** the new encoder replaces card 070's, and `rel:P`
  uses the leftover P in place of card 070's trained P. Everything
  downstream (identity codes up to noise, memory, recall's fits) is
  rebuilt from the new vectors as version 18 builds it; the fits are
  stored per encoder (card 085.3 (a)).

## 3. Dependencies

Card 054 (the recipe, `tools/card052/predflip.py`, `runs/054/b.sh`);
card 087 (property play, the ensemble); card 088 (the leftover and
card 069's probe); card 085.3 (a) (stored fits). Style as pooled feature
statistics, content as arrangement: Gatys, Ecker and Bethge 2016;
Tenenbaum and Freeman 2000 (style and content by bilinear models):
LITERATURE's current focus.

## 4. Data check

Card 054's stream as before. On the trained encoder: the share of hue in
the material parts' variance and in the arrangement parts' (report
only); keys' and doors' leftover directions, the angle card 088
measured (70°, 88°).

## 5. Feasibility gate

- **Upper bound:** the tiles' mean pixel colour: 81 of 81 pairs (card
  069); card 070's trained P: 66 of 66, 9 of 9.
- **Trivial baselines:** card 054's vectors, best weighting: 47, 4;
  card 088's leftover on card 070's encoder: 53, 7.
- **Gate:** the encoder keeps card 054's margins (pair margin and
  visibility margin met at the end of training); the property ensemble
  on it still gets card 087's part (i) right (57 of 57 on each property);
  and card 088's leftover reaches ≥ 64 of 66 left-out-colour pairs and
  ≥ 8 of 9 unseen-hue pairs (card 069's probe, only the threshold
  fitted).

**Gate result: failed in both arms** (`tools/card089/train.py`,
`gate.py`; `runs/089/`). Trained alone on the GPU (a first attempt
beside tier 3's build reached 5 of 12 checkpoints in 30 minutes and saved
nothing; `runs/089/partial/`); 12 checkpoints, about 21 minutes.

| | Mean and max | Max only | Card 054 |
|---|---|---|---|
| Pair margin, visibility margin (violations at the end) | 0, 0 | 0, 0 | 0, 0 |
| Card 087's (i): forward, pick up, toggle (of 57) | 56, 57, 57 | 57, 57, 57 | – |
| Colour read from parts 0–3 (chance 0.12) | 0.66, 0.67, **0.54, 0.38** | 0.53, 0.67, **0.22, 0.22** | 0.55, 0.65, 0.53, 0.59 |
| Share of what the properties read, made-of parts | 16% | 13% | – |
| Leftover, rank 1, 2, 4, 8: left-out colour pairs (≥ 64) | 34, 42, 28, 35 | 35, 36, 44, 40 | card 088: 53 |
| Unseen-hue pairs (≥ 8) | 3, 4, 4, 4 | 4, 3, 1, 2 | card 088: 7 |
| Hue's share of the variance on fresh hues, pooled over kinds (†: the wrong measure) | 8%, 6% | 8%, 6% | – |
| Keys' and doors' leftovers, angles | 34°, 80° | 46°, 70° | 70°, 88° |

**Why.** The made-of parts carried colour early (0.96 and 0.98 after 20
updates in the smoke test) and lost it with training: the arrangement
path sees colour too, nothing in card 054's recipe asks for colour in one
place rather than the other, and it ended there.

† **Correction** (after card 090's gate). The second half of this
diagnosis was wrong: the 6–8% pooled keys, doors and balls, so the
differences between kinds swamped hue. Measured within each kind, as card
088 did, hue is a quantity in both arms: linearly 0.43–0.72 of a kind's
variance, and recovered almost exactly from the nearest vectors (R²
0.96–0.99), as in card 070's encoder (0.98–0.99). What fails is alignment
across kinds: the made-of parts alone, as the relation, get 33 (mean and
max) and 42 (max) of 66 pairs. A key and its locked door are not drawn in
the same pixel values (the door's panel is a darker shade of its colour),
so not even what a tile is made of is the same for the two.

## 6. Success criteria and prediction

1. **The relation from ignored directions** (the gate's last part), at
   the rank admission chooses, with no term naming colour anywhere in
   training.
2. **CHARTER's tiers** with the new encoder and the leftover P, on card
   074.2's seeds: tier 1 ≥ 99% (200); tier 2 (100) not worse than
   version 18's 59% (McNemar); tier 3 once its memory builds.
3. **Card 086's recall tables** on the new encoder (decoy memory): toggle
   56 of 56, pick up 28, drop 42, with the relation the only path to the
   never-seen pair (card 084.3's admission).

**Prediction.** The material parts carry hue; the properties read
almost only the arrangement parts (most of G there), so the leftover is
the material parts and keys and doors line up there: the probe passes.
Risks: the material parts also carry kind through the share of pixels
of each colour (mean pooling: a door has more coloured pixels than a
key), which would put kind back into the leftover; max pooling alone is
the fallback, declared before the run. The arrangement parts may lose
what card 054's recipe needs (effects known), since they have 16
numbers, not 32.

**Decision rules.** Keep if the gate and 1–3 hold: the next version, with
colour never named. Revise if the gate's relation part passes and 2 or 3
fails. Stop if the relation part fails with mean and with max pooling.

**Budget.** Training about 18 minutes (card 054's), the ensemble and the
probe about 2; tiers 1–2 about an hour, tier 2's memory refitted once
for the new encoder (about 5 minutes).

## 7. Result

The gate failed with both pooling arms (section 5); the criteria were
not run.

## 8. Decision

**Stop**, by the card's rule (the relation fails with mean and with max
pooling). A path that sees only material does not keep colour there when
the other path can carry it too, and even the material of a key and its
door differs in MiniGrid's drawing (the door's panel is darker), so
"equal along what properties ignore" does not line keys and doors up.
Hue itself is a quantity within each kind in every encoder tried (†
above). Card 090 tested fresh hues in the stream.
