---
id: "090"
title: the encoder trained on fresh hues, so colour is a dimension shared by every kind
rung: 6
serves: [P3, P4, C5, C2]
status: done
verdict: fail
arch_version: 18
date: 2026-10-08
---

# 090: the encoder on fresh hues

Drafted at the user's request (2026-10-08), after card 089: discovery
needs experience in which the pattern varies (CHARTER, "Patterns are
discovered").

Approved by the user (2026-10-08: "yes").

## 1. Question

Card 088 found that what the properties ignore is mostly hue, but coded
per kind (keys' and doors' directions 70° and 88° apart); card 089 found
that an encoder trained on nine fixed colours codes colour as nine
categories (6–8% of a fresh hue's variance on any direction), and that an
architectural split does not keep colour in one place. Card 070's encoder
coded colour as a dimension (9 of 9 unseen hues) because every try had a
fresh hue, but its objective also named the relation. If card 054's
encoder, unchanged, is trained on a stream where every coloured object
takes a fresh hue each episode, the same hues on every kind and many
hues on each, with no relation term, does colour become one dimension
shared by keys and doors, so that card 088's relation (equal along what
the properties ignore) tells matching pairs apart for colours and hues it
never saw? P3, P4, C5, C2.

## 2. What changes

One component: the colours of the encoder's training stream.

| | Card 054 | This card |
|---|---|---|
| Colours in the stream | nine fixed (MiniGrid's six plus orange, cyan and white); pink, brown and teal held out | nine colour slots whose hues are drawn fresh every episode with card 070's sampler (uniform in RGB, at least 60 from MiniGrid's six and the three held-out hues); every kind in every slot |
| Pairings and rules | by colour name (a key opens the door of its name) | the same, by slot: a key opens the door of its slot, whatever hue the slot has that episode |
| Architecture, objectives, recipe | card 054's (transition model, visibility margin, pair margin 0.5) | unchanged; no relation term |

Every MiniGrid colour is now a hue the encoder never saw in training, so
every test is on unseen hues. After training, as cards 087 and 088: the
property ensemble is retrained on the new vectors and P is the leftover
at the rank admission chooses. In the agent, the new encoder replaces
card 070's and `rel:P` uses the leftover P; memory, identity codes and
recall's fits are rebuilt from the new vectors (stored per encoder).

## 3. Dependencies

Card 054 (the recipe, `tools/card052/predflip.py`); card 052's generator
(slots in place of named colours: MiniGrid draws a colour by its name's
RGB, so a slot is a name whose RGB is redrawn each episode, as card 070's
relation play did); card 070's sampler; cards 087 and 088 (properties,
leftover, card 069's probe). LESSONS: a lookup table and a general rule
fit one fixed world equally; new fillers every task made the rule the
only solution.

## 4. Data check

On 1,000 episodes of the stream: every kind appears in many hues and
every slot on every kind; no hue within 60 of a test hue; the rules
still hold (a key opens the door of its slot in 200 of 200 checked
toggles).

## 5. Feasibility gate

- **Upper bound:** card 070's trained P, 66 of 66 and 9 of 9.
- **Baselines:** card 088 on card 070's encoder, 53 and 7; card 089, 44
  and 4; card 054's vectors, best weighting, 47 and 4.
- **Gate:** card 054's margins met at the end of training; card 087's
  part (i) 57 of 57 on each property; card 088's leftover ≥ 64 of 66
  left-out-colour pairs and ≥ 8 of 9 unseen-hue pairs (only the
  threshold fitted). Reported: hue's share of the leftover's variance
  and the angle between keys' and doors' leftovers (card 088: 70°, 88°).

**Data check** (`runs/090/datacheck.json`; 1,000 episodes): all 36
kinds × slots seen, each at least 104 times; 8,994 distinct hues, none
within 60.0 of a test hue; the rule by slot 200 of 200.

**Gate result: failed** (`tools/card090/train.py`, `gate.py`;
`runs/090/`; training 12 checkpoints, about 20 minutes).

| | This card | Card 070's encoder | Card 089 (mean+max, max) |
|---|---|---|---|
| Pair margin, visibility margin (violations) | 0, 0 | – | 0, 0 |
| Card 087's (i): forward, pick up, toggle (of 57) | 57, 57, 57 | 57, 57, 57 | 56–57 |
| Leftover, rank 1, 2, 4, 8: left-out colour pairs (≥ 64) | 21, **45**, 43, 43 | 35, 41, 53, 42 | best 42, 44 |
| Unseen-hue pairs (≥ 8) | 4, 4, 4, 3 | 5, 6, 7, 6 | best 4 |
| Hue's share within a kind, linear: key, locked door, ball, open door | 0.79, 0.76, 0.62, 0.74 | 0.74, 0.71, 0.54, 0.67 | 0.43–0.72 |
| Hue from the nearest vectors (R², the same kinds) | 0.99 | 0.98–0.99 | 0.96–0.99 |
| Keys' and doors' leftovers, angles | 31°, 87° | 70°, 88° | 34–46°, 70–80° |

Hue is a quantity within each kind here, a little more linearly than in
card 070's encoder, but not one shared by keys and doors (87°). Fresh
hues on every kind did not line the kinds up: nothing in card 054's
recipe rewards it, and a key and its locked door are not drawn in the
same pixel values (the door's panel is a darker shade).

## 6. Success criteria and prediction

1. **The relation from ignored directions** (the gate's last part), with
   no term naming colour in training.
2. **CHARTER's tiers** with the new encoder and the leftover P, on card
   074.2's seeds: tier 1 ≥ 99% (200); tier 2 (100) not worse than
   version 18's 59% (McNemar); tier 3 (30 seeds, memory stored) reported
   for both.
3. **Card 086's recall tables** on the new encoder (decoy memory):
   toggle 56 of 56, pick up 28, drop 42.

**Prediction.** Hue becomes a quantity (well over the 6–8% of card 089),
and since the same hues fall on every kind and the first layers are
shared, keys and doors code it alike (small angles); the probe passes.
Risk: without the relation term nothing rewards alignment across kinds,
only economy, so colour may still be coded per kind; the angles show it.
A smaller risk: MiniGrid's own six colours, now unseen, may be coded
less crisply than before, which tier 1 would show.

**Decision rules.** Keep if the gate and 1–3 hold: the next version, card
070's relation retired. Revise if the hue is a quantity but the angles
stay large (then alignment needs a further reason). Stop if hue is still
coded as categories.

**Budget.** Training about 20 minutes; data check, ensemble and probe
about 5; tiers 1–2 about an hour.

## 7. Result

The gate failed (section 5); the criteria were not run.

## 8. Decision

**Revise**, by the card's rule: the hue is a quantity (within every kind)
but keys' and doors' directions stay far apart (31°, 87°). Variety gives
the dimension within a kind; it does not by itself say that a key's hue
and a door's hue are the same thing, which in MiniGrid's drawing they are
not, pixel for pixel. What says it is the outcome: the door opens. Card
070's relation was learned exactly that way, predicting whether a toggle
opens the door from a distance between the two tiles and nothing else,
on fresh hues; it names no colour, only the outcome. The user decides
whether that counts as discovery (CHARTER lists it as the exception).

