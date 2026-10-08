---
id: "090"
title: the encoder trained on fresh hues, so colour is a dimension shared by every kind
rung: 6
serves: [P3, P4, C5, C2]
status: approved
verdict:
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
