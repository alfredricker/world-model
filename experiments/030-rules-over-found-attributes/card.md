---
id: "030"
title: rules over found attributes
rung: 0
serves: [P3, P4, P1, C5]
status: draft
verdict:
arch_version: 5
date: 2026-09-29
---

# 030: rules over found attributes

## 1. Question

Can the agent find, from pixels alone, attributes such as shape and
colour, and state its rules over them, so that a rule carries to a
pairing it has never seen? The counted model (card 028) learns one rule
per door colour ("the red door opens when held = key red"). Can it learn
one rule, "a door opens when the held key has the door's colour", and
use it for the blue key at the blue door, a pairing withheld from its
experience? Serves P3 (here a new combination, not yet a new colour), P4
(the evidence chooses which attribute a rule uses), P1 and C5. Follows
[card 029](../029-subgoals-from-the-model/card.md) (keep), its appendix
B, and LITERATURE.md's current focus.

## 2. What changes

One component: what the model's outcome entries and rules are stated
over. The moves (card 028), conditions, subgoals and walking (card 029)
stay.

| Part | Version 5 (card 029) | Card 030 |
|---|---|---|
| Attributes | None: each exact tile is its own appearance | Found. Two tiles have the same shape when a one-to-one change of pixel values turns one into the other. A colour change is such a change, made across shapes, that leaves the learned model's outcomes unchanged (a symmetry of the model) |
| Outcome entries (pick up, drop, toggle) | One per appearance in front | Entries whose appearances a colour change relates are pooled into one when the evidence prefers it; each try is carried into one member's colours |
| Rule conditions | "held = X", "X in view" | Also "held = X in this entry's colour" (a match) and "held has X's shape" (any colour); the evidence chooses |

In the key world, with the blue key never held at the closed blue door:

```
seen:    toggle closed red door    -> open red door    if held = key red
         toggle closed green door  -> open green door  if held = key green
         toggle closed blue door   -> nothing  (tried holding nothing, key red, key green)
pooled:  toggle closed door, colour c -> open door, colour c   if held = key, colour c
test:    closed blue door, blue key in the room -> subgoal "held = key blue"
```

One pooled rule over 372 openings costs less than two rules: by card
010's evidence, about 20 nats less (appendix A). Nothing names colour or
shape: pixel values are only compared for equality (no colour channels).
Pooled entries are written back for every member, so card 029 reads them
unchanged.

**Experience.** Key world, 5,000 episodes of random play (card 028).
In blue-door rooms (a third) the door starts open and the keys are red
and green (a declared curriculum). Every appearance is seen, but the
blue key is never held at the closed blue door: a withheld pairing
(PGM's combination split), not a withheld colour (appendix B).

**Arms,** key world, 500 new layouts, each with a closed blue door, the
blue key and one other key (200-step budget):

1. Upper bound: the shortest route (evaluator); and card 029's agent
   learned from full experience (every pairing seen).
2. **Main: rules over found attributes.**
3. Trivial baseline: card 029 unchanged on the same experience. The blue
   door has never opened, so it cannot plan to open it.
4. Ablation: pooling without the match condition. Without a shared
   variable, pooling can only say "any key opens any door" (Plotkin),
   which the near misses refute.

**Control worlds,** full experience, main arm: card 029's four (key,
switch, either, both); a permuted world (red key opens the green door,
green the red, blue the blue), where "same colour" is wrong; an any-key
world, where the rule is about shape.

## 3. Dependencies

Cards 028 (the counted model; keep), 029 (subgoals, walking; keep) and
010 (the rule finder and its costs). Pooling follows
`an-algebraic-approach-to-abstraction-in-reinforcement-learni` (a
symmetry leaves the dynamics unchanged, checked against the model) and
`plotkin-inductive-generalization` (the same pair of differing terms
becomes one variable); the tests follow `1807_04225` (combination split)
and `1902_00120` (near misses). Finding the attributes is new (no paper
does it from pixels without labels), so the gate tests it on its own,
against the simulator's labels, before anything uses it.

## 4. Data check

Every step of 5,000 episodes under the experience above (seed 11;
scratch count, before storage sampling):

| Event | Count |
|---|---|
| Red door opened (red key held) | 179 |
| Green door opened (green key held) | 193 |
| Near misses at the red and green doors (another colour's key held) | 123–160 each |
| Closed blue door toggled, holding nothing / key red / key green | 1,575 / 129 / 148, never opened |
| Blue door closed by toggling | 756 |
| Blue key picked up (red- and green-door rooms) | 7,026 |
| Agent in the open blue doorway (steps) | 9,338 |
| Blue key held at the closed blue door | 0 (withheld) |

The rarest events are the openings, as in card 028 (172–213 per colour).

## 5. Feasibility gate

- **Attributes against the simulator's labels:** the shape groups and
  colour changes found are exactly those in appendix A (scratch code
  found exactly these while drafting). Pooling breaks nothing seen: held-out effects in
  the training worlds stay exact (card 028's check).
- **Upper bound:** the shortest route solves every test layout, and arm
  1's learner, having seen every pairing, reaches the goal in ≥ 99%.
- **Trivial baseline:** arm 3, predicted near 0% (the goal is behind the
  door).

Result of the gate, before the main run:

## 6. Success criteria and prediction

1. **The withheld pairing, predicted.** On held-out transitions from
   1,000 episodes of random play in test layouts (which contain the
   withheld pair), the pooled model predicts ≥ 99.9% of effects exactly,
   including every toggle of the closed blue door while holding the blue
   key; and each of the 12 toggle cases (door colour × held: nothing,
   key red, green, blue) gets the right outcome. Arm 3 gets the withheld
   case wrong by construction; arm 4 is expected to as well, or to
   predict that wrong keys open.
2. **The withheld pairing, acted.** Arm 2 reaches the goal in ≥ 98% of
   the 500 layouts, with mean steps when successful ≤ 1.15 times the
   shortest route's (card 029: 1.03 in the key world). Compared with arm
   3 (predicted near 0%) and arm 1 (≥ 99%).
3. **The evidence picks the attribute each world uses, and nothing seen
   breaks.** In each of the six control worlds, every toggle case is
   predicted right (door colour × held thing × switch on or off, 24
   cases), and acting reaches the goal in ≥ 99% of 500 layouts (card
   029: 100% in the first four). The door rule uses the kind of
   condition the world uses: the key in the door's own colour (key,
   either, both), a switch that is on (switch), any key's shape (any
   key), and not the door's own colour (permuted).

Reported with them: the attributes found; per world, the kind of
condition each rule uses and the evidence for pooling in nats; the chains
worked out on the withheld pairing; arm 4's acting.

Prediction: 1–3 pass; pooling changes only what the model says where
it has no experience. Risks: the permuted world needs partial pooling
(red and green doors under the swap of the two, blue alone); card 010's
greedy search, with three times the conditions, could settle on a longer
equivalent list; and the pixel check is loose (appendix A), so the model
check must do its part. The card cannot show transfer to a colour never
seen (appendix B), or to a new shape.

Budget: collecting 5,000 episodes in seven worlds dominates; learning
takes seconds and acting about 0.02 seconds per layout (card 029).
Estimate 10–20 minutes; over 30, handed to the user as commands.

## 7. Result

## 8. Decision

## Appendix A: the procedure

1. **Same shape.** Two tiles have the same shape when a one-to-one map
   of pixel values turns one into the other, each pixel's RGB triple
   taken as one opaque value. The map is the difference between them.
2. **Candidate recolourings.** Permutations of the appearances that keep
   each tile in its shape group, such that every value any tile changes
   changes the same way wherever it changes. A tile may keep a value
   another tile changes: in the doorway the agent's red triangle keeps
   its shades while the door's change.
3. **Model check.** Keep the candidates under which the model's certain
   outcomes are unchanged: the moves' outcomes by appearance in front,
   the draw and undraw tables, and the pick-up, drop and toggle entries
   with a single outcome. Entries with rules are left to step 4.
4. **Pooling.** For each group of outcome entries whose appearances in
   front the kept recolourings relate (for toggle, the closed doors),
   card 010's evidence compares:
   - separate entries, as card 028;
   - pooled: each try carried into one member's colours by the
     recolouring that maps its entry there, then one rule list over all
     tries.

   Conditions offered in the pooled list: "held = X" and "X in view"
   carried with the try (a match), the same uncarried (one appearance),
   and "held has X's shape" (any colour). Where several recolourings
   map a member to the reference, and for blocks of members (red and
   green pooled, blue alone), the evidence chooses. Costs are card
   010's, plus the log of the number of pooling options.
5. **Write-back.** The chosen pooled entry is carried to every member
   and stored as that member's entry, in card 028's conditions ("held =
   key blue" for the blue door); a shape condition becomes one rule per
   member of the shape. Card 029 reads the table unchanged.

**Checked while drafting** (scratch code, the 19 tiles the world
shows). The same-shape groups are the three keys, the three closed
doors, the three open doors, the three "agent in an open doorway", and
wall with goal (a coincidence: one pattern of pixels in two colours).
336 candidate recolourings pass step 2. Step 3 leaves 16: forward onto
the goal ends the episode and onto a wall does not; toggling turns each
open door into the closed door of its colour; the agent in a doorway
undraws to that doorway. The 16 recolour keys and doors together (the 6
permutations of three colours), only the keys, or only the doors. In the
key world the door rule rejects the last two kinds in step 4; in the
switch world they stay, since its door rule does not involve keys.

Evidence estimate for the key world, from section 4's counts
(unweighted): separate entries about −42.5 nats (two rule lists, and the
blue door's single outcome); pooled about −23 (one rule over 372
openings, one default over about 8,400 tries, a longer condition list,
and the pooling choice). Pooling wins by about 20 nats.

The permuted and any-key worlds, and the curriculum, are copies of the
logic-door world in the card's code (`src/` unchanged): keys drawn in
their own colours, and the door's requirement changed.

## Appendix B: a colour never seen

The user's addendum (card 029, appendix B) names a withheld colour as
the test. It is not this card's test, for one reason in our tiles: a key
and its closed door share no pixel values (the key's reds are 255, 226,
198, 170 and 85; the closed door's are 192, 161 and 114). With every
colour seen, the open door (which shares 170 and 198 with the key) and
toggling link them. With a colour never seen, the new key and the new
door are each recognisable as a recoloured key and a recoloured door,
but nothing in their pixels says they are the same new colour. There are
two routes, each a card after this one:

- **A declared prior on how pixels are made:** each pixel is a colour
  times a shade (RGB channels). The new key's recolouring then predicts
  the new door's pixels. Simple, but it builds in part of the answer.
- **The fewest new colours:** one new colour explaining both key and
  door costs less than two. The agent predicts that the new key opens
  the new door, tries it, and keeps or drops the prediction by the
  outcome (P5, P15). Nothing about pixels is declared; a near miss (two
  new colours) costs one failed try.

Both need what this card does not build: recognising a new appearance as
a recoloured known one, in every table, moves and walking included.
