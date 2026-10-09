---
id: "087"
title: properties as learned action effects, from a network on the encoder's vectors
rung: 1
serves: [P1, P4, P5, P19, P6]
status: done
verdict: fail
arch_version: 18
date: 2026-10-08
---

# 087: properties as learned action effects

Drafted at the user's request (2026-10-08), after card 086's smoke test.
CHARTER's "Current direction" now holds the user's framing: a tile's
property is what an action does to it, learned from the agent's own
tries, carried to tiles that look different but are the same kind, and
"unknown" where nothing is like it. Replaces the card 087 proposed on
2026-10-08 (walkability by identity code).

Approved by the user (2026-10-08: "yes, they do").

## 1. Question

Today what a tile does is recall over tries on that exact tile vector:
walkable is card 038's recall of a forward move on the tile in front
(`M.free`), and card 084.3's roles (pick up takes it, toggle changes it)
are recall's predictions per tile. A tile memory has not seen reads as
"not walkable": a new door colour (card 037: forward onto a new open
door right in 3 of 9 seeds; card 070's smoke test with a hue left out),
or a result tile imagined by card 038's ways (tier 2's blue door, card
068; card 086's red door, 0.24 from the real open door, 7 of 10 against
10 of 10). Does a small network on the encoder's vector, trained on the
agent's own tries in a world whose objects take a fresh hue every
episode, predict these properties for tiles it has never seen, improve
with tries (P19), know when it does not know (P5), and, as the prior for
tiles without own tries, lift the agent where walkability blocked it?
P1, P4 (drop what does not change outcomes), P5, P19, P6.

## 2. What changes

One component: where a tile's properties come from.

```
version 18:  walkable(h) = recall_forward(h) predicts "moved"         (card 038, per tile vector)
             roles(h)    = recall's pick up / toggle effect on h       (card 084.3; read by card 086)
this card:   p(h)        = (N_own(h) + β f(z_h)) / (|N_own(h)| + β)     own tries first (card 050)
             f           = a network on the tile's encoder vector z_h
```

- **The properties** (f's outputs, from the tile's 32 numbers alone):
  forward: moved onto it, blocked, or ended the episode (one outcome of
  three); pick up with an empty hand takes it; toggle can change it in
  some situation. What it takes in the present (which key, an empty
  hand) stays with recall's conditions: a locked door has "toggle can
  change it", and the conditions say when (card 086's cells, card 047's
  situations). A property names no situation, so no colour enters it.
- **The network:** an ensemble of 5 MLPs (32 → 64 → 64 → 5), each from
  its own initialisation; the mean is f, the spread is the uncertainty
  (deep ensembles, Lakshminarayanan et al. 2017). The encoder (card
  070's, the agent's) stays frozen: f is a readout of the one encoder
  (CHARTER, "One encoder").
- **Property play** (a declared curriculum, C3, like card 070's relation
  play): MiniGrid rooms with card 068's play starts, every coloured
  object a fresh hue drawn each episode with card 070's sampler (at
  least 60 from MiniGrid's six and the three held-out hues). Each play
  start faces one tile of a kind the tiers draw (wall, floor, goal, open,
  closed or locked door, key, ball, box), holding nothing, a key (half
  of them the hue of a door in the room), a ball or a box, and takes
  forward, pick up or toggle; a locked door is toggled with several
  holdings. The outcome is read from the next view as memory reads it.
  Kinds are drawn so each property's positives are at least a quarter of
  its batch (LESSONS: rare events). A hue never repeats, so no property
  can be learned per colour (LESSONS: table and rule).
- **Network or neighbours, by held-out prediction** (CHARTER): for each
  property, f replaces recall's neighbours as the prior only if it
  predicts memory's stored tries better with each identity code left
  out in turn (card 084.4's unit); otherwise that property stays as now.
  β is fitted by leave-one-try-out likelihood, as card 050's α. A
  failed try on the tile itself still overrules.
- **Unchanged:** recall for pick up, toggle and drop in a situation;
  the moves' transformations; walking (card 067), which reads walkable;
  planning and trying. Version 18 uses only walkable and ended; pick up
  and toggle are reported here and become card 086's roles when it runs.
- **Not acted on yet:** "unknown" (spread above a threshold fitted for
  calibration) is reported, not used; trying on unknown is trying's
  component, a later card.

## 3. Dependencies

Card 070 (the encoder; the hue sampler, `tools/card070/relplay.py`); card
068 (play starts); card 038 (recall of moves, the ways that imagine
result tiles); card 050 (own tries first, its leave-one-out fit); card
084.4 (identity codes left out); card 085.3's stored fits (tier 3).
Deep ensembles (`1612_01474`, Lakshminarayanan et al. 2017), and affordances
as action possibilities (Gibson 1979; Montesano et al. 2008): LITERATURE's
current focus.

## 4. Data check

Per property, in property play: tries, positives, and the kinds and
states they come from; that no hue lies within 60 of a test hue; that the
outcome read from the next view equals MiniGrid's rule in 200 of 200. In
memory (the evaluation's prior): stored forward, pick up and toggle tries
per identity code.

## 5. Feasibility gate

- **Upper bound:** f trained and tested on property play (fresh hues
  both): every property ≥ 99%.
- **Trivial baselines:** the majority answer per property; recall per
  tile (version 18) on the gate's tiles.
- **Gate** (scored against MiniGrid's rules, never against the model's
  codes, LESSONS): (i) MiniGrid's own tiles as the agent draws them,
  every kind and state in the six hues and the three held-out hues
  (about 57 tiles): every property right; (ii) the result tiles card
  038's ways imagine for a toggled door in each of card 086's colour
  folds, the red one included: all walkable; (iii) for each property,
  f as prior predicts memory's tries with identity codes left out better
  than recall's neighbours.

**Data check** (`runs/087/datacheck.json`): the tries' outcomes per kind
are MiniGrid's rules (floor and open door moved onto; goal ends; key,
ball, box taken; open, closed and locked doors and boxes changed by
toggle, the locked door only with the key of its hue). In 20,000
instances: 4,588 moved onto, 2,194 ended, 6,667 taken, 8,909 changed by
toggle; no hue within 60.0 of a test hue. Outcomes are read from a
3 × 4 room's next state (place, held tile, front tile), as memory reads
them (`grid_codes`), not from pixels.

**Gate result: passed** (`tools/card087/props.py`, `gate2.py`;
`runs/087/gate.json`, `gate2_*.json`). Ensemble of 5 on 20,000
property-play instances, 50 s.

| Part | Network | Version 18 (per-tile recall) | Majority |
|---|---|---|---|
| Upper bound: fresh hues, held-out instances | 100%, 100%, 100% | – | – |
| (i) MiniGrid's tiles, 9 colours: forward, pick up, toggle | 57, 57, 57 of 57 | forward 51 of 57 | 46, 30, 36 |
| (ii) imagined door after toggling, six folds: walkable | 6 of 6 | 3 of 6 (red, green, yellow: "blocked") | – |
| (iii) identities left out, decoy memory: forward, pick up, toggle | 27, 27, 27 of 27 | neighbours 21, 26, 25 | – |
| (iii) tier 2's memory | 30 of 30, 31 of 31, 35 of 35 | neighbours 24, 30, 34 | – |

- Version 18 reads all 39 of the agent's catalogue tiles right, but the
  same tiles drawn by card 052's drawing (L1 at most 0.054 from the
  catalogue's) flip 6 readings: the red, green, yellow, pink and brown
  open doors and the goal read "blocked". Per-tile recall breaks at a
  shift smaller than the noise between renders; the network does not.
- In tier 2's memory version 18 reads the real open green door as
  "blocked"; the network as walkable. The blue door's imagined tile is
  the real one there (card 066's memory; card 068's case was with play
  starts).
- The network is least sure of the blue open door (P(moved) 0.68, the
  imagined one 0.63), right all the same.
- Calibration on fresh hues: expected calibration error 0.0001 (pick up,
  toggle). P19: with 30 instances (7–12 positives) 53, 57, 49 of 57;
  with 300 (80–145 positives) 57 of 57 on all three.
- Box left out of property play (report): pick up reads "taken"
  (0.63–0.93) with spread 0.30 against 0.00 on the trained kinds, so the
  ensemble flags it; toggle reads "no change" at 0.00 with no spread:
  confidently wrong. The ensemble's spread catches some new kinds, not
  all (P5 partly).

## 6. Success criteria and prediction

1. **New appearances:** the gate's (i) and (ii) in the agent's own
   reading (`free_of` and the roles), against version 18's per-tile
   recall. Reported with it: P19's curve (accuracy on (i) after 10, 30,
   100, 300, 1,000 positives per property), calibration (expected
   calibration error on held-out fresh hues) and the spread on a kind
   left out of property play (the box), which should read "unknown".
2. **CHARTER's tiers** (version 18 with the properties, card 074.2's
   seeds, McNemar p < 0.05): tier 1 ≥ 99% (200); tier 2 (100) not worse
   than version 18's 59%; tier 3 (30 seeds, both versions, once the
   stored fits are built) not worse.
3. **A hue never in memory:** card 069's decoy folds with the hue left
   out of memory entirely (`--hold hue`), 100 episodes each with trying:
   success in every fold at least version 18's with only the opening
   tries removed (93–96%, card 072). Card 070 found this test failing on
   walkability (the open door of that hue never known walkable), so it
   reads the property in the agent.

**Prediction.** Walkable and pick up are right on every tile (i); the
imagined doors are walkable (ii), since an open door of any hue is
"moved onto" in property play. Toggle is the risk: "can change in some
situation" needs the curriculum to offer the matching key often enough.
Tier 1 unchanged; tier 2 rises (the blue door), not to 99% (card 086's
untraced failures). Criterion 3 rises from near random play (card 070's
smoke test) toward 93–96%; whether the key is found within a few tries
is trying's (card 072), not this card's.

**Decision rules.** Keep if the gate and 1–3 hold (the next version;
card 086's main runs then read these roles). Revise if 1 holds and 2 or 3
fails. Stop if the gate fails on (i) with fresh hues: then a readout of
this encoder does not carry properties to new appearances, and the next
question is the encoder's training.

**Budget.** Gate about 15 minutes (property play about 5, training the
ensemble about 5, scoring). Criteria: tiers 1–2 about an hour and the six
folds about 40 minutes, handed to the user as commands; tier 3 after
card 085.3's one-hour build.

## 7. Result

`runs/087/run.sh`; version 18's flags with stored fits (bit-identical,
card 085.3 (a)); `WM_PROPS=1` adds the properties (walkability only, the
one reading version 18 has). β fitted at 0.001: a tile with its own
forward tries is decided by them; the network decides only tiles memory
has not tried.

| | With properties | Version 18 |
|---|---|---|
| Tier 1 (200) | **100%**, 24.4 steps | 100%, 24.4 steps (card 074.2) |
| Tier 2 (100) | **64%** | 59%; 9 gained, 4 lost, McNemar p = 0.27 |
| Tier 3 (30, 1 hour per episode) | not run yet | **5 of 30**; 23 stopped by the hour, 2 at the 3,600-step limit |
| Hue never in memory, red / green / blue | **100 / 75 / 98%** | 5 / 11 / 1% |
| purple / yellow / grey | **97 / 100 / 96%** | 0 / 16 / 5% |

- **Criterion 1: met** (the gate's (i) and (ii), section 5).
- **Criterion 2: met** for tiers 1 and 2; tier 3 pending for the
  properties. Version 18's tier 3 baseline (`runs/087/tier3_v18.json`,
  run by Claude at the user's request, 44 GB cap, peak 25.5 GB): 5 of 30
  within one hour of wall-clock time per episode, 1.68 s per step with 20
  episodes sharing 24 cores. In the first 20 (run together) 1 succeeded
  and 19 hit the hour at 1,340–2,062 steps; in the last 10 (less
  sharing) 4 succeeded (1,415–2,362 steps), 2 failed at the 3,600-step
  limit and 4 hit the hour, with 629–1,106 random steps each. A first
  attempt crashed after an hour on a per-episode counter that held a
  list (card 085's setup statistics), fixed in `run.py`.
- **Criterion 3: met in five folds, not in green** (75% against the
  93–96% bar). Version 18 falls to random actions (29–62% of steps)
  when the hue was never in memory, as card 070's smoke test found; with
  the properties, random actions are 0% in three folds and 12–20% in
  blue, purple and grey. The 25 green failures plan nothing from the first step and
  then turn between the green key and the green door for 640 steps
  (trace, seed 1001011). Not traced further (the user, 2026-10-08):
  the green door is "openable", and recall predicts that the green key
  is picked up and opens it, so the missing plan lies elsewhere.

## 8. Decision

**Revise** by the card's rule (criterion 3 fails in one fold of six).
The properties did what they were for: every tile and imagined door is
read right, and a hue never in memory goes from 0–16% to 96–100% in five
folds and 75% in the sixth, with no tier worse. The green failure is a
planner that finds no plan with every input it needs predicted right; it
is traced after card 089, with tier 3 once its memory builds.

