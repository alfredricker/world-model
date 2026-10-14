---
id: "039"
title: what is in view as a set of things
rung: 0
serves: [P4, P12, P7, C2]
status: approved
verdict:
arch_version: 5
date: 2026-09-30
---

# 039: what is in view as a set of things

## 1. Question

Card 038's planner on codebook-free vectors (arm B) found no route, or a
false one, in the switch worlds. Part of the cause is that recall merges
what is in view into one vector, the largest value per dimension, and the
switch turning on then moves it by 0.003 under recall's weights (card 038,
section 7). If recall compares what is in view as a set of things instead,
thing by thing, does the planner read conditions that one thing's change
carries, and reach the goal in the familiar worlds? Before rung 1. Serves
P4 (keep the distinctions that change outcomes), P12 (conditions read from
memory), P7 (things as the units) and C2 (the same encoder vectors, no new
representation).

## 2. What changes

One component: the view part of recall's key, for pick up, toggle and drop.

```
card 038: view = max over the things in view, per dimension (one vector)
          distance = λv · |v − v'|
card 039: view = the set of things in view (each the encoder's vector)
          distance = Σ_{a in A} min_{b in B} λv · |a − b|  +  Σ_{b in B} min_{a in A} λv · |a − b|
```

- Each thing is matched with the most similar thing in the other view, in
  both directions, and the mismatches are summed (chamfer matching). One
  thing's change counts at its full size, whatever else is in view: the
  maximum hides it when another thing holds the largest value, and a mean
  divides it by the number of things.
- A thing is a distinct tile vector at the places in view other than the
  agent's own, the same places as card 038. Positions are not used here;
  the movement cards add them.
- **Nothing tells recall what to compare.** Every dimension of the encoder's
  vectors is compared, weighted by λv, fitted as before by leaving out one
  (front, held) pair at a time, now through the set distance. No dimension,
  part or attribute (colour, shape) is named (the user, 2026-09-30: what to
  compare must be learned).
- Unchanged: a key's own tries with the neighbours as a Dirichlet prior,
  conditions inferred from memory, the planner, walking, and arm B's
  encoders from card 038 (seeds 400–404).
- **Not changed, and expected to limit the both world:** toggle's weight on
  the held thing (0.0–0.1 in card 038). Whether a key fits a door is a
  relation between two things, the next card.

## 3. Dependencies

- Card 038's planner and recall: the upper bound passed its gate (100% in
  the four familiar worlds on oracle vectors). The card is not yet decided;
  its failures are the reason for this one.
- Arm B's encoders: passed card 038's gate (every tile distinct, 10 of 10
  seeds).
- Chamfer matching (Barrow et al. 1977, not in papi); Deep Sets
  (`1703_06114`) on what pooling a set into one vector keeps.

## 4. Data check

Stored toggles of a closed door with the switch on in view: 1,921–3,735
per world (seed-independent; the data are card 028's). Holding the
matching key with the switch on: 168–223 tries per world. In the both world
these 223 are the only tries that opened a door.

## 5. Feasibility gate

- **Upper bound:** oracle vectors with the set view reach the goal in 100%
  of familiar layouts, as with the merged view (card 038's gate).
- **Trivial baseline:** card 038's arm B, merged view, seeds 400–404:
  goal in 33–66% of layouts in the switch, either and both worlds.
- **Mechanism check, before acting** (seed 400): criterion 3's measure.
- **Shakedown:** spare seed 399, 100 familiar layouts per world.

Result of the gate (approved by the user on 2026-09-30; `slot_planner.py --check`, seed 400, 10 layouts
per world, holding nothing):
- **Switch world: 40 of 40 right.** The switch's change is visible: 0.486
  under the view weights, against 0.003 for card 038's merged view.
- **Either world: 20 of 40.** Every miss is "picking up a key opens the
  door". The view now does its part: 73% of recall's weight for that
  query is on views with the switch off. But toggle's held weight is 0.02,
  so tries holding a key count as much as tries holding nothing, and with
  the switch off they opened in 46–60% (a matching key): 49% of the weight,
  enough to tip the vote over one half. The prediction that the either
  world's loop goes was wrong: it needs what is held, as the both world
  does.
- Key and both worlds: 40 of 40, trivially (holding nothing never opens).

## 6. Success criteria and prediction

Arm B, seeds 400–404; each criterion in at least 4 of 5 seeds.

1. **Familiar worlds where one thing in view decides.** The goal reached in
   ≥ 99% of 500 layouts in the key, switch and either worlds, mean steps
   within 5% of card 029's.
2. **Nothing else lost.** Held-out effects per world at most 0.1 points
   below card 038's arm B on the same seed; in world (a), card 038's
   criteria 2 and 3 (new-colour cases ≥ 99%, goal ≥ 98% at ≤ 1.15 times the
   shortest route).
3. **The comparison does what is claimed.** Toggling a closed door holding
   nothing, in the view as it is and after each imagined change of view
   (each key picked up, the switch turned on), predicted as the world's rule
   gives it (the evaluator's) in ≥ 99% of cases, 50 layouts per world
   (switch, either). Holding nothing isolates the view from what is held.

The both world is reported, not a criterion. **Prediction:** criteria 1–3
pass. On these encoders any two tiles are at least 0.25–0.43 apart in L1
(card 038's gate), and the switch's change is no longer hidden by other
things. The either world's loop (pick up and
drop a wrong key) goes, because an imagined view without that key is near
views with the switch off, where toggling with nothing held fails. The both
world stays below 99%: among stored switch-on toggles the matching key is
223 of 3,735, and with no weight on what is held, recall cannot tell them
apart outside a key's own tries.

**Budget:** about 10 minutes per seed (card 038's arm B: 4–22, most of it in
failing layouts), so about an hour, run by the user. Set distances cost
more per prediction; seconds per layout are reported (P17).

## 7. Result

## 8. Decision
