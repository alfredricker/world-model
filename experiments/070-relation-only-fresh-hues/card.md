---
id: "070"
title: the relation as the only path, on fresh hues
rung: 6
serves: [P3, P19, P11, C5, C2]
status: done
verdict: fail
arch_version: 15
date: 2026-10-05
---

# 070: the relation as the only path, on fresh hues

Asked for by the user on 2026-10-05 ("yes" to a second attempt at
relations as learned weights, after card 069 stopped).

## 1. Question

Card 069's relation failed its gate for two reasons its diagnosis named:
the relation was not the only path (terms reading the door alone gave
each door colour its own threshold), and its tries could be memorized
(about 1,000 tries over nine fixed colours, each one fixed render). With
both removed, does "this key fits this door" become a distance under one
projection shared by both tiles, P, that holds for colours left out and
for hues the encoder never saw? P3 (relations), P19 (few tries), P11,
C5 (transfer), C2 (one encoder).

## 2. What changes

One component, the relation, as in card 069: how P is learned changes;
recall's use of it (`rel:P` replacing `rel:p`, the online refit) is card
069's, unchanged.

| | Card 069 | This card |
|---|---|---|
| What predicts a toggle's outcome in the relation term | a + u·z_front + v·z_held − g(z_front) · b · rel | **a − b · rel** only, a and b one pair for every try: no term reads one tile, so no tile can set its own threshold (the relational bottleneck, Webb et al.) |
| The tries | toggles from card 054's stream while holding something (nine fixed hues; 1,041 changed in 240,000 steps) | **relation play**: 512 fresh tries per update. The agent stands at a locked door and toggles, holding the door's key (1/2), a key of another hue (1/4), a ball or box of the door's hue (1/8) or nothing (1/8) |
| Hues | nine fixed | **a fresh hue per door and per other key**, uniform in RGB; no hue within 60 (RGB distance) of MiniGrid's six or the three held out (pink, brown, teal), none darker than 60 in every channel, and another key's hue at least 60 from the door's. No hue repeats, so no table can be learned (LESSONS: new fillers every task made the rule the only solution) |
| The outcome | the simulator's "the front tile changed" | the same, read from pixels: the front tile's render after the toggle differs from before beyond noise (card 053's threshold) |
| The rest of training | card 054's recipe and stream | the same |

Relation play is a declared curriculum (CHARTER C3), like play starts:
the world chooses where the agent starts; the agent's action and what it
sees are as always. It shows only locked doors, so nothing in it says
"locked door"; the relation must explain every outcome in it.

## 3. Dependencies

Card 069's code (`tools/card069/`: training, gate, `rel:P` in recall, the
online refit, the decoy world and its runner); card 054's encoder and
recipe; card 052's generator and tile drawing; card 053's noise
threshold. Relational bottleneck:
`the-relational-bottleneck-as-an-inductive-bias-for-efficient`,
`2012_14601` (ESBN). LESSONS, "a lookup table and a general rule fit
equally well".

## 4. Data check

Before any training, on 10 updates of relation play: the share that
opened (expected 1/2), the hues' smallest distance to the nine test hues
(at least 60), and that every try's outcome read from pixels equals the
simulator's (report only).

## 5. Feasibility gate

- **Upper bound:** the tiles' mean pixel colour as the relation: 81 of 81
  pairs (card 069).
- **Trivial baselines:** card 054's vectors under any weighting: 47 of 66
  left-out-colour pairs, 4 of 9 unseen-hue pairs; card 069's relation:
  45 and 5.
- **Step A (2 minutes): P alone on card 054's frozen vectors**, trained on
  relation play. If it passes the gate, the encoder does not change and
  step B is skipped.
- **Step B (about 17 minutes): the encoder and P,** card 054's recipe plus
  the relation term (weight 1), 12 checkpoints, as card 069; P, a and b
  start from step A's (written after step A's gate, before step B ran).
- **Gate** (each step; card 069's probe on MiniGrid's tiles as the agent
  draws them, the relation as trained, only its threshold fitted):
  ≥ 64 of 66 left-out-colour pairs and ≥ 8 of 9 unseen-hue pairs; for B,
  card 054's pair margin and visibility margin still met.

## 6. Success criteria and prediction

Run only if a step passes the gate. Card 069's test: version 15 with
card 066's memory recipe (card 068 was not kept), the decoy world's six
folds, arms with P frozen and refit online.

**Amended before the criteria ran** (the user, 2026-10-05, after step
B's gate and a smoke test). Card 069's folds left a hue out of memory
entirely; the smoke test (step A's P, fold red, 4 episodes) showed the
agent then never learns that the open door of that hue can be walked
through, so the door never becomes a condition and the agent falls back
to random actions (617 of 640 steps in the failed episode): a test of
walkability, not of the relation. Now memory holds every hue (one random
play memory of the decoy world), and each fold drops only the toggles at
a locked door of its hue while holding the key of its hue. Recall has
then never seen this key open this door; everything else, walking
through the open door included, stays. Criteria 2 and 3 read "the hue
left out" as "the hue's opening tries left out".

1. **CHARTER's tiers** with the online arm: tier 1 ≥ 99%; tiers 2 and 3
   not worse than the best version.
2. **Recall:** toggling with a key opens the door exactly when the hues
   match, ≥ 35 of 36 door and key hue pairs with the hue left out of
   memory, in every fold.
3. **Judged by tries:** decoy-world episodes whose door has the left-out
   hue: success ≥ 99%, and the decoy tried at most once per episode on
   average; online against frozen reported.

**Prediction.** Step A fails (a readout fitted on keys reads 0 of 9 door
colours on card 054's vectors, so colour is not linear across kinds).
Step B passes the six MiniGrid hues; the three unseen hues are the risk,
less than in card 069 since fresh hues fill the colour space around them.

**Decision rules.** Keep if a step passes the gate and criteria 1–3
hold. Revise once if criterion 2 fails only for some folds. Stop if both
steps fail the gate: then a relation over this encoder's vectors does not
form even with no shortcut and no repeated colours, and the next question
is the encoder's architecture, which is the user's call.

**Budget.** Step A 2 minutes; step B about 17 minutes; criteria about 45
minutes (twelve decoy runs of about 2 minutes each, tier 2 about 15).

## 7. Result

**Data check** (`runs/070/datacheck.json`; 5,120 tries of relation
play): 51% opened; the outcome read from pixels equals the simulator's
in 5,120 of 5,120; no door or other-key hue within 60.0 of a test hue;
MiniGrid's own toggle rule agrees in 200 of 200. Tiles drawn from three
renders per kind are within 1.2 levels of 255 of a direct render (noise:
4 levels).

**Gate** (`runs/070/gate_a.json`, `gate_b.json`; MiniGrid's tiles as the
agent draws them; only the threshold is fitted):

| | Card 054 | Card 069 | Step A: P on frozen vectors | Step B: encoder and P |
|---|---|---|---|---|
| Left-out colour pairs (gate ≥ 64 of 66) | 47 | 45 | 63 | **66** |
| Unseen-hue pairs (gate ≥ 8 of 9) | 4 | 5 | 8 | **9** |
| Pairs with an unseen hue | – | 22 of 45 | 40 of 45 | **45 of 45** |
| Largest same-colour relation; smallest different | – | 10.9; 3.8 | 2.9; 2.4 | **1.8; 2.9** |

- **Step A** (3,000 updates, 2 minutes): batch accuracy on fresh hues
  97%; one pair short of the gate. Against the prediction, card 054's
  frozen vectors already carry colour in a form one shared projection
  can compare; card 069 failed for its shortcuts and its data, not the
  encoder.
- **Step B** (12 checkpoints, 20 minutes; `runs/070/encoder.json`):
  **passes.** Fresh-hue accuracy 100% from the first checkpoint, relation
  loss 0.0003 at the end; card 054's pair margin (hinge 0.0002–0.0014)
  and visibility margin (0) met throughout; transitions 0.004–0.015.
  Every same-colour pair of MiniGrid's six and the three unseen hues lies
  closer under P than every different-colour pair.

**Criteria** (step B's encoder; `runs/070/decoy_*`, `runs/070/tier1.json`).

| Fold (opening tries removed) | Recall right of 36 | Its wrong pair | Success, online / frozen | Decoy tries per episode |
|---|---|---|---|---|
| red (77) | 35 | red key on red door: "no" | 7% / 7% | 0.2 |
| green (55) | 35 | green on green: "no" | 7% / 7% | 0.2 |
| blue (88) | 35 | blue on blue: "no" | 7% / 7% | 0.2 |
| purple (79) | 35 | purple on purple: "no" | 7% / 7% | 0.2 |
| yellow (69) | 35 | yellow on yellow: "no" | 7% / 7% | 0.2 |
| grey (81) | 35 | grey on grey: "no" | 7% / 7% | 0.2 |

- **Criterion 1:** tier 1 **100%** (200; 20.8 steps, as version 15).
  Tier 2 was stopped at the user's request (2026-10-05): with criterion 3
  failed it could not change the decision, and at 0% it cannot be worse.
  Tier 3 cannot be built.
- **Criterion 2: met as written, failed in substance.** 35 of 36 in
  every fold, but the one wrong pair is always the one the fold removed:
  the hue's own key on its own door, predicted not to open. "At least 35
  of 36" was too loose a bar; it admits exactly the failure the
  criterion was for.
- **Criterion 3: not met.** 7% in every fold and arm. Believing that no
  key opens the door, the agent finds no plan and acts at random (616 of
  619 steps in fold red); the identical numbers across folds are the same
  seeded random play, which colours do not change. Online refits made no
  difference (7 in 100 episodes).

**Why** (`tools/card070/why_fold.py`, fold red, report only). Recall's
toggle admits `rel:P` (and only it, beside the front tile), but its
weights per unit of distance are 38–124 for the front tile's four parts
and 3 for `rel:P`. Any stored try whose front tile differs from the
query's weighs about e⁻¹²⁰; for "red key on the red door" only tries at
the red locked door count, all failures (other keys, an empty hand):
total weight on openings 10⁻⁵⁶, on failures 0.004. Admission fits these
weights by predicting each stored try from the others, and every opening
in memory has twins at the same door with the same key, so the door's
identity predicts as well as the relation and the fit leans on it:
LESSONS' table and rule, one level up, in recall's weights.

## 8. Decision

**Revise.** The relation now exists in the encoder (the gate: 66 of 66,
9 of 9, every same-colour pair closer than every different one), but
recall weighs it only within one door, because its fit never has to
predict a combination it has not seen. The revision changes how recall
fits the weight on `rel:P` against the tiles' identity: each try
predicted only from tries of other front and held combinations (leave a
combination out, not a try), which is the question recall answers when
the agent faces a new door and key; seen combinations keep card 050's
own tries first. It needs the user's approval. Version 15 is unchanged
until then.
