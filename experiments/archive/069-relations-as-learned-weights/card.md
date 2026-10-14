---
id: "069"
title: relations as learned weights over the encoder's vectors, refit online
rung: 6
serves: [P3, P19, P11, C5, C2]
status: done
verdict: fail
arch_version: 15
date: 2026-10-05
---

# 069: relations as learned weights over the encoder's vectors, refit online

Asked for by the user on 2026-10-05: "a learned comparison of the encoded
vectors … the weights over the numbers", learned online if it can be.

## 1. Question

Can "this key opens this door" be a relation recall learns as weights over
the whole of two tokens' vectors, compared through one projection shared
by both (a relational bottleneck), refit online when a try surprises, so
that it holds for colours memory has never seen together and for hues the
encoder never saw? P3 (relations and binding), P19 (from few tries), P11
(refits run over all of memory, so nothing is erased), C5 (transfer).

## 2. What changes

One component, the relation, used in two places.

| | Version 15 | This card |
|---|---|---|
| Recall's relation candidate | `rel:p`: the front and held tiles' L1 distance within one quarter p of the vector (4 candidates) | `rel:P`: ‖P z_front − P z_held‖₁, P one learned projection (r × 32, r = 8) shared by both tokens; one candidate, admitted like any other |
| How P is learned | – | (a) with the encoder, from stored tries: the outcome of a toggle or pick up is predicted from the relation alone (the bottleneck), so the encoder must represent what the two tokens share alike in both; (b) refit online: after a try recall predicted wrong, 50 gradient steps on recall's leave-one-out likelihood over all of memory, from the current P |
| The encoder | card 054's, frozen | card 054's, trained further once with (a), then frozen while acting |

Never a query–key product (Wq a)·(Wk b): LESSONS records it fitting three
colours and carrying to none, while a distance along learned weights
carried. The projection is shared, so a hue enters it the same way from
a key and from a door.

**Step (a) as built** (`tools/card069/train.py`, written before its gate
was read). Card 054's recipe and stream unchanged (`runs/054/b.sh`), from
its saved encoder, 12 checkpoints of 2,000 updates, plus one term on 512
toggle tries per update, half of them changed, drawn from a buffer of
toggles made while the hand holds something (by pixels: the held tile is
not the empty hand's tile). The outcome (the front tile changed) is
predicted as

  logit = a + u·z_front + v·z_held − g(z_front) · b · ‖P z_front − P z_held‖₁

where u, v and g read one tile each, so the two tiles meet only in the
relation, and g (a sigmoid of z_front) lets the relation count only where
the front tile makes it matter (a closed door opens whatever is held).
Pick ups are left out: no pick-up outcome depends on two tiles together
in this stream. In 100,000 steps the stream has 58 toggles at a locked
door with the matching key (all opened) and 70 with another key (none
opened) (`runs/069/datacheck.json`).

**Why the encoder takes part** (section 5's probes, `runs/069/gate.json`).
On card 054's vectors of the tiles as the agent draws them, no weighting
tells a key and a door of the same colour from a pair of different
colours: 47 of 66 pairs right for a colour left out of the fit (always
"different" scores 60), 4 of 9 for the unseen hues; a colour readout
fitted on keys reads 0 of 9 doors right, and the conv layers' pooled
features do no better (47–49 of 66, on tiles with MiniGrid's grid lines).
The tiles' mean pixel colour gets 81 of 81. The encoder entangles colour with kind;
CHARTER's "one encoder" lets a new comparison train it (C2), and the
comparison here is what trains it.

## 3. Dependencies

Card 054's encoder and card 052's generator (MiniGrid's objects in nine
hues, three held out: pink, brown, teal); card 049's admission; card 051's
index; card 066's harness. Relational bottleneck:
`the-relational-bottleneck-as-an-inductive-bias-for-efficient`,
`2012_14601` (ESBN); online similarity learning: OASIS (Chechik et al.
2010, not in papi).

## 4. Data check

The relation needs tries where a key and a door differ. Tier 2 has none:
one key and one door of one colour per episode (card 066's memory: 0
toggles with another key; 34–57 with the matching key per colour).
Card 052's generator (BabyAI missions, several keys and doors) is checked
for toggles on a locked door with a key of each other colour before (a).
The test world below has a decoy key in every episode.

## 5. Feasibility gate

- **Upper bound:** the tiles' mean pixel colour as the relation: 81 of 81
  pairs (section 2).
- **Trivial baseline:** card 054's vectors, any weighting: 47 of 66 for
  left-out colours, 4 of 9 for unseen hues.
- **Gate for (a), before the main run:** after the encoder is trained
  through the relation, the same probe with the relation as trained
  (‖P z_door − P z_key‖₁; only the threshold is fitted, on the colours
  in the fit): ≥ 64 of 66 left-out-colour pairs and ≥ 8 of 9 unseen-hue
  pairs, with card 054's identity and its planner-level checks unchanged
  (its pair margin, recall's matching key). Tiles are drawn as the
  agent's catalogue draws them.

## 6. Success criteria and prediction

Test world (development, CHARTER): DoorKey-8x8 with a decoy key of another
colour in the first room, MiniGrid's six hues, one hue left out of memory
at a time; 100 episodes per fold whose door has the left-out hue. Arms:
P frozen after setup, and P refit online.

1. **CHARTER's tiers** with the online arm: tier 1 ≥ 99%; tiers 2 and 3
   not worse than the best version.
2. **Recall:** for every door hue and key hue (36 pairs), toggling with
   the key opens the door exactly when the hues match: ≥ 35 of 36 with
   the hue left out of memory, in every fold.
3. **Judged by tries:** in the test world, episodes whose door has the
   left-out hue: success ≥ 99%, and the decoy is tried at most once per
   episode on average; online against frozen reported.

**Prediction.** (a) passes the gate on the six hues; the unseen hues are
the risk (the encoder never saw them). Online refits matter only when
memory misleads, so the arms differ little.

**Decision rules.** Keep if all three hold; revise once if criterion 2
fails only on unseen hues; stop if the gate fails.

**Budget.** Encoder training (a) about 15 minutes (card 054's); tiers as
card 068; test world about 10 minutes per fold.

## 7. Result

**Data check** (`runs/069/datacheck.json`): in 100,000 steps of card
054's stream, 58 toggles at a locked door with the matching key (all
opened) and 70 with another key (none opened); 60 more "locked" doors
opened with an empty hand (switch doors, which look alike).

**Step (a)** (`runs/069/encoder.json`, `encoder.out`; 12 checkpoints,
17 minutes). The relation term's loss fell from 0.061 to about 0.001; its
buffer held 10,000 unchanged and 1,041 changed toggles made while holding
something. Card 054's own terms held: pair margin met (hinge 0.0003),
visibility met, transitions 0.003–0.007 (card 054's last checkpoint
0.001–0.002).

**Gate: not met** (`runs/069/gate_trained.json`; tiles as the agent's
catalogue draws them).

| | Card 054 (baseline) | This card |
|---|---|---|
| The learned relation, left-out colour pairs (gate ≥ 64 of 66) | – | **45** of 66 |
| The learned relation, unseen-hue pairs (gate ≥ 8 of 9) | – | **5** of 9 |
| Largest same-colour relation; smallest different-colour | – | 10.9; 3.8 |
| Any weighting over the 32 numbers, left-out / unseen | 47 / 4 | 50 / 6 |

Always answering "different" scores 60 of 66.

**Why** (diagnosis, report only). On clean tiles of the nine training
hues the whole predictor gets 69 of 81 door and key pairs right, but not
through the relation: the locked door's own term u is large and differs
by hue (8 to 15), so each door hue sets its own threshold on the relation
(a table by colour, which LESSONS warns a fixed set of colours allows).
Yellow's own key is predicted not to open its door. On the training
stream's noisy tiles, same-colour relations average 5.0 and different
ones 10.5, but they overlap (up to 11.5 against down to 2.4). With about
1,000 relevant tries, each a fixed render, a near-zero loss is reachable
without a rule.

## 8. Decision

**Stop**, as declared for a failed gate; criteria 1–3 were not run, and
version 15 is unchanged. What it shows: a relation trained beside terms
that read each tile alone is not forced to carry the relation, and the
stream's few, fixed tries let it fit by colour. A next attempt would need
the relation to be the only path for what depends on both tiles and many
more fresh pairs per update (LESSONS: new fillers in every task made the
rule the only solution). Whether to try that is the user's choice.
