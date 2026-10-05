---
id: "054"
title: the transition model as recall's prior
rung: 0
serves: [P8, P11, P17, P21]
status: draft
verdict:
arch_version: 10
date: 2026-10-04
---

# 054: the transition model as recall's prior

Drafted on 2026-10-04 at the user's request ("the network as recall's
prior sounds like a good idea"). It starts after card 053, whose encoder
and transition model it uses. Not approved; the numbers are proposals.

## 1. Question

When the query's own situation has no tries, does the transition model
predict better than recall's neighbours, so that recall predicts new
combinations of known tiles (a key and door of the same colour whose
match was never tried), while one try in the situation still overrules
the network? Rung 0. P8 (composition: new combinations of known
pieces), P11 (revise without erasing), P17 (cost per question that does
not grow with memory), P21 (System 1 under System 2's control).

The argument (the discussion of 2026-10-04): complementary learning
systems (McClelland, McNaughton and O'Reilly 1995; Kumaran, Hassabis
and McClelland 2016) pair a fast episodic store, which learns from one
experience and keeps exceptions, with a slow parametric learner that
extracts shared structure. Episodic control learns faster early (MFEC,
Blundell et al. 2016; NEC, Pritzel et al. 2017); mixing a network's
prediction with retrieved neighbours beats either, most on rare cases
(kNN-LM, Khandelwal et al. 2020); the less uncertain controller should
decide (Daw, Niv and Dayan 2005).

## 2. What changes

One component: recall's prior, in card 050's rule

    P = (N_own + α P_prior) / (|N_own| + α)

| Query | Version 10 (P_prior) | This card (P_prior) |
|---|---|---|
| Own situation has tries | neighbours (α ≈ 0.0001: own tries decide) | the same: own tries decide |
| No own tries, every tile part has a code | neighbours over admitted conditions | **the transition model**: its predicted after-tiles, read as codes, give the outcome class |
| A tile part reads "new" (card 035) | neighbours | neighbours (novelty judged outside the network) |

A predicted after-piece beyond 6 × every code's radius makes the
network's answer "unknown", and the neighbours answer instead.

**A change to CHARTER's Current direction**, agreed in principle by the
user, applied only if this card passes. "Weights take over from recall"
gains: "and supply recall's prior where the query's own situation has
no tries; own tries overrule them, and novelty is judged outside the
network." ARCHITECTURE.md would become version 11.

## 3. Dependencies

- Card 053: the encoder trained by transitions, version 10's recall and
  planner on it (its criteria 1–3).
- Card 050's rule (kept, version 9); card 035's "new" reading.
- Literature above; to add to LITERATURE.md (`1606_04460` and
  `1703_01988` are in the library).

## 4. Data check

The new combinations must be absent from **both** the transition
model's training and recall's memory. The encoder is retrained as in
card 053 step 1 (the arm it chose), with toggles of a locked door while
holding the key of its colour removed from the training tries and from
memory for three of the nine colours (held out: chosen before the run,
by seed). Each held-out colour still appears as a key and as a door,
with the other eight colours. Probe: 40 tries per held-out colour with
the matching key, 40 with another key.

## 5. Feasibility gate

- **Upper bound:** the transition model alone on the held-out probe. If
  it does not predict the matching key opening the door in at least 90%
  of held-out tries, the network has nothing to give, and the card stops.
- **Trivial baseline:** version 10's neighbour prior on the same probe
  (card 053's recall); "always stays locked" gets 0% and 100%.

Result of the gate: (not yet run)

## 6. Success criteria and prediction

Seed 399 first, then seeds 400–404; each criterion in at least 4 of 5.
1. **New combinations.** On the held-out probe, both cases at least 90%
   correct (matching key opens, other key does not), and the matching
   case at least 30 points above the neighbour prior.
2. **One try overrules.** In 20 held-out situations, one stored try
   whose outcome contradicts the network (written into memory for the
   test) makes recall predict that outcome there, in 20 of 20.
3. **Nothing lost.** Card 053 criterion 3's familiar worlds at least 99%
   with the new prior, and card 047's new-colour switch world (the new
   colour reads "new", so the neighbours answer) no lower than version
   10's mean of 76%.

Reported: the share of questions answered by own tries, the network and
the neighbours; time per question against memory size (P17); the
network's answers that own tries later contradicted.

**Prediction.** (2026-10-05: the premise below is withdrawn. Step T's
"40 of 40" was scored by its own codes; against the simulator its model
predicts the matching key in 0 of 40. This card waits for an encoder
that sees what actions change.) The gate passes: step T's model
predicted both colour cases 40 of 40 on pairs it had seen, and the
match is one relation across colours. Criterion 1 passes with the neighbour prior near 0–40%
on the matching case. Criterion 2 passes by construction (α ≈ 0.0001),
and checks the code path. Criterion 3: the familiar worlds pass; the
switch world is unchanged, since the network is not consulted there.

**Budget.** Seed 399: encoder retrain about 50 minutes, recall tests
about 10, the planner about 30. Seeds 400–404 only if seed 399 passes:
five retrains, two or three at a time on the one GPU, about 2.5 hours
in all, handed to the user as commands.

## 7. Result

## 8. Decision
