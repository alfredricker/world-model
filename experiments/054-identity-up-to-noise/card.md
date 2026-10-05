---
id: "054"
title: identity up to noise, and perception that keeps what is visible
rung: 0
serves: [P7, P4, P12, P3, P17]
status: approved
verdict:
arch_version: 10
date: 2026-10-05
---

# 054: identity up to noise, and perception that keeps what is visible

Drafted on 2026-10-05 from a discussion with the user, and approved with
its four steps, runs over 30 minutes allowed, for the overnight session.
The user's goal for the night: "an encoder that is well principled and
scalable that doesn't break the goal / condition hierarchy philosophy of
the model. The agent should still discover conditions."

## 1. Question

Card 053 left an encoder (transitions plus the visibility margin) on
which version 10's planner reaches 100% in the familiar worlds, but
whose learned codebook merges key colours that the vectors keep 25–60×
the noise apart, so recall's colour cases fail (45% and 85%). Can the
codebook be replaced by **identity up to observation noise**, and can
the encoder **keep every visible difference**, so that version 10's
planner matches or exceeds its results with the codebook encoders of
earlier cards and recall still discovers the conditions? Rung 0; P7
(learned units), P4 (relevance decided by recall, not by perception),
P12 (conditions stay discrete and checkable), P3 and P17.

**Compatibility with planning by conditions (GOAL.md P12; CHARTER's
Current direction).** The planner needs from codes a crisp "same tile
or not" and conditions that are met or not; CHARTER says a condition is
"a region of [the vectors]: a code's cell, or what recall treats as
alike for an action", and kinds are distributed over parts. Identity up
to noise keeps all of that: each part's code is the class of
part-vectors equal up to noise, so tiles still have code tuples, a rule
still reads only the parts it needs, and tiles equal in those parts act
as a kind for it. What goes is accidental sharing (two different
appearances in one learned cell); generalisation across appearances
comes only from recall's learned per-action similarity, the vector
level, which is where CHARTER puts it.

## 2. What changes

One component, the codes, in step A; the encoder's anchor in step B,
judged with A's codes.

| | Card 053 (V0) | This card |
|---|---|---|
| Codes | 4 parts × 8 entries, moving averages of the vectors; card 035's radius for "new"; card 036's fresh codes | **A:** per part, part-vectors closer than τ_k share a code (single linkage); τ_k = 1.25 × the largest part-distance between an untouched cell's two renders a step apart (1,000 pairs), as card 053's pixel threshold. A piece with no class within τ_k starts a new class. No codebook, no radius rule, no fresh codes |
| Encoder anchor | variance floor + identity transition of views | **B:** C-SWM's margin on every pair of batch tiles whose pixels differ beyond noise (hinge, margin m), replacing the variance floor; the identity transition of views and card 053's visibility margin kept |

Removed: the moving-average codebook, card 035's radius and α, card
036's fresh codes (step A); the variance floor (step B).

## 3. Dependencies

- Card 053: the generator stream (no tint, noise, play starts), the
  transition objective with the visibility margin, its adapters (recall
  on generator tries, the planner with a frozen encoder), all scored
  against the simulator.
- Version 10 (cards 049–051): recall, admission, the planner and its
  test set (familiar worlds, chained rooms with one door, cluttered).
- Literature: C-SWM (Kipf et al. 2020) for the margin; false-negative
  removal in contrastive learning (Chuang et al. 2020, Huynh et al.
  2022) for leaving noisy copies out; version 10's exact repeats for
  identity (card 037).

## 4. Data check

As card 053: 300 memory and 40 probe tries per kind of try, both colour
cases at a locked door present; familiar worlds 5,000 random episodes.
Step A first measures, on V0, whether a clean gap exists per part
between the largest noise distance and the smallest distance between
visibly different tiles; if not, the classes chain, and step B's margin
is what makes the gap.

## 5. Feasibility gate

- **Upper bound:** V0's transition model, matching key 97.5%, other key
  55%; version 10 on its own encoder, the familiar worlds 100%.
- **Trivial baseline:** V0 with its codebook (card 053): recall 45% /
  85%, noise agreement 97.0%.

## 6. Success criteria and prediction

Seeds 399–402 (four encoders); each criterion in at least 3 of 4.
1. **The planner matches or exceeds version 10 with its codebook
   encoder** (the user: "the planner should match or exceed the success
   of the codebook planner from previous cards"): familiar worlds ≥ 99%
   with steps within 5% of card 029's; chained rooms with one door ≥ 95%;
   cluttered ≥ 95% (version 10: 100%, 98%, 98–100%).
2. **Recall reads the colour cases and ignores noise:** both colour
   cases ≥ 90% (against the simulator); the same prediction under two
   noise draws ≥ 99%.
3. **The agent still discovers conditions** (card 049's criterion 3):
   toggle at a door admits the held tile or the front–held relation in
   the key world and the switch in the switch world; the second key
   and the vase are never admitted; at most 8 conditions per action. On
   the generator tries, toggle admits a held or relation condition.

Reported: visibly different identities sharing a code tuple, and
identities split by noise (the audit, with box colours); recall on
held-out colours (pink, brown, teal); card 047's new-colour switch
world where time allows.

**Steps and decision rules** (declared now, so the night needs no
choice from the user):
- **A** (no training; about 15 minutes): measure τ_k on V0; recall and
  the planner's familiar worlds with A's codes. If the familiar worlds
  stay ≥ 99% and recall's colour cases rise above 45% / 85%, A's codes
  are kept for B. If classes chain (two visibly different identities in
  one code), B is what should fix it; continue.
- **B** (seed 399; about 30 minutes): arms m = 0.5 and m = 1.0, 12
  checkpoints, in parallel; then A's codes and the tests above. The arm
  with the better criterion 1, then 2, then 3 goes on; if neither beats
  V0 with A's codes, V0 goes on.
- **C** (seeds 400–402; about 30 minutes): the chosen recipe trained on
  three more seeds, in parallel.
- **D** (about 45 minutes): card 051's full test set on all four
  encoders, in parallel.
- **Decision:** keep if all three criteria hold in 3 of 4 seeds; one
  declared revision is allowed overnight if one criterion fails; stop
  if the familiar worlds fall below 90%.

**Prediction.** A on V0 lifts recall's matching key well above 45%,
since the vectors keep key colours apart; noise agreement rises because
classes are wider than noise. Classes may chain for doors (noise spread
0.089, larger than keys'). B's margin closes the gap for boxes (4–8×
noise on V0). The planner holds at 100% in the familiar worlds; the
chained rooms are the doubtful test, since they were never run on a
learned-by-interaction encoder.

**Budget.** About 2.5 hours of runs in all.

## 7. Result

## 8. Decision
