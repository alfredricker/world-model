---
id: "054"
title: identity up to noise, and perception that keeps what is visible
rung: 0
serves: [P7, P4, P12, P3, P17]
status: done
verdict: fail
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

**Step A on V0** (card 053's encoder; `runs/054_gap_V0.txt`,
`runs/054/a_*_V0.json`). Codes are read online (`tools/card054/
identity.py`): a piece takes the nearest class mean within τ_k, else
starts a class; the class mean follows its members.
- **No clean gap on V0.** Noise moves parts by at most 0.072, 0.098,
  0.046, 0.298 (τ = 0.090, 0.122, 0.057, 0.372); 34 pairs of probe
  identities lie within τ in every part, so they would share an
  identity: boxes of every colour, a closed and a locked door of one
  colour, two key colours. Noisy probe copies spread up to 0.164 in
  part 1 and 0.388 in part 3, beyond τ: the 1,000 untouched pairs are
  mostly floor and walls, and V0's doors are about 7× noisier than its
  keys.
- **The planner and the conditions pass:** 100% in all four familiar
  worlds, steps as card 029's, held-out effects ≥ 99.99% exact; toggle
  admits the held tile in the key world and the switch in the switch
  world (card 049's criterion 3).
- **Recall:** matching key 100% (codebook 45%), other key 60% (85%),
  all probe tries 76.7% (85.9%), the same prediction under two noise
  draws 95.8% (97.0%). Noisy copies split into several identities
  (128 tuples among the memory tiles), which makes false "changes": the
  memory's own categories agree with the simulator in 93–95% (codebook
  99%).

A goes on to B, as declared: B's margin is what should open the gap.

**Step B, seed 399** (12 checkpoints each, 18 minutes; `runs/054/b_*`,
evaluated by `runs/054/eval.sh`). A third arm was added before any B
result was read (overnight rule 2: run the options and keep the
strongest): m = 1.0 with noisy copies pulled together (batch tiles
within pixel noise; squared distance, weight 1), because step A's
failures were noisy copies splitting, which the margin alone does not
address.

| | V0 + codebook | V0 + A | m = 0.5 | m = 1.0 |
|---|---|---|---|---|
| Recall: matching / other key | 45% / 85% | 100% / 60% | **100% / 100%** | 100% / 57.5% |
| Same prediction, two noise draws | 97.0% | 95.8% | 98.2% | 94.4% |
| All probe tries | 85.9% | 76.7% | 82.6% | 77.9% |
| Planner, four familiar worlds | 100% | 100% | 100% | 100% |
| Card 049's conditions | – | ✓ | ✓ | ✓ |
| Identity pairs within τ in every part | – | 34 | 30 | 7 |

Training (last 5 checkpoints): effects known 96.7–98.1%, changes
visible 94.5–100%, both margins met (hinges near 0). Under m = 0.5
recall fails on picking up while holding (0% on those kinds): the held
tile's renders before and after sometimes take different identities,
which records a change that did not happen. Part 3 carries most
separations and most noise (noisy copies spread up to 0.51 there; τ_3
0.49), so boxes of every colour and a closed and a locked door share
every part's class although the pixel rule calls them different (mean
absolute differences 0.09–0.12 against a noise ceiling of 0.021).

The third arm (m = 1.0, noisy copies pulled together): recall 100% /
52.5%, noise agreement 97.8%, all probe tries 81.0%, planner 100% in
all four worlds, card 049's conditions right. Picking up while holding
stays at 0%: it does under every arm and under card 053's codebook too
(0–8%), so it is not the codes. The simulator says nothing changes
there; such tries are rare (about 14 per 30,000 steps for the commonest
kind), and recall follows the many empty-handed pick-ups, where the
front tile does change. Reported as a limit of recall's data, not of
this card's change.

**Step D on seed 399** (m = 0.5, identity up to noise;
`runs/054/d_*_m05_399.json`): chained rooms with one door 100% of 100
layouts at 1.00 × the shortest route; cluttered 100% at 1.06 ×
(version 10 with its codebook encoder: 98%, 98–100%).

**B's choice: m = 0.5** (the declared order: criterion 1 equal at 100%
in every arm; criterion 2 best, 100% / 100% and 98.2%; criterion 3 equal).

**Steps C and D: m = 0.5 on four seeds** (`runs/054/{recall,planner,
d_one,d_clutter}_m05_*`). Version 10 with its codebook encoder, for
comparison: familiar worlds 100%, chained rooms 98%, cluttered 98–100%.

| Seed | 399 | 400 | 401 | 402 |
|---|---|---|---|---|
| 1. Familiar worlds (steps as card 029's) | 100% | 100% | 100% | 100% |
| 1. Chained rooms, one door (× shortest) | 100% (1.00) | 100% (1.00) | 100% (1.00) | 100% (1.00) |
| 1. Cluttered (× shortest) | 100% (1.06) | 100% (1.09) | 100% (1.00) | 100% (1.03) |
| 2. Recall: matching / other key | 100 / 100 | 100 / 62.5 | 100 / 70 | 100 / 62.5 |
| 2. Same prediction, two noise draws | 98.2% | 97.0% | 98.0% | 96.8% |
| 3. Conditions (key world: hand or relation; switch world: switch) | ✓ | ✓ | ✓ | ✓ |
| 3. Generator tries: toggle admits hand or relation | ✓ | ✓ | – | ✓ |
| Identity pairs within τ in every part | 30 | 9 | 16 | 25 |

Criterion 1 holds on all four seeds, at or above version 10 with its
codebook. Criterion 3 holds on all four in the planner's worlds; the
second key and the vase are never admitted. Criterion 2 fails on all
four: noise agreement is below 99% everywhere, and the other-key case
is below 90% on three seeds. On those seeds the transition model itself
gets the other key right only 70–85%, so part of the failure is in how
far the encoder separates colours, not in the codes. The rest is noisy
copies splitting: up to 100 classes in one part (seed 402), and the
memory's own categories agree with the simulator in only 92–97%.

**The declared revision** (criterion 2 failed): noise differs by
appearance (doors are several times noisier than floor in the encoder's
space), so each class takes its own scale: 1.25 × the largest noise
distance among the 20 untouched pairs (of 20,000) nearest the piece that
starts the class, never below τ_k (`IDENTITY=local`, `tools/card054/
identity.py`; `runs/054/r_recall_m05_*`).

| Seed | 399 | 400 | 401 | 402 |
|---|---|---|---|---|
| Recall: matching / other key | 100 / 92.5 | 100 / 75 | 100 / 85 | 100 / 70 |
| Same prediction, two noise draws | 99.0% | 97.0% | 98.7% | 98.2% |
| Memory's categories agree with the simulator | 98.7–99.1% | 96.3–97.2% | 98.0–98.5% | 96.6–98.0% |

The revision helps on every seed but seed 399's other key (100 →
92.5%), and passes criterion 2 on seed 399 only. Its effect on the
planner was not measured.

## 8. Decision

**Revise.** Criteria 1 and 3 hold on all four seeds; criterion 2 fails
after its one revision (1 of 4 seeds). What the night established: with
no learned codebook, an encoder trained only by its own transitions,
the visibility margin and the margin between visibly different tiles
keeps the whole planner working: 100% in the familiar worlds, chained
rooms and cluttered world on four seeds, and the conditions the user
asked about are still discovered. What remains is recall on the
generator's noisy tries: colour pairs the encoder separates only weakly
(the transition model's other key 70–85%), and noisy copies that still
split. The next encoder card should target colour separation where the
outcome depends on it (the relation part), not the codes. The encoder
(m = 0.5) and identity up to noise are what cards 056 and 057 build on;
the per-appearance scale is kept as an option until its planner check.
