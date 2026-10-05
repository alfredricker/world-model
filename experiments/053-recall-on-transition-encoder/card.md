---
id: "053"
title: version 10's recall and planner on an encoder trained by transitions
rung: 0
serves: [P4, P7, P12]
status: approved
verdict:
arch_version: 10
date: 2026-10-04
---

# 053: version 10's recall and planner on an encoder trained by transitions

Drafted on 2026-10-04 at the user's request, when card 052 had grown too
large. It carries card 052's R1 checks (the encoder under noise) and R2
(the agent's own recall reading that encoder). Not approved; the numbers
are proposals. Approved by the user on 2026-10-04, together with
card 054 to follow (the transition model as recall's prior).

## 1. Question

Card 052's step T trained the encoder to predict each action's effect
on the tile in front and the held tile (latent transitions). Its
transition model predicts the colour cases at a locked door (the
matching key opens it in 40 of 40 test tries, another colour's key
leaves it locked in 40 of 40), so the vectors carry the key–door match.
Card 052's stand-in recall did not read it (27% and 84%). This card
asks whether **version 10's own recall**, which can admit a front–held
relation per part as a condition, reads the match from these vectors,
and whether the planner still works on them. Rung 0; P4 (keep the
distinctions that change outcomes), P7 (learned units), P12 (conditions
the planner can chain).

The transition model is a training signal only, like the old pixel
decoder: it is not consulted when acting (GOAL.md C2, one
representation; recall stays the predictor).

## 2. What changes

One component: the encoder.

| | Version 10 | This card |
|---|---|---|
| Training | Per world, once: rebuild pixels, codebook, pair and recall terms | Once, on card 052's generator (nine colours, play starts, pixel noise, no tint): latent transitions with a target network, a variance floor, codes placed at the data |
| Codebooks | 4 parts × 8 learned codes; "new" by radius; fresh codes | The same shape and the same reading (cards 035, 036) |
| While acting | Frozen | Frozen |

Plumbing, not a component: an adapter hands the frozen encoder's vectors
and codebooks to version 10's code (`tools/card051/`), in place of the
per-world encoder. With pixel noise, memory is keyed by codes and
vectors, not by exact pixel repeats.

## 3. Dependencies

- Version 10's recall, planner and tokens (cards 049–051, kept).
- Card 052's generator and play starts (gate 1 passed), its probe tries
  stratified by kind (step 2c), and step T's harness.
- Latent transitions with a target network: SPR (Schwarzer et al.
  2021), BYOL; C-SWM (Kipf et al. 2020) in grid worlds. Variance floor:
  VICReg. Moving-average codebooks: VQ-VAE-2. To add to LITERATURE.md.
- Condition gates with an L1 penalty: Lachapelle et al. 2022
  (`2107_10098`); untested here beyond step T, where they all closed.

## 4. Data check

- Step 1: the stream as in step T; with play starts at 0.5, each kind
  of try (action, front, held) has 40 probe and up to 300 memory tries,
  including both colour cases at a locked door.
- Step 2a: the same stratified tries, rendered with pixel noise.
- Step 2b: card 051's familiar worlds, 5,000 random episodes each.

## 5. Feasibility gate (step 1: the R1 checks)

Step T's harness, seed 399, nine colours, noise, no tint, 40
checkpoints of 2,000 updates. Two changes from step T, both measured
before they are set:
- **The gates' penalty from a measured balance.** 500 updates with the
  penalty off; each gate's mean gradient magnitude from the transition
  loss is recorded; the L1 weight is set to half their median, so a
  gate stays open where the loss pulls on it more than on a typical
  gate.
- **Restarts only for long-unused codes:** an entry is restarted when it
  received no vector in the last 5,000 updates (step T: every 250).

Arms: **T** (gates, calibrated) and **T0** (no gates). An arm passes,
over the last 10 checkpoints:
1. at most 8 of 84 identities split by noise, at most 8 tuples shared;
2. at most 5% of probe tiles change code per checkpoint (step T:
   12–59%);
3. effects known at least 95% (the predicted after-tiles in the real
   after-tiles' codes).

Reported: each action's open gates, and colour read from a single
part's code (step T: 17–34%, chance 11%), since version 10's relation
is per part. Step 2 uses T if it passes, else T0; if neither passes,
the card stops here and reports.

- **Upper bound:** step T0's transition model, 99.3% of probe effects.
- **Trivial baseline:** "always stays locked" gets 0% and 100% on the
  colour cases.

**Amendments before the run** (2026-10-04, from measurements on step
T's saved weights and three short checks; criteria unchanged):
- **Gates become random on/off masks** (Gumbel-sigmoid, straight-through,
  temperature 0.5), as in CDL and Lachapelle et al. 2022. In step T the
  first layer after the gates grew 4–6× (norm 82 against T0's 15) as the
  gates shrank: a scaled gate can be rescaled by the next layer, so its
  penalty did not measure usefulness. Acting uses p > 0.5.
- **The penalty is per action**, half the median of that action's
  single-input gains (loss with the input off minus on), measured at
  2,000 updates with the gates held at 0.5. Measured at 500 updates with
  one penalty for all actions (0.0013), toggle's gates closed (0.11 by
  3,000 updates): a door's change is small in latent distance. At 2,000
  updates toggle's gains are about 0 (inputs redundant under random
  masks), so its penalty is 0 and its gates stay near 0.5: toggle's
  conditions are not found this way (reported, not a criterion).
- **Code restarts are removed**, not spaced out. Step T's T0 restarted
  25 entries by checkpoint 1 and 4 in the remaining 39, so restarts did
  not cause its code changes. Every probe piece within 0.05 of a
  boundary (28%) sat between two entries 0.02–0.06 apart, inside one
  cluster whose noisy copies spread 0.001–0.003: two codes sharing a
  cluster, which a restart at a random batch piece creates. Without
  restarts, 4–6 codes are used per part.
- Reported per checkpoint: entry pairs whose regions overlap (card 035's
  radius), and the share of probe pieces within 0.05 of a boundary.

Result of the gate: (running)

## 6. Success criteria and prediction

Step 2, on the frozen encoder from step 1:
1. **Recall reads the match** (step 2a). Version 10's recall (card 049's
   admission, card 050's own tries first), fitted on the memory tries,
   predicts both colour cases at a locked door correctly in at least 90%
   of probe tries each. Compared with version 10's own encoder trained
   on the same tries (pixel recipe), and the stand-in's 27% / 84%.
2. **Noise does not change predictions** (step 2a). The probe tries
   rendered with two noise draws get the same prediction in at least
   99%. This is the decision-level test agreed in the theory discussion,
   in place of "codes never change".
3. **The planner still works** (step 2b). Version 10's agent on card
   051's familiar worlds, clean pixels, seeds 400–404: at least 99% in
   each world (version 10: 100%), steps within 5% of card 029's.

**Prediction.** Step 1: T0 passes 1 and 3; the calibrated gates leave
toggle reading the held tile and the relation. Code stability (2) is
the doubtful one. Step 2a: criterion 1 depends on colour sitting in one
part; with colour spread over parts, as in step T, the per-part
relation is weak and the matching key falls near 60–80%. Criterion 3
passes: clean familiar worlds were never the hard case.

**Budget.** Step 1: about 50 minutes (two arms in parallel, as step
T). Step 2a: about 10 minutes. Step 2b: about 30 minutes for five seeds.

## 7. Result

## 8. Decision
