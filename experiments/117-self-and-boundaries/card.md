---
id: "117"
title: self and boundaries — the self is what every action moves; a boundary is a lasting change in anything else
rung: 2
serves: [P7, P4]
status: approved
verdict:
arch_version: 27
date: 2026-11-01
---

# 117: self and boundaries

The temporal half of event segmentation (Roadmap step 2, rung 2), on card
116's slots. It runs only if 116 is kept. Card 115 marked boundaries
by surprise, which matched condition changes no better than plain change
(0.30 against 0.31 in time and place), with false alarms on the arm. The
segmentation review gives two reasons and a fix:
- **A competent model predicts what the agent causes,** so surprise fades
  as the model improves.
- **A condition is a stable state.** A boundary is the switch between two
  stable stretches, not a spike of error. Event models in cortex hold
  steady within an event and shift at boundaries (Baldassano et al. 2017).
- **The agent's own body must be set apart.** The brain attenuates the
  sensations it causes, predicting them from a copy of the motor command
  (Blakemore, Wolpert and Frith 1998).

## 1. Question

Are the moments where a slot that is not the agent's own body changes
lastingly (stable before, stable after, different between) the moments
and places where the scene's conditions change? Rung 2, P7 (events worth
tracking), P4 (keep the distinctions that change outcomes).

## 2. What changes

The boundary signal is judged. The self is found first, by a gate step
with its own bar, because every arm of the comparison needs it.

```
116:  frames → slots (vectors + masks) per model step
117:  self:      influence of slot k = how much card 114's predicted patches inside k's mask
                 vary when the action varies; the self = the slots every action moves
      boundary:  for each other slot, a lasting change: before-window and after-window
                 means differ, each window stable within itself
```

- **Slots for every held-out frame:** card 116's model runs on 5
  interleaved streams of model steps, offset by one environment step each.
- **Influence (CAI, `2106_03443`; ACF, `2510_02484`):**
  - At every fifth held-out frame t (the model-step grid; the self needs
    only each episode's median), card 114's predictor predicts t + 5
    from the real history under the real action chunk and 8 others drawn
    from the dataset.
  - For each slot, the predicted patch vectors are pooled inside its mask
    at t. The influence is their variance across the 9 actions.
  - **The self:** the slots whose median influence over an episode is at
    least 3 times the median of all slots. No labels are used.
- **Lasting change (the judged signal):**
  - For a non-self slot at time τ: the distance between its mean vector
    over the 3 model steps before τ and the 3 after, divided by the
    spread within the two windows.
  - Boundaries are this signal's peaks (highest within ±5 steps), above a
    threshold chosen on episodes 900–999, tested on validation (as in
    card 115).
- **Matching in time and place** (card 115's diagnostic, where random
  scored 0.05): within ±5 steps, and the boundary's slot mask must touch
  the changing object (the true mask, one patch of slack).
- **Compared, all per non-self slot:**
  - plain change (slot vector now against one model step ago);
  - surprise (card 115's patch errors pooled in the slot's mask);
  - random boundaries.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The self is what every action moves | the view itself moves with every action: the self is the camera plus held items | the player sprite | the controlled sprite, where there is one | the arm; proprioception agrees | general (sense of agency) |
| Conditions are stable between boundaries | holds; day and night change slowly | holds | holds | holds | general |
| Windows of 3 model steps | a setting | a setting | a setting | a setting | declared setting |
| A held object moves with every action | holds | holds | rare | holds | open: a held cube may join the self (below) |

## 3. Dependencies

- Card 116's slots (kept), card 114's predictor and stored features,
  card 115's events, masks, surprise and matching code.
- CAI and ACF (LITERATURE's segmentation review). Influence is measured
  by varying the action in a trained predictor, with no new training.

## 4. Data check

Card 115's held-out events: 3,257 per 100 episodes (lock flips 690; drawer
starts and stops 313/314; window 371/371; cube 599/599). The gate adds:
- how many events fall on objects that card 116 binds (a slot with mBO
  ≥ 0.5);
- how often the cube is held: the time a held cube may sit in the self.

## 5. Feasibility gate

Each step under 10 minutes, capped (rule 5).
1. **The self is found.** The union of the self slots overlaps the arm's
   true mask with IoU ≥ 0.7 on validation frames, and covers at most 10%
   of other objects' pixels, apart from a held cube. If not, the card
   stops: no boundary arm can be judged without it.
2. **Upper bound:** the same lasting-change signal on true objects (DINOv2
   vectors pooled in the true masks, the arm left out by its true mask).
   It must reach F1 ≥ 0.5 in time and place, or the bars below are set
   from what it reaches before the main run.
3. **Baselines:** plain change, surprise, random; all per non-self slot.

Result of the gate: not yet run.

## 6. Success criteria and prediction

Validation episodes, threshold from episodes 900–999, matched in time and
place:
1. **Boundaries:** F1 at least 0.8 × the upper bound's.
2. **Lasting change beats the alternatives:** F1 at least 0.10 above the
   better of plain change and surprise.

Recall per kind of event, and the share of boundaries on the self (zero
by design), are reported.

Meeting both shows that boundaries can be read from learned objects,
without labels, by the rule the brain is thought to use: what I did not
do, lastingly changed.

**Prediction.**
- The self gate passes: the arm moves under every action.
- Lasting change beats surprise on lock flips and on slides stopping and
  starting, since each is a switch between two steady states.
- The cube is the risk. While held, it moves with every action, so it may
  count as self, and its starts and stops vanish into the self. The brain
  does the same: a held tool is taken into the body schema (Maravita and
  Iriki 2004). If cube recall is low for this reason, that is a finding,
  not a bug. Card 118 must then treat "held" as a condition of the self.
- Expected: criterion 1 passes, criterion 2 passes by about 0.15.

**Budget:**
- Slots on every held-out frame: 5 minutes.
- Influence (9 actions at 40,040 frames): about 6 minutes, at card 115's
  speed.
- Signals and scoring: 5 minutes.
- No training.

## 7. Result

Appended after the run.

## 8. Decision

Appended after the run: keep, revise or stop.
