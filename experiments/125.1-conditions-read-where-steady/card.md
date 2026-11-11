---
id: "125.1"
title: conditions read where steady — card 125's contrast and chain, using a patch only when it is not changing
rung: 2
serves: [P6, P4, P7]
status: done
verdict: fail
arch_version: 27
date: 2026-11-10
---

# 125.1: conditions read where steady

Roadmap step 2 (event segmentation), rung 2. Card
[125](../125-conditions-by-contrast-over-touches/card.md) found each
lock's button by contrast over remembered touches, and left two
problems caused by the arm: it covers a button just before a touch
there, so the button's state is misread (half of the touches taken as
setting the lock were not presses), and at the button places the arm
itself came out as the "condition". Card
[117.1](../117.1-self-from-proprioception/card.md) could not find the
arm per patch from the pose. The user (2026-11-10) chose the other
route: a condition is a stable state (the user, 2026-11-01), so it is
read only where its patches are steady.

Biological counterpart: event models hold steady within an event and
are updated at boundaries (Zacks et al. 2007; Baldassano et al. 2017,
LITERATURE); what is in flux is not yet part of the state.

## 1. Question

If a patch counts toward a condition only when it is steady before
the touch, does card 125's contrast still find the locks, does the arm
drop out as a condition, and can the touch that sets each lock be read
from memory? P6, P4, P7.

## 2. What changes

One component: which readings of a patch are used.

```
125:    every touch's before-features, at every patch
125.1:  a patch's before-features only at touches where it is steady
        a touch with no steady reading of a condition keeps the last known state
```

- **Steady:** the patch's variance over the 10 frames before the touch
  (ending 2 frames before contact) is under Otsu's threshold on the log
  variance over all touches and patches in memory.
- **Contrast:** as card 125, on steady readings only. A patch can be a
  candidate only if it is steady in at least half of the place's
  touches, and in at least 10 touches of each outcome.
- **State:** a touch's reading of a condition is the mean of its steady
  candidate patches, split into two states by 2-means. With fewer than
  3 steady candidate patches the touch keeps the previous touch's
  state.
- **Chain:** a touch sets the condition if the state differs before
  the next touch.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A condition's patches are steady when nothing acts on them | fails under a moving view; needs patches placed in the world | the same | holds | fixed cameras only | fixed camera, declared since card 119 |
| The agent's body is in motion near a touch | holds | the player is drawn at a fixed place: not removed by this rule | no body | holds; fails for a resting arm | holds in play data |
| An unseen condition keeps its state | holds unless others act | fails at night and with creatures | holds | fails with other agents | declared; belief without decay |

## 3. Dependencies

Card 125 (records, places, contrast; its scripts), card 119 (steady
windows), card 122 (256-pixel signals).

## 4. Data check

Fresh episodes of the state dataset: memory 600–699, test 500–599.
Neither was used by any card.

## 5. Feasibility gate

Card 125's main run and its diagnostics are the gate:
- upper bound for the contrast: card 125's criteria 1 and 2 passed
  without the rule;
- the chain read as a change of state, without the rule: 94–96% of
  presses caught, 98–100% in one place, 47–49% of setting touches are
  presses. The rule has to lift the last number.
- trivial baseline for the chain: card 125's change-map rule (8% of
  presses caught).

Not shown before the run: that the button is steady often enough at
the handle places, and that 2-means on steady readings gives the lock.

## 6. Success criteria and prediction

Written in `tools/card125.1/main.py` before the run.

1. **The contrast holds.** Card 125's criteria 1 and 2 with the rule:
   at slide places at least 6 of 10 candidate patches on the locking
   button, contrast at least 5 times shuffled, prediction within 0.03
   of the true lock; at quiet places under 5 times.
2. **The arm is not a condition.** At button places (a button flips in
   at least 30% of touches), at most 1 of the 10 candidate patches lies
   on the arm's usual patches (card 125: 8 of 10 at one such place).
3. **Chain.** Per slide place: at least 80% of the setting touches are
   presses of the locking button, at least 80% of its presses are
   caught, and at least 80% of the setting touches lie in one place.

**Prediction.** 1 passes. 2 is likely: the arm is moving before a
press. 3 is the risk: a touch near the button that is not a press
still hides it, and the carried state then depends on the touch before.

**Budget.** About 20 minutes in steps under 10 minutes, capped.

## 7. Result

Run 2026-11-10: memory episodes 600–699 (2,008 touches), test 500–599
(2,032 touches). 66% of patch readings are steady.

**1. The contrast holds: pass.**

| Place (named by truth) | Contrast against shuffled (bar 5) | Candidate patches on the locking button (bar 6) | Predicts "moved": candidates / true lock |
|---|---|---|---|
| Drawer handle | 40 times | 8 of 10 | 0.890 / 0.890 |
| Drawer handle | 33 times | 8 of 10 | 0.831 / 0.831 |
| Window handle | 18 times | 8 of 10 | 0.725 / 0.725 |
| Quiet (four) | 0.8–1.5 times | not counted | |

**2. The arm is not a condition: pass.** At the two button places 0
of 10 candidate patches lie on the arm (card 125: 8 of 10 at one).
The contrast there is still large (15 and 46 times) and now lies
elsewhere, 5 of 10 on the drawer at one place. What it has found
there was not examined.

**3. Chain: fail by the written rule.**

| | Drawer | Window | Bar |
|---|---|---|---|
| Setting touches that are presses of the locking button | 44% | 50% | 80% |
| Presses caught | 55% | 62% | 80% |
| Setting touches in one place | 96% | 97% | 80% |

**Diagnostics after the run, not judged** (`diag.py`, `diag2.py`).
- **Readings are exact.** Where a condition has a steady reading (79–80%
  of touches), its state matches the true lock in 100% of them (card
  125, without the rule: 91–92%).
- **The truth for "press" was wrong.** The rule counted a touch as a
  press if the button's state differed between sensed contact and 2
  frames after release. 24% of flips happen 1 to 5 frames before
  contact is sensed, so those presses were counted as non-presses.
- **With a press counted from 5 frames before contact:** 86–87% of
  setting touches are presses and 96–97% lie in one place. Presses
  caught stays at 56–58%, but this count is unreliable: the wider
  window nearly doubles the number of presses (322 against 170 for the
  drawer), so one flip is probably counted for two touches of a burst.
- **Scored by changes of state:** every change between two consecutive
  steady readings (216 for the drawer, 181 for the window) has a press
  between them, and every stretch with an odd number of presses is
  seen as a change (100% both ways). A change is bracketed by 2.2
  touches on average, so the single touch is not always identified,
  and two presses between readings cancel and are rightly not seen.

## 8. Decision

**Revise.** Kept: the steady rule. It leaves the contrast intact,
removes the arm as a condition, and makes the reading of a lock's
state exact. To revise: the chain's test. Its truth must allow for the
contact sensor's lag, and its unit should be the change of state
between steady readings, with the touches in between as the candidate
causes, not a single touch. That test was only run as a diagnostic on
this card's memory, so it needs fresh episodes and bars set first. Not
examined: what the contrast finds at the button places.
