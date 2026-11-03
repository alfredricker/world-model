---
id: "104.1"
title: the view's motion from a pretrained motion model — blocked moves told by the frames, not by the action's habit
rung: 1
serves: [P7, P2]
status: done
verdict: stop
arch_version: 27
date: 2026-10-13
---

# 104.1: the view's motion from a pretrained motion model

Drafted at the user's request, 2026-10-13 ("draft that as 104.1"), after the
overnight session, and approved the same day ("run 104.1"). It continues card [104](../104-sam-correspondence/card.md)'s
line: pretrained perception under GOAL C1 as amended. Series
[105](../105-one-transition-model/card.md) waits on it, because Crafter's
moves and their conditions fail on the view's motion.

## 1. Question

Card 103.4's placing reads the view's motion by matching the frame against
the episode's canvas, tempered (τ = 300), with a prior p(shift | action)
learned from the agent's own placements.
- It is right on 99.4–99.95% of moves that succeed, but only 76% of
  blocked moves (zombies 47%). With the transition model's prior instead,
  85% (105.9, stop).
- These errors are why 105.8's model misses Crafter's moves (97.6% against
  98%). They are also why 105.4's move conditions fail (blocked-move
  precision 0.40–0.46).
- Yet the two frames alone tell about 96% of these blocked moves apart
  (section 4). The tempering that stops a stepping cow outvoting the scene
  also mutes weak but real evidence, and the 90% prior wins.

Suppose the evidence comes instead from a pretrained motion model: RAFT
optical flow, or CoTracker3's point tracks. Either one separates the
camera's motion from the creatures' without our tempering. Is the view's
motion then right on 95% of blocked moves, without losing moves that
succeed? Rung 1. Serves P7 (units tracked across frames) and P2 (belief
about a changing world).

## 2. What changes

One component: the evidence term of 103.4's placing score. The prior, the
candidate shifts and the label-free choice of weight stay.

```
103.4:      score(s) = log p(s | a) + (1/τ)  Σ_q log p_pix(frame_t+1(q) | canvas, s)      τ = 300, self-consistency
this card:  F = M(frame t, frame t+1)   M frozen, pretrained; the flow on the 49 x 63 world region
            score(s) = log p(s | a) + (1/τ_M) Σ_q log[ (1 − ε) N(F_q; flow of shift s, σ²) + ε / A ]
            σ² per frame: the median squared residual at the best shift (as 103.1); ε as 103.5 fitted;
            A: the flow's range; pixels the camera carries (103.1's learned π_q ≥ 1/2, the agent's tile) left out
            τ_M chosen by 103.4's rule: the self-consistency of the canvas on training play, no labels
```

The flow is computed only on steps with more than one candidate shift. A
candidate is a shift the prior gives at least 10⁻³, so in practice these are
move steps. Every other step keeps shift 0, as now.

**Arms.** The flow is computed once per model, and the arms differ only in
how it is combined:
- **M + habit** (the decision's arm): as above, with 103.4's prior.
- **M alone:** a flat prior, so the shift the most pixels agree on wins.
  It shows what the frames say.
- **M + model's prior:** 105.9's prior in place of the habit. This arm is
  a diagnostic: 105.9 stopped, so it cannot be kept on its own, but it
  shows whether the two fixes add up.

**Models.** The stronger on the gate goes to the main run.
- **RAFT** (Teed and Deng 2020), torchvision's `raft_large`, weights
  `C_T_SKHT_V2`: dense flow between two frames.
- **CoTracker3, online** (Karaev et al. 2024), from torch hub: a grid of
  points tracked over a sliding window, read between t and t+1.

**Resolution.** The stored 64 px frames have 7 px per tile. They are
enlarged four times, bicubic (the world region becomes 196 × 252, padded to
multiples of 8). The gate compares this with card 104's native 16 px
render of the same play.

**Declared:**
- The pretrained motion model, frozen and without class labels or
  language (GOAL C1 as amended).
- The enlargement factor, an adapter as card 104's 16 px render was.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| A model pretrained on natural video reads motion in rendered sprites | to test (gate) | likely (flat colours) | yes (its training domain) | declared (C1) |
| The view moves by one 2-D shift | yes | yes where the view scrolls | no: rotation and depth need the flow's whole field | declared, as 103.1 |
| The camera's motion is the motion most pixels share | yes (creatures are few) | yes | mostly; not when a near object fills the view | general (robust likelihood) |
| A texture that repeats hides a shift of one period | yes (grass tiles repeat) | yes (uniform cells) | rarely | not removed: the prior decides there |

## 3. Dependencies

- **103.4's placing** (kept): the prior p(s | a), the candidates, the
  self-consistency rule and `runs/103.5/placing_tau300.npz`; tools
  `tools/card103.1/place.py` and `tools/card103.5/origins.py`.
- **105.9's priors** (`runs/105.9/prior_*.npy`), for the diagnostic arm.
- **RAFT:** in torchvision 0.29, installed; its weights download on first
  use.
- **CoTracker3:** torch hub `facebookresearch/co-tracker`, a download.
- **Literature:** both are in LITERATURE's current focus.
- **Card 104's play:** 16 px frames (`runs/104/hi_w10.npy`) beside the
  64 px frames of the same steps (`runs/098.5/play_w10.npz`), for the gate.
- **105.8's runner** (`tools/card105.2/run.py`), for criterion 3.

## 4. Data check

Done while drafting (`tools/card104.1/check_frames.py`), on 105.9's held-out
set. A blocked move is *told by the frames* when, with the agent's own tile
left out, frame t+1 matches frame t better under no shift than under the
one-tile shift the action would cause. "Better" means at least one
intensity level lower, as a mean over the world region.

| Held-out source | Move steps | Blocked moves | Told by the two frames |
|---|---|---|---|
| Curriculum worlds 8, 9 (always day) | 5,139 | 911 | 100% |
| Random play workers 8, 9, by day | 11,559 (day and night) | 589 | 98.1% |
| Random play, at night (daylight < 0.5, where Crafter adds noise) | (above) | 396 | 84.6% |
| All | 16,698 | 1,896 | about 96% |

- **Blockers:** plants 445, trees 305, stone 293, zombies 264, tables
  212, cows 131, water 113, furnaces 77, arrows 45, others 11.
- **Moves that succeeded:** the same test, by day, tells 97–99.7% of them
  on the curriculum but only 85–87% on random play. Open grass repeats
  every tile, so a one-tile shift is invisible there.
- **So frames alone would lose moves.** The prior must decide those
  cases, which is why the decision's arm keeps it.

## 5. Feasibility gate

- **Upper bound:** the two frames tell about 96% of blocked moves
  (section 4); the evaluator's shift gives 100%.
- **Trivial baselines:**
  - the action's habit alone: 0% of blocked moves, 100% of moves that
    succeed;
  - 103.4's placing: 76%;
  - with 105.9's prior: 85%.
- **Gate** (card 104's development worker 10: about 5,800 move steps, of
  which about 490 are blocked; not in the test set). Each model is run as
  M alone, at 64 px enlarged and at 16 px.
  - It passes if it is right on at least 90% of blocked moves by day.
  - The cheaper resolution that passes is used.
  - If no model passes at either resolution, stop without the main run.
  - *Corrected before the gate ran (2026-10-13):* only card 104's workers
    10–12 have the 16 px render, and 105.9's held-out set exists at 64 px
    only. So the main run uses 64 px enlarged. The 16 px comparison shows
    what re-rendering would give.

## 6. Success criteria and prediction

On 105.9's held-out set (curriculum worlds 8 and 9, random play workers 8
and 9), arm M + habit, with the model the gate chose:

1. **Blocked moves.** Right on at least 95% of the 1,896, reported by
   source and by day and night.
2. **Nothing lost.**
   - Moves that succeeded are right on at least 99.4% in every source
     (103.4's lowest).
   - Still steps stay at 100%.
   - The placing runs at least 20 frames per second end to end
     (card 104's bar).
3. **The model's moves.** 105.8's token model is retrained on the new
   placements (seeds 1021 and 7).
   - It places the agent's next tile right on at least 98% of move steps
     (105.8's criterion 2).
   - Makes and the role test stay within 2 points of 105.8.
   - This criterion runs only if 1 and 2 pass, and needs the flow on the
     move steps of all fourteen sources.

**Prediction:**
- **Gate:** RAFT passes at 64 px enlarged by day.
- **Criterion 1:** about 93–96%. The frames tell 96%; in the rest the
  habit wins. Random play at night is weakest, at about 85%.
- **Criterion 2** passes, because the prior decides in open grass.
- **Criterion 3:** moves reach about 98%. If they do, 105.4's Crafter
  move conditions are rerun in the next card.

**Decision rules:**
- **Keep** (the placing takes its evidence from the pretrained motion
  model) if 1–3 pass.
- **Revise** if 1 passes but 2 or 3 does not, or if blocked moves reach
  90–95%.
- **Stop** if blocked moves are no better than 85% (105.9's level) in arm
  M + habit, for both models.

**Budget,** every run capped and under 10 minutes:
- **Gate:** 2 models × 2 resolutions on about 5,800 pairs, about 5
  minutes.
- **Flow on move steps:** about 50,000 pairs over all sources (about 20%
  of 247,000 frames), in pieces of under 10 minutes, about 10 minutes in
  all.
- **Choosing τ_M and placing:** about 5 minutes.
- **Retraining:** 2 × 5 minutes.

About 35 minutes in all.

## 7. Result

Run 2026-10-13 ([results.json](results.json); `tools/card104.1/flow.py`,
`sweep.py`, `synthetic.py`). All runs were capped and took about 25
minutes in all.

**Gate** (worker 10, each model alone with a flat prior, 5,811 move steps):

| | RAFT, 64 px ×4 | RAFT, 16 px ×2 | CoTracker3, 64 px ×4 | Bar |
|---|---|---|---|---|
| Blocked moves right, by day | 100% (305) | 100% | 100% | ≥ 90% |
| Blocked moves right, at night | 100% (177) | 100% | 100% | |
| Moves that succeeded right | **5.7%** (5,329) | **45.8%** | **0.08%** | (103.4: 99.4–99.95%) |
| Steps per second, one at a time | 78 | 86 | 10.5 | ≥ 20 |

- **The gate as written could not fail.** A placing that never moves the
  view also gets 100% of blocked moves. I report it as is rather than
  re-scoring it. Both models pass its letter by reading almost every step
  as still.
- **CoTracker3 at 16 px** did not finish within the 15-minute cap. Its
  offline code takes one clip per batch.
- **RAFT needs at least 128 px,** so the 16 px render was enlarged twice
  for it.

**The pipeline is not at fault.** I ran a synthetic check on 64 pairs whose
second frame is the first moved exactly one tile. On random noise the shift
is recovered at 100% of points (RAFT) and 99.7% (CoTracker3). On Crafter
frames it is recovered at only 33% and 18%; the rest read no motion.
- **Repeated tiles match at zero.** Crafter's view is built from repeated
  tiles, mostly grass in random play. A one-tile shift of such a scene
  matches at zero displacement nearly everywhere.
- **The models spread that answer.** Their smoothness prior carries it over
  the distinctive tiles too. Even in views less than half grass, RAFT put
  69% of points at zero on real moves.

**Held out** (105.9's test set; RAFT at 64 px ×4 with 103.4's habit, swept
over every weight, since no weight passes criteria 1 and 2 together):

| τ_M | Blocked moves right (1,896) | Moves that succeeded right (14,802) |
|---|---|---|
| 1 – 1,000 | 100% | 4.1–4.8% |
| 10,000 | 99.95% | 20.7% |
| 30,000 and above | 0% | 100% |
| Best threshold on the share of points moving one tile (chosen with the answers) | 94.3% | 84.5% |
| — curriculum worlds 8, 9 | 94.4%, 95.1% | 98.0%, 99.2% |
| — random play workers 8, 9 | 94.2%, 93.6% | 80.7%, 77.0% |

1. **Blocked moves: pass only where criterion 2 fails completely.**
2. **Nothing lost: fail at every weight.** Moves that succeeded reach at
   most 20.7% while blocked moves stay above 0%. RAFT's speed would pass
   (78 steps per second); CoTracker3's would not (10.5).
3. **The model's moves: not run.**

## 8. Decision

**Stop** (2026-10-13). Pretrained motion models trained on natural video
read a world of repeated tiles as still. A one-tile shift over repeated
tiles matches at zero displacement, and their smoothness prior spreads that
answer over the distinctive content. No weighting of their evidence against
the action's habit recovers both blocked moves and moves that succeed.
Even a threshold chosen with the answers reaches only 94% and 84.5%.
103.4's pixel match already gets 99.4–99.95% of the moves that succeed.
Its shortfall on blocked moves is how it weighs the few distinctive tiles
against the prior, and that is not what these models fix.

**No further pretrained motion models.** The obstacle is the world's
periodicity, which they all share. The two frames do tell about 96% of
blocked moves (section 4), so the fix belongs in our own placing:
- **The sleep loop** (card [106](../106-sleep-phase/card.md)): re-place
  memory under the transition model until the placements stop changing.
- **Evidence counted per distinct piece of content.** Grass that repeats
  should not count as observations, and a tree should count fully. Today
  one τ for the whole frame mutes both.

Either is a new card.
