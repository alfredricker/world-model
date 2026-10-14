---
id: "108"
title: learned belief — which creature is which, matched to the transition model's predictions
rung: 1
serves: [P2, P7]
status: draft
verdict:
arch_version: 27
date: 2026-10-13
---

# 108: learned belief

Drafted 2026-10-13 at the user's request ("learned belief sounds like a
good next step"). Built on Crafter by the user's choice. Craftium has no
creature tokens yet, and CHARTER now lets a brain part be built on Crafter
while Craftium lacks its inputs. This card takes over card 105.3's
creature question; 105.3's history arms stay on hold.

**Craftium card that follows** (its portability test): creature tokens
from perception in Craftium, then this belief run there with only the
adapter changed.

## 1. Question

CHARTER now says belief supplies "which one". Each remembered token's next
state is predicted by the effect model, new tokens are matched to those
predictions, and the match corrects the memory (predict, match, correct).
Suppose creatures in Crafter are tracked this way, at every gap from one
step in view to 50 steps out of view. Do look-alike cows, zombies and
skeletons keep their identities better than with card 102.3's counted
motion? Rung 1. Serves P2 (belief about what is out of view) and P7
(units tracked across frames).

## 2. What changes

One component: belief, meaning which file (a remembered individual token)
each creature seen now continues.

```
102.3:      in view, 102.2's layers link tiles (its ground rule was later rejected by the user)
            out of view, a file spreads by its code's counted motion: P(move | last move, code)
this card:  at every gap k (1 step in view, up to 50 out of view), a file's prediction comes from the
            transition model, asked "what is here k steps later?" about a 9 x 7 patch of remembered places
            around the file's last place; creatures seen now are matched to files by those predictions
            (Hungarian assignment, with "new" as an option); a match resets the file to what was seen
```

**The transition model gains one input:** the number of steps that pass.
- **k = 1, with the agent's action:** 105.10's training, unchanged.
- **k from 2 to 50, with the actions unknown:** a "k steps pass" token takes
  the action token's place.
  - **Inputs:** the view at t.
  - **Targets:** the places in view at both t and t + k.
  - **No identities needed:** targets are place contents.
- **Masking:** some input places, and sometimes the agent's own tile, are
  masked out in training. A remembered patch is then a familiar input,
  since it has gaps, and the agent may be outside it.
- **The model stays one model.** It predicts each place's next code by
  likelihood (105.8), with positions only as differences (105.6), placed by
  the pose (105.10). No output is added.

**The belief, per step:**
1. **Predict.** Each file is asked about its patch: its creature at its
   last place, and the places around it as last seen. Places hidden by
   other files are masked. The file's place now is the model's
   probability of its code at each place, normalised over the patch.
2. **Match.**
   - **Candidates:** creature tiles in view.
   - **What counts as a creature:** a code whose content the model
     predicts leaves its place within 50 steps, with probability at
     least 0.5. That is 102.3's line between "moves" and "stays",
     computed from the model rather than counted.
   - **Score:** each candidate takes the file that predicts its place and
     code best, or is new. The cost of "new" is the density of creatures
     appearing unexplained, measured on training play (IPDA).
3. **Correct.** A matched file takes the seen place and vector, with k
   reset to 0.
   - **Unmatched files** age by one step.
   - **Existence** falls on a miss, where the file's predicted place was
     in view but no creature was seen there (IPDA, as 102.3).
4. **Capacity.** At most 8 files, the most recently seen (102.3's chosen
   K). Ranking by importance to the agent's goals needs the agent; this
   card runs on stored play.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Which tokens get files | learned | learned | learned | learned | general: from the model's predictions |
| How a file moves unseen | learned | learned | learned | learned | general: the model's k-step answer |
| A file's neighbourhood is a 9 × 7 patch of grid places | no (continuous) | yes | yes | no | declared: the Craftium card uses remembered tokens within a radius |
| A creature is one tile | no | yes | often | no | adapter (perception's tokens) |
| The pose is exact | yes | yes | yes (screen) | drifts | adapter (C1) |
| At most 8 files, by recency | yes | yes | yes | yes | general; importance later |

## 3. Dependencies

- **105.10** (kept): the transition model on tokens placed by the pose.
  Seeds 1021 and 7, retrained with the k-step query.
- **102.3** (kept): files, existence and capacity. Its code
  (`tools/card102.3/belief.py`) supplies the counted-motion baseline and
  the scoring of returns.
- **Literature** (LITERATURE.md):
  - SORT (Bewley et al. 2016): predict, assign by cost, correct;
  - IDF1 (Ristani et al. 2016);
  - IPDA (existence and misses) and PMBM (the density of new objects).
- **Data:** card 098.5's random play.
  - Workers 0–1 train and the curriculum's worlds 0–7 train, as 105.10.
  - Workers 8–9 are the test, with each creature's persistent identity
    (scoring only).

## 4. Data check

From 102.3 (workers 8–9, 60,000 steps, by day):
- **Returns within 50 steps:** 1,232 cows, 19 zombies and 229
  skeletons. The median time away is 2–4 steps.
- **k-step training pairs:** counted before training, by k (places in
  view at both t and t + k). Random play moves the view slowly, so pairs
  thin out with k; the counts are reported.

## 5. Feasibility gate

- **Upper bound:** the same matching, with each file's prediction from
  the evaluator's true next place of that creature. Identity must be at
  least 98% on returns; if not, the matching or the scoring is wrong.
- **Trivial baselines, in the same code with the pose:**
  - "stays as last seen" (zero spread);
  - 102.3's counted motion per code.
- **Check:** before the main run, the k-step model's held-out likelihood
  of the places a creature reaches must beat the counted motion's at
  k = 5 and k = 20. If not, the model has not learned motion over gaps,
  and the main run is not made.

## 6. Success criteria and prediction

On workers 8–9, by day (cows, zombies and skeletons):
1. **Identity across absence:** returns within 50 steps taken by the
   creature's own file.
   - **Bar:** at least 88%, and at least 3 points above the counted
     motion (paired over returns).
   - **Reference:** 102.3 reached 81.1% with the learned placing; its
     upper bound was 95.1%.
2. **Identity over the whole play:** IDF1 over every creature frame, in
   view and after returns.
   - **Bar:** at least 0.85.
   - **References:** 098.6's files 0.717; files on the true state 0.934;
     SAM 2.1 0.60 prompted once, 0.78 re-prompted by an oracle.
3. **Nothing lost:** with the k-step query added, 105.10's measures stay
   within 2 points in both seeds: moves, blocked moves, makes, do's
   changes and place.

**Prediction:**
- **Criterion 1 passes at 88–92%.** The model knows where a creature
  cannot go (water, trees, the agent), and the pose removes placing
  errors.
- **Criterion 2 is close** (0.83–0.88). Adjacent look-alikes that both
  move stay ambiguous at k = 1.
- **Criterion 3 passes.**
- **Zombies** gain least while far from the agent: what draws them is
  often outside the patch.

**Decision rules:**
- **Keep** if 1–3 pass.
- **Revise** if 1 or 2 fails but the model beats counted motion at the
  gate's check.
- **Stop** if the check fails or criterion 3 fails.

**Budget:**
- two trainings of about 6 minutes;
- the gate's check, about 2 minutes;
- three belief runs (the model, counted motion and stays) of about 5
  minutes each.

Every run is capped (CHARTER rule 5). About 35 minutes in all.
