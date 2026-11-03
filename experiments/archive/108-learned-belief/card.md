---
id: "108"
title: learned belief — which creature is which, matched to the transition model's predictions
rung: 1
serves: [P2, P7]
status: done
verdict: fail
arch_version: 27
date: 2026-10-13
---

# 108: learned belief

Drafted 2026-10-13 at the user's request ("learned belief sounds like a
good next step"); approved 2026-10-14 ("let's work on card 108"). Details
the draft left open were settled in code before the main runs (Appendix A). Built on Crafter by the user's choice. Craftium has no
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
   last place, and the places around it as last seen. Places never seen,
   and places last seen holding another creature, are masked. A file's
   likelihood of a candidate is the model's probability of the
   candidate's code at the candidate's place.
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

**Settled before the main runs** (2026-10-14): the details the draft left
open, and three fixes found in smoke tests, are in Appendix A. Every form
shares the bookkeeping; only the prediction differs.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Which tokens get files | learned | learned | learned | learned | general: from the model's predictions |
| How a file moves unseen | learned | learned | learned | learned | general: the model's k-step answer |
| A file's neighbourhood is a 9 × 7 patch of grid places | no (continuous) | yes | yes | no | declared: the Craftium card uses remembered tokens within a radius |
| A creature is one tile | no | yes | often | no | adapter (perception's tokens) |
| The pose is exact | yes | yes | yes (screen) | drifts | adapter (C1) |
| At most 8 files, by recency | yes | yes | yes | yes | general; importance later |
| The agent's own tile is not a creature | yes (first person: no body in view) | yes | where there is a body | yes | adapter: the pose says where the body is (C1) |

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
  - **Counts** (`runs/108/data_check.json`): the number of starting steps
    whose awake run reaches t + k falls from 132,785 at k = 1 to 69,144
    at k = 20 and 9,863 at k = 50. The curriculum's episodes are short.
  - **Places shared by the views at t and t + k:** 61 of 63 at k = 1,
    48 at k = 20 and 39 at k = 50.
  - **Creature places at t still in view at t + k:** 92,077 at k = 1,
    32,721 at k = 20 and 3,512 at k = 50.
  - So k near 50 is learned from thin data.

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
  - **What is scored:** the place and code a creature has k steps later.
  - **Sample:** workers 8–9, by day, every third step, wherever that
    place is in view.
  - **Baselines at their best:** counted motion is given the evaluator's
    last move.
  - **"Beat":** the paired difference in mean log-likelihood has a 95%
    bootstrap interval above 0.

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
   - **Arrows:** detections on an arrow are distractors and are removed
     (MOT's rule). 098.6's form, with arrows counted as objects, is
     reported. IDF1 is computed per awake segment.
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

## 7. Result

Run 2026-10-14 ([results.json](results.json)). Each training took 5.9
and 6.4 minutes. The gate's checks took 20 seconds each, and each belief
run under 2 minutes. Every run was capped.

**Criterion 3 first, since it decides the card:**

| Against 105.10 (same seed) | Seed 1021: 105.10 → 108 | Seed 7: 105.10 → 108 | Bar |
|---|---|---|---|
| Moves | 99.98% → 99.95% | 99.93% → 99.93% | within 2 points |
| Blocked moves | 99.95% → 99.68% | 99.89% → 99.79% | within 2 points |
| Makes that made a tool | 96.2% → 94.5% | 96.6% → **92.4%** | within 2 points |
| Do's changes, saplings left out (355) | 90.4% → **74.7%** | 84.6% → **81.8%** | within 2 points |
| Place, whole state | 93.5% → 94.4% | 93.4% → 94.5% | within 2 points |

**The gate:**
- **Upper bound:** 93.9% of returns, against 98%. Fails as written.
  - **The cause is perception, not matching.** 5.0% of returns come
    back with a code that is not a creature code: code 0 in dim light,
    or zombie code 39. Another 1.2% had their file dropped.
  - **Where a candidate exists, the upper bound keeps 98.8%.**
- **Check, mean log-likelihood of the place and code k steps later.**
  The model is above counted motion with a 95% interval over 578–785
  segments:

  | Seed | k = 5 | k = 20 |
  |---|---|---|
  | 1021 | +0.04 (0.00 to 0.09) | +0.13 (0.04 to 0.23) |
  | 7 | +0.06 (0.02 to 0.10) | **+0.00 (−0.08 to 0.09)** |

  Both log-likelihoods were near −3.2 at k = 5 and −4.9 at k = 20.
  - **Seed 7 fails at k = 20.**

**Belief on workers 8–9, by day** (1,465 returns; 34,452 creature frames):

| Form | 1. Returns kept | vs counted, points (95% interval) | 2. IDF1 | Skeletons kept (165) |
|---|---|---|---|---|
| **Model, seed 1021** | **83.8%** | −4.1 (−5.5 to −2.7) | **0.894** | 69% |
| **Model, seed 7** | **83.9%** | −4.0 (−5.6 to −2.6) | **0.890** | 65% |
| Model + Q (diagnostic, uses labels) | 87.5% | −0.4 (−1.3 to 0.5) | 0.911 | 94% |
| Counted motion | 87.9% | — | 0.914 | 95% |
| Stays as last seen | 38.8% | −49 | 0.343 | 36% |
| Upper bound | 93.9% | +5.9 | 0.931 | 99% |

- **Criterion 1 fails:** 83.8% and 83.9%, against a bar of 88% and 3
  points above counted motion. The model is 4 points below it
  (McNemar p < 0.001).
- **Criterion 2 passes:** 0.894 and 0.890, against 0.85. Counted motion
  passes as well (0.914). In 098.6's form, with arrows as objects: 0.852,
  0.848 and 0.896.
- **Criterion 3 fails in both seeds:** do's changes, and in seed 7 makes.

**Where the model loses** (seed 1021, share of returns):
- **No file of the creature: 7.9%** (counted motion 4.4%).
- **Linked to another file:** 1.8% (1.4%).
- **Left new:** 1.4% (1.2%).
- **No candidate:** 5.0% in every form.

Traces show why files go missing. A file takes a token of another kind,
and its own creature comes back to no file:
- an arrow's file took a skeleton, since arrows make a skeleton nearby
  likely;
- a cow's file took a plant that came into view 4 tiles away, on a
  likelihood of 0.0002.

The model predicts what a place will hold, not which token is there.
Given Q (the codes one tracked object wears), it ties counted motion.

**Prediction against result:**
- **Criterion 1:** predicted 88–92%, got 84%.
- **Criterion 2:** predicted 0.83–0.88, got 0.89.
- **Criterion 3:** predicted to pass; it failed.
- **Zombies:** 18 returns, too few to judge (33–39% in every form but
  the upper bound).

## 8. Decision

**Stop.** Verdict: fail. By the card's rule, criterion 3 failed in both
seeds and the check failed in seed 7 (k = 20). Criterion 1 failed as well.
Nothing changes in ARCHITECTURE.

**What was learned:**
1. **A prediction of place contents is not identity.** Matching needs
   the token's own part of the prediction (by intervention: with and
   without it) and which codes one token wears (102.3's Q).
   - Without Q, files take neighbours of other kinds.
   - With Q from labels, the model only ties counted motion (−0.4
     points).
2. **In Crafter, the model's better motion is too small to matter.** It
   gains 0.04–0.13 nats over counted motion. Which file is near decides
   a return, not the shape of the spread.
3. **The k-step query costs rare one-step changes in a shared model.**
   - Do's changes fell 2.8 and 15.7 points, and makes 4.2 in seed 7.
   - The query's loss (0.22–0.24) is ten times the one-step loss (0.02),
     so it takes the model's capacity.
4. **With the pose, counted motion is far above 102.3:** returns 87.9%
   (102.3: 81.1%) and IDF1 0.914 (098.6: 0.717). This baseline's Q came
   from the evaluator's tracks.
5. **What limits the upper bound is perception.** 5% of returning
   creatures carry no creature code in dim light, so they are never
   candidates.

**Open for the user:**
- **Q without labels.** Learn the codes one token wears from the belief's
  own links in view, as 102.3 did from its own tracks. Every form needs
  it.
- **The general form at a stated loss** (CHARTER). With Q, the model ties
  counted motion. Without the cost to one-step changes, it could replace
  counted motion. That needs either a query that does not take the
  one-step model's capacity, or more training.
- **The rest of the gap to the upper bound:** perception's dim-light
  codes (5% of returns), and returns whose creature has no file. For
  counted motion that is 4.4%, from files dropped or taken by other
  tokens; for the upper bound, 1.2%.

## Appendix A. Settled before the main runs

Code: `tools/card108/` (`train.py`, `belief.py`, `report.py`); runs in `runs/108/`.

**Settled before the runs:**
- **The agent's own tile is never a candidate.** The pose says where the
  body is (C1). The agent's codes are those seen at its tile in at least
  90% of their occurrences, and they are not creature codes.
- **The likelihood is not normalised over the patch.** The file's mass
  may leave the patch, or vanish. Normalising would hide both.
- **Existence.** A file is set to 1 when matched. While it goes unmatched,
  it falls by its code's rate of vanishing, then by IPDA on misses.
  - **Each place counts once per gap.** A k-step prediction from the last
    snapshot cannot carry earlier misses. Without this rule, one unseen
    place would count against the file at every step.
  - **Why the rate of vanishing is needed.** At existence 1, IPDA's miss
    leaves existence unchanged (102.3 had the same rate).
- **Counted without labels:** the cost of "new" (λ) and the rate of
  vanishing. Both are counted from creature tokens in training play that
  no creature token at the same place or a neighbour explains, one step
  before (λ) or one step after (vanishing).
- **Capacity 8 applies to files not seen now,** as in 102.3, where files
  were the memory out of view.
- **The counted motion baseline** is counted on the evaluator's tracks of
  workers 0–7, with perception's codes. That makes it stronger than
  102.3's, whose tracks were its own. Both baselines use seed 1021's
  creature codes.

**Found in smoke tests** (4,000 steps per worker, before the main runs):
- **A file's likelihood is its own part, by intervention.** The model's
  probability of a code at a place counts other creatures that may walk
  in, and that is far above λ. So remembered files took every new
  creature in their patch. Each file is now asked twice: once as is, and
  once with its creature masked out. The likelihood is the difference,
  floored at 0. IDF1 went from 0.876 to 0.897; returns were unchanged.
- **Plants are creature codes.** The model gives a plant's place a 0.54
  chance of changing within 50 steps (plants ripen or are eaten). So
  detections on any evaluator object other than a creature (arrows and
  plants) are distractors in IDF1, as in MOT.
- **A return 50 steps away is a gap of 51.** The model is asked at
  k = 50 for that last step.
- **Found, not fixed: the model's prediction is about what a place
  holds, not who.** Traces show an arrow's file taking a skeleton, since
  arrows make skeletons likely nearby. They also show a cow's file taking
  a plant that came into view. 102.3 had Q, the codes one tracked token
  wears, and the model has nothing like it.
  - **A diagnostic arm, "model + Q":** a link is allowed only to a code
    the file's code shares a tracked object with, Q from the evaluator's
    tracks.
  - **It uses labels,** so it is reported, not judged.
