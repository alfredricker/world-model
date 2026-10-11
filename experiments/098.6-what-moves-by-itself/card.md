---
id: "098.6"
title: what a followed token does next — a bank of predictors, recall on a trigger, files from explanations
rung: 1
serves: [P7, P2, P6]
status: done
verdict: fail
arch_version: 27
date: 2026-10-10
---

# 098.6: what a followed token does next

Drafted 2026-10-10 at the user's request, after card
[098.5](../098.5-tokens-that-persist/card.md)'s main run could not be made;
redrafted the same day from two literature reviews (LITERATURE, current
focus): what decides that a kind moves by itself, and which process
predicts, and when.

## 1. Question

On the encoder's tokens, 098.5's files level took things that never move
for things that do: one false "moved by itself" event made all grass a
mover (8,425 grass files against 154 for cows in 2,500 steps). Suppose
each followed token's next move comes from predictors weighted by how
well they have just predicted it, with recall run only when something
calls for it, and a file is born and kept only when "it moved" explains
the view better than "its vector drifted". Does each moving token then
keep one identity, with static ones left alone? Rung 1; serves P7
(objects from continuity), P2 (belief about a changing world) and P6
(what a token does next, which 098.4 extends).

## 2. What changes

One component: the files level's prediction of what a token does next,
and the births and existence read from it (`files.py`).

```
098.5 (revised):  next move: a tally per kind of the moves after each last move
                  a kind is a mover from its first "moved by itself" event; every changed token like it gets a file
this card:        next move: a bank of two predictors per file, weighted by how well each just predicted it (IMM)
                    prior:  most things stay; a move is one of the agent's own steps (rates from all experience)
                    recall: stored episodes of tokens like this one, given its last move
                  recall runs on a trigger only: a new file, a file found again, surprise, a changed look
                  a changed place no file takes: "a token moved here", "its own token drifted", "a new object"
                    a new file's existence = P(moved or new); then raised on each hit, lowered on each miss
                  confirmed at 0.95, deleted below 0.05; only confirmed files change belief and are reported
```

- **Every step, cheap.** Each file's grid Bayes filter predicts with the
  bank's mixture; each member's weight is multiplied by how likely it made
  the observed move (Blom and Bar-Shalom 1988; Wolpert and Kawato 1998).
  This is CHARTER's "whichever predicts better", kept per file.
- **Recall on a trigger** (Zacks et al. 2007; Yu and Dayan 2005): when a
  file is born or found again, when its look moves past the identity
  radius, or on surprise, i.e. the observed move less likely under the
  bank than under the prior alone (AXIOM's rule for a new component).
  Otherwise recall's last answer for that file is reused.
- **Recall serves the unfamiliar** (Lengyel and Dayan 2007; `1606_04460`).
  A token like nothing stored gets the prior: most things stay (Zago et
  al. 2004: priors persist).
- **Three explanations of a change** (PMBM's birth-versus-update):
  "a token moved here" (recall's prediction for the token that vanished a
  step away, under 098.5's pair test); "its token drifted" (this frame's
  noise for tokens like it, estimated per scan as trackers estimate an
  unknown clutter rate, Mahler et al. 2011); "a new object" (098.5's
  unseen-object density). A false grass "move" at night is better
  explained as drift; nothing is flagged as a mover.
- **Declared exceptions (rule 8).**
  - 098.5's unseen-object densities and appearance rates stay tallies per
    kind (a cluster of vectors) until 098.4 predicts births and vanishing.
  - Similarity is the encoder's distance at its identity radius, not a
    learned metric.
- **Not in this card:** the network member of the bank (the next card:
  trained by replay on these episodes, with a gate trained on held-out
  error deciding where recall is still called, He et al. 2021); motion
  with a visible cause; crowding; placing errors; the encoder.

### Revision after the first gate (2026-10-10, with the user)

The first gate failed because a token seen for the first time had only
one explanation, "a new object" (§5). The user generalised the question
"does a token like this move by itself?" to "how likely is an individual
token to have dynamics that produce unique outcomes?". Their example is a
cup with a hole in it: it never moves, but it does not behave like other
cups.

**The rule.** A token needs memory of its own (a file) when its own
history predicts it better than the shared predictions do: the belief
map at its place, and recall over tokens that look like it. Moving by
itself is the case this card can measure, where the outcome is the
token's position. Other outcomes come with the interaction cards.

**How it is measured.**
- **One record per token** that is followed by place or by a file. A
  record holds the token's vector when first seen, the steps it was seen
  after that, and how many of them it moved by itself. Each move is
  weighted by its probability of being the token's own move.
- **A fitted share.** For a new token, the file's starting existence is
  the share of records like it (within the identity radius) whose
  history needed their own memory. That share comes from a two-part
  mixture fitted by EM:
  - a token the map predicts is seen to move only at the rate any token
    is;
  - a token with its own memory moves at a rate of its own, drawn around
    the rate of all followed tokens with one episode's weight. This beta
    prior lets individuals differ (Kemp, Perfors and Tenenbaum 2007).
- **A record seen for few steps counts only as much as it shows.** A cow
  that left view after two steps tells little.
- **Births.** This replaces 098.5's births of "a kind's token". It
  applies to every token that no file or pair explains, at a place
  first seen or at a changed place. Pairs keep their three explanations.



- Card 098.5's revised files level: it passed its gate on the true state
  (IDF1 0.949) and MiniGrid, not its main run, which this card makes
  possible; it is judged here on 098.5's criteria.
- Card 098.3's encoder vectors, identity radius (0.771) and shifts.
- Established methods in LITERATURE: interacting multiple models and
  responsibilities (IMM; MOSAIC); recall as a kernel-weighted average
  (`1703_01988`, `1606_04460`); surprise-gated updating (Zacks; Yu and
  Dayan; `axiom`); PMBM birth and existence (`1703_04264`); clutter
  estimated per scan (Mahler et al. 2011); a hierarchical prior from all
  tokens (Kemp, Perfors and Tenenbaum 2007).

## 4. Data check

Workers 0–1 of 098.5's play (60,000 steps), encoder tokens, the agent's
true motion.

| | By day | At night |
|---|---|---|
| Static places drifting past the radius per step | ≤ 0.0005 | 0.06 – 0.16 (grass 0.12) |
| Grass appearing to move by itself, per view | 1 in 1,370,000 | 0.0079 |
| Cows, zombies, skeletons, arrows appearing to move | 0.25 – 0.81 | 0.08 – 0.28 |

Episodes for recall: cows in view in 132,338 frames, zombies 36,117,
skeletons 28,611, arrows 24,420.

## 5. Feasibility gate

- **Upper bound:** the files level on the true state, workers 8–9, IDF1
  at least 0.9 (098.5's revision: 0.949).
- **Trivial baseline:** version 27's identity by place on the tokens.
- **Prediction check:** recall built on workers 0–6, queried on worker 7.
  It passes if under 5% of static-class pair events would start a file,
  over 90% of mover pair events would by day, and recall predicts the
  followed movers' next moves better than the prior (log-likelihood).
  Reported: recall calls per step.
- **MiniGrid:** nothing moves, so no file is born.

Result of the gate, before the main run (`runs/098.6/`; code in
`tools/card098.6/`, 098.5's code untouched).

**First build: failed** (`runs/098.6/files_first_gate.py`).

| Gate item | Result |
|---|---|
| Upper bound, true state, workers 8–9 | 0.940 (cows 0.919, zombies 0.966, skeletons 0.932, arrows 0.967) |
| Prediction check, tokens | Not completed: about 10 minutes per worker against 75 s on the true state, stopped after workers 0–2 |
| MiniGrid, 10 tier-1 episodes | Identical to version 27; no file born |

On the encoder's tokens, static tokens got files. Measured on the first
1,500 steps of each of workers 0–7:
- **At least 15,584 confirmed files on static tokens** (grass: 10,765 by
  day, 3,226 at night), against 1,142 on movers.
- **12,995 of the grass files were born at places seen for the first
  time** and confirmed at once.
- **The cause was my change while building.** I had removed the "a
  token that stays" comparison at first-seen places, leaving "a new
  object" as the only explanation there. Once grass had one confirmed
  file it was a "kind", and every newly seen grass token got a file with
  existence 1.

My first report of these numbers was wrong. The diagnostic labelled
workers 1–7 with worker 0's true classes; the counts above are
re-measured with each worker's own labels. The diagnosis itself held.

**The revision** is described in §2.

**Settled while building:**
1. **A compact kernel** (1 − (d/r)² inside the identity radius) in place
   of a Gaussian. With a Gaussian of the radius's width, classes √2
   apart still get weight e⁻⁴. Grass episodes outnumber cow episodes
   about 100 to 1, so grass would leak into recall for cows.
2. **Each change counts toward the drift rate by its probability of
   being drift** (Chen et al. 2013). Before this, objects arriving
   without a file counted as grass drifting. Worker 7, true state, first
   build: 0.918 → 0.923.
3. **A tentative file not confirmed within 3 steps is deleted** (M-of-N
   confirmation, Bar-Shalom, Campo and Luh 1990).
4. **The individual part of the mixture moves at the pooled rate of all
   followed tokens, not at the rate of followed tokens that look like
   this one.** For a look whose few followed tokens never moved, the
   look-specific version makes the two parts identical, so the share is
   undetermined. On the true state, stone came out at 0.15 and water at
   0.09 that way, against 0.000 with the pooled rate.
5. **The beta prior has the weight of one episode.** With a fixed rate
   instead, a skeleton that waited 17 steps counted as almost certainly
   scenery: the share for skeletons was 0.49, against 0.67 with the
   beta prior (true state, 2,000 steps per worker).
6. **Pairs use the same mixture.** A pair's "moved" explanation is the
   share for tokens like the one that vanished, times a followed
   token's chance of making that move. The rest is "drifted" or "an
   unmodelled cause" at the rate any token is seen to move. Before
   this, the pair used recall over all tokens like it, which already
   mixes scenery with individuals.
7. **At a place seen for the first time**, the alternative to a file's
   match is "a token the map predicts", weighted by 1 minus the share.
   Without this alternative, any match confirmed a file. In MiniGrid,
   where every episode starts with no records (share 0.5), that changed
   5 of 10 tier-1 episodes. With it, 10 of 10 were identical and no file
   was confirmed.
8. **The long-term memory** is saved and reloaded between runs: workers
   0–3, then 4–7, then 8–9. This is the same as running them in order,
   in runs of 20 minutes or less.
9. **Changes made and then removed, because they were reactions to the
   mislabelled diagnostic:** a move shared with its surroundings did not
   count as the token's own, and a file's matched move was also weighed
   against an unmodelled cause. Without them, 165 cows, 120 arrows, 34
   zombies and 18 skeletons were confirmed, and 17 static tokens (first
   1,000 steps of workers 0–7, tokens).

**Revised build:**

| Gate item | Result |
|---|---|
| Upper bound, true state, workers 8–9 | **0.934** (cows 0.919, zombies 0.997, skeletons 0.787, arrows 0.986); no file on a static token; motion right on all 49,352 followed steps (7,816 at night); recall 0.09 calls per step |
| True state, worker 7: first-seen static tokens that start a file | 0 of 25,929 by day, 0 of 897 at night |
| True state, worker 7: movers that start a file | all; 89% confirmed at once at first sight by day, 67% at night (12) |
| True state, worker 7: recall against the prior on movers' moves | mean log-likelihood −1.11 against −2.69 (27,369 moves) |
| Tokens, worker 7: static pair events that would start a file | 0% of 50 by day, 2.6% of 1,296 at night (pass: under 5%) |
| Tokens, worker 7: mover pair events that would start a file, by day | 98.8% of 251 (pass: over 90%); at night 57.5% of 146 |
| Tokens, worker 7: recall against the prior on movers' moves | −1.13 against −2.80 (20,729 moves; pass) |
| Tokens, worker 7: first-seen static tokens that start a file | 0 of 25,300 by day, 0 of 560 at night |
| Tokens, workers 4–7: recall calls per step | 0.14–0.16 (true state 0.09–0.10) |
| MiniGrid, 10 tier-1 episodes | Identical to version 27; 343 tentative births, none confirmed |
| Placing the agent's own motion, tokens, no files | Right on 98.2% of steps (workers 8–9) |

The upper bound is below the first build's 0.940, because of
skeletons (0.787 against 0.932). Skeletons often wait in place. One seen
for the first time while waiting is about a coin toss between "a
skeleton" and "scenery that looks like one". It gets its file when it
first moves.

**The gate passes.** The upper bound is above 0.9, the prediction check
passes all three tests, and MiniGrid confirms no file.

## 6. Success criteria and prediction

098.5's criteria, unchanged:

1. **Identity kept.** IDF1 (cows, zombies, skeletons, arrows) on the
   tokens, workers 8–9, at least 0.90 and 95% of the upper bound. Files on
   static tokens count against it.
2. **Motion read right** in at least 95% of followed steps (90% at night);
   the agent's own motion placed right as often as without files.
3. **MiniGrid unchanged:** tiers 1–3 and the 100 fresh seeds identical to
   version 27, episode by episode; no file confirmed.

Reported: files on static tokens; recall calls and time per step; the
bank's weights; night; placing errors.

**Prediction.** Recall runs far less than once per file per step, mostly
for births at night. Grass gets almost no files, because drift explains
its false moves. By day movers keep their identities about as on the true
state; at night they lose files more often. IDF1 near the bar (0.85–0.92).

**Decision rules.**
- **Keep** if 1–3 hold: version 28, files level in the agent; then the
  network card and 098.4.
- **Revise** if criterion 1 fails only at night or only through placing
  errors.
- **Stop** if files on static tokens exceed 1% of confirmed file frames,
  or the upper bound falls below 0.9.

**Budget.** True state 8 minutes, tokens 20, MiniGrid 6, baseline 3: about
40 minutes, in runs of 20 minutes or less.

## 7. Result

`runs/098.6/tokens.json` (main run, tokens, workers 8–9, from the memory
of workers 0–7), `truth.json` (upper bound), `nofiles.json` (baseline),
`c_tier*.json` and `fresh_c_tier3.json` (MiniGrid); numbers also in
`results.json`.

| | Tokens (this card) | True state (upper bound) | Identity by place (baseline) |
|---|---|---|---|
| IDF1, movers pooled | **0.717** | 0.934 | 0.009 |
| ID recall / ID precision | 0.632 / 0.827 | 0.924 / 0.945 | 0.259 / 0.005 |
| ID recall: cows, zombies, skeletons, arrows | 0.68, 0.48, 0.51, 0.70 | 0.92, 1.00, 0.79, 0.99 | 0.21, 0.47, 0.32, 0.19 |
| File detections on static tokens | 0.39% | 0% | 98.3% |
| Motion right (followed two frames running) | 99.0% of 37,680 (night 99.1% of 1,397) | 100% of 49,352 | |
| Agent's own motion placed right | 98.05% (without files 98.18%) | given | |

The baseline counts every token like a confirmed kind as an object, by
place. On tokens most of those are scenery.

**Where identity is lost** (share of the movers' object frames, workers
8–9):

| | Its own file | Another file | No file |
|---|---|---|---|
| Tokens, day (47,109 frames) | 72.3% | 14.5% | 13.2% |
| Tokens, night (8,546) | 13.6% | 4.1% | **82.4%** |
| True state, day | 91.9% | 5.8% | 2.3% |
| True state, night | 95.3% | 3.2% | 1.4% |

On tokens by day, cows' files break into pieces (18.7% of cow frames go
to another file, against 7.9% on the true state). Skeletons and zombies
go without files (33% and 23%). At night almost nothing gets a file.

**Cause: similarity at the identity radius,** this card's declared
exception. Workers 0–1 were measured with the true classes (6,000
frames). At the encoder's identity radius (0.771):
- **By day, 31% of mover tokens** have more static tokens than movers
  among their neighbours.
- **At night, 96.5% do,** with a median of 303,003 static neighbours.
  Night tokens fall inside one another's radius.

Recall over "tokens like this" therefore mixes movers with scenery. The
share of tokens needing memory of their own comes out near zero for
movers at night, and files are not born. At radius 0.5, no mover by day
and 43% at night have static neighbours outnumbering the movers; at 0.3,
none by day and 15.5% at night. On the true state, the same files level
keeps identity day and night.

**Criteria.**
1. **Identity kept: fails.** IDF1 0.717 against 0.90, and 77% of the
   upper bound against 95%.
2. **Motion read right: passes.** 99.0% (night 99.1%). The agent's own
   motion is placed right on 98.05% of steps with files and 98.18%
   without, a difference of 78 of 60,000 steps.
3. **MiniGrid unchanged: passes.** All 430 episodes (tier 1: 200, tier
   2: 100, tier 3: 30, fresh tier 3: 100) are identical to version 27,
   step by step, and no file was confirmed. There were tentative births,
   and the time per step rose from 0.0017–0.0116 s to 0.0031–0.0308 s.

**Prediction:** wrong on IDF1 (predicted 0.85–0.92, measured 0.72). It
was right that grass gets almost no files, that recall runs far less
than once per file per step (0.15 calls per step on tokens), and that
night is worse.

## 8. Decision

**Revise.** The decision rules did not foresee this case:
- **Not keep:** criterion 1 fails.
- **Not stop:** files on static tokens are 0.39% of file detections,
  under 1%, and the upper bound is 0.934, over 0.9.
- **Not quite the "revise" rule:** criterion 1 fails mostly at night,
  but also by day (72% against 92% on the true state).

Both the day and the night losses trace to one measured cause: the
identity radius is too wide a "like this" on the encoder's vectors.
The files level, the share of tokens needing memory of their own, and
the bank hold on the true state.

What the revision needs is the user's choice:
- **The width of "tokens like this"**, chosen by held-out prediction
  instead of the identity radius (the standard way to set a kernel's
  width). This is testable by day.
- **Night**, which needs tokens that keep movers apart from scenery in
  the dark: the encoder's line, cards 098.2 and 098.3.

Version 27 stays. The files level is not added to the agent.
## Appendix A. Settled before building

- **Stored episodes.** Three reservoirs of 50,000 each, uniform samples of
  their streams, each episode weighted by how many it stands for:
  - every step of every file: vector, last move, move, change of look;
  - every pair event, weighted by P(moved) from the explanation;
  - one in 50 views of a token that stayed with every place a step away
    in view.
- **Recall.** A Gaussian kernel of the identity radius on vectors, times
  agreement of the last move (at half a step length). With less than one
  episode's weight, the last move is dropped. The prior, all episodes
  pooled, joins with one episode's weight.
- **Bank.** Weights renormalised each step, with a switching probability
  of 0.05 as in IMM, so a member can recover.
- **Drift.** P(a token like b drifts past the radius) is this frame's
  share of such places among those showing tokens within the radius of b,
  pooled over the last 10 frames with one pseudo-frame of the long-run
  share. The distance it drifts is a kernel density over those changes.
- **Existence.**
  - On a hit: r ← rL / (1 − r + rL), where L is the bank's probability
    times the look match, over the drift and new-object explanations.
  - On a miss: r ← r(1 − q) / (1 − rq), where q is the file's
    probability inside the view.
  - Births at or above 0.95 are confirmed at once.
