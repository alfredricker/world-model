---
id: "098.7"
title: how wide "tokens like this" is — the kernel width chosen by held-out prediction
rung: 1
serves: [P7, P2, P10]
status: done
verdict: fail
arch_version: 27
date: 2026-10-11
---

# 098.7: how wide "tokens like this" is

Drafted 2026-10-11 at the user's request ("do 1. first"), after card
[098.6](../098.6-what-moves-by-itself/card.md) (revise).

## 1. Question

In 098.6, recall over "tokens like this" used the encoder's identity
radius (0.771) as its kernel width. That radius mixed movers with
scenery: 31% of mover tokens by day, and 96.5% at night, had more static
neighbours than mover ones.

If instead the width is the one that best predicts held-out episodes,
do object files keep the movers' identities on the encoder's tokens?

Rung 1. Serves P7 (objects from continuity), P2 (belief about a
changing world) and P10 (recall of the relevant experience).

## 2. What changes

One component: the kernel width used for "tokens like this" in
`files.py`. Nothing else in 098.6's files level changes.

```
098.6:      like this = within the identity radius (0.771), for recall, drift and the records' share
this card:  like this = within width h, for the same three uses;
            h = the candidate that best predicts held-out stored episodes' next moves
                (leave-one-out log-likelihood, weighted as the stream; Duin 1976, Silverman 1986)
            candidates: the identity radius times 1, 2/3, 4/9, ... down to 1/40 (a geometric grid)
            refit every 1,000 steps on 3,000 stored episodes
```

- **Identity stays at the identity radius.** These uses keep it: whether
  a token is unchanged, whether a file's token is the same one, the pair
  test, and kinds.
- **The width is chosen by prediction, not set by hand.** It changes
  only if the data change, for instance after a new encoder, so it needs
  no rework when the encoder improves.
- **Declared exceptions carried over from 098.6 (rule 8).** Unseen-object
  densities and appearance rates stay tallies per kind until 098.4. The
  distance is still the encoder's; only its width is chosen.
- **Not in this card:** night tokens (the encoder's line, 098.2–098.3);
  the network member of the bank; 098.4.

## 3. Dependencies

- Card 098.6's files level. On the true state it reaches IDF1 0.934, and
  MiniGrid is unchanged. Its tokens run (0.717) is this card's baseline.
- Card 098.3's encoder vectors and shifts.
- Choosing a kernel width by held-out likelihood: Duin 1976 and
  Silverman 1986, added to LITERATURE.

## 4. Data check

Done while drafting. The memory after workers 0–7 on tokens
(`state_tokens_0to7.pkl`) holds 113,273 stored episodes. 1,500 were
held out per store, each predicted from all the others. Mean
log-likelihood of the held-out next move:

| Width (× identity radius) | 1 | 0.5 | 0.25 | 0.18 | 0.08 | 0.05 | 0.03 | 0.015 |
|---|---|---|---|---|---|---|---|---|
| Followed tokens' steps | −1.34 | −1.13 | −1.11 | **−1.10** | −1.12 | −1.14 | −1.16 | −1.21 |
| Pair events | −11.36 | −7.09 | −6.37 | −6.24 | −5.85 | **−5.75** | **−5.75** | −5.78 |
| Views of tokens that stayed | −0.000 | −0.000 | −0.000 | −0.001 | −0.001 | −0.001 | −0.001 | −0.002 |

What the table shows:
- **The identity radius is the worst width for the episodes that
  matter.** Followed tokens are predicted best near 0.18 times the
  radius, and pair events near 0.05.
- **Stays lose almost nothing at any width.**
- **Adaptive widths (the k nearest episodes) did worse than a fixed
  width on pair events.** The encoder's float16 vectors repeat: only
  18,853 of 126,000 tokens in 2,000 frames are distinct, so the k-th
  neighbour is often at distance 0.

Neighbours by true class (098.6, workers 0–1). This is the share of
mover tokens with more static neighbours than mover ones:

| Radius | 0.771 | 0.5 | 0.3 | 0.2 |
|---|---|---|---|---|
| By day | 31% | 0% | 0% | 0% |
| At night | 96.5% | 43% | 15.5% | 11.7% |

## 5. Feasibility gate

- **Upper bound:** the true state. Its one-hot vectors are either equal
  or √2 apart, and every candidate width is below √2. So the result is
  098.6's 0.934 by construction. A run of 2,000 steps per worker checks
  that the files are identical.
- **Trivial baseline:** 098.6 on tokens, 0.717. Identity by place
  scored 0.009.
- **Prediction check** (worker 7, memory from workers 0–6), 098.6's
  tests: under 5% of static pair events start a file; over 90% of mover
  pair events do by day; recall beats the prior on movers' moves.
  Reported: the chosen width, at each refit.
- **MiniGrid:** 10 tier-1 episodes identical to version 27, no file
  confirmed. (Run before the protocol change below.)

Result of the gate, before the main run (`runs/098.7/`, code in
`tools/card098.7/`):

**Settled while building:**
1. **Refits at doubling intervals** (1,000, 2,000, 4,000 … steps) on 500
   held-out episodes per store, instead of every 1,000 steps on 3,000.
   One refit over the full memory (113,273 episodes, 10 widths) takes 7.8
   s.
2. **The held-out sample has its own random stream,** and recall is
   marked stale only when the width changes. With this, the true state
   is identical to 098.6.
3. **Each file's prediction carries its own list of moves.** The list
   can grow within a step. In 098.6's code a file then read the wrong
   entry. It raised no error in 098.6's runs, and it crashed this card's
   first smoke test.

| Gate item | Result |
|---|---|
| True state, 2,000 steps of each of workers 0–7 | Files identical to 098.6 on every step (11,868 file detections); upper bound therefore 0.934 |
| Chosen width (refits at steps 1k–128k) | 0.51, 0.51, 0.51, 0.34, 0.15, 0.34, 0.15, **0.15** (0.2 × the radius) |
| Tokens, worker 7: static pair events that would start a file | **17.4% of 86 by day, 7.9% of 1,290 at night: fails** (under 5%; 098.6: 0% and 2.6%) |
| Tokens, worker 7: mover pair events that would start a file, by day | 99.7% of 1,425 (pass) |
| Tokens, worker 7: recall against the prior on movers' moves | −1.09 against −1.68 (pass) |
| Tokens, worker 7: movers first seen, confirmed at once | **0%** by day (098.6: 77%) |
| Tokens, worker 7: IDF1 | 0.587 (098.6: 0.713) |
| Tokens, worker 7: movers' frames followed by their own file | 54.6% by day, 2.0% at night |
| MiniGrid, 10 tier-1 episodes | Identical to version 27, no file confirmed |

**The gate fails.** The main run was not made.

## 6. Success criteria and prediction

098.5's criteria, unchanged:

1. **Identity kept.** IDF1 (cows, zombies, skeletons, arrows) on tokens,
   workers 8–9, at least 0.90 and at least 95% of the upper bound.
   Files on static tokens count against it.
2. **Motion read right** in at least 95% of followed steps (90% at
   night). The agent's own motion is placed right as often as without
   files.
3. **MiniGrid unchanged, run only on keep.** Tiers 1–3 and the 100
   fresh seeds are identical to version 27, episode by episode, and no
   file is confirmed. *Changed with the user, 2026-10-11, before the main
   run:* the tiers run only if criteria 1–2 call for a keep, before
   version 28 is adopted.

Reported:
- the share of movers' frames followed by their own file, another file
  or none, by day and night;
- files on static tokens;
- the chosen width;
- recall calls and time per step.

**Prediction.**
- **The chosen width settles near 0.1 times the identity radius.**
- **By day,** movers' frames followed by their own file rise from 72%
  toward the true state's 92%. Births no longer pull in scenery, and
  cows' files break into pieces less often.
- **At night,** the share rises from 14%, but far less. At any width,
  about 12–15% of night movers have mostly static neighbours, and the
  identity radius still decides matching.
- **IDF1 about 0.80–0.88:** criterion 1 is likely to fail, at night.

**Decision rules.**
- **Keep** if criteria 1–2 hold. MiniGrid's tiers then run, and
  version 28, with the files level in the agent, is adopted only if
  criterion 3 holds.
- **Revise** if criterion 1 fails only at night: by day, movers' frames
  followed by their own file are within 5 points of the true state's
  92%. Then the encoder's night card comes next.
- **Stop** if by day the share stays below 80%, or files on static
  tokens exceed 1% of file detections. "Like this" would then not be
  the limit, and 098.6's diagnosis was wrong.

**Budget.**
- The data check is done.
- The true-state identity check takes 3 minutes.
- The tokens run takes about 40 minutes, in three runs of 20 minutes or
  less (workers 0–3, 4–7 with the check, 8–9).
- MiniGrid takes about 25 minutes, and runs only on keep.
- In all, about 45 minutes of wall time, or 70 minutes on keep.

## 7. Result

Not run; the gate failed (section 5).

**Why the chosen width made things worse** (`tools/card098.7/share_by_width.py`,
1,200 tokens of worker 8 with the evaluator's classes). The narrow width
predicts next moves better, but the records' share for a token then
rests on few records:

| Width | Movers by day: median share, ≥ 0.95 | Movers at night: median share, records under the kernel | Static by day: median share |
|---|---|---|---|
| 0.771 (identity radius) | 0.991, 56% | 0.005, 11,333 | 0.001 |
| 0.517 | 0.976, 55% | 0.012, 134 | 0.000 |
| 0.339 | 0.957, 51% | 0.048, 8.7 | 0.000 |
| 0.231 | 0.957, 52% | 0.048, 4.0 | 0.000 |
| 0.154 | 0.948, 49% | 0.048, 0.7 | 0.000 |

(This table is 098.6's memory. With 098.7's own memory, built while the
width moved, movers by day were at 0.91 at the radius and 0.87 at 0.15,
and 7% were at or above 0.95.)

- **By day, the share is the same at every width.** Movers come out near
  0.95–0.99, about half of them at or above 0.95, and static tokens at 0.
- **At night, no width works.** At the radius, night movers are swamped
  by scenery. At narrow widths they have about one record each, and the
  share falls back to the pooled prior (0.048).
- **The width that predicts moves best is not a width for pooling
  evidence.** Narrow neighbourhoods also left this card's memory with
  fewer confirmed files, and so fewer recorded moves for movers.

## 8. Decision

**Stop.** The gate fails, and the measurement answers the question
without the main run: the width of "tokens like this" is not the limit.
- **By day,** births are as good at the identity radius as at any
  narrower width.
- **At night,** no width separates movers from scenery on these vectors.

098.6's diagnosis was therefore half wrong. Night is the encoder's
problem, as it said. But the day loss (72% of frames followed by their
own file, against 92% on the true state) is not caused by "like this".
In 098.6 by day, cows' files broke into pieces on 18.7% of cow frames
(7.9% on the true state), and skeletons and zombies went without files
on 33% and 23%. The cause of that needs its own diagnosis.

098.6's files level (identity radius) remains the best on tokens.
