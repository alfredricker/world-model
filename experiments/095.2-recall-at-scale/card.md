---
id: "095.2"
title: recall at scale — a learned gate, a nearest-neighbour index, and forgetting that shrinks memory
rung: 1
serves: [P17, P10, P19, P6]
status: done
verdict: fail
arch_version: 20
date: 2026-10-09
---

# 095.2: recall at scale (offline)

A step of [card 095](../095-state-change-prediction/card.md)'s series,
on [095.1](../095.1-which-tokens-change/card.md)'s arm C (arm B's vote
over stored token instances, with arm A's network as its prior), before
arm C goes in the agent (095.3). The user, 2026-10-09: "We should have a
card that implements all of this because it should be relatively easy
to diagnose. The router forgetting should prune from memory as well to
save RAM and disk size", and: do this "first to try and get speed boost
before doing a full agent eval". Offline: the agent is not changed, so
CHARTER's tiers are not run; the predictor 095.3 puts in the agent is
the one this card leaves.

**One card, three pieces.** CHARTER asks for one component per card. The
component here is recall's retrieval (which stored instances a query
reads, and which are kept), and the user asked for its three pieces
together. Each piece is switched on alone and then together, with its
own numbers, so a loss can be traced to the piece that caused it.

## 1. Question

Arm B's vote reads every stored group: on tier 3, 14,000–127,000 per
action, growing with experience, for each of the 13–18 tokens of a
state, at every query the planner makes. Memory holds every stored try
(about 300,000 per world; 490,000 rows for tier 2 on disk). Can recall
read only a few dozen instances per uncertain token, and keep only the
instances that carry its predictions, with its predictions and learning
from one try unchanged? Rung 1; P17, P10, P19, P6.

## 2. What changes

```
095.1 arm C: every token → B's vote over every stored group (+ A as prior)      memory: every stored try
this card:  1 gate:      A answers tokens it is sure about; only the rest go to B
            2 index:     B's vote over the k nearest stored groups, found by an approximate nearest-neighbour index
            3 forgetting: stored instances that change no vote are merged or dropped, in RAM and on disk
```

1. **A learned gate.** A small head on arm A's state network, trained
   on training episodes (095.1's leave-one-try-out totals) to predict
   whether B's vote would
   move A's answer by more than δ for this token. Tokens below the gate's
   threshold take A's answer. About 85–90% of tokens are expected to
   stop at the gate (an interaction changes at most 2 of 13–18; A's mean
   predicted change elsewhere is 0.00002).
2. **An index.** B's distance is already a learned metric: the sum over
   admitted conditions of λ times a distance between parts, or a place
   mismatch. Each stored group is written as one vector in which plain
   L1 distance equals B's distance (each admitted part scaled by its λ;
   a place as a scaled one-hot), and put in an approximate nearest-
   neighbour index (HNSW, Malkov and Yashunin 2018, or FAISS, Johnson et
   al. 2017). A query votes over the k nearest (k = 64 to start). An
   exact match is at distance 0 and is always found. Card 091's learned
   router embedding is the alternative if admitted conditions grow too
   many to embed this way.
3. **Forgetting that shrinks memory.** A stored instance is kept for
   what it adds to votes:
   - instances equal on every admitted condition are merged into one
     group with summed counts (lossless for today's conditions);
   - per group, a bounded sample of raw tries is kept (with every try
     whose outcome differs from its group's majority), so a condition
     not yet admitted can still be found later (card 096's note: "a
     condition never stored cannot be learned");
   - a group whose removal changes no held-out prediction by more than
     δ (its influence, measured by leaving it out) is dropped.
   The pruned memory is written as the agent's memory file; the full
   memory is kept beside it until this card's decision, so the step can
   be undone.

**Unchanged:** arm A, arm B's admitted conditions and λ; the agent.

## 3. Dependencies

095.1 (arms A, B, C), 049 (admission), 091
(the router embedding, the alternative metric), 051 (grouping on
admitted conditions). Literature to add: Neural Episodic Control
(Pritzel et al. 2017), an agent's episodic memory read by nearest
neighbours over learned keys; kNN-LM (Khandelwal et al. 2020),
nearest-neighbour retrieval over billions of entries; HNSW and FAISS;
for forgetting, influence of a training point (Koh and Liang 2017) and
prototype condensation (Hart 1968, condensed nearest neighbour).

## 4. Data check

Per tier and action: tokens per state and the share the gate would pass
(from 095.1's held-out predictions); stored groups, raw tries and bytes
on disk and in RAM; the share of B's kernel weight in its k nearest
groups (if the top 64 hold over 99% of the weight, the index loses
almost nothing); groups whose leave-one-out influence is below δ.

## 5. Feasibility gate

- **Upper bound:** each piece alone and all three together against the
  full vote (095.1's arm C), on 095.1's held-out tries and counterfactual
  boxes: held-out log-likelihood within 0.001 per try, exact next state
  not lower, counterfactual box test not lower.
- **Trivial baseline:** arm A alone (no recall), and the full vote.

## 6. Success criteria and prediction

1. **Predictions kept:** within the feasibility gate's tolerance, each
   piece and all together, on tiers 1–3 and the counterfactual boxes;
   and **one try still counts**: a single contradicting try added to
   memory (a pick up of a box that fails) changes the prediction for its
   own state as it does under the full vote, and is never gated out,
   indexed past or forgotten.
2. **Query cost:** stored groups read per query, and time per query
   (one thread, batched as the planner asks), at least 10 times lower
   than the full vote on tier 3.
3. **Memory:** stored memory in RAM and on disk, per tier, at least 10
   times smaller than memory 066's on tier 3, with the full memory kept
   beside it.

**Prediction.** The gate passes 10–15% of tokens; the top 64 groups hold
over 99% of B's weight; merging on admitted conditions alone shrinks
memory about 20–100 times (tier 3: 1.2 million distinct instances per
action into 14,000–127,000 groups), and influence pruning a further 2–5
times. Predictions within the tolerance; reads per query 100–1,000 times
fewer, time per query 10–50 times lower.

**Budget.** Data check: about 10 minutes. Building the gate, index and
pruned memory and scoring them: about 30 minutes (tier 3's fit took 24
minutes in 095.1; the pruning reuses it). No agent runs.

## How it was built (differences from the plan)

`tools/card095.2/scale.py`, `radius.py`; `runs/095.2/scale_<tier>.json`.
Arm B is rebuilt from 095.1's admitted conditions (not refitted), and
095.1's saved arm C predictions are the reference (checked against the
rebuilt arm on 300 queries per action).

- **Index:** no nearest-neighbour library is installed (FAISS, hnswlib),
  so the index is an exact radius search built on B's own structure: per
  admitted condition, the stored values within distance R of the query's
  (R = 12, a kernel weight of 6 × 10⁻⁶), combined depth first with the
  strongest condition first, and looked up in a hash of the groups.
  Exact within R, so nothing near is missed.
- **Forgetting:** a group with no change is dropped when its neighbours
  (itself left out) predict a change below 0.02 on at least its own
  weight; then any dropped group near a stored group whose P moved by
  more than 0.01 is restored (up to three passes). Influence is measured
  on stored groups, not on held-out tries, which would leak. Changed
  instances are kept, merged when equal. Raw tries kept: surprises (none
  found) and 2% of training tries, for conditions not yet admitted.
- **Gate:** as planned, trained on 095.1's 20,000 leave-one-try-out
  instances per action; its threshold passes 99.9% of their positives.

## 7. Result

Held-out episodes, all actions together (log-likelihood per try of which
tokens change; exact next state):

| | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Full vote (095.1's arm C) | −0.000014, 100% | −0.000057, 99.990% | −0.00018, 99.838% |
| Index | −0.000014, 100% | −0.000049, 99.994% | −0.00018, 99.834% |
| Forgetting | −0.000014, 100% | −0.000057, 99.990% | −0.00019, 99.838% |
| Gate | −0.00076, 99.987% | −0.0088, 99.929% | −0.021, 99.716% |
| All three | −0.00076, 99.987% | −0.0088, 99.933% | −0.021, 99.715% |

Counterfactual boxes (tier 2, the hand holds the box in front): full,
index and forgetting 100% for every colour; gate and all three 94–100%
(the gate held back 2–19% of the hand tokens, where arm A answered).

One try: a failed try added at the state of 200 held-out tokens
predicted to change moves P the same under every configuration (largest
difference 0.0008); with tens to thousands of stored tries per state, one
try moves P by at most 0.1 (tier 2 toggle: 0.96 → 0.86).

Cost, tier 3 (one CPU thread; stored groups read per token):

| | Pick up | Drop | Toggle |
|---|---|---|---|
| Full vote: groups, ms per token | 14,300, 2.2 | 52,604, 3.0 | 127,100, 2.3 |
| Index on kept groups: groups, ms | 1.8, 0.012 | 8.1, 0.031 | 5.4, 0.24 |
| Speed-up, index | 190× | 96× | 9.7× |
| Gate passes | 2.6% | 0.2% | 1.4% |

Toggle's index reads few groups but walks many value combinations on its
two weak conditions (weight 2 and 0.5). Tier 2 pick up: 7×; tier 2
toggle: 0.8× (the full vote there is only 6,835 groups).

Memory, tier 3:

| | RAM | Disk |
|---|---|---|
| Interaction tries (095.1's states) | 283 MB | 12.5 MB |
| Memory 066 (every action, moves too) | | 12.5 MB |
| Store: groups and changed instances | 20.9 MB (13.6×) | 1.37 MB (9.1×) |
| After forgetting, + 2% raw tries | 15.3 MB (18.5×) | 1.39 MB (9.0×) |

Groups after forgetting: tier 3 194,004 → 62,863 (pick up none dropped,
drop 64% and toggle 77%); tier 2 47,883 → 34,921. Most of the saving is
the store itself: changed instances merged on equal fields (tier 3 pick
up 106,314 → 58,880), not the per-try record.

1. **Predictions kept:** met by the index and by forgetting (within
   0.00002 per try; exact next state within 0.004 points, tier 3 index;
   counterfactual 100%; one try unchanged). **Not met** by the gate:
   0.02 per try lower on tier 3, 20 times the tolerance.
2. **Query cost ≥ 10× on tier 3:** index and forgetting 190× and 96× on
   pick up and drop, 9.7× on toggle: met on two of three actions, just
   short on toggle. With the gate 700× or more, but the gate fails 1.
3. **Memory ≥ 10× on tier 3:** RAM 18.5× met; disk 9.0×, just short.

**Why the gate fails.** Arm A disagrees with the full vote by more than
0.01 on 0.01–0.6% of tokens, as often on training episodes as on held-out
ones, so it is not A overfitting. The gate saw only 2–108 such tokens per
action among its 20,000 training instances, too few to learn where they
lie, and its threshold, set on those few, does not carry to new
episodes.

**Prediction:** gate passes 10–15% (it passed 0–2.6%, and wrongly);
index reads 100–1,000× fewer groups (it read 6,000–24,000× fewer); time 10–50×
lower (10–190×); memory 20–100× smaller from merging (14× in RAM, 9× on
disk).

## 8. Decision

**Revise** (the user, 2026-10-10: "keeping the index and forgetting but
dropping the gate"). The index and forgetting keep every prediction and,
weighted by each action's held-out tokens, make B's vote 31× faster on
tier 3 (7× on tier 2), with 18× less RAM; they go into 095.3 as they are.
The gate is dropped: the index already makes B's vote cheap, and a gate
trained on so few disagreements cannot find them. If a gate is wanted at
larger scale: train it on every training instance (the index makes their
leave-one-out totals cheap), or read A's own uncertainty.
