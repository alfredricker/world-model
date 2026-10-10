---
id: "099"
title: version 24 without the parts it no longer reads, arm C fitting its own weight of own tries
rung: 1
serves: [C2, C3, C4, P17]
status: done
verdict: pass
arch_version: 24
date: 2026-10-10
---

# 099: version 24 without its dead parts

Drafted 2026-10-10 at the user's request ("any more cleanup that should
be done with v24 … redundant parts, priors that are MiniGrid specific,
parts that don't have functionality"). It follows a read-only audit of
version 24's code, which three readers did against the run counters of
card 095.5's runs. [Card 100](../100-one-held-intention/card.md) merges
the overlapping parts that keep choices between steps; this card only
removes parts no action depends on.

## 1. Question

Version 24 still builds, at setup, the recall that arm C replaced (cards
042, 047, 049–051, 085, 091). Nothing reads it when the agent chooses an
action, with one exception: arm C borrows the weight of this episode's
own tries from card 050's fit. If that weight is fitted by arm C on its
own store, and everything else unread is no longer built, does every
episode stay the same, with setup at least twice as fast? C2 (one
learner, one representation: one store answers each question), C4 (a
component counts only if it changes behaviour), C3 (declared priors: the
architecture document describes what runs), P17.

## 2. What changes

One component: recall for pick up, drop and toggle becomes arm C alone.
Its predecessors are no longer built. The table lists what goes and why
it is safe to remove.

| Part | Version 24 | Evidence it is unread |
|---|---|---|
| Card 091's router, 091.1's identity-first vote | Loaded; embeds every stored key at setup (5 s on tier 3) | `router_queries` 0 on every tier; arm C overrides its `predict`, card 087 its forward answer |
| Card 049's admission, card 051's groups and own-situation index, for the three interactions | Fitted at setup (tier 3: 89.5 s, toggle 83.4) | Arm C overrides `predict` and `result` (statechange.py:173–202); `old_reads` 0 since 095.5 |
| Card 042's two levels, card 047's situation classes, `templates`, `opened`, card 038's ways, for the three interactions | Fitted or loaded at setup | Read only by the parts above, or by version 20's conditions, which 095.5 replaced |
| Card 085's sampled fits, 085.3's stored fits | Supply the fits above | Their only readers are the fits above. Removed together, since dropping the sampling alone triggers an exact fit that runs out of memory on tier 3 |
| Card 069/070's relation rel:P, as a candidate condition | Admitted for toggle and drop | Read only by admission. Card 070's encoder stays: `WM_REL_ENCODER` sets every token's vector |
| Card 051's routes through two tokens (`walk2.walk`, `clear`) | Installed, then overwritten by card 067 (propagate.py:198) | Its install still installs commit, threats and index; that part stays |
| Card 074.2's cache of recall's pair distance | On | `pd_calls` 0 on every tier |
| Card 074.1's split into "hold h" and a view need | On (`WM_SPLIT=1`) | 095.5's conditions name only the hand; `split_hand_met`, `split_view_held` and `steps_spliced` are 0 |

**What replaces the one number still read.** The weight β sets how much
this episode's own tries at the same (action, front, hand) outweigh arm
C's vote: P = (N_own + β P_C) / (N_own + β) (statechange.py:185). Today
β is card 050's α, fitted for version 18's neighbour vote. Its values
are the grid's floor, 0.000123, for every action on tiers 1–2 and for
pick up and drop on tier 3; tier 3's toggle has 12.18. Here arm C fits β
on the stored tries its store is built from. Each try's outcome category
is predicted from the other stored tries with the same (action, front,
hand) as its own tries, with arm C's vote as the prior. β is chosen on
card 050's grid (`ALPHAS`, 181 values from e^−9 to e^9) by the summed
log likelihood, as card 050 did. One approximation is declared: the
vote is not refitted without each try, because the store is fixed. Over
about 50,000 tries per action, one try barely moves it.

**Not changed:**
- Moves, forward, draw and undraw keep their recall: card 028's pose
  graph, card 087's walkability and card 042's code identity.
- Card 072's trying stays. It never fired on these seeds, but it could
  on others; card 101 is to decide it.
- Card 073 is not loaded and needs nothing.

Code: `tools/card099/lean.py` (to write). It wraps card 066's setup so
that the parts above are not built, and gives arm C its own β.

**ARCHITECTURE.md, on keep.** Rewrite the body (In brief, Learning,
Acting, Data structures, Components) to describe version 25 as it runs.
The version notes at the top stay. Correct the priors list with what the
audit found undeclared:
- Each step's observation is the encoder's vector of a clean render of
  the simulator's tile, looked up from a catalogue of at most 255
  appearances. It is not an encoding of the frame; for MiniGrid's
  noise-free tiles the vectors are the same.
- The world is static except for the agent's actions: remembered tokens
  stay as last seen, and changes at the front or the hand are credited
  to the action.
- Interactions change only the front and the hand, and arm C is asked
  only about those.
- Arm C's results snap to catalogue tokens.
- Six action ids are fixed.
- Which random-play steps are stored is decided by changes in the
  simulator's full grid, and arm C's "changed" labels come from the true
  view.
- The networks trained offline in MiniGrid are card 087's property
  network and arm A.
- Thresholds and caps: inform's "sure" is 0.8; each episode is capped at
  300 s; card 072 tries at most 6 candidates over 100 steps.
- Memory is 3.2 million random steps per tier (tier 3: 888 episodes),
  not "5,000 episodes per world".

## 3. Dependencies

- Version 24 (card 095.5) and its runs as the reference: `runs/095.5/cond_tier*.json` and `fresh_cond_tier3.json`.
- Arm C's store (095.2–095.3).
- Card 050's fit of the own-tries weight. This is the same likelihood, on a different store.

## 4. Data check

- Own-try groups per action in arm C's stored tries: the number of distinct (front, hand) pairs, and how many hold more than one try. This is reported before the fit.
- Version 20's groups were 237 for pick up, 2,168 for toggle and 238 for drop on tier 3.

## 5. Feasibility gate

- **Upper bound:** arm C's own β, per tier and action, compared with version 24's borrowed β. When they fall on the same grid value, the same episodes are expected.
- **Trivial baseline:** version 24 as it runs.

Gate run: the β fit alone, offline, for tiers 1–3 (a few minutes). Result
of the gate, before the main run:

**8 of 9 the same; tier 3's toggle differs** (`runs/099/beta_tier*.json`,
`runs/099/gate.sh`). On arm C's store every (action, front, hand) pair's
other tries predict each try's category exactly, so the fit lands on the
grid's floor, 0.000123, for all nine. Version 24 borrows that value for
eight of them. For tier 3's toggle it borrows 12.18, from version 20's
groups, which also split by the doors in view and by the tokens a toggle
produced. Under 12.18, a toggle the agent has tried still follows arm C's
vote until about twelve tries at the same tokens. Under the floor, one
try decides (card 050's own-tries-first). So tiers 1 and 2 should be
identical, and tier 3 and the fresh seeds are judged by CHARTER's rule
(criterion 1's second branch).

| Tier | Pairs (pick up / toggle / drop) | β (own, all) | Borrowed |
|---|---|---|---|
| 1 | 13 / 13 / 12 | 0.000123 | 0.000123 |
| 2 | 183 / 211 / 186 | 0.000123 | 0.000123 |
| 3 | 237 / 246 / 238 | 0.000123 | 0.000123, **12.18 (toggle)**, 0.000123 |

Setup for tier 3, up to the fit, took 13 s, against version 24's 125 s.

## 6. Success criteria and prediction

1. **Same episodes.** If the gate gives every β the borrowed value: every
   episode of tiers 1–3 and of the 100 fresh tier 3 seeds (1003100–199)
   is step for step version 24's (430 of 430). Otherwise, CHARTER's
   tiers apply: tier 1 ≥ 99%; tiers 2 and 3 and the fresh seeds not
   worse than 100/100, 30/30 and 99/100. Every differing episode is
   traced to β.
2. **Faster setup.** Tier 3's model setup takes at most half of version
   24's 117.6 s. Time per step is not above 0.034 s.
3. **Nothing removed was needed.** The run completes with the parts
   above not built. A read of one of them fails, so the run itself is
   the check.

**Prediction.** The fit lands on the floor wherever own tries decide
alone, which covers five of the six tier 1–2 values and pick up and drop
on tier 3. For tier 3's toggle, the store's groups are not version 20's,
so its β may move. If it does, a few tier 3 episodes differ but none is
lost. Setup on tier 3 is about 30 s: 117.6 less 89.5 of admission and 5
of the router, plus the β fit.

**Budget.** The gate takes a few minutes. Tiers 1–3 and the fresh seeds
take about 15 minutes, as for 095.5.

## 7. Result

`runs/099/main.sh` (the same tiers, seeds and caps as 095.5's runs),
`runs/099/lean_tier*.json` and `fresh_lean_tier3.json`. The comparison is
`tools/card099/compare.py`, which writes `runs/099/compare.json`. Code:
`tools/card099/lean.py` (`WM_LEAN=1`). The flags `WM_ROUTER`,
`WM_ROUTER_ID`, `WM_SAMPLED*`, `WM_STORED_FITS` and `WM_SPLIT` are dropped,
and card 095.5's check for `WM_SPLIT` accepts card 099.

| | Tier 1 | Tier 2 | Tier 3 | Fresh 100 seeds | Tier 3 model setup | Tier 3 s per step |
|---|---|---|---|---|---|---|
| Version 24 | 200/200, 24.3 | 100/100, 30.2 | 30/30, 128.8 | 99/100, 144.5 | 117.6 s | 0.034 |
| **Version 25** | **200/200**, 24.3 | **100/100**, 30.2 | **30/30**, 128.8 | **99/100**, 144.5 | **10.0 s** | **0.012** |
| Episodes the same | 200/200 | 100/100 | 30/30 | 100/100 | | |

"The same" means every count the runner keeps per episode is equal, for
example steps, success, exploration, wrong predictions, plan tests,
orders found and followed, kept ties, arm C's queries, conditions asked,
and intention continued and deliberated. Times are left out, as are four
fields that are reports rather than behaviour:
- **`weighings`** (card 074's record of each order test) is compared
  with card 042's code ids dropped. Those ids are numbered in first-seen
  order, and version 24 numbered them while fitting the removed recall at
  setup. With the ids dropped, all 430 are equal.
- **`rel_weight_end`** is card 069's report on the weight of rel:P, a
  removed part.
- **`sc_new` and `props_prior_tiles`** count misses of arm C's and card
  087's caches. Those caches persist across all the episodes a worker
  process runs, so the counts depend on which episodes share a worker.
  They already differ between version 23's and version 24's runs of
  identical episodes.

The time from start to the first episode on tier 3 is 14 s, against 125 s.
Time per step fell to about a third on tier 3. The removed parts were
also updated while acting (card 051's index, on every try); how much of
the time each part took was not traced.

1. **Same episodes:** met, 430 of 430. Tier 3's toggle β moved from 12.18
   to the floor (the gate) and changed no decision.
2. **Faster setup:** met. Tier 3's model setup takes 10.0 s against
   117.6 s, and time per step is 0.012 s against 0.034 s.
3. **Nothing removed was needed:** met. Every run completed with the
   removed parts not built and their reads raising.

**Deviation, recorded:** the gate's first write compared tier 3's betas
against version 24's under swapped action ids (4 is drop, 5 is toggle).
It was corrected and re-run before the main run; the table in section 5
is the corrected one.

## 8. Decision

**Keep:** architecture version 25. Recall for pick up, drop and toggle is
arm C alone, with its own weight of this episode's tries. Version 20's
recall, card 091's router, card 085's fits and card 074.1's split are no
longer built. Every episode is the same as version 24's, setup on tier 3
is about nine times faster, and each step is about three times faster.
ARCHITECTURE.md is rewritten with this keep, as section 2 said. What is
left of the audit is merging the overlapping parts (card 100), trying
against inform (card 101), and moves by arm C.
