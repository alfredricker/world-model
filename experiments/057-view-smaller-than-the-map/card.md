---
id: "057"
title: a view smaller than the map
rung: 1
serves: [P2, P15, P12, P17, C5]
status: approved
verdict:
arch_version: 10
date: 2026-10-05
---

# 057: a view smaller than the map

Drafted and started overnight on 2026-10-05 under the user's standing
instruction for the night ("then the egocentric view that is small than
the size of the map"; AGENTS.md, "Overnight sessions"). The numbers below
were fixed before any run.

## 1. Question

When the agent sees only MiniGrid's 7 × 7 view (six rows ahead, three
columns to each side, nothing behind) instead of a window that always
holds the whole map, does version 10 still reach its goals in the
familiar worlds and the chained rooms? It must keep what it has seen
(its tokens, card 044) and go and look for what it has not. P2 (what is
believed now includes what is out of view), P15 (seek what the agent
does not know), P12 and C5 (the same tasks with less in view; the rest
familiar). Rung 1's tasks under partial view; BabyAI's view.

## 2. What changes

One component: what the agent does with a partial view.

| | Version 10 | This card |
|---|---|---|
| View | 13 × 13 around the agent: the 8 × 8 map is always fully in view | MiniGrid's 7 × 7 ahead of the agent; places outside it are not observed (no occlusion by walls) |
| Placing a view | least distance between the view and every placement's predicted view | the same over observed places only, a place whose token was never seen counting as no evidence |
| Online tries | stored with the real view | stored with the believed view (the remembered tokens around the agent), the form recall's memory and queries already use |
| No chain of conditions | random action | **arm B:** walk to the nearest pose from which a never-seen place would be in view (frontier exploration, Yamauchi 1997); **arm A:** random, as version 10 |

Declared exception (rule 8): the stored memory before acting is version
10's, learned from full views. Removing it means re-collecting play
under the 7 × 7 view and storing each try with the believed view; that
is the next card if this one passes.

## 3. Dependencies

- Card 054's encoder with identity up to noise; version 10 (cards
  049–051); card 044's tokens and placements; card 045's walking.
- Literature: frontier-based exploration (Yamauchi 1997, not in papi);
  MiniGrid's agent view (Chevalier-Boisvert et al. 2023).

## 4. Data check

Familiar worlds: version 10's 30 test layouts per world; chained rooms
with one door (13 × 8): card 051's 100 layouts. With the 7 × 7 view the
agent sees on average about half of the 8 × 8 map's interior at once,
and less than a third of the chained rooms.

## 5. Feasibility gate

- **Upper bound:** version 10 with the full view on the same encoder
  (card 054: 100% in every familiar world and the chained rooms).
- **Trivial baseline:** arm A (random when there is no chain).

## 6. Success criteria and prediction

Encoder of seed 399, then 400–402 if it passes; each criterion in at
least 3 of 4.
1. **Familiar worlds:** arm B succeeds in ≥ 95% in every world, with mean
   steps at most 1.5 × the full-view agent's.
2. **Chained rooms with one door:** arm B ≥ 90%, steps at most 1.5 × the
   full view's.
3. **Belief stays right:** placing every view needs no mismatch on
   observed places (largest distance 0), and arm B beats arm A on both
   tests.

Reported: exploration steps, random steps, the believed tiles that
disagree with the simulator at the end of each episode.

**Prediction.** The familiar worlds pass with a few extra steps of
looking around; arm A loses most where the goal is behind the agent at
the start. Chained rooms are the doubtful test: the outer room's key or
goal may be out of view until the agent walks to the middle.

**Decision rules.** Keep if all three hold (version 11: partial view).
One declared revision if one fails. Stop if arm B is below arm A.

**Budget.** About 10 minutes per seed.

## 7. Result

## 8. Decision
