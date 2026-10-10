---
id: "096.5"
title: episodes stored as the condition changes that mattered, read for how soon the goal is
rung: 1
serves: [P12, P10, P19]
status: done
verdict: fail
arch_version: 24
date: 2026-10-10
---

# 096.5: episodes as condition changes

A step of [card 096](../096-structure-memory-by-consolidation/card.md)'s series (the user,
2026-10-09: episodic memory as sequences of the condition changes that
mattered, not frames), written and approved overnight (2026-10-10,
overnight authority; new card 6 of the night). Offline reading of
episodes the agent records; the agent's actions do not change.

## 1. Question

If each episode is kept as the sequence of condition changes that
mattered (each interaction that changed the token in front or the hand,
with its step), can the agent read from its stored episodes how soon the
goal will be reached after a given change, better than from how many
changes or steps have passed? And does prediction error (arm C's
surprise) mark those changes, as event segmentation suggests? P12 (from
any state, how soon a goal is), P10, P19.

## 2. What changes

Nothing in the agent's choices. A recorder (`tools/card096.5/episodes.py`,
`WM_EPISODES=1`) keeps each real interaction try: step, action, tokens in
front and in the hand before and after, and arm C's surprise (minus the
log probability of the outcome's category). Read offline:
- **how soon, from the same change:** the median, over the other stored
  episodes that made the same change (same action, tokens before and
  after), of the steps from that change to their goal;
- **baselines:** the same median over stored episodes after their k-th
  change (the same count done), and after the same elapsed step.

## 3. Dependencies

Version 24; card 095.3's arm C (surprise). Literature: Kurby and Zacks
2008, EM-LLM (boundaries from prediction error); Soar's episodic memory
(changes, not views).

## 4. Data check

Version 24's episodes: tier 2's 100 test seeds; tier 3's 30 test and 100
fresh seeds (130). A held-out episode is predicted from all the others
of its tier.

## 5. Feasibility gate

- **Upper bound:** the true remaining steps (error 0).
- **Trivial baseline:** the two baselines of section 2.

## 6. Success criteria and prediction

1. **How soon:** on tier 3, the median absolute error from the same
   change is lower than from either baseline, and it is better at more
   change points than worse against the elapsed-step baseline (sign
   test, p < 0.05). Tier 2 reported.
2. **Coverage:** at least 80% of tier 3's change points have a stored
   episode with the same change.

Reported: the share of condition changes arm C found surprising
(surprise above 0.1) and which tries surprised it.

**Prediction.** The same change predicts how soon better than counts or
elapsed steps, because tier 3's chains differ in length by which room
the ball is in and the change names the room's door. Surprise marks few
changes (box contents, a few first-time events): in a world the agent
predicts well, what matters is better marked by "it changed the state"
than by "it surprised".

**Budget.** Recording 10 minutes; analysis a minute.

## 7. Result

`runs/096.5/howsoon_tier2.json`, `howsoon_tier3.json` (version 24's
successful episodes, each predicted from all the others of its tier).
The recorder changed no episode (tier 2 100/100 in 30.2 steps).

| | Tier 2 (100 episodes, 630 changes) | Tier 3 (129 episodes, 3,087 changes) |
|---|---|---|
| Steps still to go (median, truth) | 12 | 63 |
| Error from the same change (median / mean) | 1 / 2.2 | 35 / 42.1 |
| from the same count of changes done | 1 / 2.2 | 34 / 40.8 |
| from the same elapsed step | 2 / 3.0 | 34 / 41.6 |
| Same change better / worse than elapsed step | 340 / 138 | 1,479 / 1,557 |
| Changes with a stored same change | 100% | 99.9% |
| Changes arm C found surprising (above 0.1) | 8.3% (locked door opened with its key, P 0.9) | 0% |

Every interaction try the agent made changed something (630 of 630;
3,087 of 3,089): version 24 does not try what it predicts will fail.

1. **How soon:** not met on tier 3 (the same change is no better than
   counting); on tier 2 it ties counting and beats elapsed steps.
2. **Coverage:** met (99.9%).

On tier 3 the steps still to go depend on where the ball's room is and
which doors are already open; which door was just opened, or which key
picked up, says nothing about that. On tier 2 the chain is the same in
every episode, so any index works. Arm C's surprise is over whether the
front and the hand change, and every category is predicted; only the
contents of a box are uncertain (card 095.7), which the category does
not show.

## 8. Decision

**Stop** (as a reader of how soon). Condition changes recur across
episodes (99.9% have a stored match), so they are material for compiled
chains, but alone they do not carry how soon a goal is: that needs the
situation the change happened in (where the goal is relative to it),
which is what walking's closeness already computes. Prediction error
on outcome categories marks nothing in a world the agent predicts; for
"what mattered", "it changed the state" is the usable test here. The
recorder (`WM_EPISODES=1`) stays available for card 097.3.
