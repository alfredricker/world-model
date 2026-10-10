---
id: "098.1"
title: Crafter interactions predicted by recall over the learned tokens (offline)
rung: 1
serves: [P6, P10, P19, P4]
status: done
verdict: uninterpretable
arch_version: 24
date: 2026-10-10
---

# 098.1: Crafter interactions by recall

Written and approved overnight (2026-10-10, overnight authority; new
card 8 of the night). The second offline evaluation in Crafter (card
098's plan). The agent is not run in Crafter.

## 1. Question

In MiniGrid, recall predicts what pick up, toggle and drop do from the
token in front and the hand (arm C; card 096.1: right on 85–100% of
events after one occurrence). Crafter has 11 interactions (do, sleep,
five place and six make actions), an inventory in place of the hand,
and night. On card 098.2's tokens, learned from pixels, does the same
recall predict what an interaction changes (the token the agent acts on
and the inventory), after few occurrences, by day and by night? P6
(effects), P10 (recall), P19 (few occurrences), P4.

## 2. What changes

Nothing in the agent. New, offline:
- **Codes for every frame** from card 098.2's encoder and prototypes.
- **The place acted on, learned**: Crafter's view is north-up and the
  player's heading shows only in its own tile's appearance. For each
  code of the agent's own tile, the neighbouring place whose code most
  often changes after "do" is its front (as card 028 learned the front
  place in MiniGrid). Judged against the true heading (evaluator's).
- **Events**: each interaction step: action, front code, inventory codes
  before; outcome: the front code after and the codes after of the
  twelve inventory slots that are not vitals (sapling to iron sword).
  Health, food, drink and energy also change by themselves every few
  dozen steps, so they would cap any recall; their changes are reported
  separately. Set before any result was seen.
- **Recall** (arm B's identity level, card 091.1's order): the most
  common outcome among stored events with the same action, front code
  and inventory codes; if none, the same action and front code; if
  none, "nothing changes".

## 3. Dependencies

Cards 098 and 098.2 (data, tokens); card 091.1 (identity first); card
096.1 (the few-occurrence curve, scored on events that change
something).

## 4. Data check

Card 098's 100,000 random-play steps: events per action, the share that
change something, and how many distinct (action, front code) keys occur.
Stored: workers 0–7; judged: workers 8–9 (other worlds).

## 5. Feasibility gate

- **Upper bound:** the same recall on the evaluator's true state (the
  front tile's semantic class and the true inventory counts): what
  tokens could reach at most.
- **Trivial baseline:** "nothing changes".

## 6. Success criteria and prediction

1. **Front learned:** the learned front place matches the true heading
   in at least 95% of "do" steps.
2. **Recall:** on judged events that change something, the outcome is
   predicted right in at least 80% overall, at least 90% of the upper
   bound's rate, and at least 75% at night (light below 0.3). The key's
  inventory part is the twelve non-vital slots.
3. **Few occurrences:** at least 70% right on events whose key was
   stored 1–3 times.

Reported: by action; events that change nothing (against the baseline);
what the misses are.

**Prediction.** Front learned near 100% (the four headings are distinct
appearances). Collect from trees and water near the upper bound;
conditional events (stone needs a pickaxe; place table needs wood) are
right when the inventory key matches and wrong when it falls back to
action and front alone. Sapling from grass (probability 0.1) is
unpredictable by any recall: its "change" outcomes cap the overall rate.

**Decision rules.** Keep (recall's identity level carries to Crafter) if
1–3 hold; revise if 1 holds and 2 fails only through conditional events
(a condition card next); stop if 1 fails.

**Budget.** Codes for 100,000 frames about 2 minutes on the GPU; the
rest a minute.

## 7. Result

### First run (`runs/098.1/results_v1.json`): the test could not be passed

56,002 stored and 14,179 judged interaction events; 353 judged events
change something by the tokens (404 by the evaluator's state).

| | Tokens | True state (upper bound) |
|---|---|---|
| Front learned (agrees with the true heading) | 24.2% | — |
| Right on events that change something | 21.3% | 37.1% |
| Right on events that change nothing (baseline 100%) | 98.9% | — |

Two defects of the setup, not answers to the question:
- **The tokens merge the agent's headings.** Only 2 codes hold the
  agent's own tile; the embedding keeps the heading (nearest neighbour
  86%), but the four headings' centroids are 0.20–0.27 apart and the
  code radius 0.75. Headings change only on move steps, which card
  098's still pairs excluded by action, so they were never contrasted.
- **Absolute outcomes do not transfer.** When the full key (with the
  inventory) was never stored, recall fell back to action and front and
  returned another situation's inventory codes ("wood 2" where the
  agent had none), so even the true state scores 37%. Outcomes must be
  changes relative to the state before, per part: does the front change
  and into what, does each slot change and into what (for a slot, its
  code's learned successor), as arm C reads results relative to the
  try's tokens in MiniGrid.

The feasibility gate fails (the upper bound, 37%, is below criterion 2's
80%), so the first run is not a test. The tokens are fixed first (card
098.3: still pairs chosen by the agent's own check that the view did not
shift, which includes blocked moves); this card's revision then uses
them, with outcomes as changes per part.

### Revision (the card's one): card 098.3's tokens, outcomes as changes per part

`tools/card098.1/interact.py --relative`, `runs/098.1/results_r1.json`.
Per part (the front and each of the twelve slots): does it change (the
majority of stored events with the same action, the agent's own code,
front code and slot codes; else the same action, own code and front
code), and into what (the front's most common code after; a slot's
learned successor: what the same slot showing the same code became
under the same action). One correction during the run: the successor
first ignored which slot it was, so an empty slot gaining wood and one
gaining a sapling were mixed; a slot is a token with a where.

| | Tokens (098.3) | True state |
|---|---|---|
| Front learned (all / awake) | 86.2% / 100% | — |
| Right on events that change something | 30.9% | 38.6% |
| awake / asleep | 33.3% / 12.8% | 39.9% (awake) |
| at night | 20.8% | — |
| key stored 1–3 times | 52.0% | — |
| Right on events that change nothing (baseline 100%) | 99.8% | — |
| "do", changing events | 48 / 131 | 49 / 134 |
| "place plant" | 47 / 78 | 69 / 82 |

**Why even the true state reaches only 39%.** In random play most
events that "change something" are not caused by the action: a cow
walks into or out of the tile in front during a sleep or a failed make
or place (sleep 2 of 16, the six make actions 21 of 112 right even on
the true state), and grass gives a sapling with probability 0.1. "Do" on
a tree is deterministic when awake (180 of 180; the 30 failures were
all asleep). Criterion 2's absolute bars cannot be met on this event
set, so the feasibility gate fails: **uninterpretable** as a test of
criteria 2–3.

What the run does show: on "do" recall over the learned tokens equals
recall over the true state (48 of 131 against 49 of 134); the tokens'
loss is in conditional events ("place plant" needs a sapling: 47 of 78
against 69 of 82), where an exact key over twelve slots is too specific
and the fallback ignores the inventory, and at night (20.8%).

## 8. Decision

**Revise.** The next version scores only changes the action causes:
for each event, the same situation's changes without the action
(stored noop and failed steps) are the baseline, and an event counts as
changing only where the outcome differs from it (the counterfactual
the agent itself can read from memory). Conditions over the inventory
come from arm C's kernel over the slot tokens (which slot matters),
not an exact key. Card 098.3's tokens are its input.
