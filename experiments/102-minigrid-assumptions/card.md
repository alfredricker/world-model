---
id: "102"
title: MiniGrid's assumptions out of the model — the series' plan
rung: 1
serves: [P7, P3, P6, P2, P18]
status: draft
verdict:
arch_version: 27
date: 2026-10-11
---

# 102: MiniGrid's assumptions out of the model

Opened 2026-10-11 by the user: "let's make a card series fixing the set
of minigrid specific assumptions that the model employs". This card is
the series' plan; each sub-card is its own experiment.

The number was used before for a card that was never written ("where
to drop", named in card 101 and set aside). That idea is now unnumbered.

## 1. Question

Adapting the model to Crafter and ARC has taken weeks, because
mechanisms took the shape of MiniGrid's facts:
- an action changes only the front and the hand;
- every object fills one tile;
- the world changes only when the agent acts;
- the view follows the agent;
- an action has no place.

These were narrowed "with no loss in MiniGrid", which MiniGrid cannot
show. LESSONS calls this the lookup-table lesson: in one fixed world the
specific form and the general one fit equally well.

The series asks: can each assumption be replaced by a general form that
keeps MiniGrid's results, works in Crafter, and passes a breaking world
built to defeat the assumption? If so, a new environment changes only
the adapter (frames to tokens, the list of actions, the goal) (CHARTER
rules 9 and 10). Serves P7, P3, P6 and P2, and P18 through the next
environments.

## 2. The assumptions and the cards

From ARCHITECTURE's "Built-in priors" (version 27) and the Crafter cards:

| Card | Assumption now | Where it breaks | General form | Breaking world |
|---|---|---|---|---|
| [102.1](../102.1-conditions-between-any-tokens/card.md) | Arm C asks only about the front and the hand, and only those two change | Crafter's "table near" and crafting, ARC's "click here, change there", robots | Every token can change; conditions between any tokens, with relations learned from raw positions | 1, "near, anywhere" |
| [102.2](../102.2-move-together-act-together/card.md) | One token per tile | ARC's shapes, robots' objects, Crafter's lakes | A token is the tiles that move together or change together | 2, "shapes" |
| [102.3](../102.3-short-term-memory/card.md) | A token out of view stays as last seen; only movers get files (Crafter) | Creatures out of view, robots | Short-term memory: every attended token kept, its place spreading as its kind moves, fading when unseen | 3, "out of sight" |
| [102.4](../102.4-where-content-went/card.md) | A step's change is read by written rules: ground under movers, tiles that are not ground cannot vanish, rigid one-step shifts (102.2's layers) | Any world the rules were not written for; robots' scene flow | Where each place's content came from, read from arm A's learned pointer; static learned per code | 2 and 3, reused |
| [102.5](../102.5-the-agent-as-a-token/card.md) | The agent's motion is a fixed shift or turn per action, and the view follows the agent | ARC (a sprite in a fixed view, or no body), robots | The agent is a token, found by which tokens the actions move; its motion predicted like any token's | 4, "a body in a fixed view" and "no body" |
| [102.6](../102.6-actions-with-a-where/card.md) | Actions are fixed ids with no place; where an action acts is the front | ARC's clicks, robots' reaching | An action may carry a where; where it acts is a learned relation | 5, "clicks" |
| [102.7](../102.7-tokens-off-the-grid/card.md) | Tokens are the world's own tiles, and the view moves by whole tokens | Minecraft, robots' cameras | Patches cut by the adapter at any size and phase; motion between them | Crafter cut off its grid |

**Order.** 102.1's pair function gives the relations that the others
use. 102.2 defines what a token is, which 102.3 keeps in memory. 102.4
replaces the written rules both read change by with the world model's
own pointer. 102.5 moves the agent's own motion into 102.3's rule. 102.6 needs 102.1's
relations and 102.2's shapes.

**Progress.** 102.1 done (2026-10-11). Its first run's form is kept by
the user: arm C asks every token, with softmax attention on raw
positions, and crafting's conditions are learned. Relations do not
reliably carry to new positions; three revisions that tried to fix this
each lost crafting. Carrying is left to the training format or to
102.5's locus. 102.2 **keep** (2026-10-11): tiles that share fate are
one token, learned without declaring shape, colour or adjacency. A pair
layer within each token predicts pushes far better (92% against 75%),
short of its bar; the user adopted it. 102.3 **keep** (2026-10-11):
every tracked token is kept out of view with a spread learned per code
(zero for codes never seen to move). Where it comes back is calibrated
and better than "stays as last seen" and 098.6's rule, with 4–8 files.
The user then rejected the written rules 102.2 reads change by (ground,
and "tiles that are not ground cannot vanish"): dynamics are to be
learned by the general architecture. 102.4 (drafted 2026-10-12)
replaces them; 102.5 and 102.6 were renumbered to follow it.
Overnight 2026-10-12 (the user's portability objective): 102.4 revised
(the shortest description reads unambiguous moves in a fixed view, not in a
moving one); 102.5 revise (the agent found by contingency in every world;
Crafter's own motion 93% against 98%); 102.6 revise (a click's where
carries to cells never clicked; shapes need groups); 102.7 added, to
measure the tile-grid prior.

**Not yet planned** (from "Built-in priors"; each to be judged by rule 9
when reached):
- the token lattice sized in advance;
- walking by value iteration over grid placements;
- occlusion computed from tile names (MiniGrid's adapter);
- goals written as a tile read from the simulator;
- results snapped to a catalogue of known tokens;
- an outcome acted on only when its probability is at least one half
  (chance);
- the world changing by itself, covered by card 098.4, whose world joins
  the breaking worlds;
- the planner choosing an action's where (after 102.6).

## 3. How each card is judged

- On Crafter, as CHARTER says, and on its own breaking world.
- Every breaking world built so far runs as a smoke test (rule 10).
- MiniGrid's three tiers run only when a card is kept with a new
  architecture version. MiniGrid must not lose anything: each general
  form has the old one as a special case.
- The breaking worlds come from one small engine,
  `tools/card102/worlds.py`. Card 102.1 builds it, and each card adds
  its world. They give pixels and actions only, drawn as 7 × 7 tiles so
  that Crafter's adapter reads them unchanged. The evaluator keeps the
  state.
- No ARC-AGI-3 environment is installed. Its cases are covered by
  breaking worlds 2, 4 and 5 until one is.

## 4. Series decision

Closed when every row of the table is kept or stopped. The adaptation
cost of the next environment (what had to change beyond the adapter) is
the series' measure, reported when ARC or Minecraft is entered.
