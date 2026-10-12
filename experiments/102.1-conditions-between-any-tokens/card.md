---
id: "102.1"
title: conditions between any tokens — arm C asked about every token, with relations learned from raw positions
rung: 1
serves: [P6, P3, P12, P17]
status: done
verdict: fail
arch_version: 27
date: 2026-10-11
---

# 102.1: conditions between any tokens

Drafted 2026-10-11 at the user's request ("our model should learn
conditions between any tokens, not just hand and front") as card 095.8.
The same day the user moved it to the 102 series, which removes
MiniGrid's assumptions from the model one card at a time (102.1–102.6).
The user also removed its hand-built relation features: "The only thing
declared is that each token has a where, its raw position. Which
positions count as a relation is learned from outcomes." Offline; the
agent is not run.

## 1. Question

Version 27's arm C answers one question: given an action, the token in
front and the hand's token, does each of the two change, and into what?
Every query is keyed on those two places, and the rest of the view is
marginalised (card 095.3). Card 095.1 had arm C read the whole state and
found by itself that in MiniGrid only the front and the hand change.
Card 095.3 then narrowed the query to those two places "with no loss in
MiniGrid", which MiniGrid cannot show. This is the assumption this card
removes.

Crafter has a condition that is not in front. A pickaxe or sword is
made only when a table is anywhere in the 3 × 3 square around the agent,
and the iron tools also need a furnace there. The effects fall on
several carried tokens at once: making a wood pickaxe takes a wood and
adds a pickaxe.

Can arm C predict, for every token, whether it changes and into what,
from conditions between any tokens? The relations must be learned from
the tokens' raw positions, not declared. And does it then predict what
Crafter's actions change where the two-place query cannot? Rung 1.
Serves P6 (effects on any token), P3 (a condition names a role, "a
table near", not a place), P12 (goals over any condition) and P17 (cost).

## 2. What changes

One component: arm C's query, and where its conditions come from.

```
version 27:  (action, front token, hand token) → P(front changes), P(hand changes), what each becomes
this card:   (action, every token with its raw where) → for each token i: P(i changes), what i becomes
             relations: learned by attention over pairs of tokens (whats and raw wheres)
             the vote compares stored instances by what that attention reads around i
```

- **The state** is every token in view, the carried tokens and the
  agent's own token, each a what (encoder vector) and a where.
- **The where is the token's raw position:** the place where the
  observation shows it. In Crafter, carried tokens are shown in the
  inventory rows, and in MiniGrid in the held row. Nothing else is
  declared: no offsets, distances, neighbourhoods, front or hand.
- **Arm A learns the relations.** A pair function reads token i and
  token j (both whats, both raw wheres, and the action) and gives an
  attention weight and a message. Token i's context is the
  attention-weighted sum of messages over all j. From it, arm A gives
  P(i changes) and what i becomes (card 095.1's four ways). Which pairs
  matter, and at which positions, is whatever predicts outcomes.
- **Arm B's vote compares instances by that context.** A stored token
  instance counts toward a query token by:
  - identity first: same code before similar vectors (cards 042, 091.1);
  - then nearness of their contexts, with the kernel width chosen by
    held-out likelihood (card 098.7).

  This keeps recall's strengths: a new try counts at once, and one
  failed try corrects (P19). The network supplies only the relations.
- **Every token can change.** Version 27's four outcome categories (the
  front and the hand) become one prediction per token, which also covers
  several carried tokens changing at once.
- **Unchanged:** P = (B + α·A) / (N_B + α) with arm A as the prior;
  this episode's own tries first; results as relations (keep, copy from
  another token, shift, set); card 095.2's grouping.
- **Not in this card:** the agent and the planner (a later card, where
  "be near a table" becomes a subgoal); creatures' motion (098.6, 102.3);
  caused against the world's own changes (098.4); objects of many tiles
  (102.2); an action's own where (102.6). Vitals and creature tokens are
  left out of scoring, as in card 098.1, set before any result.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| A token has a raw position in the observation | yes | yes (cell) | yes (pixel or 3D point) | declared |
| Positions lie on a grid of places | yes | yes | no | not used: the pair function reads raw numbers |
| One token per tile | yes, except lakes and tunnels | no | no | kept; removed by 102.2 |
| A fixed set of action ids | yes | partly (clicks carry a place) | no | kept; removed by 102.6 |
| Every token is in view or carried | creatures leave view | yes | no | kept; removed by 102.3 |

## 3. Dependencies

- Arm C: cards 095.1 (whole-state arms; roles found by themselves),
  095.2 (grouping, index) and 095.3 (in the agent). Identity first in a
  learned metric: card 091.1. Kernel width by held-out likelihood: card
  098.7.
- Crafter tokens: card 098.3's, the best kept, stated as such. Crafting
  episodes are by day, where they hold (day purity 99.9%). Inventory
  slots are encoded like any tile (card 098: 99.99% of slots right).
- Literature:
  - Pasula, Zettlemoyer and Kaelbling 2007 (`1110_2211`): rules that
    name objects by their relation to the action's target, in LITERATURE;
  - interaction and relation networks (Battaglia et al. 2016; Santoro et
    al. 2017): one pair function shared across all pairs of objects, to
    be added to LITERATURE;
  - Hintzman 1986: the exemplar vote.
- LESSONS: a learned dot-product score between two things is a lookup
  table, and a relation carries only when the training distribution
  makes the general rule the only solution. Here that means a relation
  learned at the places seen may not carry to new ones. Criterion 2's
  breaking world tests exactly that.

## 4. Data check

From card 098.5's random play (`runs/098.5`, evaluator's state; counted
while drafting):

| | Training worlds (238,564 steps) | Test worlds (59,648 steps) |
|---|---|---|
| Steps with a table in the 3 × 3 square | 1,184 | 317 |
| Make tries with a table there (each make action) | 69–77 | 10–29 |
| Tools made | 8 (wood pickaxe 4, wood sword 4) | 0 |
| of them with the table beside, not in front | 5 | 0 |
| Tables placed | 64 | 19 |

**Random play almost never crafts.** A declared curriculum supplies the
experience (C3; Appendix B). Counted when the data was made
(`runs/102.1/curr_w*.npz`, evaluator's state):

| | Training worlds 0–7 | Test worlds 8–9 |
|---|---|---|
| Steps | 101,618 | 25,467 |
| Tools made | 989 (wood sword 357, wood pickaxe 303, stone pickaxe 154, stone sword 152, iron pickaxe 12, iron sword 11) | 236 (iron tools 5) |
| Makes per near place of the table (each of the 8) | 103–190 | 18–51 |

Every near place has enough successes, and the iron tools are too few
to judge alone (reported only).

**Breaking world 1, "near, anywhere"** (built by this card; Appendix C).
A lamp lights when a box stands beside it. The agent pushes boxes, and
neither the lamp nor the box is the agent. The lamps stand in the left
half of the grid in training worlds and in the right half in test worlds.
Lightings per world are counted when it is built. As built
(`runs/card102/world1_*.npz`): training has 3,000 episodes, 300,000
steps, 6,589 pushes, 545 lamps lit and 392 put out. Test has 1,000
episodes, 100,000 steps, 2,132 pushes, 204 lit and 140 put out. The
world's encoder (card 098.3's recipe on the training play) gives 7 codes,
each holding one kind only.

## 5. Feasibility gate

- **Upper bound:** the same arms on the evaluator's state (true kinds,
  inventory counts and heading as tokens). They must predict held-out
  makes exactly in at least 99% of tries. Crafting is deterministic
  given the table, the furnace and the counts.
- **Trivial baselines:**
  - "nothing changes";
  - **the two-place query:** version 27's form on the same tokens and
    data, with the learned front (card 098.3) and the carried tokens.
    This comparison isolates the card's change;
  - random play's own data, with no curriculum, reported for the record.
- **Timing:** training arm A and the vote on one training world first.
  If the full run would exceed 30 minutes, it goes to the user as a
  command.

Choices settled while building, and the gate in detail, are in
Appendix D.

**Result of the gate** (`runs/102.1/crafter_truth.json`):
- **Passed on make:** the upper bound gets the carried tokens exactly
  right in 99.84% of 7,697 held-out make tries. The whole scored state is
  right in 98.86%, since creatures and plants change by themselves.
- **Missed:** all 5 iron tools (23 in training).
- **Role test:** 95.4% for the upper bound, so criterion 2's bar is just
  reachable.
- **Relations found:** in layer 2, the made tool's slot puts 94% of its
  attention on the table.

## 6. Success criteria and prediction

Judged on Crafter's worlds 8–9, breaking world 1's test worlds, and card
095.1's held-out MiniGrid episodes.

1. **Crafter's effects.**
   - Make actions: the log-likelihood per try of which scored tokens
     change must be above the two-place query's (paired bootstrap,
     p < 0.05). The share of tries whose whole scored next state is
     right must be at least 95% of the upper bound's.
   - Do and place: no worse than the two-place query, within 0.01 nats
     per try.

   This would show that a condition away from the front changes the
   prediction where it decides the outcome (C4).
2. **A role, not a place.**
   - **Crafter:** on held-out states where a tool was made, the
     evaluator moves the table's token to each of the 8 places around
     the agent and the 16 places two steps away. P(made) must be at
     least 0.5 at all 8 near places and below 0.5 at all 16 far ones,
     in at least 95% of these 24-place tests.
   - **Breaking world 1:** with the lamps at positions never seen in
     training, P(lights) must be at least 0.5 when a box is pushed beside
     the lamp and below 0.5 otherwise, in at least 95% of pushes.

   In Crafter the agent always stands at the centre of the view, so a
   relation learned per position could pass there; breaking world 1
   cannot be passed that way. This would show that the condition is a
   relation between tokens, learned from tries.
3. **Nothing lost in MiniGrid.** On card 095.1's held-out tries of tiers
   1–3, the share of tries whose whole next state is right must be within
   0.01 points of version 27's arm C. Card 095.1's box test must stay at
   100% for every colour. This would show that the two places are a
   special case of the wider rule.

**Reported:**
- which pairs arm A's attention weights for each Crafter action, at each
  position;
- accuracy against the number of successes in training (10, 30, 100,
  300; P19);
- the furnace as a second condition for the iron tools;
- cost per query and store size against version 27's (P17).

**Prediction.**
- For make, attention falls on the carried wood and a table near the
  agent. Do and place match the two-place query.
- **Main risk:** criterion 2 in breaking world 1. With raw positions
  only, the pair function may learn "lamp at x, box at x + 1" per
  position, which would not carry to the right half. If it fails there,
  the honest finding is that raw positions need a learned relative form
  (a later card), not a declared one.
- MiniGrid is unchanged, since card 095.1 found the same roles.

**Decision rules.**
- **Keep** if 1–3 hold: the wider query replaces the two-place one, and
  a later card puts it in the agent and the planner.
- **Revise** if only breaking world 1 fails (the relation does not carry
  to new positions).
- **Stop** if the upper bound itself misses criterion 1, or if attention
  never weights a token other than the one that changes.

**Budget.**
- Making the curriculum's 128,000 steps takes about 5 minutes, and
  breaking world 1 about 2 minutes.
- Arm A's training and the vote take about 20 minutes, timed first.
- MiniGrid's check takes about 10 minutes.

## 7. Result

First run: `runs/102.1/chain.sh`, 2026-10-11. Numbers are in
[results.json](results.json); code is in `tools/card102.1/`. The test
sets:
- **Crafter:** held-out worlds 8–9, 16,147 awake steps that are not
  moves.
- **World 1:** 99,000 test steps, with lamps at columns never seen in
  training.
- **MiniGrid:** card 095.1's held-out tries.

| Held-out Crafter, per try | Every token | Two-place query | Upper bound | Nothing changes |
|---|---|---|---|---|
| Make: log-likelihood | **−0.336** | −0.542 | −0.333 | −0.737 |
| Make: whole scored next state right | **99.3%** | 96.7% | 98.9% | 96.4% |
| Tries that made a tool (236): right | **96.2%** | 39.0% | 95.8% | 0% |
| Do: log-likelihood | −0.741 | **−0.595** | −0.515 | −1.001 |
| Place: log-likelihood | −0.415 | −0.415 | −0.321 | −3.910 |

**Criterion 1 fails, on "do".**
- **Make:** +0.206 nats per try against the two-place query (95%
  interval 0.17 to 0.24, bootstrap over 400 episodes). Holds.
- **Place:** −0.001 nats (−0.03 to 0.03). Holds.
- **Do:** −0.146 nats (−0.22 to −0.08). Fails. Crafter's view does not
  turn with the agent, so the tile "do" acts on depends on the heading
  only the agent's sprite shows. The two-place query is given that tile;
  every token must find it (stone mined: 31 of 31 against 7 of 31).

**Criterion 2 fails narrowly.**
- **Crafter, the table moved to each of the 24 places around the
  agent:** 93.9% of 196 tests pass (bar 95%; upper bound 95.4%;
  two-place query 0%). The mean P(made) is 0.95 at the near places and
  0.001 at the far ones.
- **World 1:** the tokens run catches 100% of lamp changes and is right
  on 95.7% of non-changes, a pass. But only two of four runs pass (two
  seeds each of tokens and upper bound). Every run is weakest at the
  column farthest from training: one catches 59% there, against 100% at
  the training columns. The relation is learned position by position.

**Criterion 3 fails narrowly.** Exact next state on MiniGrid against
version 27's arm C: tier 1 100% (100%), tier 2 99.97% (100%), tier 3
99.70% (99.83%); bar 0.01 points. The box test is 100% for every colour.

**Reported:**
- **Attention:** in layer 2, the made tool's slot puts 47% of its
  attention on the table (upper bound 94%).
- **The vote:** it adds little to whether a token changes (narrow
  kernel, α = 1), but supplies the results (place results 98% right,
  against 49% from arm A alone).
- **Iron tools:** the upper bound makes 0 of 5 (23 in training). The
  vote compares by arm A's context, so a condition arm A has not learned
  is invisible to recall.
- **Cost:** 0.11 ms per step's query for all 81 tokens; 118,554 stored
  context centroids; 10 minutes of training.
- **Scoring correction:** results are labelled by code and place, since a
  carried slot draws the count and the slot names the item. The first
  tokens result (8% of tools made right) was this mislabelling. The
  correction was applied to every form.

## 8. Revisions

The first run's failures point to one cause: relations from raw
positions are learned position by position. Three revisions tried to
make them carry. Each changed one component, was judged by the same
criteria, and was approved by the user. Revisions 2 and 3 came after a
literature review, whose papers are rows in [LITERATURE](../../LITERATURE.md).

| | What changed | Crafter: tools made right / role test | World 1: lamp changes caught | Lamp's weight on walls and empty rows | Why it failed |
|---|---|---|---|---|---|
| First run | Softmax attention, raw positions | **96% / 94%** | 100% and 63% (two seeds) | | Relations learned per position; carry in 2 of 4 runs |
| 1 | Positions only as differences between tokens | 6% / 0% | 67% and 96% | 42% | "A table near the agent" became two steps, not learned. Fixed tokens give position back |
| 2 | Independent gates per pair, closed by default, pruned to the fewest within 5% of the loss; offsets from the action's locus | 0% / 0% | 42% | **0%** | Gates collapsed when pruned and could not reopen. Rare relations fit inside the 5% margin |
| 3 | Gates never pushed shut | 9% / 0% | 72% | 56% | Gates stayed open (23,000 of 26,000) and leaked again |

World 1's revisions 2 and 3 are seed 1021 only, since their runs were
stopped once the outcome was clear. Code: `tools/card102.1/arms.py`
with `--rel` (1), `--gates --locus` (2) and `--soft --locus` (3). Runs:
`runs/102.1/chain_rel.sh`, `chain_r2.sh` and `chain_r3.sh`.

**What was learned:**
- **Softmax must spend its weight.** A lamp with nothing to read spends
  it on walls and empty rows, which give its position back. Gates that
  can close removed this, but they need a cost that keeps rare relations
  (P19) and a schedule that does not collapse them.
- **Removing absolute position** turns relations to the agent into two
  steps. Measuring positions from where the action happens keeps them
  one step (deictic references, Pasula et al. 2007). That needs the
  action's locus, which card 102.5 finds.
- **The user's reading:** this is the repo's generalisation problem. In
  one fixed world a lookup table and a general rule fit equally well
  (LESSONS), and the training format can make the general rule the only
  solution.

## 9. Decision

**Keep the first run's form** (the user, 2026-10-11). The criteria
failed (verdict fail), but this is the best form found.
- **What is kept:** softmax attention over every token, raw positions,
  and the vote over its context. This is arm C for cards 102.2–102.6.
- **Unchanged for now:** version 27's agent and ARCHITECTURE. MiniGrid's
  tiers and the breaking worlds run when the form enters an agent
  (CHARTER rule 10).
- **Limits carried forward:**
  - relations do not reliably carry to new positions;
  - "do" lags the two-place query;
  - MiniGrid is 0.13 points below version 27 on tier 3.

  Carrying is to be taken up through the training format, or once card
  102.5 gives the action's locus.

## Appendix A. Arm C on every token

- **Instances.** For every stored step and every token i in the state:
  the action, i's what and raw where, arm A's context for i, and the
  outcome (did i change, and its result as a relation).
- **Arm A.** Each token's input is its what, its raw where (as numbers)
  and the action. The pair function is a small network on the
  concatenation of both tokens' inputs, giving an attention logit and a
  message. Per token, the softmax over j weights the messages, and the
  heads give P(change) and card 095.1's ways. It is trained on the
  training worlds' stored steps, as card 095.1 trained it on MiniGrid's.
- **The vote.** For a query token, instances of the same action count by:
  - the same code first;
  - then exp(−‖context_q − context_k‖ / h), with h chosen by
    leave-one-out likelihood among a few values.

  Own tries come first, as in version 27.
- **Cost.** Pairs are all tokens against all, about 80 × 80 per state.
  Unchanged instances equal in code and context are grouped (card
  095.2).

## Appendix B. The crafting curriculum

Declared data policy (C3). It sets only where episodes start. The agent
still receives pixels and its own actions, and the evaluator's state is
used only for scoring.

- Each episode starts by day, so card 098.3's tokens hold.
- Wood, stone, coal, iron and sapling are each drawn independently,
  0–4, so that no material's count goes with another's.
- With probability 0.8 a table, and independently 0.5 a furnace, is put
  on a free grass, sand or path place 1–3 steps from the agent (steps
  counted as the 3 × 3 square counts them), uniformly over the places.
- 64 uniformly random actions follow, with no doing-nothing copies.
- 2,000 episodes on card 098.5's world seeds: worlds 0–7 train, worlds
  8–9 test.

## Appendix C. Breaking worlds

A small grid engine, `tools/card102/worlds.py`, built by this card for
the whole series (CHARTER rule 10).
- **What it gives:** pixels and actions only. Each cell is drawn as a
  7 × 7 tile of one colour with a simple shape. The evaluator's state is
  kept for scoring.
- **World 1, "near, anywhere":** a room with an agent (move in four
  directions; walking into a box pushes it), 2–3 boxes and 1–2 lamps. A
  lamp lights while a box stands in one of the four cells beside it.
  - **As drafted:** a 12 × 12 room, with lamps in columns 1–5 in training
    worlds and 7–10 in test worlds.
  - **As built:** a 9 × 7 room seen whole, the size of the view Crafter's
    adapter reads, with lamps in interior columns 1–3 in training and
    5–7 in test. Random play with no-op and the four moves.
- **Tokens:** card 098.3's encoder recipe (still pairs; there is no
  night), trained on the world's own random play, with Crafter's adapter
  (7 × 7 tiles). No adapter is written for the breaking worlds.
- Later cards add their worlds to the same engine.

## Appendix D. Building notes and the gate in detail

**Settled while building, before any result:**
- **Crafter's steps:** awake steps that are not moves. A move shifts the
  view, so "the token at this place changed" would describe the view, not
  the world; the agent's own motion is card 102.5's. Steps where the
  agent falls asleep or wakes are left out too, since sleep darkens the
  whole frame.
- **Labels:** arms are trained on the agent's own labels (a token's code
  changed one step later) and scored on the evaluator's.
- **The two-place query** is the same arms with each token's attention
  limited to itself, the token in front and the carried rows. It still
  predicts every token, so it is a stronger baseline than version 27,
  which predicts every other token unchanged. In world 1 the front is the
  cell the move goes into.
- **Raw wheres** enter the pair function directly and linearly, besides
  the token's state, so that a relation can be a difference of positions
  if the data asks for one. Nothing declares it.
- **Breaking world 1's criterion 2** is read on lamp tokens at pushes:
  P ≥ 0.5 in at least 95% of those that changed, and P < 0.5 in at least
  95% of those that did not. Most lamps are far from any push, so a
  single rate over all of them would be inflated.
- **Arm C's fit:** arm A is fitted on nine tenths of the training
  episodes. The kernel width and α are chosen by the own-label
  likelihood of the last tenth, and the store then holds every training
  step.

**The gate in detail** (arm A trained for 20,000 steps of 128 states, about 10 minutes):

- **Upper bound, held-out makes (worlds 8–9, 7,697 tries):**
  - **Carried tokens exactly right:** 99.84%. This passes the gate on
    this reading.
  - **Whole scored next state right:** 98.86%. The misses are mostly
    creatures and plants changing by themselves next to the agent, which
    crafting's determinism does not cover.
  - **Tries that made a tool (236):** carried tokens right in 96.6%.
    Wood and stone tools are right in 98.7% (231); the iron tools in
    0 of 5 (23 in training).

  The gate's "exactly" is read on the carried tokens, because its stated
  reason ("crafting is deterministic given the table, the furnace and
  the counts") holds only for them. That reading was chosen after the
  whole-state figure was seen, so both are given.
- **Upper bound, role test:** 95.4% of the 196 tests pass. The mean
  P(made) is 0.97 at the near places and 0.0004 at the far ones; near
  places reach 0.5 in 97.3% of cases and far places stay below it in
  100%. Criterion 2's bar (95%) is therefore just reachable.
- **Relations found:**
  - In layer 2, the made tool's slot puts 94% of its attention on the
    table's token.
  - In layer 1, it puts 75% on the carried slots (the materials).

  So the stop rule "attention never weights a token other than the one
  that changes" does not apply.
- **The vote adds little in Crafter.** The held-out tenth chose a narrow
  kernel (0.05 of the median context distance) and α = 30, so arm C's
  P(change) is nearly arm A's (make log-likelihood −0.333 against
  −0.348). Results are another matter. Given their own pseudo-count
  α_r, chosen the same way, the vote's result wins wherever it has one
  (α_r = 0: 93.9% of held-out changes right, against 74.6% from arm A
  alone). Place results are 98.2% right, against 49.2% from arm A alone.
- **The iron tools show recall's limit here.** The 11 stored iron
  successes do not count for a new iron try, because arm A's context
  never learned what separates them from failures. The vote compares by
  that context, so a condition arm A has not learned is invisible to it
  (P19).
- **Trivial baselines:**
  - "nothing changes" is scored with the main results;
  - random play with no curriculum was not run, since it made 8 tools in
    238,564 steps (section 4), too few to learn from.
- **Changed while building:**
  - **Results as relations.** Card 095.1's copy way is kept: a result
    equal to the token arm A's pointer picks is stored as "copy from it".
  - **Results need their own α.** Weighing the vote's result against
    α·P_A, with the α fitted for change, let arm A's poorer results win
    (place 55% right). Results now get their own pseudo-count α_r, chosen
    on the held-out tenth.

  Both changes were made before any tokens or two-place run.
