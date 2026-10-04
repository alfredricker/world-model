---
id: "049"
title: recall through the conditions that matter
rung: 0
serves: [P4, P10, P12, P17, P19]
status: done
verdict: fail
arch_version: 8
date: 2026-10-03
---

# 049: recall through the conditions that matter

Drafted 2026-10-03 at the user's request, after tracing card 048's
failures. The staged encoder moved to card 050, which will use this
card's recall if it passes. Approved by the user on 2026-10-03.

## 1. Question

Version 8's recall compares the whole view of a stored try with the whole
present view. Can it instead read only the conditions that changed an
action's outcome in memory, so that tokens irrelevant to the action
cannot veto a memory, while losing nothing in the familiar worlds? Before
rung 1. Serves:
- P4, distinctions kept only where they change outcomes;
- P10, retrieval that still works when the present view holds new
  combinations;
- P12, conditions read straight from recall: the admitted condition the
  present lacks is the need;
- P17, query cost set by the number of conditions, not the size of the
  view (Minecraft's inventory, health, hunger and time are candidates
  like any token);
- P19, a condition admitted from few contrasting tries.

Version 8 cannot get there. In card 048, seed 401, a second door in view
added about 23 to the view distance of every stored toggle of the red
door, and "the door opens" fell from 1.00 to 0.38. In seed 399, a key
missing from the floor did the same. That key lay in every stored view,
so recall had never seen a toggle without it. The similar-tile level that
should carry experience over is off: β = 0.000–0.003 in all five seeds.

## 2. What changes

One component: recall's key and its match, for pick up, toggle and drop.
The planner, tokens, walking, moves' recall and memory are unchanged.

```
version 8:  weight of stored try i = [front and hand codes equal] x exp(-λ1 chamfer(view_i, view_now))
            mixed with β x exp(-λ2 (L1(front) + L1(hand) + chamfer(view)))
card 049:   weight of stored try i = exp(-Σ over admitted conditions c of λ_c |f_c(try i) - f_c(now)|)
```

- **Candidate conditions,** all read from the one encoder's codes and
  vectors, per action:
  - the tile in front and the held tile, one candidate per codebook part
    (4 each), compared by distance within that part;
  - a relation per part: the front and held tiles' distance within that
    part (the relational bottleneck; never a query–key product, LESSONS);
  - presence in view: "a token with code k in part p lies within 6
    tiles", one candidate per code used in memory.
- **Admission.** Starting from the front tile alone, the candidate that
  most raises recall's leave-one-out likelihood of the stored outcomes is
  added, while the gain exceeds a cost of log(number of candidates) per
  condition. Card 010's description cost and Griffiths and Tenenbaum's
  causal support make the same choice. A token present in every stored
  try adds no likelihood and stays out, and so does one redundant with an
  admitted condition (blocking). λ is then refitted on the admitted
  conditions, as recall's weights are fitted now.
- **Match.** Only admitted conditions enter the distance. A token in view
  that no condition reads costs nothing. A condition whose presence lowers
  the outcome (a preventer) counts like any other; its sign shows in the
  outcomes.
- **Rule 8.** Everything is recall over kept tries. Nothing is tallied or
  merged into kinds, and the codes stay names.
- **Declared exceptions:** card 043's spliced check for conjunctions in
  the planner; card 045's route procedure.
- **Not in this card:** "no match" read as unknown, prompting a try (a
  planner change); the encoder (card 050).

## 3. Dependencies

- Version 8 (kept with card 047): planner, tokens, walking, encoders and
  memory of seeds 400–404.
- Recall's leave-one-out fit (cards 037–042), here also used to choose
  which candidates enter.
- Admission by evidence with a description cost: card 010, which passed on
  exact variables.
- Literature (LITERATURE.md, "Current focus"): Cheng 1997; Griffiths and
  Tenenbaum 2005; Pasula et al. 2007 (`1110_2211`); Kruschke 1992 and
  2001; Webb et al. (relational bottleneck).

## 4. Data check

- **Contrast exists where it must.** In the switch world, toggles of the
  door are stored with the switch both on and off. In the key world, they
  are stored with and without the matching key in hand. Counts are taken
  from memory before the run.
- **No contrast where it must not.** The key world's second key lies in
  view in every stored toggle, so it should not be admitted.

## 5. Feasibility gate

- **Upper bound:** recall with the view cut by hand to the tokens the
  evaluator knows matter (the switch in the switch world, nothing else).
  It must solve card 048's seed-401 layouts and the cluttered rooms
  below.
- **Trivial baselines:**
  - version 8: seed 401 one door 46%;
  - recall with no view at all, which must fail the switch world,
    showing that view conditions are needed.
- **Shakedown:** spare seed 399, 30 layouts per test, timed on its
  slowest test.

Result of the gate, before the main run (2026-10-03; code
`tools/card049/conditions.py` and `worlds.py`; outputs `runs/049_gate_*`
and `runs/049_shake_*`):

- **Upper bound passes.** Version 8's recall with the view cut to the
  floor and the switch: card 048's seed-401 one-door layouts 98% (100
  layouts, 1.08 times the shortest route; version 8: 46%); the cluttered
  world 100% (seed 399, 30 layouts).
- **Baselines.** No view at all: switch world 3% (30 layouts), so view
  conditions are needed. Version 8 in the cluttered world: 90% (seed 399;
  3 of 30 layouts act at random for 139–190 steps).
- **Two corrections before the main run:**
  - *The cluttered world's objects.* The tile set has no balls or boxes
    in other colours, so the extra objects are keys of colours the door
    does not take (the third colour, and yellow and purple, never seen in
    the key world), switches and vases. A layout counts only if the
    shortest route moves nothing but the door's key; the first version
    allowed moving the second key, and one layout walled the door's key
    off behind it.
  - *View conditions name a whole token.* The first shakedown read
    presence per code piece and failed the switch world (27%) and the
    both world (0%). Pieces are shared across tiles: the switch on and
    off differ only in part 1, and part 1's "on" code is also the wall's,
    so it was present in every view. A view candidate is now "a token
    with this code tuple is in view" (codes for identity).
- **Shakedown, seed 399, after the corrections** (30 layouts per test):
  - familiar worlds: key, switch, either and both all 100%, steps 0.93–
    0.97 times card 029's, effects exact: criterion 1's shape passes;
  - chained rooms, one door: 100% at 1.03 times the shortest route;
  - cluttered world: 100% (the same two layouts take random steps under
    every recall, the upper bound included, so not recall's doing);
  - conditions admitted for toggle at a door: key world, the held tile's
    part 1 and the front–held relation in part 0; switch world, "the
    switch off is in view" (a preventer); either and both, the relation
    and the switch off. The second key and the vase are never admitted;
    at most 3 conditions beyond the front tile;
  - two doors: 0 of 30, but by plan (3% random steps; version 8 had
    no plan). Traced on layout 3: the agent walks to key A, picks it up,
    drops it, picks it up again, for all 200 steps. The drop chain of
    appendix A now forms (holding key A, "hold key B" needs an empty
    hand, so it drops key A), but nothing keeps "hold key A" until door A
    is open, so the planner undoes one subgoal for the other. That is
    subgoal ordering in the planner, not recall. Every step is also
    flagged as mismatching the real view (the same distance, 23.2, each
    time); not traced.
- **Time.** Setup about 3 minutes per seed (admission itself 1–5 s per
  action); 0.8–3.4 s per layout with six runs sharing the machine.

## 6. Success criteria and prediction

Seeds 400–404; each criterion in 5 of 5 seeds unless stated.

1. **Nothing lost.** Card 045's criteria: all four familiar worlds ≥ 99%,
   steps within 5% of card 029's; unseen rooms ≥ 99% at ≤ 1.15 times the
   shortest route.
2. **Irrelevant tokens cannot veto.**
   - Card 048's chained rooms, one door: ≥ 95% at ≤ 1.15 times the
     shortest route (version 8: 46% in seed 401, 97–98% elsewhere).
   - Cluttered key world: each layout adds 2–4 balls and boxes in random
     colours that no action or goal involves, placed so that no route
     needs one moved; every view is a new combination. Memory comes from the plain key world. Goal reached in
     ≥ 95%, against version 8 on the same layouts.
3. **The right conditions are admitted,** in at least 4 of 5 seeds:
   - toggle at a door reads the held tile or the front–held relation in
     the key world, and the switch in the switch world;
   - the key world's second key and the vase are never admitted;
   - each action admits at most 8 conditions.

Reported:
- card 048's two doors;
- card 047's new-colour switch test (version 8: mean 39.6%);
- whether the relation or per-colour parts were admitted for the key;
- the P19 curve: admitted conditions from 3, 10 and 30 contrasting tries;
- time per query and per layout against version 8;
- an ablation arm with only the view through conditions (front and hand
  as in version 8).

**Prediction.**
- Criteria 1–3 pass.
- Two doors: uncertain. With the floor key no longer admitted, the
  spliced check passes, and the chain needs the planner to reach "drop
  key A" through the hand need (appendix A). If it does, two doors rise
  well above card 048's 2 of 150; if not, they stay below 50%.
- New colours: little change. Version 8's encoder puts a new colour far
  away in every part, so no part matches until card 050.
- With three colours, per-colour parts and the relation fit equally, and
  the cost decides (LESSONS).

**Budget.** Admission is a greedy search over about 50 candidates per
action, each step a leave-one-out fit (about 25 s per world now). That is
roughly 10–20 minutes per seed, with the five seeds in parallel. The
shakedown gives the time; if the main run exceeds 30 minutes, it goes to
the user as commands.

## 7. Result

Main run 2026-10-03 by the user (`runs/049_main.sh`; seeds 400–404, the
five in parallel). Numbers: [results.json](results.json).

| Seed | Familiar worlds (steps vs card 029) | Unseen rooms | Chained, one door | Cluttered (version 8) | Toggle at a door reads (key world; switch world) | New-colour switch door, yellow / purple | Two doors |
|---|---|---|---|---|---|---|---|
| 400 | 100% (0.99–1.00) | 100%, ≤ 1.08 | 98%, 1.08 | 100% (100%) | relation part 2; switch on | 95% / 95% | 0 of 30 |
| 401 | 100% (0.99–1.00) | 100%, ≤ 1.08 | 98%, 1.08 | 100% (92%) | held part 3, relation part 0; switch on | 14% / 14% | 0 of 30 |
| 402 | 100% (0.99–1.00) | 100%, ≤ 1.08 | 98%, 1.08 | 100% (100%) | relation part 0; switch off | 95% / 92% | 0 of 30 |
| 403 | 100% (0.99–1.00) | 100%, ≤ 1.08 | 98%, 1.08 | **91%** (95%) | held part 0, relation part 0; switch on | 95% / 95% | 0 of 30 |
| 404 | 100% (0.99–1.00) | 100%, ≤ 1.08 | 98%, 1.08 | 100% (92%) | held part 2; switch off | 71% / 95% | 0 of 30 |

- **Criterion 1, nothing lost: pass, 5 of 5.** All four familiar worlds
  100% with effects exact, steps 0.99–1.00 times card 029's; the six
  unseen rooms 100% at 1.01–1.08 times the shortest route.
- **Criterion 2, irrelevant tokens cannot veto: fail, 4 of 5.**
  - Chained rooms, one door: 98% at 1.08 in every seed (version 8: 46%
    in seed 401, 97–98% elsewhere). The two failing layouts (78 and 88)
    fail in every seed and under the upper bound, so not recall.
  - Cluttered world: 100% in four seeds, 91% in seed 403 (version 8:
    92–100%, at least 95% in 3 of 5). Version 8's failures act at random
    (vetoes); seed 403's do not. Traced on its layout 2: with seed 403's
    conditions a purple key (never seen) reads as a key for toggle, so
    the agent picks it up and tries the red door. Then, still holding it,
    it tries to pick up the red key 180 times; every try fails and the
    prediction never changes. Pick up reads the front tile and one
    front–held relation, not the held tile itself, and recall gives the
    agent's own repeated failures no more weight than each of the
    thousands of stored successes they resemble. Version 8's same-tile
    level let a situation's own tries outweigh its neighbours (card 038's
    back-off); this card dropped it.
- **Criterion 3, the right conditions: pass, 5 of 5.** Toggle at a door
  reads the held tile or the front–held relation in the key world, and
  the switch (as "on present" or as "off present") in the switch world.
  The second key and the vase are never admitted. At most 4 conditions
  beyond the front tile.

Reported:
- **Two doors: 0 of 150,** by plan; the same oscillation as the
  shakedown (key A picked up and dropped in turn).
- **New-colour switch door: mean 76.1%** over the ten runs (card 047:
  39.6%), above 90% in 7 of 10; seed 401 14%. The prediction ("little
  change") was wrong.
- **First-sight key test:** right in 2 of 10 runs (card 047: 2 of 5
  seeds).
- **Which parts:** different in every seed (part 0, 2 or 3; held or
  relation), because version 8's encoder spreads colour and kind across
  all four parts.
- **Time:** familiar worlds 5.8–11.2 s per layout, against version 8's
  0.42–0.72 s in card 047's run; the new-colour switch tests 31–85 s.
  Cost grows with memory, since a query is compared with every stored
  key on every admitted condition, with no cache. The cluttered world
  0.7–3.0 s (version 8: 0.47–1.27 s).
- **Not run:** the ablation arm (view only) and the P19 curve.

## 8. Decision

**Revise** (the user, 2026-10-03; card 051). Recall through admitted
conditions does what the theory says: irrelevant tokens no longer veto
(card 048's seed 401 from 46% to 98%), the conditions are the right ones
in every seed, nothing familiar is lost, and new colours carry over far
better (76% against 40%). It fails criterion 2 in one seed for a reason
the theory names: trying must be able to correct what recall infers
(CHARTER, "Recall replaces counting"). The revision is one change to
recall: a query's own stored tries (the same key) count as its own
evidence, with the admitted-condition neighbours as the prior, as card
038's back-off did. Whether it also lifts seed 401's new-colour test
(14%) is untraced. The
cost per query needs a cache before scaling. Two doors are the planner's
subgoal ordering, a separate card.

## Appendix A: notes for later

Raised by the user, 2026-10-03.

**Splicing may stop mattering here, but it is still wrong.** When the
planner needs a different hand, it writes that hand into the present
state rather than imagining the actions that produce it (card 043's
declared exception). In card 048's two-door layouts this made key A
vanish from the imagined room, and version 8's recall rejected the room
because every stored toggle had a key on the floor. With this card's
recall the floor key is not a condition, so the spliced room should pass:
the admitted condition is the held key against the door (by relation or
by colour), and key B against door B meets it. The imagined state is
still one that cannot occur, though. A spliced room would also miss a
consequence that matters, for example key A dropped in front of door B
and blocking it (seen in card 048's random steps). If two doors still
fail, or a later world depends on where a dropped object lands, the fix
is card 048's first follow-up: imagine the drop and the pick up, so the
planner only asks recall about states its learned effects produce.

**Or the agent never learns to put one object down to pick up another.**
The pieces are in memory: in the key world, a pick up with a full hand
changes nothing, and a drop empties the hand. The planner's conditions
can turn this into a need: pick up key B works in stored situations with
an empty hand, so "change the hand" becomes a need, and drop achieves
it. Whether the chain actually forms (hold key B ← pick up key B ← empty
hand ← drop key A) was not traced in card 048, whose two accounts differ
(its section 5 says the empty-hand need was never made; its section 7
places the failure in recall's view). This card's two-door report
decides it. If two doors still fail, trace that chain first.

