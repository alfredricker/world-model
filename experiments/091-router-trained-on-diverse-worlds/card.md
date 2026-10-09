---
id: "091"
title: a router to stored tries, trained on diverse worlds, as recall's prior
rung: 1
serves: [P10, P17, P6, P19, C2]
status: done
verdict: pass
arch_version: 18
date: 2026-10-08
---

# 091: a router trained on diverse worlds

Sketched at the user's request (2026-10-08), after rereading cards
076–083 and the user's note in STATUS (everything as tokens, a network
between state and recall, smart forgetting, System 1, position
relations, linear cost). The first of three: 091 the router; then
forgetting by what the router never uses; then the router's System 1
(actions read directly), after card 092 (view effects in recall). Each is its own card. The user (2026-10-08): no
reasoning network for now (a vote over the retrieved tries); pruning is
the forgetting card's.

Approved by the user (2026-10-08: "let's start 091"); tier 3 is run only
if tier 2 shows a clear gain.

## 1. Question

Cards 076–083 trained routers and reasoners on one world's few examples
(about 400 orders, mostly tier 2's; about 36 key and door combinations)
and judged them on transfer to another world or a left-out colour: the
regime where networks memorise (Chan et al.; LESSONS). Card 087's
network, trained on varied data, beat recall's neighbours on every
property. If a router that maps a situation (its tokens, what and where)
to an embedding, and predicts an action's outcome from the stored tries
nearest in that embedding, is trained on many worlds with every stored
try as a target, does it predict a world it never trained on at least
as well as recall fitted on that world, and does the agent with it as
recall's prior do as well on the tiers, at a cost that grows linearly
with memory? P10, P17, P6, P19, C2.

## 2. What changes

One component: recall's prior for a query with no identical stored try
(version 18: the neighbour vote over admitted conditions, card 049).
Own tries first is unchanged (card 050: one failed try overrules).

```
version 18:  P = (N_own + β P_vote) / (|N_own| + β)        P_vote: weights per number, fitted per world
this card:   P = (N_own + β P_route) / (|N_own| + β)
             e = f(situation)                               tokens (what: 32 encoder numbers; where: offset,
                                                            the hand its own token) → attention → embedding
             P_route = Σ_{k ∈ top K by ‖e_q − e_k‖} w_k · outcome_k,  w ∝ exp(−‖e_q − e_k‖)
```

- **The router f** reads the situation as tokens (card 044's what and
  where; positions kept, which removes card 044's declared exception for
  this reading) and the action as an input, one network for every
  action. A distance in e, not a dot product (LESSONS: a learned dot
  product fitted three colours and carried to none).
- **Training data: many worlds.** Card 052's generator (1–4 rooms of 5–10
  tiles; keys, balls, boxes, closed, locked and switch doors, switches,
  lava, decorative tiles; nine hues, half the episodes with fresh hues as
  card 090's stream), plus the stored tries of tiers 1 and 2 and the
  decoy world. Tier 3 is never trained on: it is the unseen world.
- **Training target: every stored try** (millions, not dozens): hide a
  try, retrieve from the rest, predict its outcome. Single tries and
  whole (front, held) combinations hidden (card 082.1); the loss weighted
  per combination (card 071), so rare openings are not drowned.
- **Network or vote, by held-out prediction** (CHARTER): per action, the
  router replaces the vote as the prior only where it predicts held-out
  tries better.
- **Cost:** the router is trained once, offline; memory's embeddings are
  computed once per memory (linear) and retrieval is a nearest-neighbour
  search, so building tier 3's memory needs no per-world weight fit (card
  085.3: about 85 minutes).
- **Recorded for the forgetting card** (report only here): how much retrieval weight
  each stored try receives over held-out queries.

## 3. Dependencies

Card 052's generator; card 044's tokens; card 050 (own tries first);
cards 076 and 082 (router code, combinations hidden, weighted loss);
card 087 (varied data). Pritzel et al. 2017 (`1703_01988`), Goyal et al.
2022 (`2202_08417`), Wu et al. 2022 (`2203_08913`), Chan et al. 2022
(`2205_05055`).

## 4. Data check

Per world and action: tries, outcome classes, how often each rare
outcome (an opening, a pick-up from a box) occurs; the generator's hues
and kinds per episode.

## 5. Feasibility gate

- **Upper bound:** on a world it trained on, the router predicts held-out
  tries at least as well as version 18's recall fitted on that world.
- **It reads memory:** with the retrieved tries' outcomes shuffled, its
  predictions get worse (Kossen et al.'s control); otherwise it has
  memorised in its weights.
- **Trivial baseline:** each action's outcome frequencies.

## 6. Success criteria and prediction

1. **An unseen world:** on tier 3's stored tries, held-out log-likelihood
   per try not worse than version 18's recall fitted on tier 3 itself.
2. **CHARTER's tiers** with the router as prior: tier 1 ≥ 99%; tier 2
   not worse than the best version; tier 3 against version 18, both on
   the same 30 seeds.
3. **Cost:** tier 3's setup within 20 minutes (memory's embeddings and
   index), time per step not above version 18's.

**Prediction.** The router reaches recall's in-world accuracy on pick
up, drop and moves; toggle is the risk (rare openings). On tier 3 it
does better than recall where tier 3 differs from the other worlds
(boxes holding keys, many doors), since recall's weights must be fitted
on whatever tier 3's memory happens to hold.

**Decision rules.** Keep if the gate and 1–3 hold. Revise if 1 holds and
2 or 3 fails. Stop if the gate fails (it does not read memory, or cannot
fit a world it trained on).

**Budget.** From the sizing run below: token data for four worlds 22 s;
6,000 training steps 2.5 minutes; evaluation a few minutes. The agent's
integration and the tiers (about an hour each for tiers 1–2) dominate.

## Sizing (before the gate)

`tools/card091/data.py`, `router.py`; `runs/091/`. Trained on tiers 1, 2
and the decoy world's memories (code-based worlds; card 052's generator
not yet added); tier 3 never trained on. Each try's tokens: what the
agent sees (7 × 7, occlusion), the held tile, and believed tiles out of
view (18.5 tokens on average, at most 49).

- **A normalised vote is overconfident** (first try, 1,000 steps): with
  a query's (front, held) combination hidden, pick up in tier 1 scored
  −0.56 per try against −0.23 for the action's outcome frequencies, and
  −9.2 on the rarer outcomes. A softmax over candidates commits to the
  nearest, however far. **Fixed** with recall's own form: kernel-weighted
  counts plus the action's frequencies with a learned weight α.
- **Second try** (6,000 steps; α 0.08, τ 0.90), mean log-likelihood per
  try, whole combinations hidden (rows hidden: router and the
  combination's own counts both near 0):

| | Router | Action's frequencies | Rarer outcomes: router, own combination |
|---|---|---|---|
| Tier 3 (unseen), forward / pick up / drop / toggle | −0.0001 / −0.0001 / −0.0009 / **−0.106** | −0.69 / −0.35 / −0.36 / −0.17 | toggle −1.51, −3.17 |
| Tier 2, the same | −0.0000 / −0.0002 / −0.0001 / −0.003 | −0.69 / −0.32 / −0.32 / −0.011 | toggle −1.07, −6.66 |
| Tier 1, the same | −0.0005 / −0.22 / −0.18 / −0.0003 | −0.70 / −0.23 / −0.22 / −0.034 | pick up −3.70, −2.82 |

  In a world it never trained on, the router predicts combinations its
  memory has never seen from that world's other tries. Toggle is the
  weak point (tier 3's openings). Tier 1 holds one key colour, so a
  hidden key combination has nothing like it in its memory.
## Gate

`tools/card091/gate.py`, `shuffle.py`; `runs/091/gate_*.json`,
`shuffle_*.json`. Version 18's prior is its own neighbour vote, read from
the agent's World (card 051's groups for pick up, drop and toggle; card
038's vote for forward), with the same tries hidden as for the router.
Version 18's four categories are the router's four outcomes (bit 0 the
front tile changed, bit 1 the hand changed). Mean log-likelihood per
try; router `runs/091/router_a.pt` (trained on tiers 1, 2 and the decoy
world, never on tier 3).

| Hidden from memory | World | Forward | Pick up | Drop | Toggle |
|---|---|---|---|---|---|
| (front, held) combination: version 18 / router / frequencies | tier 1 | −0.69 / −0.0003 / −0.70 | −0.04 / −0.30 / −0.23 | −1.16 / −0.19 / −0.22 | −0.91 / −0.0008 / −0.03 |
| | tier 2 | −0.83 / −0.0000 / −0.69 | −1.38 / −0.0004 / −0.32 | −1.33 / −0.0001 / −0.32 | −0.85 / −0.003 / −0.011 |
| | **tier 3 (unseen)** | −0.86 / −0.0002 / −0.69 | −0.22 / −0.0008 / −0.35 | −0.99 / −0.002 / −0.36 | −1.13 / **−0.12** / −0.17 |
| every try with the same front tile | tier 1 | −0.69 / −0.05 / −0.70 | −0.69 / −0.30 / −0.23 | −1.16 / −0.19 / −0.22 | −0.91 / −0.001 / −0.03 |
| | tier 2 | −0.83 / −0.04 / −0.69 | −1.67 / −0.0005 / −0.32 | **−0.14 / −0.32** / −0.32 | −2.37 / −0.002 / −0.011 |
| | tier 3 | −0.86 / −0.02 / −0.69 | −1.97 / −0.0008 / −0.35 | −0.99 / **−0.67** / −0.36 | −2.82 / −0.13 / −0.17 |

- **Upper bound: passes.** With single tries hidden the router is at
  −0.001 or better on every action of every world, as are the
  combination's own counts (version 18's own-tries level).
- **It reads memory: passes.** With the candidates' outcomes shuffled,
  tier 3's router falls to −0.71 / −0.70 / −0.70 / −0.34 (forward, pick
  up, drop, toggle) from −0.0002 / −0.0008 / −0.002 / −0.12; tier 2 the
  same.
- **Criterion 1 (tier 3, unseen): met** with combinations hidden, on
  every action. Version 18's vote is worse than the action's frequencies
  on most cells: its weights are fitted to tell stored situations apart,
  not to predict a hidden one.
- **Where the router loses:** drop with every same-front try hidden
  (hiding the empty floor removes nearly every successful drop, and the
  nearest tries left are drops against walls); and tier 1's pick up,
  where one key colour means a hidden key leaves nothing like it.
  Per action, by held-out prediction (section 2), the router is the prior
  for every action; drop is the one to watch.

## Integration: the router in recall's key form

The agent's queries are recall's keys, not views: (front, held, the
believed view's tiles, at most 6), and forward's key is the front tile
alone. The router was retrained on that form (`tools/card091/keys.py`
dumps each world's stored keys with their outcome counts;
`krouter.py`; tokens: the agent's own 32-number vectors with a role,
front, hand or view; τ 0.85, α 0.31; 4,000 steps, 18 s). Trained on the
keys of tiers 1, 2 and the decoy world (about 2,900 per interaction,
66 forward); each query's candidates are its own world's keys.

Tier 3 (unseen, about 54,000 keys per interaction), mean log-likelihood
per try, combination hidden, version 18's prior (gate above) against the
router: forward −0.86 / −0.26; pick up −0.22 / −0.001; drop −0.99 /
−0.05; toggle −1.13 / −0.03. Every same-front try hidden: drop −0.99 /
−0.61 (frequencies −0.33), the one cell where the router is worse than
the frequencies.

In the agent (`tools/card091/routed.py`, `WM_ROUTER=<router>`): pick up,
drop and toggle through card 051's `IndexKind.predict`, forward through
the forward kind's `cat_of`, each P = (N_own + β P_route) / (|N_own| +
β) with version 18's β (α for forward). Stored keys are embedded once at
setup; an episode's new keys and the queries as they come, on the CPU.
Smoke: 8 tier 1 episodes, all solved, about 120 router queries each.

## Result

`runs/091/run.sh`; version 18's runs are card 074.2's, on the same seeds.

| Tier | Router | Version 18 | Card 087 | Judged |
|---|---|---|---|---|
| 1 DoorKey-8x8 (200) | **200/200**, 24.4 steps | 200/200, 24.4 | – | passes (≥ 99%) |
| 2 BlockedUnlockPickup (100) | **79/100**, 72.5 steps when solved | 59/100, 95.2 | 64/100, 101.0 | **better** |
| 3 ObstructedMaze-Full (30) | **0/30**; 28 turn in place for most of the 3,600 steps | 5/30 (an hour per episode) | – | **worse** |

- **Tier 2 on the same seeds:** 21 episodes solved only with the router,
  1 only by version 18 (sign test on the 22 that differ, p < 0.0001).
- **Random actions rose** from 21% to 36% of steps on tier 2, and 7
  episodes ran out of time (version 18: 4). The router answers more
  queries with an outcome (P ≥ 0.5) where version 18 answered "unknown";
  where that answer is wrong, the planner finds no plan and acts at
  random. To read in the failures, not yet traced.
- **Criterion 3 (cost)** is measured on tier 3.
- **Tier 2's 21 failures** (counters; traces of seeds 1002093, 1002081
  and the router-only win 1002006, `runs/091/trace_*.log`): 13 never
  open the locked door, 8 open it and still fail. No plan exists until
  the box is in view: every step before that is the fallback
  (exploration, card 072's untested guesses, random actions), and the
  door opens only when the fallback happens to toggle it with the key
  in hand (1002006 at step 17, with and without the router; 1002081 at
  step 169). The guesses tried are near-zero ones (pick up a wall,
  toggle the door holding the ball), and one guess is continued for 100
  or more blocked forwards into a key or ball. After the door opens, the
  router's gain is in the plan to the box (1002006: version 18 planned
  326 steps and failed; the router solved it in 130).

- **Tier 3's first attempt** (the user, 2026-10-08) was stopped by the
  memory cap 2.5 minutes in (44 GB): `routed.py` built a (queries ×
  keys × 32) array of differences against tier 3's 54,000 keys in each
  of 20 workers. Now `torch.cdist` (queries × keys only); a 2-episode
  smoke peaked at 5 GB. One smoke episode (seed 1003001) spent 3,600
  steps exploring with 6 wrong predictions, unlike version 18's run of
  the same seed (26 explore steps): read in the full run.

## Decision

**Keep** (the user, 2026-10-08, before tier 3's score: "I want to mark
091 a keep regardless"). The gate passed (it reads memory; tier 3's
held-out tries, a world it never trained on, predicted far better than
version 18's vote on every action); tier 1 100%; tier 2 79% against
59% (21 seeds won, 1 lost). Version 19.

**Tier 3 (overnight, after the decision): 0/30, worse than version
18's 5/30.** In 28 episodes the agent spends 3,000 or more of its 3,600
steps exploring (95% of all steps; version 18: about 1%), and a trace of
seed 1003001 shows it turning in place before a blue door for over
3,000 steps: the step chosen points it away and the next step's plan
points it back. The same 2-seed smoke with version 18's flags explores
25 and 26 steps; with card 087's forward prior added to the router it
still loops, so the router's predictions for pick up, drop or toggle
cause it. Seconds per step 0.12 (version 18: 1.68), since turning costs
little. Under CHARTER's keep rule (no tier worse) this would be revise;
the user's keep stands, and the loop is traced in card 092's night.
