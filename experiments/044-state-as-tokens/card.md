---
id: "044"
title: the state as tokens
rung: 0
serves: [C2, P12, P2, P6, P7]
status: done
verdict: pass
arch_version: 6
date: 2026-10-01
---

# 044: the state as tokens

## 1. Question

Version 6 holds what the agent believes as a grid: the tile at each of 182
fixed places in the frame of the episode's first view, plus a pose, one of
484 standing places and headings composed from the counted move maps.
Where a thing is lives only in an array index and the pose table. Recall
reads "ahead" and "held" as two fixed places, and what is in view as the
set of distinct appearances, positions dropped (card 039). Card 041 would
add where-tokens for walking alone: a second account of where things are
(C2).

Suppose instead the state is a set of tokens, every tile and the held
thing included. Each token has a what (the encoder's vector, with its
codes) and a where (its offset from the agent, or the hand). A move
transforms every where, and "ahead" and "held" become values of where that
recall and the conditions read. Does the planner keep card 043's results?
Before rung 1. Agreed in principle with the user, 2026-10-01 ("tokenizing
everything"). Serves C2 (the state, recall's roles, conditions and later
walking read the same tokens), P12 (facing a thing is a condition on its
where), P2 (a token keeps its where out of view; the grid holds only the
first view's frame), P6 (a move's effect is a transformation of where)
and P7 (things as the units, ready for segments in place of tiles).

## 2. What changes

One component: the agent's state, and how its readers address things.

```
version 6:  state = (tile at each of 182 first-frame places, pose ∈ 484 counted poses, ended)
            ahead = the place the pose puts in front; held = place 169; in view = the pose's places
            a move = the next pose, from the pose table
card 044:   state = ({token: what, where}, ended)
              what  = the tile's encoder vector (its codes read from it)
              where = (x, y), the offset from the agent (x right, y ahead), or "hand"
            ahead = the token at (0, 1); held = the token at hand; in view = tokens with |x|, |y| ≤ 6
            a move m = where → M_m where + b_m for every token but the hand's,
                       when recall predicts the move happens (forward: moved, blocked or ended)
```

- **Tokens.** Every tile is a token, floor and walls included, and so is
  the held thing. A token keeps its identity while it exists, so a
  condition can name it ("face this door").
- **Moves as transformations.** M_m and b_m are fitted by least squares to
  the place correspondences card 028 counts from pixel transitions. This
  is card 041's T_m, moved here because tokens need a way to move.
- **Seeing.** The real view's tiles replace the tokens inside the window
  (matched by where); tokens outside it keep their moved where. Placing
  the view stays as now: the placement nearest the real view (L1 over
  vectors), among those reachable by composing the move transformations.
  There are 676: card 028's 484 poses plus the first view's outer ring,
  where the agent never stands.
- **Pick up, toggle and drop** change the what of the token ahead and of
  the hand, as recall predicts, as now.
- **Declared exceptions** (CHARTER rule 8):
  - recall still compares what is in view as distinct appearances without
    positions. Comparing token sets with positions is the user's "optimize
    memory later" (2026-10-01): section 5 finds 545 times the work per
    comparison and about 1,400 times as many distinct views;
  - walking is not yet conditional. It keeps its hand-set closeness, now
    computed from where, and still checks moving closer, and the move
    ways, by stepping moves forward in imagination (now moving tokens).
    Card 041 learns how soon; card 045 removes the imagined checks.
- **Unchanged:** card 043's planner and conditions, card 042's recall, arm
  A's encoders (seeds 400–404), the data.
- **Card 041 after this:** only the learned how-soon remains; its
  where-tokens and move transformations are this card's (agreed with the
  user, 2026-10-01). Obstacles as conditions become card 045.

## 3. Dependencies

- Card 043's planner with card 042's recall: kept, 5 of 5 seeds.
- Card 028's move correspondences (the simulator's moves exactly).
- This card's gate (section 5): the transformations are exact and the
  pose table holds nothing that where does not.
- Methods: tokens as content plus position, as transformer patch tokens
  (Dosovitskiy et al. 2020, not in papi); a move as a transformation acting
  on the state (`1812_02230`).

## 4. Data check

Per world: about 13,000 held-out left turns, 13,000 right turns and 6,400
forward steps that changed the view; about 300,000 stored pick up, toggle
and drop tries, each view 168 tokens plus the hand.

## 5. Feasibility gate

Done before drafting (`tools/card044/token_check.py`,
`runs/044_token_check.json`, 14 seconds). The four worlds gave identical
results.

- **Moves are exact transformations.** Least squares on each move map's
  correspondences gives left and right as quarter turns about the agent
  and forward as a one-tile shift, with zero residual. The counted maps
  pair only places whose appearance varies (121 of 169 for turns, 110 for
  forward) and treat the rest as "entering" with a fixed appearance; the
  transformations cover those too.
- **The pose table holds nothing that where does not.** Over all 484
  poses, all 33,172 view places whose first-frame place the table knows
  agree with the agent's place and heading. Another 17,904 lie inside the
  first frame but get a fixed appearance; tokens carry the tile seen.
- **Stepping tokens.** On the held-out moves, the tokens moved by the
  transformation give the view after at every place that did not enter
  (168 per turn, 154 per forward step; the agent's own place and the one
  it leaves are drawn as now): 100%. The hand is unchanged in 100%.
- **Size.** 168 tokens per view plus the hand, against 7.2 distinct
  appearances (at most 8). Matching views token by token costs 545 times
  as much; distinct stored views rise from 130–144 to about 206,000.
  Hence recall's declared exception.
- **Upper bound:** the check; tokens carry exactly what grid and pose do.
  **Trivial baseline:** card 043 (100% in four worlds, 5 of 5 seeds).
- **Shakedown,** before the main run: spare seed 399, 30 layouts per
  world, against card 043's shakedown on the same layouts.

Result of the gate, after building (`tools/card044/tokens.py`, approved by
the user 2026-10-01):
- **Views at the start of 100 layouts per world, from every placement.**
  Where the agent stands on a tile it can walk onto, the only placements
  walking uses, tokens and the pose table give the same view at all 1.6
  million places per world. From placements outside the room, the pose
  table's fixed guess (wall) differs from the tile tokens keep at 192,832
  of 8.1 million places, which changes the set in view of 26,640 of
  48,400 views. Walking never uses those placements.
- **Shakedown** (`runs/044_shakedown_399.json`, 2 minutes): identical to
  card 043's in all four worlds and in the four new-colour tests. Goal,
  moves (482, 500, 436, 630), wrong predictions, refusals, conditions
  computed per move and the kinds of steps taken all match. Held-out
  effects are 1.0 everywhere.
- **Time.** First 15–50% slower per layout, from reading the token ahead
  through a general lookup on every imagined step. With a direct lookup:
  0.614 and 0.640 s per layout in the key and switch worlds, against card
  043's 0.593 and 0.616 (`runs/044_dev.json`).

## 6. Success criteria and prediction

Arm A, seeds 400–404, card 043's command with this card's state.

1. **Nothing lost.** Card 043's criteria 1 and 2 in at least 4 of 5
   seeds: held-out effects ≥ 0.999, goal in ≥ 99% of 500 layouts per
   world, mean steps within 5% of card 029's; the key world with yellow and
   with purple keys ≥ 99%.
2. **The same decisions.** In every seed and world, the total number of
   moves is within 1% of card 043's on the same seed and layouts. A
   change of representation should not change what the agent does.
3. **Reported, no bar:** seconds per layout against card 043's 0.26–0.92
   (P17); places where a token carried a seen tile that the pose table
   would have filled with a fixed appearance.

**Prediction.** Both criteria pass in 5 of 5 seeds. Any differences in
moves come from the 17,904 places above, where an imagined view may now
hold a remembered tile, and from tie-breaking order. Time per layout is
about the same.

**Budget.**
- Building: the token state, and its readers in the planner. Card 038's
  planner and card 029's and card 028's walking and conditions read the
  pose table in 43 places. This is most of the work.
- Shakedown: about 3 minutes.
- Main run: about 20 minutes (card 043's took 17), after approval.

## 7. Result

Main run, arm A, seeds 400–404 (`runs/044_armA.json`, 18 minutes; card
043's took 17). Compact numbers are in [results.json](results.json).

| Seed | key | switch | either | both | Criterion 1 | Key world, yellow / purple | Moves against card 043 |
|---|---|---|---|---|---|---|---|
| 400 | 100% | 100% | 100% | 100% | pass | 100% / 100% | identical |
| 401 | 100% | 100% | 100% | 100% | pass | 100% / 100% | identical |
| 402 | 100% | 100% | 100% | 100% | pass | 100% / 100% | identical |
| 403 | 100% | 100% | 100% | 100% | pass | 100% / 100% | identical |
| 404 | 100% | 100% | 100% | 100% | pass | 100% / 100% | identical |

- **Criterion 1: pass, 5 of 5 seeds.** Held-out effects were 1.0 for
  every action in every seed and world. Mean steps were 16.38, 17.27,
  15.42 and 21.77, as in card 043, and card 029's to within 0.3%.
- **Criterion 2: pass.** In the four familiar worlds and the key world
  with new colours, every count matches card 043's in every seed: goal,
  moves, wrong predictions, refusals, conditions computed per move and
  the kinds of steps taken.
- **Reported:**
  - the switch world with a new-colour door (no bar, as in card 043)
    differs in 3 of 10 runs, all where the agent acts at random in 90–99%
    of steps. In seed 402 with purple, moves were 12,860 against 12,984
    (0.95%), and the goal 60% against 59%. In seed 400 with purple and
    seed 403 with yellow, the moves were the same but refusals differed
    (16 against 9, 31 against 16). The cause was not traced. These runs
    have 9–72 wrong predictions; after each, the agent places its view by
    matching every placement, and tokens order the placements differently
    and keep remembered tiles at those outside the room. Either can break
    a tie differently;
  - time: 0.31–1.09 s per layout against card 043's 0.26–0.92, 6–18%
    slower (12% on average). The shakedown's measure (about 1.5%, section
    5) understated it. The likely cost is the larger state (638 token ids
    against 182 places) copied and interned at every imagined step;
  - first-sight effects of the new colours (the run file's "criterion_2",
    card 038's label): unchanged from card 043, 2 of 5.
- **Prediction:** met for both criteria; time is 12% slower, not about the
  same.

## 8. Decision

**Keep** (the user, 2026-10-01: "looks good"). The state is now a set of tokens,
every tile and the hand a what and a where, moved by fitted
transformations, with no pose table; and the agent does exactly what card
043's did in every familiar world. Nothing in this world needed the
change: tokens and the grid carry the same information here (section 5).
Its value is what reads it next: card 041's how-soon from where, card
045's obstacles as conditions on tokens, recall comparing tokens with
positions (the deferred memory work), and segments in place of tiles.
Open: the 12% time cost, recall's declared exception, and walking that
still imagines moves.
