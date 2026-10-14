# Lessons

Durable lessons, each backed by evidence. At most three pages: merge or delete
rather than append.

`old:` is `~/Projects/demilabs/experiments/research/trajectories/`; "phwm
record" is `old:phwm/docs/05-experimental-record.md`. *Confirmed* means a
control or baseline supports it; *suggestive*, one seed or a diagnostic.

## Designing a test

- **Check that the events a test depends on are common in training, and
  that the answer can be known from what the model sees.** Unlocks were
  0.018% of transitions; pick up and toggle stayed at 0.19–0.24 until half
  of each batch was interactions (0.76–0.87). One egocentric view matched
  the best action 0.30–0.33 of the time (chance 0.27). *Evidence:* old:gamma
  chained-rooms stage A; card 001 gate. *Confirmed.*
- **Measure the upper bound in the test world itself, and check that the
  bar is reachable.** Card 072 took 100% as the bound where knowing the key
  gives 94%. Card 105.2 asked for 80% of "do" changes when 44% are
  saplings Crafter grants with probability 0.1. On a grid a smallest 90%
  region holds more than 90% (card 102.3). Check too that the trivial
  baseline fails the gate: "the view never moves" passed card 104.1's.
- **Score effects against the simulator, never against the model's own
  codes.** Codes that never saw doors open scored "nothing happens" 40 of
  40; against the simulator, 0 of 40. *Evidence:* card 052. *Confirmed.*
- **Examples of a goal are evidence only if drawn the way the agent's own
  experience was** (card 056: 95–97% only with play like the agent's).
- **A hand-built mechanism that passes is not a result, and offline gains
  need checking in live behaviour.** *Evidence:*
  old:grl/notes/2026-09-10-reset.md. *Confirmed.*

## Generalisation

- **In one fixed world a lookup table and a general rule fit equally well;
  only the task distribution or the model's form can prefer the rule.**
  PHWM's "tool tier ≥ resource tier" stayed at 0.16–0.23 until new fillers
  per task made the rule the only solution (0.980). A threshold per door
  colour formed beside a relation (card 069: 45 of 66) and vanished as the
  only path with a fresh hue per try (card 070: 66 of 66). A pair network
  on raw positions learned the lamp column by column (102.1: 2 of 4 runs
  carried). A transformer reading positions only as differences, through a
  bounded bias, carried in every seed (105.6: 3 of 3). *Evidence:* phwm
  §5.4–5.5; cards 069, 070, 084.3, 102.1, 105.6. *Confirmed.*
- **"No loss in this world" narrows the model into the world's shape, and
  the narrowing becomes structure.** Arm C asked only about the front and
  the hand (095.3) and moves kept their own models (095.4). The planner then
  needed five hand-written condition forms that do not cover Crafter.
  Before building on a closure that cites "no loss", test it against the
  next world. *Evidence:* the version 27 review; cards 087, 095.3, 095.4.
- **Verify that held-out cases are held out, and report every case
  separately.** PHWM's first held run left 98% of queries in support;
  colour decoded at .982 while the held cell scored 0.0. *Evidence:* phwm
  §5.2–5.6; card 027. *Confirmed.*
- **Nothing in a reconstruction or effect objective factors an attribute
  across kinds.** Hue is a quantity within each kind (R² 0.96–0.99) but not
  lined up across them (33–53 of 66 pairs). *Evidence:* cards 031, 032,
  070, 088–090. *Confirmed.*
- **Novelty depends on the action; an exact lookup over codes cannot use
  "like a key but not a known key".** Recall's per-action weights got all
  four cases (10 of 10) where one radius per code could not. *Evidence:*
  cards 034, 035, 037. *Confirmed.*

## Conditions and acting

- **Conditions read from a model inherit the errors of its training
  targets.** Replacing one token at a time gave world 1's conditions of
  pushes and moves exactly (precision and recall 1.0; baseline precision
  0.09–0.14). In Crafter, perception had misplaced about a quarter of
  blocked moves, as if the agent walked on, more often near creatures. The model
  learned it, and conditions on distant cows and zombies followed
  (blocked-move precision 0.23–0.31). Placed by the agent's own pose, the
  same model gave 0.95–0.97. *Evidence:* cards 105.4, 105.10.
- **Check a condition only in a situation the learned effects produce.**
  Splicing memories gave a base-rate guess (88–91%); checking in the
  imagined result of the achieving action gave 100% (5 of 5 seeds). In
  card 105.4, 493 of 501 false conditions of a lamp going out came from a
  state the world cannot produce (a lit lamp with no box beside it). The
  model's own dynamics cannot flag such states: trained only on states the
  world produces, it predicted no change under no-op for 0 of 39,744
  impossible ones (card 106). *Evidence:* cards 042, 043, 105.4, 106.
  *Confirmed.*
- **Softmax attention must spend its weight, so it learns from tokens that
  explain nothing** (a lamp put 42% on walls, which give its position
  back). Gates that can close fix it, but pruning closes rare relations for
  good. *Evidence:* card 102.1 revisions 1–3, one seed each.
- **Contrast successes with failures; one smooth distance cannot mark
  conditions.** *Evidence:* cards 001, 003, 004, 010. *Confirmed* for the
  pattern.
- **Choose what to act on by the action's effect on its parent condition,
  and keep a choice while it still gives a plan.** Refusing moves that end
  a condition met higher up gave 100% (41 of 1,000 layouts looped before).
  Choices priced afresh flipped (71–73%); keeping the last gave 100%.
  *Evidence:* cards 027, 029, 058, 060, 074. *Confirmed* for looking.
- **A declared front hides a relation the model must learn elsewhere.**
  Crafter's view does not turn with the agent: given the front, stone was
  mined in 31 of 31 predictions, asked of every token in 7 of 31.
  *Evidence:* card 102.1, one seed.
- **Trust a learned distance only within the reach its input shows.**
  Learned walking matched 7,430 of 7,431 short routes but 26 of 33 that
  left the view, all copies agreeing. *Evidence:* card 097.1.

## Networks and objectives

- **A loss must neither drop rare certain changes nor forecast random
  ones; score the next code by its likelihood.**
  - **Average error drops rare certain changes.** A 4-bit code merged the
    goal square (0.04% of rows) with a door (cards 005, 006). Under plain
    squared error, Crafter's model predicted "the tree stays" on all 138
    steps that gave wood (105.7).
  - **Weighting changes forecasts random ones.** Weighted toward change,
    it predicted a cow stepping into grass on half of the no-op steps
    (105.2).
  - **The likelihood fixes both.** The next code's likelihood, as a
    softmax of negative distances to perception's codes, fixed both: do's
    certain changes 83–87%, no-ops 94–96% (105.8).
  - **Pick the read-out.** Under it, read a place by probability, not by
    distance (17% → 97%).
  - **A harder second query takes capacity:** k-step queries cut do's
    changes by 2.8 and 15.7 points (108).
  - *Evidence:* cards 005, 006, 027, 102.2, 105.2, 105.7, 105.8.
    *Confirmed.*
- **Train each place's source where the content changed; a blend of ways
  is free.** Trained on every place alike the model learned "nothing
  changes" (32%); training copy's pointer by its own cost where content
  changed gave 92%. *Evidence:* card 102.4.
- **Query a network only in the form it was trained on, and check that
  copies agree.** Arm A queried on two tokens was right by luck
  (retrained copies: 0%). Counting an intervention only where two seeds
  agree raised Crafter's make-condition precision from 0.77–0.84 to 0.99,
  but copies share blind spots they were never shown. *Evidence:* cards
  096.1, 096.2, 106.
- **An unbounded learned bias can stop a model fitting its own data.**
  Relative attention with an unbounded position network failed to fit
  in one seed of two (final loss 25× the others). The same network with
  log-spaced input and a bounded output, as in Swin V2's code, fit in 3
  of 3. *Evidence:* cards 105.1, 105.6.
- **Test action sensitivity before training; latent self-prediction
  erases what actions change** (shuffled ratio 1.00–1.04; a door moved its
  vector 0.03 against noise 0.019). *Evidence:* GRL-105, GRL-106; cards
  052, 053. *Confirmed.*
- **Some objectives carry too little or the wrong signal:** distance to a
  goal state is not distance to a condition (rank correlation 0.01–0.05),
  and a reachable-future bonus collapsed to sleeping. *Evidence:* card
  001; GRL-108.

## Perception (Crafter, Craftium)

- **To learn invariance to a change shared by a whole frame, put both
  conditions' candidates in one softmax** (night tiles coded as by day:
  12.8% → 94.0%). *Evidence:* cards 098, 098.2.
- **Appearance cannot tell look-alikes apart; correspondence needs a
  prediction of where each will be.** A walk over every place linked 284
  of 290 moved cows to grass (098.8). SAM 2.1, prompted at each creature
  and re-prompted with its true identity whenever lost, reached IDF1 0.78.
  Prompted once it reached 0.60, as hand-built files did (0.62); its tracks
  swapped to the next cow. In Craftium's 3D pasture, CoTracker3 keeps a
  cow 78% of the time while it stays in view, but only 13% after it has
  left and come back. A belief in the world's frame, with exact positions
  and the pose, brings every cow back within 100 steps. *Evidence:* cards
  098.6–098.8, 103.5, 104; smoke tests of 2026-10-13 (card 107, section 1).
- **Pretrained depth sees terrain, not creatures.** Tuning Depth Anything
  3's head by view synthesis with the known pose fixed the scale (0.74 →
  0.99 of true; 30% → 74% of pixels within 25%). A cow 10–20 pixels across
  is still lost in the patch-scale depth (31% too near), and the loss's
  mask drops movers anyway. *Evidence:* card 107.
- **Common fate groups tiles into objects once ground is learned and a
  mover must leave its place** (99.5% of moved shapes one token; no false
  group in 21,947 Crafter states). *Evidence:* card 102.2.
- **Place a view by the agent's own motion, not by matching alone;** a
  wrong placement that sees only new places matches perfectly. Learned
  view motion reached 99.3% held out, but placed only 76–85% of blocked
  moves right. Pretrained motion models read repeated tiles as still (4–5%
  of real moves). The pose now places the view (C1, 105.10). *Evidence:*
  cards 057, 061, 098.8, 102.5–102.8, 103.1–103.4, 104.1, 105.9.

## Memory

- **A longer context is not memory** (1.716 against 1.713 transfer CE);
  memory claims need a test where the past is necessary. *Evidence:*
  RLM-003, -007. *Confirmed.*
- **Recall by a similarity vote needs a space where shared properties are
  near and new things far** (on card 031's encoder, 10 of 10 wrong).
  *Evidence:* card 033. *Confirmed.*
- **A planner needs crisp identity: recall in two levels, mixed, not
  switched.** A failed try rules out its situation, not the thing (30% →
  17% when switched). *Evidence:* cards 042–047, 091, 091.1. *Confirmed.*
- **A prediction of what a place will hold is not identity.** Score a
  file by its own part (with the token minus without it), and link it
  only to codes one token wears. Otherwise an arrow's file takes a
  skeleton. The model kept 83.8% of returns, against 87.9% for counted
  motion; given Q from labels, it tied. *Evidence:* card 108.
- **Recall that compares by a network's context cannot see a condition
  the network has not learned** (11 iron-tool successes lost among 5,900
  failures). *Evidence:* card 102.1.

## Process

- **Change pass/fail criteria at most once; then open a new experiment.**
- **Name counters apart from the records they merge into, label a
  diagnostic from each run's own record, and log every loss term
  separately.** *Evidence:* cards 074.2, 098.6.
- **Time a change under the run's own load, and profile before a long
  run.** Parallel threads made tier 3 3.6 times slower (card 092). A gather
  in a backward pass made a step 1.5 s instead of 21 ms (card 105.1).
- **Wait on a file a run writes, not on a process name**; `pgrep -f`
  matches the waiting command itself (2026-10-13).

## Not yet mined

old:legacy/mgrid/c1–c4 and sam (c4 reportedly has a composition split that
removed every matched case).
