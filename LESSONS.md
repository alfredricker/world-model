# Lessons

Durable lessons, each backed by evidence. At most three pages: merge or delete
rather than append.

`old:` is `~/Projects/demilabs/experiments/research/trajectories/`; "phwm
record" is `old:phwm/docs/05-experimental-record.md`. *Confirmed* means a
control or baseline supports it; *suggestive*, one seed or a diagnostic.

## Designing a test

- **Check that the events a test depends on are common in training, and
  read a capability only once its prerequisite is learned.** In chained
  rooms unlocks were 307 of 1.7M transitions (0.018%); after 40k updates
  pickup and toggle effects scored 0.19 and 0.24 against 0.999 for turns,
  which left the relation, memory and transfer screens unreadable. Half of
  each batch at interactions reached 0.76–0.87 in 1,000 updates; replay
  priority on windows, not transitions, left unlocks at the uniform share.
  *Evidence:* old:gamma/notes/2026-09-21-chained-rooms-stage-a.md,
  2026-09-22-chained-rooms-stage-a2.md. *Confirmed.*
- **Check that the answer can be known from what the model sees.** One
  egocentric view explained 48–66% of the variance of true
  steps-to-condition, and the best action per view matched the truth on
  movement forks 0.30–0.33 of the time (chance 0.27): no memoryless model
  could pass a 0.8 bar. *Evidence:* card 001 gate (exact). *Confirmed.*
- **A hand-built mechanism that passes is not a result, and offline gains
  need checking in live behaviour.** A hand-chosen lattice fit the Crafter
  renderer exactly and failed on lighting changes; gains on count worlds
  did not survive live play. *Evidence:*
  old:grl/notes/2026-09-10-reset.md (RLM-133),
  old:grl/INHERITED.md (RLM-057, -061, -062). *Confirmed.*

## Generalisation

- **In one fixed world a lookup table and a general rule fit equally well;
  only the task distribution or a prior can prefer the rule.** On PHWM's
  crafting grid (a tool works when its tier ≥ the resource's) the
  just-sufficient pair stayed at 0.16–0.23 through relational bottlenecks,
  memory, sparsity and parameter sharing, as the pair table re-formed. New
  random fillers in every task, with the query cell held out, made the rule
  the only useful solution: 0.980 over 6 seeds, against 0.494 with outcomes
  shuffled. With
  three fixed colours, per-colour rules and "same colour" fit our data
  equally; card 010's cost per condition is what can separate them, and
  new colours per episode is the stronger test. A learned dot-product score
  between two things, (Wq a)·(Wk b), is such a table: on one-hot vectors
  it fit three colours and carried to none, while a distance along learned
  weights carried. *Evidence:* phwm record §5.4–5.5;
  old:phwm/docs/06-achievements-and-limitations.md §6.1.1; card 040
  appendix A. *Confirmed.*
- **Verify that held-out cases are really held out.** PHWM's first held
  run left about 98% of query pairs in the support set; corrected, held
  cells scored .917 (6/6 seeds above .90). *Evidence:* phwm record §5.6.
  *Confirmed.*
- **New combinations and new members are different claims.** Combination
  split (every colour seen, one pairing withheld) .943±.004 against a .976
  ceiling; member split (a colour never seen in the role) .808±.053, 3/6
  seeds above .75. *Evidence:* phwm record §5.3. *Confirmed.*
- **A property that can be read out is not one that is used; report every
  case separately.** PHWM decoded colour at .982 while the held matching
  cell scored 0.0 in 3 seeds, and later the wrong-key cell stayed at 0, an
  "any key opens" shortcut that average accuracy barely penalised. Mgrid
  colour probes scored 99.81% against a 99.23% majority baseline. A pixel
  network trained on 14 appearances' counted kinds placed withheld keys and
  doors by colour and brightness, not shape (4 of 33 right). *Evidence:*
  phwm record §5.2; old:legacy/mgrid/README.md (t1); card 027 run 4.
  *Confirmed* (the last *suggestive*).
- **Codes that rebuild pixels keep every distinction but do not factor a
  thing's shape from its colour; nothing in the objective prefers it.**
  Four codebooks of 8 kept all 20 tiles apart and the counted model lost
  nothing (10 of 10 seeds). But a key or door in a new colour came out
  "new" or as a blue one, never as its shape: 0 of 10 seeds, the same
  without the change penalty or with one codebook of 64. The model over
  ideal label codes carried to it fully. Key entries per group, not one
  set per action: one set made "drop" name each key colour, which gives a
  new colour nothing. A penalty on the number of codes merged tiles
  before it made a shorter code. Where it left fewer codes, a new colour
  fell on the nearest known one (the purple key took the blue key's codes
  in 8 of 10 seeds). That carried the shared effects (pick up and drop
  right in 9 of 10) and wrongly the colour rules too. *Evidence:* cards
  031, 032. *Confirmed* (label arm as control).
- **An exact lookup over codes cannot use "like a key but not a known
  key"; novelty depends on the action.** One radius per code, the same for
  every action, either lets a new appearance share familiar codes (it
  carried picking up a new key in 9 of 10 seeds, but gave new open doors a
  familiar door's codes in 5–7) or marks it "new" (every new tile told
  apart, 10 and 8 of 10, and nothing predicted, 0 of 10). Recall's
  per-action weights over the same vectors predicted all four new-key and
  open-door cases in 10 of 10. *Evidence:* cards 034, 035 (both rules on
  the same encoders). *Confirmed* (label arm as control).
- **Recall-weighted entries carry what a new thing does, but a code tuple
  cannot name a tile that mixes a familiar part with a new one.** Entries
  built from recall-weighted tries lost nothing familiar and carried
  picking up and dropping a new key (8 of 9 seeds per colour; counting on
  the same codes, 0–2 of 10). Forward onto the new open door was exact in
  3 of 9, although recall's vote was right in 9 of 9: the agent in a new
  doorway usually has a code no familiar doorway has, and no familiar
  statement produces it. State each tile's outcome from its own tries;
  over a pooled group, statements pick up coincidences. *Evidence:* card
  037. *Confirmed* (label arm as control).

## Conditions and acting

- **Check a condition only in a situation the learned effects produce.**
  Splicing the hand of one memory with the view of another assumes the
  parts are independent; "holding the green key while it lies in view"
  got a base-rate guess and the either world fell to 87.8–90.6% (card
  042). Checked in the imagined result of the achieving action instead:
  100% in 5 of 5 seeds (card 043). *Confirmed.*
- **Chains of prerequisites are the hard part, and one smooth distance
  cannot mark conditions.** PHWM: one-step held combinations 0.54–0.71,
  furnace → iron pickaxe 0.11; card 001: one-step conditions 0.66–0.88,
  two-step chains at the goal-swapped level. In a deterministic world each
  shortest-path step lowers the distance by exactly 1, so a key pickup
  looks like a move (card 003). Contrasting an action's successes with its
  failures recovered exact conditions from 10 successes, and "walk to X"
  as an action with conditions gave the key → door → goal chain (card
  004). *Evidence:* phwm record §5.12; cards 001, 003, 004. *Confirmed*
  for the pattern.
- **Admit conditions by evidence, and require every rule to be a route to
  success.** A Bayesian rule list (Beta-Bernoulli rules, a cost per
  condition) recovered exact rules for 19/19 goals in five worlds and
  excluded an irrelevant vase, with no thresholds; without the route
  constraint the greedy search stuck on an equivalent list of failure
  rules. *Evidence:* card 010 (exact variables; pixel networks gave the
  same 19/19).
- **Choose what to act on by the action's effect on its parent condition,
  not by whether the route depends on the thing.** Dependence-chosen
  targets looped in 41 of 1,000 layouts (a key dropped back where it
  blocked, a switch turned off again). Refusing moves that end a condition
  met higher up, and marking actions that do not make the parent true,
  gave 100%. *Evidence:* card 027 runs 1–2, card 029.
- **Values can define conditions ("the achieving action works within
  walking reach": 99.2% of new layouts, card 007), but only values trained
  through the encoder see deep ones** ("door open" found in 27% of frames
  on frozen features, 98% trained through; cards 008, 012). A fixed
  threshold on 0.95^steps flickered into a spurious condition (card 008).
  Learned walking reached 30% against 95% for exact walking, failing in a
  rarely seen doorway (card 012). *Evidence:* 1 seed each.

## Networks and objectives

- **Networks trained on average error drop rare but certain differences
  and hold conditions redundantly; let counting decide discrete units and
  train networks to recognise them.** A 4-bit code gave the goal square
  (0.04% of rows) the green door's code; Dirichlet-multinomial evidence
  found all 8 kinds, about 260 nats apart (card 027). Pixel networks needed
  about 30× the unlocks exact counting needs (right in 4%, 45%, 90%, 100%
  of cases at 10, 30, 100, 300), erring as "it will not work" (card 005).
  A network reading out every needed fact at 0.9995 still gave fragmented,
  unswitchable rules (card 006). *Evidence:* cards 005, 006, 027.
- **Test action sensitivity and frame–action pairing before training.**
  Latent prediction without a stop-gradient ignored actions (shuffled
  ratio 1.00–1.04) where reconstruction did not (GRL-106); a latent
  transition learned nothing on an allocentric grid (ratio 1.0) and did on
  egocentric frames (50–200; card 001). An off-by-one pairing hid death
  anticipation (GRL-105). *Evidence:* old:grl/records/GRL-105.json,
  GRL-106.json; card 001 gate. *Suggestive.*
- **Some objectives carry too little or the wrong signal.** Distance to
  a goal state is not distance to a condition (rank correlation 0.01–0.05
  from example frames, 0.57–0.78 from the exact state; card 001, 7 runs).
  Inverse-action questions cannot beat the action prior beyond lag 2
  (old:gamma/DECISIONS.md); a contrastive reachable-future bonus drove
  collapse to sleeping (GRL-108, 2 of 3 seeds).

## Memory

- **A longer context is not memory.** A history model scored 1.716
  transfer CE against 1.713 for a current-frame model; memory claims need
  that control and a test where the past is necessary. Learned recurrent
  updates drifted on repeated evidence (.052 mean, .991 max). *Evidence:*
  old:gamma/notes/2026-09-21-continuous-learnability.md; RLM-003, -007.
  *Confirmed* (drift *suggestive*).
- **Recall by a similarity-weighted vote needs a space where shared
  properties are near and new things are far.** On label codes the vote
  found "same colour" and opened a new-coloured door at the first try.
  On card 031's encoder nothing carried (the yellow key's pick-up was
  wrong in 10 of 10 seeds). Replay into that encoder carried pick-up and
  drop (10 of 10) and found "same colour" in 2 of 10. It also pushed
  familiar tiles off their codebook codes in 9 of 10 seeds, which broke
  acting. A new door's vote then weighed as much as a familiar door's
  (1.3–13 against 2.6–11.1; labels 0.15 against 2.0), so one success
  could not correct it. *Evidence:* card 033. *Confirmed* (label arm as
  control).

- **A planner needs crisp identity; one learned weighting of vectors per
  action cannot supply it.** Colour had to count at doors and nowhere
  else: adding colour weight fixed doors (held-out −0.73 → −0.16) and hurt
  everything else (−0.010 → −0.040). Recall in two levels, equal codes for
  the same thing and vectors for similar things, matched the counted
  planner in all four worlds (cards 042–043). *Confirmed.*

## Process

- **Change pass/fail criteria at most once; then open a new experiment.**
  *Evidence:* old:grl/notes/2026-09-10-reset.md.
- **Log every loss term separately;** a combined value hid a scale failure
  (old:legacy/mgrid/README.md). **Check cheap-model extraction;** of 61
  lessons Haiku drew from the old repo, about 10 survived unchanged.

## Not yet mined

old:legacy/mgrid/c1–c4 and sam (c4 reportedly has a composition split that
removed every matched case).
