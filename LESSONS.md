# Lessons

Durable lessons, each backed by evidence. At most three pages: merge or delete
rather than append.

`old:` is `~/Projects/demilabs/experiments/research/trajectories/`; "phwm
record" is `old:phwm/docs/05-experimental-record.md`. *Confirmed* means a
control or baseline supports it; *suggestive*, one seed or a diagnostic.

## Designing a test

- **Check that the events a test depends on are common in training, and
  read a capability only once its prerequisite is learned.** Unlocks were
  0.018% of chained-rooms transitions; after 40k updates pick up and toggle
  scored 0.19 and 0.24 (turns 0.999), leaving later screens unreadable.
  Half of each batch at interactions reached 0.76–0.87 in 1,000 updates.
  *Evidence:* old:gamma/notes/2026-09-21/22 chained-rooms stage A.
  *Confirmed.*
- **Check that the answer can be known from what the model sees.** One
  egocentric view explained 48–66% of the variance of steps-to-condition;
  the best action per view matched the truth 0.30–0.33 of the time (chance
  0.27), so no memoryless model could pass 0.8. *Evidence:* card 001 gate.
  *Confirmed.*
- **On a grid, a smallest 90% region holds more than 90%; count its
  boundary place in proportion.** A calibrated belief read 96.8% inside
  its region before, 91.2% after. *Evidence:* card 102.3.
- **Measure the upper bound in the test world itself.** Card 072 took tier
  1's 100% as the bound for the decoy world, where knowing the key gives
  94%, so its 99% criterion could not be met. *Evidence:* card 072.
- **Score effects against the simulator, never against the model's own
  codes.** Codes that never saw doors open scored "nothing happens" 40 of
  40 and effects 99%; against the simulator, 0 of 40. *Evidence:* card 052
  step T correction. *Confirmed.*
- **Examples of a goal are evidence only if drawn the way the agent's own
  experience was.** Frames from shorter play admitted an intact vase,
  from a few long episodes a key of another colour; one frame per episode
  of play like the agent's gave 95–97%. *Evidence:* card 056. *Confirmed.*
- **A hand-built mechanism that passes is not a result, and offline gains
  need checking in live behaviour.** A hand-chosen lattice fit Crafter's
  renderer and failed on lighting; count-world gains did not survive live
  play. *Evidence:* old:grl/notes/2026-09-10-reset.md, INHERITED.md.
  *Confirmed.*

## Generalisation

- **In one fixed world a lookup table and a general rule fit equally well;
  only the task distribution or a prior can prefer the rule.** PHWM's
  "tool tier ≥ resource tier" stayed at 0.16–0.23 under many
  architectures as the pair table re-formed; new random fillers per task
  made the rule the only solution (0.980, shuffled 0.494). A dot-product
  score (Wq a)·(Wk b) is such a table; a distance along learned weights
  carried. A relation beside terms that read each tile alone became a
  threshold per door colour (card 069: 45 of 66); as the only path, with
  a fresh hue per try, it was learned (card 070: 66 of 66, 9 of 9 unseen
  hues). Recall's fitted weights and Bayesian admission favour a door's
  identity for the same reason (every try has twins at that door: card
  084, 0 of 6 folds); admitting conditions that name no tile first fixed
  it (card 084.3). Positions behave the same way: a pair network on raw
  positions learned "a box beside the lamp" at the training columns (100%
  in held-out episodes); at new columns two of four training runs passed
  and two faded with distance (100%, 96%, 59% at columns 5–7). Feeding
  only position differences did not fix it: walls and other fixed tokens
  give the position back (43% of the lamp's attention on the agent, 42%
  on walls and empty rows, 2% on the box), and a relation to the agent
  became two steps that were not learned (made tools 96% → 6%; card
  102.1 revision 1). *Evidence:* phwm record §5.4–5.5; cards
  069, 070, 084–084.3, 102.1. *Confirmed.*
- **Verify that held-out cases are held out; keep new combinations apart
  from new members.** PHWM's first held run left 98% of queries in the
  support set. Combinations .943 (ceiling .976), new members .808.
  *Evidence:* phwm record §5.3, §5.6. *Confirmed.*
- **A property that can be read out is not one that is used; report every
  case separately.** Colour decoded at .982 while the held matching cell
  scored 0.0; an "any key opens" shortcut barely moved average accuracy.
  *Evidence:* phwm record §5.2; card 027 run 4. *Confirmed.*
- **Nothing in a reconstruction or effect objective factors an attribute
  across kinds.** Codes kept all 20 tiles apart but put a key of a new
  colour on "new" or on a blue one, never on its shape (0 of 10). On
  card 070's encoder, hue is a quantity within each kind (R² 0.96–0.99)
  but not lined up across kinds (card 088: 53 of 66 pairs; cards 089–090:
  33–45 of 66); "same colour" across kinds is a relation the outcome must
  teach.
  Measure an attribute within each kind. *Evidence:* cards 031, 032,
  070, 088–090. *Confirmed.*
- **Novelty depends on the action; an exact lookup over codes cannot use
  "like a key but not a known key".** One radius per code either shared
  familiar codes (new open doors took a door's codes in 5–7 of 10) or
  predicted nothing (0 of 10); recall's per-action weights got all four
  cases in 10 of 10 (cards 034, 035). State each tile's outcome from its
  own tries: a code tuple cannot name a tile that mixes a familiar part
  with a new one (card 037). *Confirmed.*

## Conditions and acting

- **Softmax attention must spend its weight, so it learns from tokens that
  explain nothing.** A lamp put 42% of its weight on walls and empty rows,
  which give its position back. Independent gates that can all close
  gave 0%. Pruning such gates after a dense phase closed nearly all of
  them in 1,000 steps, and closed gates do not reopen. A margin on the
  average loss gives up rare relations (make, a lamp lighting). Gates
  never pushed shut stayed open (23,000 of 26,000) and leaked again (56%).
  *Evidence:* card 102.1 revisions 1–3, one seed each.

- **Check a condition only in a situation the learned effects produce.**
  Splicing the hand of one memory with the view of another gave a
  base-rate guess (87.8–90.6%); checking in the imagined result of the
  achieving action gave 100% in 5 of 5 seeds. *Evidence:* cards 042, 043.
  *Confirmed.*
- **One smooth distance cannot mark conditions; contrast successes with
  failures.** Distances scored at the goal-swapped level; contrast gave
  exact conditions from 10 successes, and a rule list in which every rule
  is a route to success gave 19/19 goals. *Evidence:* cards 001, 003,
  004, 010. *Confirmed* for the pattern.
- **Choose what to act on by the action's effect on its parent condition.**
  Targets chosen by dependence looped in 41 of 1,000 layouts; refusing
  moves that end a condition met higher up gave 100%. *Evidence:* cards
  027, 029.
- **Keep a choice between steps while it still gives a plan.** Choices
  priced afresh flip when a turn changes which is cheaper (cards 058,
  060: 71–73%); keeping the last step's choice gave 100%. Keep the need
  last pursued even when it is the only one unmet (card 074.1), never
  override an order the test finds (card 074: decoy 94% → 100%), and
  make an act chosen to learn its outcome whatever the model predicts
  (card 101). *Confirmed* for looking.
- **Only values trained through the encoder see deep conditions** ("door
  open" in 27% of frames on frozen features, 98% trained through; cards
  008, 012). A threshold on 0.95^steps flickered into a spurious
  condition. *Evidence:* 1 seed each.
- **A declared front hides a relation the model must learn elsewhere.**
  MiniGrid's view turns with the agent, so the front is always the same
  place. Crafter's does not, so the tile "do" acts on is a relation to
  the agent's heading. Given the front, stone was mined in 31 of 31
  predictions; asked of every token, 7 of 31. *Evidence:* card 102.1, one
  seed.
- **Trust a learned distance only within the reach its input shows;
  copies trained on the same data share its blind spots.** Learned
  walking matched the exact field on 7,430 of 7,431 short routes but on
  26 of 33 that left the view, all copies agreeing. *Evidence:* card
  097.1.

## Networks and objectives

- **Networks trained on average error drop rare but certain differences;
  let counting decide discrete units and train networks to recognise
  them.** A 4-bit code merged the goal square (0.04% of rows) with a door;
  evidence counting found all 8 kinds. Pixel networks needed about 30×
  the unlocks counting needs, and one reading out every fact at 0.9995
  still gave fragmented rules. Where a network must learn them, draw
  steps by their last loss, unweighted: pushes and door changes (2–3%
  of steps) were never learned from uniform draws, and replaying events
  or weighting the replay skewed or cancelled it. *Evidence:* cards
  005, 006, 027, 102.2.
- **A blend of ways is free; code each place by one way and learn the
  latent source where the content changed.** Trained on every place
  alike, arm A learned "nothing changes" (gate 32%, "keep" 43%); with a
  change head it drew pushed shapes as a blend of keep and set (0% of
  shapes moved). Scoring each way alone by its code length (the mixture's
  −log Σ p e^−cost) still left the pointer untrained wherever keep was
  cheaper (gate 56%); training the source by copy's own cost where the
  content changed gave 92% and moved shapes of several colours (83%).
  *Evidence:* card 102.4.
- **Query a network only in the form it was trained on, check that
  copies agree, and score rare events on events that change something.**
  Arm A, queried on two tokens after training on whole states, was right
  by luck (retrained copies: 0% of tier 2's pick ups); scored per event, a
  "nothing happens" prior won and its weight overrode memory. *Evidence:*
  cards 096.1, 096.2.
- **Test action sensitivity before training; latent self-prediction
  erases what actions change.** Without a stop-gradient, latent
  prediction ignored actions (shuffled ratio 1.00–1.04). Against an EMA
  target with a variance floor, a door's opening moved its vector 0.03
  against noise 0.019, and a planner on it succeeded 0–3%. An off-by-one
  pairing hid death anticipation. *Evidence:* GRL-105, GRL-106; cards
  001, 052, 053. *Confirmed.*
- **Some objectives carry too little or the wrong signal.** Distance to a
  goal state is not distance to a condition (rank correlation 0.01–0.05);
  inverse-action questions do not beat the action prior beyond lag 2; a
  reachable-future bonus collapsed to sleeping. *Evidence:* card 001;
  old:gamma/DECISIONS.md; GRL-108.

## Tokens from frames (Crafter)

- **To learn invariance to a change shared by a whole frame, put both
  conditions' candidates in one softmax.** Tiles paired across light got
  the same code by day and night for 12.8% of night tiles with only the
  night frame's candidates, 94.0% with the day frame's among them. At
  night no pixel threshold separates same from different. *Evidence:*
  cards 098, 098.2.
- **No kernel width rescues tokens the encoder runs together.** Files
  that held movers on the true state (IDF1 0.934) reached 0.717 on
  tokens; night movers sit among scenery at the identity radius and have
  about one record each at a fifth of it. The width that best predicts
  moves is not a width for pooling evidence. *Evidence:* cards 098.6,
  098.7.
- **A walk over every place prefers "they swapped" to "it moved"; tracking
  needs a way to be hidden or revealed.** The grass a cow covers has no
  right continuation, and "the cow became grass, the grass a cow" is
  one-to-one, so the cycle rewards it: motion trained through the walk
  learned "nothing moves" (284 of 290 moved cows linked to grass). Fitted
  to appearance's links it drifted (appearance cannot say where an
  identical tile went); neighbours in the input split kinds (335 codes)
  and lost movers (67%); a change branch trained through the walk took
  over. A tile has one right continuation, so edge dropout forces wrong
  ones. *Evidence:* card 098.8.
- **Common fate groups tiles into objects once ground is learned, a
  mover must leave its place, and the odds of one object are the
  world's own.** Moved shapes became exactly one token in 99.5% of
  states (colour and touching: 56%), and Crafter formed no false group
  in 21,947 states. Without the last two rules, 49% of Crafter's states
  held one: a new arrow read as its skeleton moving, a spawned cow as a
  change. Read a group through a pair layer within it, not a mean of its
  tiles: true objects predicted pushes 100% against 88%. *Evidence:*
  card 102.2.
- **Match a token across absence by the codes tracked tokens wear, not
  its own code.** 098.3's codes include the ground beneath a creature,
  so 76% of lost Crafter identities came back as another code. Weighting
  the match by how often a token of one code is seen as the other lifted
  identity from 74% to 80% (card 102.3, smoke test).
- **Read the view's motion in the world's own units, and compare
  hypotheses on the same evidence.** Cut into patches that do not fit the
  tiles, the view's motion was never read (8-pixel patches: 0% of moves;
  3 pixels off phase: −1 point); read in pixels, 94.9%, with the step
  length (±7 px) learned. Summing mismatch over known places favours
  shifts into the unseen; pricing the unseen as new content favours "no
  move"; either moves the result by several points. *Evidence:* cards
  102.5, 102.7, 102.8, one run each.
- **Count evidence per independent observation, not per pixel; choose the
  count by the agent's own self-consistency.** With the camera's pixels
  learned and left out (26 of 26 in the agent's tile), a cow stepping on
  plain grass outvoted the action's prior (44 pixels × ~19 nats against
  3–5): still steps 99% → 91%; the sprite had been a hidden "no move" vote.
  Dividing the evidence by τ = 300 pixels gave 99.3% held out (night 96.9%;
  the declared agent tile 98.1–98.5%). τ was the value whose placements
  left re-seen world content most often as written; this peaked where the
  evaluator's accuracy did. Letting false files remove evidence instead
  cost 11 points. *Evidence:* cards 103.1, 103.3, 103.4, one run each.
- **Place a partial view with the agent's own motion, not by matching
  alone.** A wrong placement that sees only never-seen places matches
  perfectly (0% success); placing only among the placements the action
  could lead to gave 100% (cards 057, 061). In Crafter, two frames miss a
  fifth of real moves on uniform ground; placing against belief gets 98%
  (card 098.8), 99.3% with tempered evidence (card 103.4). *Confirmed* for
  these worlds.

## Memory

- **A longer context is not memory.** A history model scored 1.716
  transfer CE against 1.713 for a current-frame model; memory claims need
  a test where the past is necessary. *Evidence:* RLM-003, -007.
  *Confirmed.*
- **Recall by a similarity vote needs a space where shared properties are
  near and new things far.** On label codes the vote opened a new-coloured
  door at the first try; on card 031's encoder nothing carried (10 of 10
  wrong). *Evidence:* card 033. *Confirmed.*
- **A planner needs crisp identity: recall in two levels, mixed, not
  switched.** Equal codes for the same thing and vectors for similar
  things matched the counted planner (cards 042–043). Switching to a
  thing's own tries let one failed toggle make a new door never openable
  (30% → 17%); a failed try rules out its situation, not the thing. A
  router voting over every key gave a locked door 0.27–0.53 "opens" after
  1,544 failures; voting among keys of the same tiles first ended the loop
  (cards 046–047, 091, 091.1). *Confirmed.*
- **Recall that compares by a network's context cannot see a condition
  the network has not learned.** The 11 stored iron-tool successes
  counted for nothing when a new iron try came, because arm A's context
  did not separate them from the 5,900 failures (0 of 5 made). *Evidence:*
  card 102.1, upper bound.

## Process

- **Change pass/fail criteria at most once; then open a new experiment.**
  *Evidence:* old:grl/notes/2026-09-10-reset.md.
- **Name counters apart from the records they merge into, and label a
  diagnostic from each run's own record.** A counter `steps` overwrote
  episode step counts (card 074.2); a per-class count read worker 0's
  classes for all workers (card 098.6).
- **Time a change under the run's own parallel load.** 24 math threads per
  forked worker and a per-step map search made tier 3 3.6 times slower
  in parallel. *Evidence:* card 092.
- **Log every loss term separately;** a combined value hid a scale failure.
  **Check cheap-model extraction;** of 61 lessons Haiku drew from the old
  repo, about 10 survived.

## Not yet mined

old:legacy/mgrid/c1–c4 and sam (c4 reportedly has a composition split that
removed every matched case).
