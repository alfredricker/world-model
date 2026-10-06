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
- **Score a model's effects against the simulator, never against its own
  codes.** Card 052's step T scored "the door opens" by the predicted
  code tuple equalling the real one; its codes did not see doors open,
  so "nothing happens" scored 40 of 40 and effects 99%. Against the
  simulator: 0 of 40. *Evidence:* card 052 step T correction.
  *Confirmed.*
- **Examples of a goal are evidence only if drawn independently and the
  way the agent's own experience was.** Inferring a goal from five frames
  by contrast with experience (the size principle) admitted an intact
  vase when the frames came from shorter play than the agent's, and a
  key of another colour when they came from a few long episodes (door
  green, 77.5%); one frame per episode of play like the agent's raised
  every goal with enough episodes to 95–97%. *Evidence:* card 056.
  *Confirmed* (four encoders, identical).
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
  weights carried. A distance is not enough when terms reading each tile
  alone sit beside it: card 069's relation, trained on nine fixed colours
  next to such terms, became a threshold per door colour (45 of 66 pairs
  right for a colour left out). With the relation as the only path and a
  fresh hue for every try, the same encoder learned it (card 070: 66 of
  66, and 9 of 9 for hues never seen). The same holds for recall's
  weights: fitted by predicting each stored try from the others, they
  favour the door's identity, since every try has twins at the same door
  (card 070). *Evidence:* phwm record §5.4–5.5;
  old:phwm/docs/06-achievements-and-limitations.md §6.1.1; card 040
  appendix A; cards 069, 070. *Confirmed.*
- **Verify that held-out cases are really held out, and keep new
  combinations apart from new members.** PHWM's first held run left about
  98% of query pairs in the support set (corrected: .917). Combination
  split .943±.004 against a .976 ceiling; member split (a colour never
  seen in the role) .808±.053. *Evidence:* phwm record §5.3, §5.6.
  *Confirmed.*
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
  Four codebooks of 8 kept all 20 tiles apart (10 of 10 seeds), but a key
  or door in a new colour came out "new" or as a blue one, never as its
  shape (0 of 10; label codes carried fully). A penalty on the number of
  codes put a new colour on the nearest known one (8 of 10), which
  carried the shared effects and wrongly the colour rules too.
  *Evidence:* cards 031, 032. *Confirmed* (label arm as control).
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
- **One smooth distance cannot mark conditions; contrast successes with
  failures.** Two-step chains scored at the goal-swapped level (PHWM;
  card 001), and a key pickup looks like any step (card 003). Contrasting
  an action's successes with its failures gave exact conditions from 10
  successes and the key → door → goal chain (card 004); a Bayesian rule
  list in which every rule is a route to success gave 19/19 goals with no
  thresholds (card 010). *Confirmed* for the pattern.
- **Choose what to act on by the action's effect on its parent condition,
  not by whether the route depends on the thing.** Dependence-chosen
  targets looped in 41 of 1,000 layouts (a key dropped back where it
  blocked, a switch turned off again). Refusing moves that end a condition
  met higher up, and marking actions that do not make the parent true,
  gave 100%. *Evidence:* card 027 runs 1–2, card 029.
- **Keep a choice between steps while it still gives a plan.** Choices
  priced afresh every step flip when a turn changes which is cheaper:
  clearing the way or walking round (card 058: both world 73%), which
  door to open and look past (card 060: chained rooms 71%). Keeping the
  last step's choice: 100% (card 060; achievers since card 051).
  *Confirmed* for looking; card 058's clearing chain still flips inside.
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
  anticipation (GRL-105). A target network is not enough: predicting its
  own next vectors against a moving-average target, with a variance
  floor, an encoder erased what actions change (a door's opening moved
  its vector 0.03 against noise 0.019; keys of three colours one code),
  so every transition was trivially predictable; a planner on it
  succeeded in 0–3% (card 052 step T, card 053). *Evidence:*
  old:grl/records/GRL-105.json, GRL-106.json; card 001 gate; card 052
  step T correction. *Confirmed* for latent self-prediction.
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
  opened a new-coloured door at the first try; on card 031's encoder
  nothing carried (10 of 10 seeds wrong). Replay into that encoder
  carried pick-up and drop but pushed familiar tiles off their codes (9
  of 10), which broke acting. *Evidence:* card 033. *Confirmed* (label
  arm as control).

- **A planner needs crisp identity; one learned weighting of vectors per
  action cannot supply it.** Colour had to count at doors and nowhere
  else: adding colour weight fixed doors (held-out −0.73 → −0.16) and hurt
  everything else (−0.010 → −0.040). Recall in two levels, equal codes for
  the same thing and vectors for similar things, matched the counted
  planner in all four worlds (cards 042–043). But the levels must be
  mixed, not switched: when a thing's own tries replaced similar things,
  one failed toggle (switch off) made a new door never openable, and the
  new-colour switch tests fell from 30% to 17%. Judging every remembered
  situation by the mixed prediction gave 40%, with the switch turned on by
  plan (cards 046–047). A failed try rules out its situation, not the
  thing. *Confirmed.*

- **Place a partial view with the agent's own motion, not by matching
  alone.** With a 7 × 7 view, a wrong placement that sees only
  never-seen places matches perfectly, and after a pick up the true one
  shows the changed tile and hand: 0% success and wrong memories in every
  episode. Placing the view only among the placements the action could
  lead to gave 100% and no wrong remembered tile in 220 episodes. The
  motion itself can be learned from the small view: count correspondences
  only where both places were observed and vary (far places are seen
  almost only as wall and "predict" anything), then trim to one rigid
  transformation; 16–35 shared places give every move exactly (card 061).
  *Evidence:* cards 057, 061. *Confirmed* for this world.

## Process

- **Change pass/fail criteria at most once; then open a new experiment.**
  *Evidence:* old:grl/notes/2026-09-10-reset.md.
- **Log every loss term separately;** a combined value hid a scale failure
  (old:legacy/mgrid/README.md). **Check cheap-model extraction;** of 61
  lessons Haiku drew from the old repo, about 10 survived unchanged.

## Not yet mined

old:legacy/mgrid/c1–c4 and sam (c4 reportedly has a composition split that
removed every matched case).
