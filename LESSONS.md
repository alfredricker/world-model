# Lessons

Durable lessons, each backed by evidence. At most two pages: merge or delete
rather than append.

Paths starting `old:` are under
`~/Projects/demilabs/experiments/research/trajectories/`. Imported 2026-09-24:
each lesson was extracted, then checked against its source; only claims that
survived checking are here. *Confirmed* means a control or baseline supports
it; *suggestive* means one seed, small n, or a diagnostic.

## Designing a test

- **Check that the events a test depends on occur often enough in the
  training data before running it.** In chained rooms, unlocks were 307 of
  1.7M transitions (0.018%); after 40k updates pickup and toggle consequences
  scored 0.19 and 0.24 against 0.999 for turns. A label-driven fine-tune with
  half of each batch at interactions reached 0.76–0.87 on all four screens in
  1000 updates. *Evidence:* old:gamma/notes/2026-09-21-chained-rooms-stage-a.md,
  old:gamma/notes/2026-09-22-chained-rooms-stage-a2.md. *Confirmed.*
- **Interpret a capability only after its prerequisite is learned.** Stage A's
  relation, memory and transfer screens were unreadable because both arms
  predicted "nothing happens" for every interaction. *Evidence:* stage A note,
  final verdicts. *Confirmed.*
- **Check that a test's answer can be known from what the model sees.** In
  chained rooms one egocentric view explains 48–66% of the variance of true
  steps-to-condition, and on movement forks the best action per view matched
  the truth 0.30–0.33 of the time (chance 0.27): a memoryless model could
  not pass a 0.8 movement bar. Only 57% of forks were decidable from the
  view. *Evidence:* card 001 gate, third round. *Confirmed* (exact
  computation over all reachable states).
- **Verify that held-out pairs are really held out.** PHWM's first held-query
  run left about 98% of query pairs in the support set; the corrected run
  scored .917 on held cells (6/6 seeds above .90). *Evidence:*
  old:phwm/docs/05-experimental-record.md. *Confirmed.*
- **A probe that decodes a property does not show the model uses it.** PHWM
  decoded colour at .982 while held matching-cell accuracy was 0.0 in 3 seeds.
  Mgrid colour probes scored 99.81% against a 99.23% majority baseline: always
  report the majority baseline. *Evidence:* phwm record above;
  old:legacy/mgrid/README.md (t1). *Confirmed.*
- **Transfer to new combinations and to new members are different claims.**
  PHWM: combination split .943±.004 (ceiling .976); member split .808±.053,
  only 3/6 seeds above .75. *Evidence:* phwm record above. *Confirmed.*
- **Chains of prerequisites are the hard part, and a working step is not a
  used step.** In PHWM, one-step held combinations scored 0.54–0.71 but
  furnace → iron pickaxe 0.11; a sleep request that executed at 0.86–0.92
  was proposed once in 15 lives; request chains were flat from one
  demonstration to all. Card 001's gate repeats the pattern: one-step
  conditions 0.66–0.88, two-step chains at the swapped-goal level. One
  smooth distance cannot mark conditions: in a deterministic world every
  shortest-path step lowers the true distance by exactly 1, so a key pickup
  looks like a move. Under random play, long-range jumps are dominated by
  the final approach (card 003). Contrasting an action's successes with its
  failures at the achieving step recovered exact conditions from 10
  successes, with supplied variables; treating "walk to X" as an action
  with conditions gave the whole key → door → goal chain without naming
  places. *Evidence:* cards 003, 004 (exact, supplied variables).
- **From pixels, rare conditions need about 30× the examples exact counting
  needs, and the error is "it will not work", not "the wrong key works".**
  Toggle-with-matching-key was right in 4%, 45%, 90%, 100% of test cases
  with 10, 30, 100, 300 unlocks in training; wrong-key cases stayed ≥ 99.7%
  right throughout. *Evidence:* card 005 (1 seed per count).
- **A network trained only to predict holds conditions redundantly; they
  cannot be found as switchable parts afterwards.** Card 005's network
  encoded every needed fact (linear read-out 0.9995–1.0), yet a sparse
  dictionary and raw units both gave fragmented rules, and switching off a
  rule's part left the predicted success unchanged in 0% of cases for most
  parts. Separable conditions must be asked for in training.
  *Evidence:* card 006 (1 network, 1 dictionary).
- **A condition can be defined by the agent's own values: "the states where
  this goal's achieving action works are within walking reach".** From one
  supplied goal it discovered door open → matching key → empty hands, each
  behaving as a condition (100% / 0%), and acting on them solved 99.2% of
  new layouts. *Evidence:* card 007 (1 seed; encoder shaped by other
  supplied goals).
- **A state trained on one goal does not notice that goal's deeper
  conditions; train every value that defines a condition through the
  encoder.** Trained on the goal square only, the key was read out at 0.73
  (card 008); a reach value on those frozen features found "door open" in
  27% of true frames (AUC 0.57), and the same value trained through the
  encoder in 98% (AUC 0.998). In the shared network the value loss needs
  its own large batch of walking steps (≈130 walking rows per update: 4%;
  1024: 98%) and a cross-entropy loss (a squared error on probabilities
  near 0 barely moved the encoder: 38–61% → 98%). *Evidence:* cards 008,
  012 (1 seed each).
- **Right conditions are not enough: one flat walk value per way fails on
  long walks through rare regions.** Card 012's learned conditions with
  exact walking reached the goal in 95% of new layouts, with learned walking
  in 30%: walking was 94% right toward a near, common target and 61%
  toward the goal square through an opened door (1.5% of the data), where
  the agent spun in the doorway. *Evidence:* card 012 run 5 (1 seed).
- **Do not read "reachable" off a discounted value with a fixed threshold.**
  0.95^steps near 0.1 at the far end of a room made "within walking reach"
  flicker as the agent walked, creating a spurious condition. *Evidence:*
  card 008.
- **Admit conditions by evidence, and require every rule to be a route to
  success.** A Bayesian rule list (or of ands, Beta-Bernoulli rules, cost
  per condition) recovered exact rules for 19/19 goals across key, switch,
  either, both and no-drop worlds, excluded an irrelevant vase, with no
  thresholds; without the route-to-success constraint the greedy search
  stuck on an equivalent "failure rule" list. From pixels, each world's
  network learned these requirements well enough that the same rules were
  read from its predictions (19/19). *Evidence:* card 010 (exact variables;
  pixels read in the simulator's vocabulary).
  *Evidence:* old:phwm/docs/05-experimental-record.md §5.12,
  06-achievements-and-limitations.md; card 001 gate. *Confirmed* for the
  pattern.
- **Choose what to act on by the action's effect on its parent condition,
  not by whether the route depends on the thing.** Removing a thing shows
  the route needs it, not that acting on it helps: dependence-chosen
  targets looped in 41 of 1,000 layouts (a key dropped back where it
  blocked, a switch turned off again, the wrong side of a thing). Refusing
  moves that end the way's condition and marking actions that do not turn
  the parent true gave 100% in both worlds. *Evidence:* card 027 runs 1–2.
- **Offline or component gains need checking in live behaviour.** Gains on
  count worlds and crafting instruments did not survive live play.
  *Evidence:* old:grl/INHERITED.md (RLM-057, -061, -062). *Confirmed.*
- **A hand-built mechanism that passes is not a result.** A hand-chosen
  lattice fit the Crafter renderer exactly and failed on lighting changes.
  *Evidence:* old:grl/notes/2026-09-10-reset.md, RLM-133. *Confirmed.*

## Sampling and training data

- **Put replay priorities on transitions, not windows.** Unlock anchors had
  mean priority 2.23 against 1.18 overall, yet the same 0.03% sampling share
  as uniform, because one transition barely moves a window's summed
  priority. *Evidence:* stage A note, diagnosis. *Confirmed.*
- **The learner's own "nothing happened" margin finds the rare
  interactions; surprise and learning progress concentrate too weakly.**
  Margin: 0.5% of all transitions, 47–63% of interactions; priority mass on
  interactions 19–28% against 0.36% uniform. Surprise raised event sampling
  1.9–13×; learning progress left it at 0.4%. *Evidence:* stage A2 note.
  *Suggestive.*
- **A network trained on average error drops rare but certain
  differences; evidence from counts keeps them.** A 4-bit code trained to
  predict effects gave the goal square (0.04% of rows) the green door's
  code in both worlds; balancing rows by outcome only moved the merge to
  another minority. Grouping the same appearances by Dirichlet-multinomial
  evidence found all 8 kinds: about 260 nats separate goal from door, under
  0.001 nats per training row. Let counting decide discrete units; train
  networks to recognise them. *Evidence:* card 027 runs 2–4 (both worlds,
  1 seed each).
- **A few labelled tiles teach colour, not shape.** A pixel network
  trained on 14 appearances' counted kinds placed each withheld key and door
  by colour and brightness (4 of 33 right; open doors called floor, the
  green door the goal). *Evidence:* card 027 run 4 (3 seeds). *Suggestive.*

## Objectives

- **Test action sensitivity before training: true versus shuffled actions.**
  Latent prediction without a stop-gradient became action-insensitive in a
  synthetic moving-square world (permutation ratio 1.00–1.04) while a
  reconstruction model passed the same check. The proposed fix (detached
  target plus capped reconstruction auxiliary plus collapse checks) is
  untested. *Evidence:* old:grl/records/GRL-106.json (inconclusive).
  *Suggestive.*
- **Unit-test that training and acting pair frames with the same action.**
  An off-by-one in GRL-102 hid death anticipation; after the fix, imagined
  survival over the last 5 ticks fell from 0.99 to 0.36–0.66. Survival itself
  did not change. *Evidence:* old:grl/records/GRL-105.json. *Suggestive.*
- **Inverse-action questions alone carry too little information.** Under a
  uniform policy an exact frame-pair lookup cannot beat the action prior
  beyond lag 2, and matched toggles were 339 of 1,030,054 transitions.
  Masked inverse models with history, current-frame and image-blind inputs
  all stayed near the action prior; a continuous-only control failed the
  same way, so quantization was not the cause. *Evidence:*
  old:gamma/DECISIONS.md, old:gamma/notes/2026-09-21-continuous-control.md.
  *Confirmed* (quantization part: one seed).
- **A contrastive reachable-future exploration bonus rewards the action with
  the least distinguishable futures.** In GRL-108 it drove collapse to
  sleeping (2 of 3 seeds); leave-one-out attributed the collapse to that
  bonus. *Evidence:* old:grl/records/GRL-108.json. *Suggestive.*
- **Generic spread regularizers (VICReg-style) did not produce binding;
  pairwise constraints did.** *Evidence:* phwm record above, §5.7.
  *Suggestive,* no number extracted.
- **A reachability head trained on single-state goals measures distance to
  states, not to conditions.** Shown the goal as 4 example frames from
  other worlds, its rank correlation with true steps-to-goal was 0.01–0.05,
  against 0.57–0.78 with the exact goal state, in all four gate runs.
  Label-free goal sets (4 frames from a later stretch of the same episode)
  did not change this, even at 100k updates. Supplying the goal condition
  (examples plus a success signal) lifted pooled top-1 from 0.29 to 0.62,
  0.2 above the goal-swapped control, but only for one-step conditions;
  two-step chains stayed at the swapped level. *Evidence:* card 001 gate.
  *Confirmed* (7 runs, 1 seed each).
- **Latent-regression transitions need observations that change with the
  action.** On an allocentric state grid, where a step changes one cell, T
  learned nothing (shuffled-action error ratio 1.0); on egocentric frames
  it learned (ratio 50–200). *Evidence:* card 001 gate. *Suggestive* (1 seed).
- **QRL's transition loss needs QRL's push-up objective.** Added to an
  expectile-trained head, it shrank one-step distances to about 0.02 and
  tied all actions. *Evidence:* card 001 gate, `runs/gate_try2`.
  *Suggestive* (1 seed).
- **Log every loss term separately.** A combined regularizer value hid a scale
  failure. *Evidence:* old:legacy/mgrid/README.md (t1). *Confirmed.*

## Memory and relations

- **A longer context is not memory.** A history model scored 1.716 transfer
  CE against 1.713 for a trained current-frame model. Memory claims need a
  trained current-frame control and a test where the past is necessary.
  *Evidence:* old:gamma/notes/2026-09-21-continuous-learnability.md.
  *Confirmed.*
- **Relational heads may retain unseen events better than content heads.**
  With the switch out of view, the relational arm scored 0.861 against 0.361
  for content and 0.0 for current-frame (n=36, transfer_colour). This was
  before interactions were learned. *Evidence:* stage A note. *Suggestive.*
- **Learned recurrent belief updates drift on repeated evidence.** Duplicated
  evidence moved predictions by .052 mean and .991 max; the fix used
  supplied max pooling, not learning. *Evidence:* old:rlm/records/RLM-003.json,
  RLM-007.json. *Suggestive* (supplied objects).

## Process

- **Change pass/fail criteria at most once; after that, open a new
  experiment.** *Evidence:* old:grl/notes/2026-09-10-reset.md. *Confirmed* as
  a failure pattern.
- **Cheap-model extraction needs checking.** Of 61 lessons Haiku extracted
  from the old repo, about 10 survived checking unchanged. *Evidence:* this
  import. *Confirmed.*

## Not yet mined

old:legacy/mgrid/c1–c4 and sam (c4 covers learned units, JEPA latents and a
composition split that reportedly removed every matched case); stage A2
results once they exist.
