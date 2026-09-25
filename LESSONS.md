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
  interactions.** Only 0.5% of all transitions carry a margin, against 47–63%
  of interaction transitions; priority mass on interactions rose to 19–28%
  from 0.36% under uniform. *Evidence:* stage A2 note. *Suggestive:* the full
  stage A2 run had not reported when this was written.
- **Surprise and learning-progress priorities concentrate too weakly.**
  Surprise variants raised event sampling 1.9–13×; learning progress left
  event mass at 0.4% after 2000 updates. *Evidence:* stage A2 note.
  *Suggestive.*

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
  all stayed near the action prior. *Evidence:* old:gamma/DECISIONS.md.
  *Confirmed.*
- **Quantization did not cause those failures.** A continuous-only control
  failed the same way. *Evidence:* old:gamma/notes/2026-09-21-continuous-control.md.
  *Suggestive* (one seed).
- **A contrastive reachable-future exploration bonus rewards the action with
  the least distinguishable futures.** In GRL-108 it drove collapse to
  sleeping (2 of 3 seeds); leave-one-out attributed the collapse to that
  bonus. *Evidence:* old:grl/records/GRL-108.json. *Suggestive.*
- **Generic spread regularizers (VICReg-style) did not produce binding;
  pairwise constraints did.** *Evidence:* phwm record above, §5.7.
  *Suggestive,* no number extracted.
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
