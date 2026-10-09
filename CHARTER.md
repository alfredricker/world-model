# Charter

Changes to this file need the user's explicit agreement.

## Objective

See [GOAL.md](GOAL.md): the constraints (C1–C6) and properties (P1–P21) the
finished system must satisfy. This file governs how we move toward them.

## Current world

**MiniGrid's own environments, as three evaluation tiers** (decided with the
user 2026-10-05, replacing the chained rooms of 2026-09-24 as the standing
evaluation). Every card that changes the agent reports all three, on the
same environment seeds each time.

| Tier | Environment | Size | Task | How it is judged |
|---|---|---|---|---|
| 1 | `MiniGrid-DoorKey-8x8-v0` | 8 × 8 | key → door → goal square | **≥ 99% success** (pass/fail) |
| 2 | `MiniGrid-BlockedUnlockPickup-v0` | 11 × 6 | move the ball blocking the door → key → door → pick up the box | against the best version so far |
| 3 | `MiniGrid-ObstructedMaze-Full-v1` | 16 × 16 | keys hidden in boxes, balls blocking doors, locked doors across rooms; 3,600 steps | against the best version so far |

- **Tier 3 is v1** (the user, 2026-10-05): MiniGrid's fix of v0, whose
  layouts can have a ball covering a key and so be unsolvable.
- **Against the best version:** success on the same seeds, compared pair
  by pair: better or worse when the paired difference is beyond chance
  (McNemar's test, p < 0.05), otherwise no different; steps to success
  reported alongside. A change is kept only if tier 1 passes and neither
  tier 2 nor tier 3 is worse; the card's own question decides the rest.
  STATUS names the best version and its three scores.
- **The learner receives:** MiniGrid's 7 × 7 egocentric view with
  occlusion, as tile images, and its own previous action: turn left, turn
  right, forward, pick up, drop, toggle (MiniGrid's "done" is not used).
  The goal is the environment's success signal (episode end), or its
  mission given as a goal condition or example frames (GOAL C1).
- **The learner never receives:** simulator state, object positions or
  identities, event labels, rewards as supervision unless a card declares
  it, or the evaluator's probe labels.
- **Experience:** random play in each environment, collected as card 034's
  (declared per card); the encoder is shared across environments (C2).
- **Development worlds:** the key, switch, either and both worlds and the
  chained rooms stay for the ladder's tests and for diagnosis.

The next world (Crafter) is entered only when the ladder rungs that these
worlds can test have passed.

## Capability ladder

Each rung is one capability with one test. A rung is interpreted only after
every rung below it has passed.

| Rung | Property (GOAL.md) | Capability | Test that shows it | Status |
|---|---|---|---|---|
| 1 | P12, P6, P19 | Tell how close a goal is from any state, and how each action changes that, including through rare interactions | Goal-conditioned fork: at a development probe step, the learner is shown a goal condition (holding a key of a given colour, a given door open, a given box open) as a set of example frames from other episodes in which it holds. For each of the 5 actions, the evaluator computes the true fewest steps to the goal after that action by searching a copy of the simulator (unreachable counts as infinite; the learner never sees these numbers). The learner must rank the actions by it. Scored on interaction-decisive forks (the best action is a pickup or toggle) and movement-decisive forks that are decidable from the current view: some action is best in every reachable development state that shows the same egocentric view (decided 2026-09-26; the rest need memory and are reported, and become the memory rung's test). A pickup or toggle that changes nothing must not be ranked best. Two pass criteria, each at a threshold fixed by the card's feasibility gate: interaction-decisive top-1 above a goal-swapped control (the same model shown another goal), and the rank correlation between predicted and true steps-to-goal over fork states. Also report the score against the number of distinct interaction events in training (10, 30, 100, 300) | open |
| 2 | P12, P16 | Infer the conditions a goal needs | Subgoal choice: in a state where a condition of the goal is unmet, the learner ranks candidate conditions, each shown as example frames like a goal, by how much achieving them brings the goal closer; the evaluator's search gives the truth. Chains up to box → key → door. Scored above a goal-swapped control. The candidates are a test device; the learner never trains on them | sketch |
| 3 | P12 | Act to reach an unrewarded subgoal that enables a goal | Defined when rung 2 passes | sketch |
| 4 | P21 | Deliberate over chains of conditions | Defined when rung 3 passes | sketch |
| 5 | P2 | Remember an unseen event | Toggle at a switch door whose switch was pressed out of view: history model above a trained current-frame model by ≥ 0.1, n ≥ 30 | sketch |
| 6 | P3 | Apply a relation to new participants | Locked door with matching key, on `transfer_colour` (colours never seen) | sketch |
| 7 | P8 | Compose known consequences into longer chains | Held-out key → door → switch sequences | sketch |
| 8 | P9 | Learn a reusable skill | Defined when rung 7 passes | sketch |

"Sketch" rungs get their exact test and thresholds in a card when they are
reached, not before.

The order follows GOAL.md's main insight (decided 2026-09-26): how close a
goal is, then which conditions it needs, then acting on those conditions as
subgoals, then deliberating over them.

**Theory first** (agreed with the user 2026-09-26). The theory of
conditions ([card 003](experiments/003-conditions-theory-check/card.md),
section 2) is checked with exact computation in a small key-and-door world
before architectures are compared: a condition is what an achieving action
needs, found by contrasting successes with failed attempts, and each
condition becomes a subgoal. Rung 1's test and the current world will be
redefined from that card's result; the architecture screen waits for it.

**Architecture selection** (decided 2026-09-25). Before rung 1, while there
is no model (arch_version 0), the first architecture is chosen by one
bounded screen: at most five whole candidate architectures, one design card,
the same data and the same rung-1 feasibility screen for all. Runs are as
long as an informative result needs (decided 2026-09-26), set by a
throughput profile in the card; runs over 30 minutes are handed to the
user. The winner becomes arch_version 1 and then takes rung 1 properly;
the losers and their scores stay in the card so the choice can be
revisited with evidence. The screen may have one follow-up round of at most
three combinations of the first round's parts, each justified by first-round
diagnostics. This is the only exception to "one component per experiment",
and it applies once.

## Current direction

Agreed with the user between 2026-09-28 and 2026-09-29. New cards build on
these; rule 8 guards against sliding back.

- **Effects of actions.** The primitive is how an action changes
  conditions, learned from the agent's own outcomes. Moves are actions
  too; walking is move effects in the same learner. Nothing is read from
  the simulator.
- **One encoder.** A new comparison or readout reads, and if needed
  trains, the existing encoder that feeds the codebooks. Networks inside
  a component (an MLP as recall's metric, for example) are used where
  they improve on the current mechanism in a test; what is ruled out is a
  second representation beside the encoder's, such as a separate encoder
  or output head (C2; the user, 2026-10-06).
- **One latent space** (agreed 2026-09-29, after card 034). Codes,
  recall, planner conditions, relations and later fast weights all read
  the same encoder vectors. Codes quantise them. Recall is a learned
  metric on them, one per action. A condition is a region of them: a
  code's cell, or what recall treats as alike for an action. A relation is
  a constraint between two things' vectors in one part, such as sharing a
  code there. There is no second output head and no second family of
  codebooks, since either would be a second representation (C2; P12's
  goals "in the same terms as experience"). When objectives conflict
  inside the space, the answer is to organise it into parts, with what
  things share apart from what varies among them, not to add a space.
- **Codes are names, and kinds are distributed.** Tiles, later segments,
  become vectors, then several codebooks. What a code means is never
  engineered. A kind is not a label: each rule reads only the codebooks it
  needs, and the tiles sharing codes there act as a kind for that action.
  New codes for a new thing are fine.
- **Recall replaces counting.** Experiences are kept, not reduced to
  tallies or merged into hard kinds, so that old experience can be re-read
  when new structure appears. Similarity is graded, specific to the action,
  and learned. Trying confirms or corrects what recall infers.
- **Recall and networks, each where it predicts better** (the user,
  2026-10-06; replaces "weights take over from recall"). Recall over
  stored experience and trained networks, including networks that reason
  over recalled experience, are judged by one general rule. In a given
  situation, the answer comes from whichever has better predicted the
  agent's own held-out experience in situations like it, measured the
  same way for every action and world. Where neither has, the agent
  tries. No feature, grouping or arbiter is designed for one
  environment: inputs come from the encoder's vectors and the agent's
  own predictions, and a mechanism must work for every action without
  changes made for one.
- **Similarity and relations come from what things do.** Relations such
  as "this key fits this door" sit on top of the codes, as codes or in
  weights, rather than being forced into the encoder's numbers (card 033).
- **Properties are learned action effects** (the user, 2026-10-08). A
  tile's property is what an action does to it: walkable (moving forward
  steps onto it without being blocked), pickupable (pick up takes it),
  unlockable (toggle changes it). Each is learned from the agent's own
  tries, improves with experience (P19), carries over to tiles that look
  different but are the same kind (not a table of exact tiles), and says
  "unknown" where no experience is like the tile, which leads to trying.
  A property says what a tile can do in some situation; conditions say
  when it holds now (a locked door becomes unlockable once the matching
  key is held). A network on the encoder's vectors may learn them where
  it predicts better than recall (bullet above); its training data then
  varies appearance, so a property cannot be learned as per-colour facts.
- **Patterns are discovered** (the user, 2026-10-08). The agent finds by
  reflecting on its experience which attribute governs a relation, as a
  person sees from the table "red key opens red door, blue key opens blue
  door" that the rule is about colour: the attribute is the one that
  explains stored tries most simply, chosen from candidates the agent
  forms itself, not named in advance. Card 070's relation, trained to
  match colours, is the current exception until a card replaces it.
  Discovery needs experience in which the pattern varies (the user,
  2026-10-08): an attribute becomes a dimension the agent can compare
  across kinds only if its values are many and never repeat (else a
  table fits as well) and vary independently of kind (the same values
  on many kinds, many values on each). Where the world does not supply
  this, a declared curriculum does (C3); cards 069 and 070. Variety
  makes the dimension within each kind; lining it up across kinds took
  the outcome that depends on it (cards 088–090): what properties
  ignore finds the attribute, outcomes say which values correspond.
  Colour transfer in MiniGrid is paused (the user, 2026-10-08): new
  colours are learned within a few tries, and the principle is taken up
  again in a richer world (Crafter), not tuned to MiniGrid's colours.
- **Transfer is judged by tries.** How quickly a new thing is mapped onto
  known ones matters more than getting it right at first sight.
- **Goals are any condition** over codes, positions and relations.

## Rules

Agreed with the user 2026-09-24.

0. **Every proposal names its property.** A card, a component change or a
   change of direction states which GOAL.md property it serves and why the
   current approach cannot reach it.
1. **Switching direction.** Allowed only when a rung fails after its
   feasibility gate passed (an upper-bound fit showed the test is passable and
   the learner still failed), with a one-page post-mortem card and the user's
   sign-off. The session after a switch is thinking only: no code, no runs.
2. **Replacing a component.** Allowed only when an ablation or a component
   test places the failure in that component. Only that component changes.
3. **Components before integration** (decided 2026-09-24). Each component
   gets its own small test before it is combined with others: in isolation
   where that is informative, otherwise attached to parts that have already
   passed (rule 7).
4. **Documents.** Only those listed in AGENTS.md. Cards have a length cap.
5. **Runs.** No long run without a card the user has read and approved.
6. **Success criteria declared first.** Before a run, the card states the
   metrics, the threshold for each, what it is compared against, and why
   those numbers would show the capability. They are not changed after the
   run starts; a second change of criteria means a new card.
7. **Build from primitives; no assumed dependencies.** A component test is
   valid only if everything it relies on is either (a) already built here and
   passed its own test, or (b) an established method with evidence in a
   setting like ours, cited in LITERATURE.md. If a dependency is neither (for
   example object-bound latents with no standard method for this
   architecture), that dependency becomes the next card first; the test that
   needs it waits.
   If a component's behaviour only means something inside the larger system,
   test it attached to the parts that have already passed, not in isolation
   and not with a stand-in. Oracle or label-driven stand-ins may show a test
   is passable (a feasibility gate) but never show the component works.
8. **No regressions** (agreed with the user 2026-09-29). A new card keeps
   every point of "Current direction". Where it still relies on something
   the direction replaces (for example the counted model while recall is
   not yet in the planner), section 2 lists that as a declared exception,
   with the reason and the card that will remove it. It must not
   reintroduce a replaced mechanism as a new part: merging into hard kinds,
   counting in place of recall, a side network, or engineered code
   meanings. Changing "Current direction" itself needs the user's explicit
   agreement.
