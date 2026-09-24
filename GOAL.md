# Goal

Only the user changes this file.

## What we are building

An agent that learns, from its own pixels and actions, a world model it can
reuse: it discovers the things in its world and how they relate, predicts
what its actions will cause, remembers what matters, and uses all of that to
reach goals it was not trained on. Benchmarks proceed in increasing complexity.
Simple models should begin with simple benchmarks, i.e. MiniGrid. Progress
could look like `Mgrid -> Crafter -> Minecraft`.

Every experiment serves one or more properties below. A proposed change or a
change of direction must say which property it moves toward and why the
current approach cannot get there.

## Constraints on every solution

- **C1 Learned, not supplied.** Objects, relations, events, skills and goals
  are learned. The environment gives pixels and actions only; labels and
  simulator state are for the evaluator. A hand-built mechanism can be a
  baseline or a probe, never a result.
- **C2 One learner, one representation.** Perception, memory, prediction,
  goals and imagination use the same learned state, not separate models
  joined by adapters.
- **C3 Declared priors.** Every built-in assumption (architecture, curriculum,
  data policy) is listed in ARCHITECTURE.md.
- **C4 Behavioural use.** A capability counts only if it changes the
  predictions or decisions that need it. Being decodable by a probe is not
  enough.
- **C5 Transfer with retention.** Tests use new instances or combinations,
  say what changed and what stayed familiar, and check that earlier
  capabilities survive.

## Properties

Each has a one-line meaning, an example, and the shape of a test.

### Represent

- **P1 Grounded meaning.** A representation means what it lets the agent
  predict about experience and action.
  *Example:* "key" means what happens when you pick it up and bring it to a
  door, not a decodable label. *Test:* the prediction changes when the
  situation changes.
- **P2 Belief about a changing world.** Tell apart what was seen, what is
  remembered, and what is believed true now, including things out of view.
  *Example:* a switch pressed earlier and now out of sight still opens the
  door; night eventually turns to day. *Test:* two histories with the same
  current view but different pasts give different predictions.
- **P3 Relations and binding.** Separate roles from the things filling them,
  so a known relation applies to new participants.
  *Example:* "key opens door of the same colour" works for a colour never
  seen. *Test:* held-out colour or combination.
- **P4 Task-relevant abstraction.** Keep the distinctions that change
  outcomes, drop the rest. *Test:* performance holds when irrelevant
  appearance changes, fails when a relevant one is removed.
- **P5 Uncertainty and applicability.** Predictions carry calibrated
  confidence and a sense of when they apply. *Test:* calibration on held
  situations; confidence drops outside the training range.

### Predict and discover

- **P6 Action consequences.** Predict what each available action will cause,
  including rare interactions, delays, failure and harm. *Test:* for the same
  state, different actions get different, correct predictions.
- **P7 Learned units.** Discover the parts, objects and events worth tracking
  from continuity and interaction, not from a supplied vocabulary.
  *Test:* the learned units track things across frames and matter for P6.
- **P8 Composition.** Combine known pieces into predictions for new
  combinations and longer chains. *Test:* held-out compositions.
- **P9 Temporally extended actions.** Learn reusable skills (e.g. "fetch the
  key") together with what they achieve, how long they take and when they
  fail. *Test:* a skill works from different start positions and with
  different participants.

### Remember and revise

- **P10 Memory.** Keep particular experiences, retrieve the relevant ones
  for a new question, and consolidate repeated patterns. *Test:* retrieval
  helps a prediction that the current view alone cannot make.
- **P11 Revise without erasing.** Contradicting evidence can change a belief,
  split a concept or narrow a rule, while other skills survive.
  *Test:* change one rule of the world; the agent adapts and keeps the rest.

### Act toward goals

- **P12 Grounded goals and prerequisites.** Goals are expressed in the same
  learned terms as experience; the agent infers what must happen first.
  *Example:* "have 3 wood" differs from "collected wood once"; get the key
  before the door. *Test:* reaches an unrewarded subgoal because it enables
  the goal.
- **P13 Deliberation.** Imagine and compare candidate plans in the learned
  state, with backtracking; more thinking helps harder decisions.
  *Test:* planning in imagination beats acting greedily on new tasks.
- **P14 Execute and recover.** Carry out a plan with feedback; notice when
  effects did not happen and revise. *Test:* recovers after a perturbation.
- **P15 Informative exploration.** Seek experience that resolves the
  agent's own uncertainty, weighing time and risk. *Test:* learns a rare
  interaction faster than random exploration does.
- **P16 Learning from demonstration.** Infer goal and prerequisites from a
  few demonstrations, then do it independently in changed conditions.

### Across all of them

- **P17 Scalability.** Memory, retrieval and planning stay usable as
  experience grows; report their costs alongside capability.
- **P18 Integrated competence.** Component successes eventually show up in
  autonomous play from ordinary starts (Crafter score).

## Suggested order

Proposal, to be settled in the CHARTER ladder: P6 (consequences, including
rare interactions) → P2 (belief and memory of unseen events) → P3 (relation
transfer) → P8 (composition) → P9 (skills) → P12 (goals). Each rung tests
one property in the smallest world that can show it.

## Not assumed

Language input, pretrained semantic models, a symbolic program language, fixed
object slots, a supplied event or skill vocabulary, pixel reconstruction as the
definition of imagination.

## Avoid

Slot attention