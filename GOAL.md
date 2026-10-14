# Goal

Only the user changes this file.

## What we are building

An agent that learns, from its own pixels and actions, a world model it can
reuse: it discovers the things in its world and how they relate, predicts
what its actions will cause, remembers what matters, and uses all of that to
reach goals it was not trained on. Benchmarks proceed in increasing complexity.
Simple models should begin with simple benchmarks, i.e. MiniGrid. Progress
could look like `Mgrid -> Crafter -> Minecraft`.

The main insight of the model is P12. The agent should work towards satisfying
latent goals through a hierarchy. First, it should be able to infer proximity to 
completing a given goal based on its state. When a goal is achieved, either by
demonstration or exploration, the agent should be able to infer the *conditions* that
allowed for successful completion. These conditions themselves are sub goals
of a higher order goal, as they must be met in order to achieve the higher order
goal. This hierarchy extends down to primitive actions, and should allow for 
System 1 thinking (P21).

Every experiment serves one or more properties below. A proposed change or a
change of direction must say which property it moves toward and why the
current approach cannot get there.

## Constraints on every solution

- **C1 Learned, not supplied.** Objects, relations, events, skills and goals
  are learned. The environment gives pixels, actions and the agent's own
  pose; labels and all other simulator state are for the evaluator. The
  pose is proprioception: where the agent's body is and how it moved, as a
  robot's odometry gives it. It says nothing about the world (the user,
  2026-10-13: "Let's do proprioception", "and amend C1"). A declared,
  pretrained perception front end with no language or class labels may
  supply objects' outlines and their correspondence across frames; what
  objects are, what they do and what they mean is learned (the user,
  2026-10-13: the agent need not be learned end to end). Any other hand-built mechanism can be a
  baseline or a probe, never a result. Until P20, goals may be supplied as
  tasks: a target condition shown as example observations or a success
  signal. What a goal requires and how to reach it are always learned.
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
- **C6 Predict abstract consequences, on the horizon they unfold.** What the
  model predicts is how actions change the abstract variables that matter
  (holding an iron pickaxe, a door being open, a skeleton in range), and how
  those variables change which goals can be reached, how likely and how
  soon. There is no fixed prediction window: how long a change takes is
  itself predicted, a few steps for an arrow, hundreds for a crafting chain.
  Frames are the evidence these variables are grounded in, not the thing
  being predicted. Which variables matter is learned (C1, P4). A consequence test scores that effect, not the next frame; a fixed horizon
  in a test is a measuring device, not the target.

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
  outcomes, drop the rest: a variable matters if changing it changes what
  happens later or what can be reached. *Test:* performance holds when irrelevant
  appearance changes, fails when a relevant one is removed.
- **P5 Uncertainty and applicability.** Predictions carry calibrated
  confidence and a sense of when they apply. *Test:* calibration on held
  situations; confidence drops outside the training range.

### Predict and discover

- **P6 Action consequences.** Predict what each available action, or
  sequence of actions, will cause in the abstract state (C6), including rare
  interactions, delays, failure and harm.
  *Example:* crafting an iron pickaxe is progress toward diamonds, although
  nothing in the next frame says so; a skeleton drawing its bow means harm a
  few steps away, not in the next frame. *Test:* for the same state,
  different actions get different, correct predictions of the effect at the
  horizon where it shows. A next-frame score alone does not count.
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

- **P19 Learn important consequences from few examples.** An event that is
  surprising and whose effect persists is learned from a handful of
  occurrences, not only after many repetitions. "Important" is judged
  without labels: how surprising the outcome was, whether its effect lasts,
  whether it changes what the agent predicts afterwards, and whether it
  changes how likely or how soon a goal can be reached (it met or broke a
  condition, P12).
  *Example:* after opening a locked door a few times, the agent predicts
  it for a new door. *Test:* accuracy against the number of distinct
  occurrences in training (for example 10, 30, 100, 300).
  Every experiment on a rung that serves P19 reports this curve.

### Act toward goals

- **P12 Goals as conditions, subgoals from conditions.** A goal is a
  condition in the learned state (holding diamonds), expressed in the same
  terms as experience. The agent knows which conditions a goal needs, and
  evaluates them from its current state and its memory. It does not
  forecast when the goal will be reached (the user, 2026-10-14, replacing
  "how likely and how soon").
  - **Statements.** What the agent knows about conditions takes the form of
    statements: an effect, the conditions it needs, and its spoilers.
    A spoiler is an effect that ends the goal or one of its conditions, and
    it has conditions of its own. Making an iron pickaxe needs a table and a
    furnace nearby and wood, coal and iron in hand. A parent stops a child
    reaching for cheese when the parent is awake and sees or hears the
    child. When a goal is reached, by exploration or in a demonstration
    (P16), the agent infers which conditions made that possible, by
    contrast with its failures.
  - **Evaluation.** For each condition a plan needs, the agent forms a
    belief that it holds:
    - from what it sees now;
    - else from memory, carried forward by what it has learned about how
      that kind of fact changes while unseen (sleepers stay asleep unless
      woken; a mined diamond never comes back);
    - else from a learned prior, given what it can see (deep caves
      sometimes hold diamonds).

    The unmet or doubtful conditions that matter most become subgoals, and
    so on down to actions. Alternatives are compared by these beliefs and
    by their cost.
  - **Two levels of effect model.** System 1's model predicts every token's
    next state at every step: a cow's step, an arrow's flight, the agent's
    own motion. System 2's statements are abstract and selective. They are
    read from that model and from memory. A condition is evaluated only when
    a plan needs it or a surprise forces it, since evaluating it from
    memory is expensive. Both levels use the same learned state (C2).
  - **Not supplied.** Conditions are not supplied as forms. A condition is
    a fact about the learned state that the agent's own model of effects,
    or its record of successes and failures, is found to depend on.
  - **Goals are re-weighed as the situation changes.** A threat that could
    end every goal makes survival the immediate goal, while the agent tries
    to keep the conditions it has built up. Survival need not be supplied:
    it follows from death making every goal unreachable.

  *Example:*
  - Wanting diamonds makes an iron pickaxe a subgoal, which makes iron ore
    a subgoal. A cave already searched is believed empty, so the agent
    goes where it has not been. A skeleton in range interrupts all of them.
  - A child who wants cheese at night knows the parents stop it when they
    see or hear it. It believes they are asleep, so it keeps them asleep
    by avoiding the stair that creaks.

  *Test:* for a goal it was not trained on, the agent:
  - names the conditions the goal needs, checked against the world's
    rules;
  - holds beliefs in them that are calibrated against the truth, across
    states and histories;
  - chooses differently after two histories with the same view, where
    memory decides;
  - pursues an unrewarded condition because the goal needs it;
  - avoids a threat and then resumes.
- **P14 Execute and recover.** Carry out a plan with feedback; notice when
  effects did not happen and revise. *Test:* recovers after a perturbation.
- **P15 Informative exploration.** Seek experience that resolves the
  agent's own uncertainty, weighing time and risk. *Test:* learns a rare
  interaction faster than random exploration does.
- **P16 Learning from demonstration.** Infer goal and prerequisites from a
  few demonstrations, then do it independently in changed conditions. Form an understanding of what conditions were required for the demonstration to succeed.
- **P20 Self-set goals.** The agent proposes its own goals and subgoals from
  its learned state, beyond those it is given. Early on the experimenter
  supplies goals (C1); later the agent chooses conditions worth achieving,
  for example ones it cannot yet reach reliably, or ones that open up many
  other conditions. *Test:* practice on self-chosen goals makes the agent
  better at supplied goals it has never seen, compared with practice on
  random or supplied goals only.
- **P21 System 1 and System 2.** Simple decisions, and ones that need an
  immediate response, go through a fast System 1: the network maps state
  and goal straight to an action. Decisions that need long chains go
  through a slower System 2 that weighs conditions, memory and other
  relevant factors. System 2 deliberates over chains of conditions
  ("condition A enables condition B, which enables the goal"), comparing
  candidate chains and backtracking when one fails, not by simulating many
  latent steps; this keeps its compute reasonable. Chains are searched
  from both ends. Backward from the goal finds which conditions matter.
  Forward from the present, one imagined effect at a time, checks whether
  they hold together: things used up, side effects, and the world's own
  changes. The hierarchy of
  conditions (P12) is what lets the lower levels run as System 1.
  *Example:* dodging an arrow is System 1; planning the route to diamonds
  is System 2. *Test:* on new tasks, deliberating over conditions beats
  acting greedily, more deliberation helps harder decisions, and its
  compute is reported (P17).

### Across all of them

- **P17 Scalability.** Memory, retrieval and planning stay usable as
  experience grows; report their costs alongside capability.
- **P18 Integrated competence.** Component successes eventually show up in
  autonomous play from ordinary starts (Crafter score).

## Suggested order

Proposal, to be settled in the CHARTER ladder: P6 with the predictive side
of P12 (how likely and how soon a goal is from each state, and how each
action, including rare interactions, changes that) → P12 conditions (infer,
from reached goals and demonstrations, which conditions made them possible;
P16) → P12 acting (pursue unmet conditions as subgoals, down to actions) →
P21 (deliberate over chains of conditions) → P2 (belief and memory of
unseen events) → P3 (relation transfer) → P8 (composition) → P9 (skills) →
P20 (self-set goals). Each rung tests one property in the smallest world
that can show it.

## Not assumed

Language input, pretrained semantic models, a symbolic program language, a
supplied event, condition or skill vocabulary, pixel reconstruction as the
definition of imagination, next-frame (t+1) prediction as the definition of a
consequence, a fixed prediction horizon.
