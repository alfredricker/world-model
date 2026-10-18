# Charter

Changes to this file need the user's explicit agreement.

## Objective

See [GOAL.md](GOAL.md): the constraints (C1–C6) and properties (P1–P21) the
finished system must satisfy. This file governs how we move toward them.

## Current world

**Craftium** (the user, 2026-10-13: "I like the idea of craftium as the
main world"). Cards are judged in Craftium (Malagón et al. 2025): the
Luanti engine (formerly Minetest) running VoxeLibre, a Minecraft clone.
It is GOAL's "Minecraft" step, a 3D world between Crafter and robotics.

- **The learner receives:**
  - the rendered frame (640 × 360, field of view 90° vertical);
  - its own pose: position in blocks, yaw and pitch (proprioception,
    GOAL C1);
  - its previous action, from Craftium's discrete list: forward, back,
    left, right, jump, sneak, dig, place, choose slot 1–5, and turning by
    the mouse left, right, up and down.
- **The learner never receives** the game's state: creature positions or
  identities, the voxel map, or event labels. Rewards are not supervision
  unless a card declares it. The probe mod's log and the voxel map are for
  scoring only (card 107).
- **Settings:** lockstep, one game frame per step, as in card 107, with
  Craftium's other defaults, including its physics multiplier (×20). A card
  declares any setting it changes.
- **Crafter becomes a development world,** as MiniGrid's worlds did. Its
  cards stand as evidence, and brain parts may be diagnosed there.
  - **Brain parts may be built on Crafter while Craftium lacks their
    inputs** (the user, 2026-10-13). Each such card names the Craftium card
    that follows it, which is its portability test.

**MiniGrid's own environments, as three evaluation tiers** (decided with the
user 2026-10-05, replacing the chained rooms of 2026-09-24 as the standing
evaluation). *Amended by the user, 2026-10-11:* cards were judged on
Crafter, and from 2026-10-13 on Craftium (above). The three tiers are run only when a card has decided keep on a
significant architecture change (a new architecture version), before
that version is adopted, on the same environment seeds each time.

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
  reported alongside. A kept change is adopted only if tier 1 passes and
  neither tier 2 nor tier 3 is worse; the card's own question decides
  the rest.
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
worlds can test have passed. *Amended by the user, 2026-10-10:* work moves
to Crafter now, with rung 1 still open; and on 2026-10-13 to Craftium.

## Capability ladder

Each rung is one capability with one test. A rung is interpreted only after
every rung below it has passed.

| Rung | Property (GOAL.md) | Capability | Test that shows it | Status |
|---|---|---|---|---|
| 1 | P12, P6, P19 | Know the conditions of effects and goals | **Effects:** at each held-out step where an effect occurs (any action's, or the world's own; a spoiler is an effect like any other), the learner names the tokens the effect depends on. **Goals:** shown a goal as example frames (GOAL C1), it names the effects that make the goal hold and their conditions. The truth is the world's own rules, written out for scoring only (as in card 105.4), and is asked only of states the world can produce. Pass: precision and recall at least 0.9 for every kind of effect with at least 100 held-out events, and for every goal, each above the baseline "every token near the change". Both are also reported against the number of distinct occurrences in training (10, 30, 100, 300; P19). Judged in Craftium; in Crafter while Craftium lacks the tokens, each card naming its Craftium follow-up. Replaces the steps-to-goal fork (the user, 2026-10-16), since P12 no longer asks how soon a goal is reached | open: in Crafter, one-step makes and moves pass (card 106: precision 0.99–1.0, recall 0.95–1.0); goals, other effects and Craftium not yet tested |
| 2 | P12, P2, P10 | Evaluate conditions from state and memory | For a condition a goal needs, about a place out of view, the learner forms a belief that it holds: from the view, else from memory carried forward by how such facts change while unseen, else from a prior given what is remembered nearby. Scored as card 109 scores it: the log-likelihood of the truth above a view-only and a static-memory baseline and within a margin of an upper bound; the place chosen is the right one; and in paired histories with the same view, the choice follows memory. This absorbs the former rung 5, an event remembered out of view (the user, 2026-10-16). Thresholds are fixed by the card's feasibility gate (109: 0.15 nats, 90%, 90%) | open: card 109 passed choice (95–99%) and memory (98–100%), and missed the bound by 0.21 nats against 0.15 |
| 3 | P9, P12 | Learn reusable skills and when they fail | A skill runs toward a target condition about a target token (approach it, mine it, dodge it) and ends when the condition holds, when it stalls or when a spoiler fires (Roadmap step 4). It must work from new start positions and with new participants, and the learner must know the conditions of its success, found by contrasting its successes with its failures (Roadmap step 5). Moved from rung 8 (the user, 2026-10-16), since rungs 4 and 5 act and plan through skills | sketch |
| 4 | P12 | Act to reach an unrewarded subgoal that enables a goal | Defined when rung 3 passes | sketch |
| 5 | P21 | Deliberate over chains of conditions | Defined when rung 4 passes | sketch |
| 6 | P3 | Apply a relation to new participants | Locked door with matching key, on `transfer_colour` (colours never seen) | sketch |
| 7 | P8 | Compose known consequences into longer chains | Held-out key → door → switch sequences | sketch |

"Sketch" rungs get their exact test and thresholds in a card when they are
reached, not before.

The order follows GOAL.md's main insight (decided 2026-09-26; rungs 1–3
changed by the user 2026-10-16, after P12 was replaced): which conditions
effects and goals need, then evaluating them from state and memory, then
skills that reach conditions, then acting on those conditions as subgoals,
then deliberating over them.

**Theory first** (agreed with the user 2026-09-26). The theory of
conditions ([card 003](experiments/archive/003-conditions-theory-check/card.md),
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

- **One transition model over vector states** (the user, 2026-10-13:
  "INTEGRATED LEARNING!" and "it should be as simple as latent
  transitions of vector states. There is no 'choice'"). This widens
  "effects of actions" (2026-09-28).
  - **The state.** A set of token vectors, as perception gives them.
    Appearance and position live in the same vector.
  - **A step's change.** The change of those vectors, including tokens
    that appear or vanish.
  - **What it covers.** Every action, moves included, and the world's own
    changes when the agent does nothing, all in one model.
  - **No vocabulary of events.** Moving, opening, being mined and being
    blocked are names we give to regions of that change. The model does
    not choose among them, and no list of kinds of change is written:
    possible events are learned from experience. What the agent carries
    or holds is whatever tokens perception shows, not a declared slot.
  - **Relations are learned.** Which tokens a change depends on is
    learned. No place such as "the front" or "the hand" is declared, and
    nothing is read from the simulator.
  - **Version 27 drifted from this.**
    - Moves kept their own models (card 095.4 was closed).
    - Arm C read only the front and the hand, in four outcome categories.
    - Its results were chosen among keep, copy, shift and set.
    All three are special cases of one residual, attention-reading
    prediction of the next vector.
- **A general form may cost some loss at first** (the user, 2026-10-13).
  - **Why.** Hand-engineered priors are hard to match in the world they
    were built for, but they fight scalability and slow every step up
    the ladder.
  - **Rule.** A change toward a unified, general mechanism is kept with
    a stated loss against the special case it replaces (AGENTS.md gives
    the default margin for MiniGrid's tiers).
  - **Not this.** A card must not narrow a general form back into a
    special case because the special case scores higher in one world.
- **Perception may have several parts; prediction has one** (the user,
  2026-10-13). Declared front ends may supply tokens and their
  correspondence across frames (GOAL C1): a tracker, a segmenter, later
  depth. Everything they produce enters one token state, which one effect
  model and one planner read.
  - **Perception supplies "where"** (the user, 2026-10-13). It gives tokens
    at positions in a world frame, placed by the agent's own pose
    (proprioception, GOAL C1).
    - **Per world:** in Craftium, the position, yaw and pitch the game
      reports, with learned depth along each pixel's ray (card 107); in
      Crafter, the pose the game reports for the agent's body; in
      ARC-AGI-3, the screen's own coordinates; in robotics, odometry, with
      learned depth for where things are.
    - **Not the brain's job.** The brain reads positions only as
      differences and never supplies perception's frame. Its predictions
      are not fed back to place the view.
    - **Belief supplies "which one"** (the user, 2026-10-13).
      Perception's correspondence holds while a token stays in view. When
      a token reappears, or perception's link is unsure, it is matched to
      a remembered token: the effect model predicts each remembered
      token's next state, and new tokens are matched to those predictions
      in the same latent space (predict, match, correct). This decides
      only which remembered token a new one continues. It never places
      the view and never overwrites what perception saw. Which tokens are
      remembered, and how they may move while unseen, are learned, not
      declared.
    - **Priority** (the user, 2026-10-14, after card 108 stopped): telling
      look-alikes apart is deferred. Belief's first job is to evaluate the
      conditions a plan needs (System 2, below).
- **Two levels of effect model; conditions evaluated from state and
  memory** (the user, 2026-10-14, with GOAL P12 replaced; this widens
  "conditions come from the effect model", 2026-10-13).
  - **System 1** is the one transition model above. It predicts every
    token's next state at every step, for every action and the world's own
    changes. It is cheap, and it is always on.
  - **System 2 holds statements:** an effect, the conditions it needs, and
    its spoilers. A spoiler is an effect that ends the goal or one of its
    conditions, and it has conditions of its own.
    - **Where statements come from** (the user, 2026-10-17, replacing
      imagined alteration of System 1's input, cards 105.4–110). A
      condition is a fact about tokens that an effect is found to depend
      on, in three ways:
      - **deduction from memory:** what remembered successes of the effect
        share and the failed tries lack;
      - **experiment:** where facts always occurred together in memory, or
        memory holds too few cases, the agent changes one in the world and
        watches (P15);
      - **others:** a demonstration adds successes (P16).

      Imagined alteration (replacing a token in System 1's input) is not a
      source. It returns only what the model already learned, and it asks
      about states the world cannot produce (card 110). Which remembered
      changes count as the same effect is read from the learned state.
      Statements are stored and merged in sleep (card 106). They are not
      supplied forms.
    - **Evaluation, not forecasting.** System 2 does not predict when a
      goal will be reached. For each condition a plan needs, it forms a
      belief that the condition holds:
      - seen now;
      - else remembered, carried forward by what has been learned about
        how that kind of fact changes while unseen;
      - else a learned prior, given what can be seen.
    - **Selective.** Evaluating from memory is expensive. So a condition is
      evaluated only when the plan's chain asks for it, or a surprise
      forces it.
  - **Chains.** An effect makes facts true, and those facts are conditions
    of other effects. Chains of conditions are paths through this graph.
  - **Searched from both ends.**
    - Backward from the goal finds which conditions matter. These include
      the conditions that keep a spoiler from firing.
    - Forward from the present, one imagined effect at a time (not frame
      by frame), checks that a chain holds together. It looks for things
      used up, side effects that undo a condition (the stair that wakes
      the parents), and the world's own changes in the meantime. A break
      found forward becomes a new condition.
  - **One representation.** Statements and beliefs are over the same
    token vectors as System 1 (C2). They are not a second model of the
    world.
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

## Roadmap

The model's trajectory: the parts the brain needs, in the order cards
build them (the user, 2026-10-14). Every card names the step it serves.
Between sessions, the next card comes from the first step that is not
built. Steps and their order change only with the user. Perception's
adapter for Craftium (tokens and their places from the pose) runs in
parallel; its cards name no step.

| Step | Part | What it is | Built so far | Status |
|---|---|---|---|---|
| 1 | System 1, one step | One transition model over token vectors, for every action and the world's own changes | Series 105, with the view placed by the pose (105.10) | built in Crafter; "do" changes 85–90% |
| 2 | Conditions of one-step effects | Facts that remembered successes of an effect share and failed tries lack, separated by experiment where they always occurred together | By imagined alteration (105.4, 106; replaced 2026-10-17). By contrast over memory (111–111.2): finds what memory holds evidence for (0.94–1.0), and closes only complete answers; the open ones are left to experiment | memory part built, closing by co-changes (111.3); opportunistic experiment stopped (112) |
| 3 | Belief over conditions | b(condition \| state, memory) for what is out of view: seen now, else remembered and carried forward by how such facts change, else a prior from what is remembered nearby. Recall and a memory network compared by the rule for recall and networks. Trained by predicting what is seen when a place next comes into view | Card 109 (revise): a network over the remembered places around the query, within 0.04 nats of the bound on remembered places; never-seen places queried from afar missed | in progress |
| 4 | Skills (options) in System 1 | A skill runs toward a target condition about a target token and ends when it holds, or when it stalls or a spoiler fires: approaching a token, mining it, dodging an arrow. Targets are the conditions statements name. Policies are learned from the agent's own play by hindsight. Primitive actions are skills of one step | 097.1 learned walking in MiniGrid and stopped | not started |
| 5 | Statements over skills | ⟨skill and its target token, conditions at the start, effect, spoilers, reliability⟩. Conditions are regions of the encoder's latent space, for the target token and for tokens related to it. They come from contrasting a skill's successes with its failures, and from experiments. Reliabilities come from recall or a network | One-step statements as exact codes and offsets (105.5, 106) | not started |
| 6 | Planning over statements | Backward: the subgoal is the condition whose belief, set true, most raises the plan's evaluation per cost. Forward, one statement at a time: side effects, the world's own changes and spoilers; a break becomes a maintenance condition | 105.5 (world 1, one-step statements) | partial |
| 7 | Tests with the shape of GOAL P12's examples | Sleeping safely at night in Crafter, then Craftium: a spoiler from another agent, a condition kept over an interval, alternatives compared by belief | | not started |

The planner's evaluation of a plan is a belief, not a time: the product
of its statements' reliabilities, the beliefs in the conditions it relies
on, and the beliefs that no spoiler fires while a condition is held. The
products assume independence, as a first approximation.

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
   Every run, smoke tests included, is capped so that it cannot crash the
   machine (the user, 2026-10-12, after an uncapped run used all memory;
   the machine has 60 GB and no swap). Launch it as
   `timeout <limit> systemd-run --user --scope -q -p MemoryMax=16G
   -p MemorySwapMax=0 -p CPUQuota=800% nice -n 10 bin/prun python …`
   with `OMP_NUM_THREADS=4`. Then a blow-up kills only the run. Before
   scaling a smoke test up, check that no array grows with the square of
   a count the change could enlarge.
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
9. **Portability** (agreed with the user 2026-10-11). Every card's
   section 2 has a portability check. It lists each assumption the
   mechanism makes about the environment (a place or slot it names, a
   grid, objects of one tile, a front, fixed action ids, a view that
   follows the agent, a world that changes only by the agent). For each,
   it says whether the assumption holds in Craftium, Crafter, ARC-AGI-3
   and robotics. An assumption that does not hold everywhere is an
   environment-specific prior: it is removed, or declared with the card
   that will remove it. Narrowing a mechanism because nothing is lost in
   the current environment is such a prior, since one environment cannot
   show the loss. A narrowing for speed is kept only as a cache checked
   against the general form. A new environment should change only the
   adapter: frames to tokens, the list of actions and the goal. Anything
   more is recorded as a finding.
   - **Perception may be specific to an environment; the brain may not**
     (the user, 2026-10-13, after card 104.1: "we might need crafter
     specific perception. As long as the 'brain' of the model is fully
     portable: belief over tokens, recall of tokens, and planning of tokens
     in the latent space, then we will have succeeded").
   - **What this means for a card.** Perception's assumptions are declared
     in the adapter and need no card to remove them. The brain's
     assumptions must hold in every environment. The brain is belief,
     recall, the transition model and planning, all over tokens.
10. **Breaking worlds** (agreed with the user 2026-10-11). These are
    small worlds, each built to defeat one environment-specific
    assumption:
    - a condition between tokens away from the agent;
    - an object of many tiles;
    - a token that leaves view and comes back moved;
    - a body in a fixed view, or no body;
    - an action aimed at a place;
    - a world that changes by itself.

    They come from one engine (`tools/card102/worlds.py`), give pixels
    and actions only, and are drawn so that Crafter's adapter reads them.
    Card series 102 builds them. A card that changes the architecture
    runs every breaking world built so far as a smoke test, alongside
    MiniGrid's tiers, before its version is adopted. A world that a
    change breaks again is a regression under rule 8.
