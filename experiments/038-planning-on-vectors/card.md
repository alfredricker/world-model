---
id: "038"
title: planning on vectors
rung: 0
serves: [P8, P10, P12, P19, P11, C2, C5]
status: approved
verdict:
arch_version: 5
date: 2026-09-30
---

# 038: planning on vectors

## 1. Question

Suppose the planner holds what it sees as encoder vectors and never labels
them. Recall predicts every effect and answers every "is this like that?"
question, and working backward retrieves past tries by what they achieved.
Does the planner then act on keys and open doors of a colour never seen,
with nothing lost? And does the encoder still need its codebooks? Serves:
- P8, predictions composed for new combinations;
- P10, retrieving the relevant experience;
- P12, subgoals from conditions;
- P19 and P11, correcting a wrong prediction from a few tries;
- C2, one representation;
- C5, nothing lost.

[Card 037](../037-recall-in-the-planner/card.md) placed its failure in the
planner's labels. Recall said the agent can step onto the new open door in
9 of 9 seeds, but a label built from familiar codes named the right tile in
only 3 of 9.

## 2. What changes

The card changes two components, agreed with the user on 2026-09-29/30 as
one card with separate arms: the planner's representation, and the
encoder's codebooks. Three revisions followed shakedowns, each agreed with
the user on 2026-09-30 (below; evidence in Appendix C).

| | Planner of cards 028–037 | Card 038 |
|---|---|---|
| A place holds | a label (tile ID or code tuple) | the encoder's vector |
| Whether a need is met | the labels are equal | recall's prediction (revision 1) |
| Whether an action does something | a table entry per label | recall's vote over tries keyed on the thing in front (and, for pick up, toggle and drop, the thing held and what is in view) |
| What results | a label from the table | the neighbours' change carried onto the vector, part by part |
| Working backward | operators read from the table | conditions inferred from memory (revision 3) |
| Memory while acting | fixed | every try added (online learning) |

- **Carrying a change over.** Each changed part is handled in one of four
  ways:
  - kept;
  - copied from the other changed place (a picked-up key goes to the hand);
  - shifted by the neighbours' weighted mean change;
  - set to their weighted mean result.

  Leave-one-out over the stored things of the kind picks one way per part.
  Weights and priors are card 037's recall (a learned λ per action, prior
  1/4, an outcome needing probability ≥ 1/2), with a key's own tries
  joined to that vote as in revision 2.
- **Conditions** are predictions, checked by recall: "the episode ends";
  "the thing at place j can be walked onto"; "facing"; and part
  conditions (revision 3).
- **Online learning.** Each real try joins recall's memory, and that
  kind's cached predictions are dropped. λ and α are not refitted. Memory
  carries across the roughly 12 layouts each worker plays in turn.

**Revisions** (2026-09-30, each agreed with the user after a shakedown):
1. **Needs by prediction.** The "like" test (k ≥ 1/2 under the asking
   action's λ) is removed. Needs are checked by recall's prediction, and
   what is in view joins recall's key for pick up, toggle and drop.
2. **Recall's weighting.** The other keys' vote q (card 037's, with its
   prior) becomes a prior of strength α on a key's own tries: P_c =
   (n_c + α q_c) / (n + α), a hierarchical Dirichlet smoothing (MacKay and
   Peto 1995). α is fitted per world and kind by leave-one-try-out
   likelihood over the stored tries (grid of log α from −9 to 9). What
   results is carried over with the same weights. A key with no tries of
   its own is predicted exactly as before; card 037's rule is the special
   case α = W + 1 (W, the neighbours' weight). Draw and undraw are
   unchanged.
3. **Working backward by conditions from memory.** For an unmet "doing a
   on u makes c true", the (held, view) pairs of recall's stored tries of
   that kind on things like u are split into those with which recall
   predicts c and those without. The parts of the key (thing held, what is
   in view) that separate the two groups are the conditions.
   - Each condition is met when recall predicts success with the current
     value of that part and the other part as in a successful try. It is
     achieved by an action whose predicted effect changes that part.
   - Successful pairs needing the same parts are one alternative, stood
     for by the pair nearest the present one under recall's metric
     (declared).
   - Walking and card 029's order and backtracking stay. A search over
     imagined sequences of up to 4 actions was withdrawn unrun: it
     simulates latent steps instead of reasoning over conditions (GOAL.md
     P21).

**Not changed:** in arm A the agent's look on a new open door is wrong in
3 of 4 seeds, against Appendix B's 18 of 20; in arm B it is right in 4 of
4 (Appendix C). The fault lies in arm A's vectors, and the user does not
need arm A corrected (2026-09-30), so the procedure stays as built.

**Arms** (test colours yellow and purple, never in training), all with
the same planner:
1. **Upper bound:** oracle vectors, one-hot in four parts (kind, colour,
   state, agent), from the evaluator's tile names. Gate only.
2. **A, encoder with codebooks:** card 037's saved encoders. The planner
   never reads the codes.
3. **B, encoder without codebooks:** the same recipe minus the codebook
   terms.

**Baselines,** not rerun: card 037's arms 2 and 3, same encoders and seeds.

**Declared exceptions** (CHARTER rule 8):
- Arm B leaves out the codebooks that "codes are names" and "goals over
  codes" assume. That is what it tests. Changing the direction needs the
  user's agreement.
- The move maps and poses are card 028's, fitted on exact repeats of the
  vectors (pixels repeat exactly in this world).
- What is in view is one vector: the largest value per dimension among
  the things in view, the agent's own place aside (a set pooled into one
  vector, `1703_06114`). It replaces card 010's context rules.

## 3. Dependencies

- Card 029's walking, poses and depth-limited working backward (passed),
  kept in structure.
- Card 037's recall: weights, λ fit and prior (its weighting revised,
  section 2). It lost nothing familiar in
  9 of 10 seeds, and its vote was right on the new cases in 9 of 9.
- Card 033: recall keyed on the held thing learned "toggle holding the
  yellow key opens" within 1–8 tries. Card 010's evidence score (passed).
- Methods (LITERATURE.md):
  - a shift in latent space: `1911_12247`, `2002_11963`;
  - copy with substitution: `gentner-structure-mapping`, `1910_05065`;
  - kernel readout and immediate storage: `1703_01988`, `1606_04460`;
  - tries indexed by what they achieved: `1707_01495`, `1906_05253`;
  - the pair term: `2002_02886`;
  - own tries with the neighbours as a prior: MacKay and Peto 1995 (not
    in papi).

## 4. Data check

As in card 037: in world (a), for each new colour, picking up the key
4,060 times, dropping it 3,985 and forward onto the open door 503. The
agent drawn onto a tile has 5 stored examples (floor, 3 open doors, goal).

## 5. Feasibility gate

- **Upper bound:** arm 1 passes criteria 1–3. Without it, nothing about
  the encoders can be read.
- **Trivial baseline:** card 037's arm 3 failed criterion 2 in 10 of 10.
- **Arm B's encoders:** every training tile distinct, and pixel rebuild
  error at most twice arm A's median.
- **λ:** fitted better than at its start, per kind, as in card 037.
- **Shakedown:** spare seed 399, familiar worlds only, before the main run.

## 6. Success criteria and prediction

Per arm, each criterion must hold in at least 8 of 10 seeds (400–409), for
yellow and for purple. A predicted place is right when the real tile
nearest its vector (L1, among the 52 tiles the view can show) is the one
there.

1. **Nothing lost.** In the four familiar worlds, held-out effects right in
   ≥ 99.9%. The goal reached in ≥ 99% of 500 layouts, with mean steps within
   5% of card 029's.
2. **New keys and open doors predicted.** In world (a), picking up the new
   key, dropping it and forward onto the open new door each right in ≥ 99%
   of occurrences, from memory before acting.
3. **Acted.** In world (a), the goal reached in ≥ 98% of 500 layouts, with
   mean steps at most 1.15 times the shortest route's (online learning on).

The card passes if arm A or arm B meets all three. **Codebooks, decided
before the run:** if B passes each criterion in at least as many seeds as
A, I propose dropping them.

Also reported: world (b); toggling the new doors; tries curves (success
and wrong predictions by position in a worker's sequence); the ways chosen
per part; seconds.

**Prediction.** The main risk is building the planner, which is why arm 1
gates everything. Given arm 1:
- criterion 1 passes in A and B, because a familiar thing's own tries
  dominate its vote;
- pick up and drop pass by copying;
- forward onto the open door is right in about 18 of 20 cases (Appendix
  B's shift check);
- criterion 3 follows, helped by online learning;
- opening and closing new doors mostly fail at first sight (Appendix B: 9
  and 3 of 20), and the tries curve shows whether online learning mends it;
- I have no prediction on A against B.

**Budget.**
- Building comes first; it is not a run.
- Arm B's encoders take 2 minutes.
- Arms 1, A and B have an unknown per-seed cost. The planner does vector
  arithmetic instead of table lookups, and card 029 took 0.02 seconds per
  layout.
- The shakedown measures the cost, and the runs are then split so each
  process stays under 30 minutes. I guess about an hour of wall time.

## 7. Result

**Feasibility gate, 2026-09-30** (after revisions 1–3; main runs not yet
started):
- **Upper bound, passed** (`runs/038_arm1.json`, 226 seconds). Familiar
  worlds: goal in 100% of 500 layouts at 0.997–1.00 of card 029's steps,
  held-out effects 100%. World (a), yellow and purple: every criterion
  case 100%; goal 100% at 1.05 times the shortest route, with no wrong
  predictions. World (b): goal 100% at 1.07 times.
- **Arm B's encoders, passed** (`runs/038_gate.json`): every training
  tile distinct in 10 of 10 seeds (smallest L1 distance 0.25–0.43), and
  rebuild error 0.000068–0.000201 against a limit of 0.000218 (twice arm
  A's median). Training took 4.5 minutes, against 2 budgeted.
- **λ** fitted better than at its start in every kind and world (oracle).
- **Trivial baseline:** card 037's arm 3, not rerun.

**Arm B, seeds 400–405** (`runs/038_armB.json`; the user stopped the run
after six seeds and set five per arm). Criterion 1 holds in 0 of 6
seeds, criterion 2 in 5 of 6, criterion 3 in 6 of 6. The key world
reaches the goal in 100% everywhere; the switch, either and both worlds
fall to 33–66%. The same values recur in different seeds (either 65.8% in
seeds 400, 402 and 404), and success does not change with a layout's
place in its chunk, so the layouts, not online learning, decide.

**Arm A, seeds 400–402** (`runs/038_armA.json`; the user stopped it in
seed 403 to free the machine for card 039, judging three seeds enough).
Criterion 1 in 0 of 3 seeds, criterion 2 in 1 of 3 (seeds 400 and 401:
the agent in the new open doorway wrong, card 037's failure), criterion 3
in 3 of 3. The same worlds fail as in arm B (either 66–90%, both 15–45%;
seed 401's key world 36%, 96% of moves random). On the seeds both arms
finished, arm B passes each criterion in at least as many seeds as arm A
(0 and 0, 2 and 1, 3 and 3), so by the rule set before the run,
codebooks are dropped.

**Diagnosis** (serial probes on seed 400, fresh memory per layout):
- **Recall cannot use what is held when toggling a door.** Toggle's
  fitted weight on the held part is 0.0–0.1 in all four worlds (oracle
  vectors: 1.3–2.6, against about 35 on the front). The door's rule is a
  relation, a key of the door's colour. Leaving one (door, held) pair out
  to fit λ, the held key's identity points only to other doors' failures,
  so the fit turns it off. The neighbours' vote on toggling a door is then
  the world's base rate, whatever is held: 0.33, 0.49, 0.65 and 0.17 in
  the key, switch, either and both worlds. Familiar cases are right only
  because a key's own tries decide (α near 0).
- **Every combination the planner imagines that memory has not stored
  exactly gets that base rate.** In the either world (0.65) every
  imagined change "opens" the door. Where a wrong-coloured key is nearer,
  the planner picks it up and drops it for 200 steps (5 of 12 probed
  layouts), every prediction of what pick up and drop do being right. In
  the both world (0.17) no route is ever predicted, so 66–89% of moves are
  random. The upper bound passed with the same weakness: its base rates
  (0.01–0.44) fall below one half, so its errors are "will not open",
  which trying corrects.
- **The merged view hides the switch** (arm B only): switch-on and
  switch-off views differ by 0.003 after recall's weights, so the switch
  route cannot be read either. On oracle vectors they stay apart.
- **Smaller:** seed 405 predicts toggling the switch as no change (toggle
  right in 70.7% of held-out rows, key world, where the switch opens
  nothing). In seed 402 the agent in the new purple doorway is predicted
  as the red one, card 037's failure, now in arm B for one colour.

## 8. Decision

## Appendix A: the procedure

`tools/card038/`, reusing card 028's poses and walking and card 029's
search order. Nothing in it gives a vector a label.
- **Facts:** a vector for each place of the first view's frame, and one
  for the hand. Search recognises a state it has already seen only when
  the numbers are exactly the same; nothing is merged by similarity.
- **Placing:** the pose whose predicted view has the least total L1
  distance to the real one.
- **Recall per kind,** as in card 037:
  - the kinds are left, right, forward (the thing in front), the agent
    drawn onto and undrawn from a tile, and pick up, toggle and drop (the
    thing in front, the thing held and what is in view);
  - weights k = exp(−Σ λ|x − x′|), with own tries at weight 1;
  - λ is fitted by card 037's leave-one-out, with every key of the same
    thing in front and held (a pair) left out together;
  - outcome categories are only which places change (for moves: moved,
    blocked or ended);
  - an outcome needs probability ≥ 1/2 with prior 1/4.

  Forward decides whether the agent enters a tile. The draw and undraw
  kinds supply only how the agent looks on it.
- **What is in view** (revised): the largest value per dimension among
  the things at the view's places, the agent's own place aside.
- **Ways per part:** keep, copy, shift or set. The choice is made by
  leaving out each pair's keys together over the category's stored keys,
  with equal weight per key. A category with one pair uses shift.
- **Working backward** (revised):
  - Conditions: "episode ended", "place j walkable", "facing", and part
    conditions: "doing a on thing u (at place j), with the part held or
    in view as now and the other part as in a successful try, makes c
    true" (recall's result written into the facts, then c checked).
  - Achievers of "episode ended": forward onto a thing predicted to end
    it. Of "place j walkable": pick up or toggle the thing at j. Of a part
    condition: a pick up, toggle or drop on a thing in the facts whose
    predicted effect changes that part so that the condition holds. Each
    achiever's needs are its conditions from memory (section 2), then
    facing its thing (`conditions`).
  - Places in the way: a thing memory has seen made walkable (k ≥ 0.01 on
    the front part of pick up's or toggle's λ) is checked as in the way.
  - Needs are then taken in card 029's order.
- **Online learning:** each real step's try (things before and after,
  held, view) joins its kind's memory, and that kind's cached predictions
  are dropped.
- **Upper bound's vectors:** four one-hot parts from the tile names: kind
  (floor, wall, key, door, goal, switch, …), colour, state (open, closed,
  on, off, none), agent (present or not).
- **Arm B's encoder:** card 035's `encoder` without the codebook,
  commitment and reseeding terms. The decoder reads the unit-length
  pieces, and the pair term and recall term (μ = 0.01) stay.

## Appendix B: drafting check

On card 037's saved encoders (seeds 400–409), with no training: a new
tile's vector plus the mean change on red, green and blue, compared by L1
with the 52 tiles the view can show.

| Effect on a new-colour door | Nearest tile right (of 20) | With a way chosen per part (of 20) |
|---|---|---|
| Agent steps into the open door | 18 | 18 |
| Door opens | 9 | 6 |
| Door closes | 3 | 2 |

The agent's step changes the vector the same way for every colour.
Opening and closing do not: in pixels an open door is an outline and a
closed one a filled square of its colour. Parts in the encoder are a later
card.

## Appendix C: shakedowns before the main run

All on spare seed 399, familiar worlds, 100 layouts, unless stated.

- **Revision 1.** As first approved, a need was met when a thing was
  "like" the target. On oracle vectors that failed: λ is fitted only to
  predict which places change, so under toggle's λ holding nothing looks
  like holding the green key (k = 0.998), and switch on looks like switch
  off (0.98). Arm A showed the same (0.96–1.00). The goal was reached in
  0–5% of layouts, although held-out effects were right in 98–100%.
- **Revision 3, first shakedown** (oracle): key, switch and either worlds
  100% (0.96–0.98 of card 029's steps; switch was 64% before). Both world
  90%, at 2.5 times card 029's steps, with 67% of moves random.
  - The cause was recall's vote, not the planner. Holding the red key with
    the switch on, the key's own 12 tries all opened the red door, but 35
    neighbouring keys outvoted them (open 0.485 < 1/2). Toggle's fitted λ
    nearly ignores the held part (at most 0.09 per dimension against 2–4
    for the front), so doors with nothing held count as near-equal.
    Held-out door openings in the both world: 35 of 39 right (other
    worlds 100%).
  - Diagnostic only: with a key's own tries alone deciding it, the both
    world reached 100% at 0.98 of card 029's steps.
- **Revision 2, check.** α fell at the grid's lower end (e⁻⁹) in every
  world and kind, for oracle and arm A vectors alike: a key's own tries
  never disagree in this world, so the fit lets them decide. Held-out door
  openings in the both world: 39 of 39 on oracle vectors (card 037's rule:
  35) and on arm A's (card 037's rule: 0). Arm A's lowest held-out effect
  per action rose from 0.986–0.991 to 0.994–0.999. New-colour cases were
  unchanged.
- **Revisions 1–3 together** (oracle): all four worlds 100%, 0.96–0.98 of
  card 029's steps, no random moves, 1.4–2.2 seconds per layout.
- **The agent's look on a new open door.** Forward onto the open yellow
  door has the right outcome (the agent moves) in every arm. In arm A the
  drawn agent is nearest another tile: right in 1 of 4 seeds (399–402).
  Seed 399's lands nearest the closed green door, 400's and 401's nearest
  a familiar-colour doorway. Draw's neighbours are weighed on forward's λ,
  which puts the floor (k = 0.70) as near an open door as the doors are
  to each other (0.71), and the floor and goal pull the way per part
  toward copy (seed 399: copy, copy, shift, shift). Appendix B weighed
  only the three open doors. In arm B, with the same procedure, the look
  is right in 4 of 4 seeds (399–402; shift in all or three of four
  parts). Not changed (section 2).
- **Cost.** A first main-run process (arm B, seeds 400–402) was stopped at
  its 30-minute limit inside seed 400, before any result was saved. It had
  finished the key world (criterion cases 100%, goal 100% in world (a))
  and the familiar switch world (100%). World (b) alone took 21 minutes:
  its goal was reached in 9% of layouts, so most ran the full 200 steps.
  - About 70% of a failing layout's time was recall's weighted distance,
    recomputed after every stored try. Summing it one dimension at a time
    (the same numbers to about 1e-14) made the oracle shakedown identical
    at 0.26–0.29 seconds per layout (from 1.4–2.2). Twenty workers doubled
    their throughput (6.9 against 14.8 seconds on the same work).
  - `--layouts-b N` caps world (b)'s acting layouts, which are reported
    only.
- **What is in view hides a small change (arm B, seed 399).** In a
  familiar switch layout the planner found no plan in 200 steps. It
  inferred the right condition (what is in view, for toggling the door),
  but no action was predicted to meet it. Switch on and switch off differ
  by 0.32 (L1) in that encoder, but pooled with the other things in view
  by the largest value per dimension, the view changed by 0.001: others
  already hold the largest values where the switch differs. On oracle
  vectors and on seed 400's arm B encoder the switch survives pooling. The
  pooled view is a declared exception (section 2); the main runs measure
  how often this happens.
