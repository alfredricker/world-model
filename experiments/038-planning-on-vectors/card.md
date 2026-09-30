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
encoder's codebooks.

| | Planner of cards 028–037 | Card 038 |
|---|---|---|
| A place holds | a label (tile ID or code tuple) | the encoder's vector |
| Whether a need is met | the labels are equal | recall's forward prediction (revised; see below) |
| Whether an action does something | a table entry per label | recall's vote over tries keyed on the thing in front (and, for pick up, toggle and drop, the thing held and what is in view) |
| What results | a label from the table | the neighbours' change carried onto the vector, part by part |
| Working backward | operators read from the table | the shortest imagined sequences of actions that recall predicts make the need true |
| Memory while acting | fixed | every try added (online learning) |

- **Carrying a change over.** Each changed part is handled in one of four
  ways:
  - kept;
  - copied from the other changed place (a picked-up key goes to the hand);
  - shifted by the neighbours' weighted mean change;
  - set to their weighted mean result.

  Leave-one-out over the stored things of the kind picks one way per part.
  Weights and priors are card 037's recall: a learned λ per action, prior
  1/4, and an outcome needing probability ≥ 1/2.
- **Conditions** are predictions, checked by recall:
  - "the episode ends";
  - "the thing at place j can be walked onto";
  - "doing a on thing u, with what is held and in view now, makes
    condition c true".
- **Working backward: open, awaiting the user.** A search over imagined
  sequences of up to 4 actions was written but withdrawn unrun. It
  simulates latent steps instead of reasoning over conditions (GOAL.md
  P21).
  - Proposed instead: for an unmet "doing a on u makes c true", recall's
    stored tries of that kind are split into those that achieved c and
    those that did not. The conditions are the parts of the key (thing
    held, what is in view) that separate the two groups.
  - Each condition is met when recall predicts success with the current
    value of that part and the other part as in a successful try. It is
    achieved by an action whose predicted effect changes that part.
  - Walking and card 029's order and backtracking stay.

**Revision, 2026-09-30 (agreed with the user after the shakedown).** As
first approved, a need was met when a thing was "like" the target (k ≥ 1/2
under the asking action's λ). On oracle vectors that failed. λ is fitted
only to predict which places change, so under toggle's λ holding nothing
looks like holding the green key (k = 0.998), and switch on looks like
switch off (0.98). Arm A showed the same (0.96–1.00). The goal was reached
in 0–5% of layouts, although held-out effects were right in 98–100%.

The "like" test is removed. Needs are checked by recall's prediction, and
what is in view joins recall's key for pick up, toggle and drop.
- **Online learning.** Each real try joins recall's memory, and that
  kind's cached predictions are dropped. λ is not refitted. Memory carries
  across the roughly 12 layouts each worker plays in turn.

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
- Card 037's recall: weights, λ fit and prior. It lost nothing familiar in
  9 of 10 seeds, and its vote was right on the new cases in 9 of 9.
- Card 033: recall keyed on the held thing learned "toggle holding the
  yellow key opens" within 1–8 tries. Card 010's evidence score (passed).
- Methods (LITERATURE.md):
  - a shift in latent space: `1911_12247`, `2002_11963`;
  - copy with substitution: `gentner-structure-mapping`, `1910_05065`;
  - kernel readout and immediate storage: `1703_01988`, `1606_04460`;
  - tries indexed by what they achieved: `1707_01495`, `1906_05253`;
  - the pair term: `2002_02886`.

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
  - Conditions: "episode ended", "place j walkable", "doing a on thing u
    (at place j), with what is held and in view now, makes c true"
    (recall's result written into the facts, then c checked), and
    "facing".
  - Achievers of "episode ended": forward onto a thing predicted to end
    it. Of the others: open (see section 2). The imagined-sequence search
    in `last_steps` is withdrawn and must be replaced before any run.
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
