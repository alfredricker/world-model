---
id: "109"
title: conditions from memory — believing what holds out of view, from what was seen and how such facts change
rung: 2
serves: [P12, P2, P10]
status: draft
verdict:
arch_version: 27
date: 2026-10-14
---

# 109: conditions from memory

Drafted 2026-10-14 at the user's request ("I think that would be a fine
first card"). It is the first card of System 2 as GOAL P12 now reads: the
agent evaluates the conditions a plan needs from its state and its memory,
and does not forecast when the goal will be reached. Rung 2 is the
proposed ladder's "evaluate a goal's conditions from state and memory";
the ladder awaits the user.

**Worlds that follow** (portability tests, only the adapter changed):
- **Crafter:** coal and iron never come back once mined; saplings ripen
  while unseen.
- **Craftium:** searched caves and ores.

## 1. Question

A child who has searched a cave and found no diamond does not search it
again. Diamonds do not appear, so a remembered empty place stays empty,
while an unsearched deep cave may hold one.

Suppose the agent evaluates a condition, "a gem is at place x", from:
- what it sees now;
- else what it remembers, carried forward by recall of how such places
  changed while unseen in its past;
- else, for a place never seen, recall of what places in surroundings
  like these held when first seen.

Does that belief match the truth? And does it choose where to look as
memory requires, including after two histories with the same view?
Rung 2 (proposed). Serves P12 (conditions evaluated), P2 (belief about
what is out of view) and P10 (retrieval helps where the view cannot).

## 2. What changes

One component: System 2's evaluation of a condition at a place out of
view, in two arms judged by CHARTER's rule for recall and networks. System
1 (series 105) is not used beyond its tokens. This is the first card of
series 109, belief over conditions (CHARTER "Roadmap", step 3).

```
version 27:  a place out of view is as last seen, forever (a declared static world)
102.3:       a token out of view spreads by its kind's counted motion; places themselves are as last seen
this card:   b(code c at place x), asked only for the places a query names; in view, perception; otherwise
             arm R (recall)    remembered: recall over stored revisits, places seen with content like x's last
                                 sighting and seen again after a gap like x's; b = the weighted share that then
                                 showed c
                               never seen: recall over stored first sightings of places whose remembered
                                 surroundings were like x's; b = the weighted share that held c when first seen
             arm N (network)   a small transformer: a query token (the code asked, at x) attends over the
                                 remembered places within 4 tiles of x (vector, offset from x, age; places never
                                 seen absent) and gives b
```

**Both arms learn from the same signal, without labels.** Whenever a
remembered or never-seen place comes into view, what is seen there is the
target for the belief held one step before: predict what you will see
when you look. Arm N is trained on these targets with the log-likelihood.
The arm used for the criteria is the one with the better held-out
log-likelihood on validation episodes, as CHARTER's rule says. Both arms
are reported.

**Details:**
- **Memory.** The places as last seen (102.3's map, placed by the pose):
  each place's code, vector and step.
- **Recall (arm R).** Weights use codes first and vectors for similarity (card
  042's two levels). The gap enters as the log of its ratio. Each
  bandwidth is fitted by leave-one-out on training episodes.
- **Surroundings of a never-seen place (arm R).** The mean vector of
  remembered places within 3 tiles. This pooling is declared and simple;
  arm N reads the places themselves.
- **Selective.** A query names its places, and nothing else is evaluated.
  Here the evaluator names the cave ends; a later card lets the planner
  name them. The cost per query is reported.
- **The choice.** For the goal "gem" or "berry", go to the named place
  with the highest belief, ties to the nearest.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Places in a world frame, from the pose | yes (blocks) | yes | screen | odometry, drifts | adapter (C1) |
| A place counts as seen when it is in the view | no: the frustum and occlusion decide | yes | yes | no: the same | adapter: perception says what was seen |
| A place's content is one token | voxels yes, creatures no | yes | often | no | adapter |
| Which facts last and which change while unseen | learned | learned | learned | learned | general: recall of revisits |
| An unseen place is judged by remembered places near it | yes | yes | yes | yes | general; the 3-tile radius is declared |
| The places asked about are named by the evaluator | | | | | test device; the planner names them later |

## 3. Dependencies

- **102.3** (kept): memory of places out of view, placed by the pose
  (105.10).
- **042** (kept): recall in two levels, codes for identity and vectors
  for similarity.
- **Arm N's form:** 105.6's transformer, positions read only as
  differences through the bounded bias, with the age as one more input.
- **The tokens:** 098.3's encoder recipe on this world's play, as 102.3
  did for world 3.
- **The engine:** the breaking worlds (`tools/card102/worlds.py`); world
  6 is added (Appendix A).
- **Literature:** exemplar recall (Hintzman 1986, in LITERATURE); a place
  seen empty as negative information (IPDA, in LITERATURE).

## 4. Data check

Before training, count on the training play:
- revisits of cave ends, by content at the first sighting and by gap
  (10, 30, 60 and 120 steps);
- bushes that ripened while unseen;
- first sightings of cave ends, by cave kind.

Ripening after 30–60 steps can only be learned from revisits at such
gaps. If fewer than 100 exist, the policy's revisits are raised before
the run.

## 5. Feasibility gate

- **Upper bound:** the evaluator's probability, given the history:
  - a gem is certainly gone once taken, and absent once its place is seen
    without one;
  - an unseen dark cave holds a gem with p = 0.5;
  - a bush follows the ripening rule.
- **Trivial baselines,** in the same code:
  - **view only:** every place out of view gets the never-seen rule;
  - **static memory:** version 27's, where the last sighting holds
    forever.
- **Check before the main run:**
  - at least 90% of probes have one best cave under the upper bound;
  - view only fails criterion 3. If it does not, the test does not need
    memory.

## 6. Success criteria and prediction

Probes on 200 test episodes: the agent in the corridor with no cave end
in view, asked about every cave end for both goals. Paired probes use the
same world and the same final place and step, after two scripted visit
orders, with a different best cave.

1. **Belief.** The mean log-likelihood of the truth at probed places is
   above both baselines (paired bootstrap over episodes, p < 0.05), and
   within 0.15 nats of the upper bound.
2. **Choice.** The chosen cave has the highest true probability, given
   the history, in at least 90% of probes for each goal.
3. **Memory decides.** In paired probes the choice follows the history in
   at least 90% of pairs.

**Reported:** beliefs by source (seen, remembered, never seen), by kind
and by gap; the cost per query; criterion 1 against the number of
training episodes (10, 30, 100, 300; P19).

**Prediction:**
- **Gems pass easily.** Revisits of a taken gem's place never show a gem.
- **Berries are harder.** The ripening hazard needs revisits at each gap.
  The ripening rule is the most likely miss on the 0.15 nats.
- **Criteria 2 and 3 pass** at about 95%.
- **The baselines fail where expected.** View only searches dark caves
  again; static memory never returns to an eaten bush.

**Decision rules:**
- **Keep** if 1–3 pass.
- **Revise** if one kind fails, or if 2 or 3 fail with 1 passing.
- **Stop** if, on remembered places, belief is no better than static
  memory.

**Budget:** about 17 minutes, every run capped (CHARTER rule 5):
- the world and its play, about 2 minutes;
- the encoder, about 5 minutes;
- arm N's training, about 5 minutes;
- recall, probes and scoring, about 5 minutes.

## Appendix A. World 6, "searched caves"

- **Layout.** A corridor 3 tiles high and about 25 long, with six dead-end
  caves (three up, three down), each 1 tile wide and 6 deep. The 9 × 7
  view follows the agent. From the corridor it shows the first two tiles
  of a cave. A cave's end shows only from at least three tiles inside.
- **Two kinds of cave,** told apart by their floor along the whole cave:
  - **dark:** the end holds a gem with p = 0.5;
  - **sand:** the end holds a bush, ripe with p = 0.7.
- **Gems.** Walking onto a gem takes it into the carried row and leaves
  floor. Gems never come back.
- **Berries.** Walking onto a ripe bush eats the berry and leaves a bare
  bush. It ripens again 30–60 steps later, seen or not.
- **Episodes.** 300 steps each; layout and contents are random per
  episode. 500 training episodes and 200 test episodes.
- **The play (declared, C3).** The agent walks to a random cave's end and
  back to the corridor, then picks another, with 10% random actions. The
  evaluator keeps the state.
