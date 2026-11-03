---
id: "109"
title: conditions from memory — believing what holds out of view, from what was seen and how such facts change
rung: 2
serves: [P12, P2, P10]
status: done
verdict: fail
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
  seen empty as negative information (IPDA, in LITERATURE). Memory
  formed by predicting what will be sensed is MERLIN's principle (Wayne et
  al. 2018, `papi` 1803_10760); a memory laid out by place and read by
  attention is Neural Map's (Parisotto and Salakhutdinov 2018, `papi`
  neural-map). Both were checked after the draft; their stored summaries
  are placeholders.

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

## 7. Result

Run 2026-10-14 ([results.json](results.json); `tools/card109/belief.py`,
`runs/109/`). World 6's play and tokens took 1 minute. The encoder's 8
codes in use each matched one kind of tile (purity 1.0). The belief run
took 163 seconds, within the stated 17 minutes, capped.

**Two departures from the plan:**
- **The smoke test was the main run.** A bug in reading the arguments
  made the smoke test run the declared configuration: 8,000 training
  steps for arm N and 100 pairs per goal. It is kept as the main run;
  code and seeds are as declared. The bug is fixed.
- **Recall's bandwidths were fitted on held-out training episodes**
  (400–449, by a grid), not by leave-one-out. Two of the three ended at
  the grid's lower edge: s = 0.1, h = 0.125, s_c = 0.03.

**Data check (section 4): passes.**
- **Revisits:** bushes last seen bare were seen again 480 times after
  30–60 steps, against the 100 required. Every gap had at least 32
  revisits for every content.
- **Ripening unseen:** 1,105 bushes left bare were ripe when next seen.
- **First sightings of cave ends:** 647 gems, 686 empty dark ends, 1,213
  ripe bushes, 152 bare ones.

**Gate:**
- **The upper bound** passes criteria 2 and 3 at 100%.
- **View only fails criterion 3** (0% of pairs for both goals), as
  required.
- **"At least 90% of probes have one best cave" fails as written:** 46%
  for gems, 27% for berries. Ties under the upper bound are part of the
  world: two unsearched dark caves both hold a gem with p = 0.5, and two
  bushes seen ripe are both certain. Criterion 2 was therefore scored
  where not every cave ties (70% of probes for gems, 99% for berries),
  with any of the tied best caves counted right. Paired probes have one
  best cave by construction.

**Arm chosen: N, the network.** Held-out log-likelihood on episodes
450–499 at every place coming into view was −0.044 for N and −0.074 for
recall (−0.18 and −0.71 at cave ends alone).

**Test probes** (200 episodes, 4,285 probes, each cave end for both
goals). Log-likelihood is per place and condition, in nats; 0 is
perfect.

| Form | Log-likelihood | Remembered places | Never-seen places | 2. Choice right, gem / berry | 3. Pairs following the history, gem / berry |
|---|---|---|---|---|---|
| Upper bound | −0.109 | −0.029 | −0.256 | 100% / 100% | 100% / 100% |
| **N, network** | **−0.317** | −0.073 | **−0.764** | **99.0% / 95.3%** | **98% / 100%** |
| R, recall | −0.415 | −0.060 | −1.068 | 89.1% / 88.5% | 79% / 100% |
| Static memory (version 27) | −0.943 | −0.874 | −1.068 | 92.8% / 84.8% | 79% / 100% |
| View only | −1.356 | −1.021 | −1.970 | 33.7% / 53.3% | 0% / 0% |

**Criteria:**
1. **Belief: fail.** N is above both baselines (paired bootstrap over
   episodes, p < 0.001): +1.04 nats against view only (95% interval
   0.99 to 1.09) and +0.63 against static memory (0.58 to 0.67). It is
   0.208 nats below the upper bound (0.191 to 0.225), against the 0.15
   allowed.
2. **Choice: pass,** 99.0% for gems and 95.3% for berries, against 90%.
3. **Memory decides: pass,** 98% of gem pairs and 100% of berry pairs,
   against 90%.

**Where the 0.208 nats arise:**

| Places | Share of the gap, nats |
|---|---|
| Never-seen ends of sand caves already seen (ripe bush) | 0.080 |
| Never-seen ends of dark caves already seen (gem) | 0.042 |
| Ends of caves never seen at all (both conditions) | 0.058 |
| Remembered places (nearly all bushes left bare 1–60 steps before) | 0.029 |

**Never-seen ends: the network's belief fades with the agent's
distance** (mean belief against the upper bound's):

| The agent's distance from the end | Gem in a dark cave: N / bound | Ripe bush in a sand cave: N / bound |
|---|---|---|
| 0–6 steps (about 600 probes each) | 0.42 / 0.50 | 0.89 / 0.91 |
| 7–12 steps (about 1,950) | 0.30 / 0.50 | 0.77 / 0.92 |
| 13–40 steps (about 1,530) | 0.11 / 0.50 | 0.45 / 0.94 |

The cause is where the training targets arrive. A cave's end is first
seen just after its mouth, so every target paired a never-seen end with
surroundings seen a step before. Asked from afar, the network met old
surroundings it had never been trained on, and read them as "not an
end". Recall's mean of the surroundings did not single out the ends at
all: 0.05 for a gem at every distance.

**Ends of caves never seen** get almost 0 from both arms, while the
upper bound gives 0.25 for a gem. Here the bound uses what the agent
cannot know: that a cave is there. The probe named the place, and the
agent had seen nothing near it.

**Remembered places.** Gems and empty dark ends were exactly right in N
and in static memory, at every gap. Bushes left bare cost the most:

| Gap since a bare bush was last seen | N | R | Static | Bound |
|---|---|---|---|---|
| 1–29 steps (2,470) | −0.375 | −0.291 | −0.636 | −0.087 |
| 30–60 steps (1,682) | −0.856 | −0.542 | −4.460 | −0.424 |
| 61–300 steps (2,894) | −0.013 | −0.011 | −6.908 | −0.001 |

Recall predicts ripening better than the network: by 0.08 nats in the
first range and 0.31 in the second.

**Also reported:**
- **Cost per query,** batched on the GPU: 13 µs for recall, 23 µs for
  the network.
- **Recall against the number of training episodes** (P19): −0.475 with
  10 episodes, −0.453 with 30, −0.427 with 100, −0.416 with 300. The
  network's curve was not run; it needs four trainings.

**Prediction against result:**
- **Gems:** remembered gem places were exact as predicted; never-seen
  dark ends were not.
- **The 0.15 nats:** missed, as predicted, but mostly on never-seen
  places (0.18 of 0.21), not on ripening (0.03).
- **Criteria 2 and 3:** passed at 95–100%, as predicted.
- **Baselines:** failed where expected. View only chose the best gem
  cave in 34% of probes. Static memory lost 6.9 nats on bushes left bare
  more than 60 steps before.

## 8. Decision

**Revise.** Criteria 2 and 3 pass, and the belief is far above both
baselines (+0.63 nats over static memory), so this is not a stop. On
remembered places it lies within 0.044 nats of the bound. Criterion 1
fails on its margin, and the misses are localised.
- **Never-seen places queried from afar (0.12 nats):** a gap in the
  training signal, not in the form. The next card would build each
  target also from memory as it stood k steps before the sighting, with
  ages counted to the sighting. The target is still what is seen; only
  some recent sightings are left out, as a query from afar leaves them
  out. One component, the training targets.
- **Ends of caves never seen (0.06 nats):** a flaw in this card's upper
  bound. The next card names only places the agent could know of, here
  the ends of caves whose mouth it has seen.
- **Recall beat the network on ripening.** CHARTER's rule ("each where
  it predicts better") allows choosing per source rather than per card.
  That would be a separate card.

## Appendix A. World 6, "searched caves"

- **Layout.** A corridor 3 tiles high and 23 long, with six dead-end
  caves (three up, three down, at least 3 columns apart), each 1 tile
  wide and 4 deep. The 9 × 7 view follows the agent. From the corridor
  it shows the first one to three tiles of a cave, never its end. A
  cave's end shows from the cave's first tile inward, and from inside a
  neighbouring cave less than 5 columns away (nothing occludes). As
  drafted the caves were 6 deep; 4 fits the view, and the property
  tested (the end is out of view from the corridor) is unchanged.
- **Two kinds of cave,** told apart by their floor along the whole cave:
  - **dark:** the end holds a gem with p = 0.5;
  - **sand:** the end holds a bush, ripe with p = 0.7.
- **Gems.** Walking onto a gem takes it into the carried row and leaves
  floor. Gems never come back.
- **Berries.** Walking onto a ripe bush eats the berry and leaves a bare
  bush. It ripens again 30–60 steps later, seen or not. A bush bare at
  the start ripens 30–60 steps into the episode.
- **Episodes.** 300 steps each; layout and contents are random per
  episode. 500 training episodes and 200 test episodes.
- **The play (declared, C3).** The agent walks to a random cave's end and
  back to a random place in the corridor, then picks another, with 10%
  random actions and 5% noops. The evaluator keeps the state.
- **Paired histories (criterion 3).** One layout, two scripted visit
  orders, the same final place and step:
  - **gems:** dark caves D1 and D2 hold gems and D3 none; history A takes
    D1's gem and looks into D3, history B takes D2's and looks into D3;
  - **berries:** sand caves S1 and S2 hold ripe bushes; history A looks
    at S2 and eats S1 last, history B the reverse.
  A pair is kept only if each history has one best cave under the upper
  bound and the two differ.
