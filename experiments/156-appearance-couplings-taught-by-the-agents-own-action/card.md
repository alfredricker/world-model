---
id: "156"
title: appearance couplings taught by the agent's own action
rung: 2
serves: [P7]
status: done
verdict: fail
arch_version: og4
date: 2026-11-20
---

# 156: appearance couplings taught by the agent's own action

Roadmap step 2, rung 2, on the direction under trial. Card
[155](../155-object-modules-fed-by-binding/card.md) measured the cost of
removing the masks: an object exists for the model only after it has
moved. The user, 2026-11-20: couplings as a function of appearance need
"a training incentive ... instead of trying to architect a perfect loss
or prior", and "an interactive environment where the agent learns by
what objects do may also help"; "yes" to one card with both. Runs take
about seven minutes per world. Biological counterpart: infants group by
common motion first and by static cues later, the static cues learned
from the moving cases (Kellman and Spelke; Needham); active exploration
of what is not yet understood (Gibson's exploratory action).

**Terms**, beyond cards 153.1 to 155. Binding couplings $J_{pq}$: the
log-odds that points $p$ and $q$ are parts of one object, accumulated
from motion ($+a$ a step moved together, $-b$ a step in which one moved
and the other did not). **Appearance couplings** $A_{pq}$: the same
quantity predicted from the frame alone.

## 1. Question

Can binding couplings predicted from appearance, taught only by the
couplings that the agent's own action produced, bind an object before it
has ever moved, and does an agent that chooses what to act on teach them
as well as a script that is told where the objects are? P7.

## 2. What changes

Two parts, separated by the arms (the user asked for both in one card).

```
appearance couplings (tools/card156/appearance.py)
  a small convolutional embedding e_p of the frame around each point, and a network on (|e_p - e_q|, (e_p + e_q)/2, offset)
  with three states per pair:  together | apart | no evidence
  labels    the binding layer's own J at the end of each episode:  J >= 3a together,  J <= -3b apart,  else no evidence
            (the same number of pairs of each state per frame; the states' true frequencies are put back at read-out)
  read-out  A_pq = J_in P(together) + J_out P(apart);  the pairs whose likeliest state is "no evidence" are centred
            (their mean is subtracted), so that what is expected to show nothing pulls neither way
  use       E(b) = -1/2 sum_pq (J_pq + A_pq) b_p.b_q  at first sight; a group bound this way that does not move is
            stored as a pattern too, with no couplings of its own
the agent (tools/card156/agent.py), in place of card 152's script, which walks to the true place of each object
  it lists the groups its binding layer sees (reading the pair states at equal odds: what could be an object), goes
  to the side of each, applies a grip and moves; if the group's points moved with the hand it carries and lets go,
  if not it tries the other grip; with no group left it moves to random places with random grips
```

$J_{in}$, $J_{out}$ are the medians of the couplings stored with the
memory's patterns. Nothing reads a mask; $A$ is never a training label.
Hand-set: the network's size, 1,500 pairs of each state per frame, 300
training steps every 10 episodes, evidence from 3 steps (card 154's
number), a carry of 3 to 7 steps.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Whether two points belong together can be read from a 7 by 7 neighbourhood of each and their offset | partly | yes | partly | partly | declared: the embedding's reach is a setting |
| The agent can act on a group by going to its side and gripping | no | partly | no | yes | declared: the action that tests a group is the world's |
| The view does not move, so the ground gives no evidence | no | no | yes | partly | declared, as card 147 |
| Transition tokens and the hand's place are given | no | no | no | partly | declared, as card 155 |

## 3. Dependencies

Card 154 (binding with feedback, stored patterns), card 155 (object
modules and dynamics classes on that binding), cards 152 and 153.1 (the
tests, the classes).

## 4. Data check

Development worlds (seed 0, by colour and by shape), 100 episodes. The
binding layer's own couplings agree with the simulator (40 episodes of
the script, by colour, any evidence counted): 98.9% of pairs with
positive coupling are of one thing, 96.5% of pairs with negative
coupling are not. But 82% of the pairs of a thing that did not move in
the episode have no coupling, and those are half of all same-thing
pairs. The script moves 119 and 139 of 168 and 194 movable objects, the
agent 111 and 117 of 175 and 185.

## 5. Feasibility gate

Never-met objects, no mask at test, development worlds (by colour, by
shape):

- **Upper bound:** appearance couplings taught from the simulator's
  masks: movable objects bound at first sight 100.0 and 95.5%; the take
  action right for 100.0 and 97.9%.
- **Trivial baseline:** no appearance couplings (card 155): 0% bound,
  take 39.1%.

Passed.

## 6. Success criteria and prediction

Test worlds: seeds 1 to 5, by shape and by colour, 100 training
episodes; never-met objects (new textures, and widths never seen); no
mask anywhere. "Bound" is card 154's "found". The method is the agent
arm.

1. **Bound before it moves.** Never-met takeable and sticky objects
   bound at first sight in at least 90.0% of cases, on the mean of the
   10 worlds (card 155: 0).
2. **End to end.** The take action right for at least 80.0% of never-met
   objects on the mean of the 10 worlds (card 155: 39.1; card 153.1
   with masks: 99.8).
3. **Choosing what to act on against being told.** The agent arm's take
   on never-met objects within 10.0 points of the script arm's, on the
   mean.

Development (the readings of the pair states were chosen here: the
expected coupling, each pair's likeliest state, with and without the
states' frequencies; the one kept is in section 2). Script arm: bound
98.4 and 97.7%, take 100.0 and 91.7. Agent arm: bound 79.1 and 97.7,
take 80.9 and 86.9; in the world by colour the agent moved 6 sticky
objects in 100 episodes and no sticky class was learned; a second run of
the same world bound 100% of never-met sticky objects, so the agent's
result varies from run to run.

Prediction: 1 fails narrowly, 2 passes narrowly, 3 fails (the agent is
about 12 points under the script). Budget: about 55 minutes of runs,
none over 10 minutes.

## 7. Result

Numbers in [results.json](results.json). Ten worlds (seeds 1 to 5, by
shape and by colour), 100 training episodes, never-met objects, no mask
anywhere. "Bound": card 154's "found", for takeable and sticky objects.
"Take": the share of objects for which the proposed action is right in
the simulator.

| Mean of 10 worlds, % | Bound at first sight | Take | Take, by shape / by colour |
|---|---|---|---|
| Card 155, no appearance couplings | 0 | 39.1 | |
| Script's experience (told where objects are) | 96.3 (lowest 83.6) | **93.9** | 87.9 / 100.0 |
| The agent's own experience | **96.9** (lowest 89.0) | 75.0 (lowest 42.5) | 72.8 / 77.1 |
| Card 153.1 with masks | | 99.8 | |

Both arms: objects of a width never seen are bound as often as the rest
(97.8 and 97.0%); three dynamics classes in 19 of the 20 runs (four in
one), purity 95.6% or more, the third being the fixed objects, of which
92% are bound where behaviour goes with shape and 22 to 30% where it
goes with colour; 0.5 to 3.2 labels a scene lie on the ground. The
script moves 73.3% of the movable objects it meets, the agent 67.6%.

| Criterion | Verdict |
|---|---|
| 1. Agent arm: never-met takeable and sticky objects bound at first sight, at least 90.0% on the mean | **pass**, 96.9 |
| 2. Agent arm: take right for at least 80.0% of never-met objects on the mean | **fail**, 75.0 |
| 3. Agent arm's take within 10.0 points of the script arm's | **fail**, 75.0 against 93.9 |

**Why the agent's take is low** (checked). The binding is not the
cause: the agent's couplings bind as well as the script's, and its
classes are as pure. The test asks for the action that leaves an object
held without moving it. The agent always moves as it grips, because
that is how it sees whether the group came: of 103 takes in 50 episodes
none is without a move (the script: 35 of 150). Couplings have no answer
for a combination their records lack (card 149's lesson): with the true
classes given, couplings fitted to the agent's transitions also get 0%
on takeable objects in the four agent worlds I looked at, against 100%
on the script's. My prediction of a blind spot in what the agent
explores was not what happened.

What the card shows.

- **Appearance couplings can be taught by motion alone.** With no mask,
  no similarity loss and no grouping prior, a never-met object is bound
  before it moves in 96 to 97% of cases, whoever gathered the
  experience. End to end, the script arm takes never-met objects right
  in 93.9% of cases against 39.1 without the couplings.
- **The agent does not need to be told where objects are** to teach
  them: 96.9 against 96.3.
- **Fixed objects become objects known not to move** where they look
  like movable ones (a third class, bound 92%), and stay scenery where
  their colour marks them (22 to 30% bound), which is what the
  experience says in each case.

Where it is weak.

- **The agent's narrow action** (above): criterion 2 and 3.
- **Sticky objects where behaviour goes with shape**: the script arm's
  take is 23.9 to 84.8% for them (100% by colour), card 155's weakness
  again: shape read from a group's mask is blurred.
- **Labels on the ground**: up to 5.2 a scene in one world. They become
  still patterns (80 to 225 a run, with the still objects) and cost the agent tries.
- **The read-out was chosen in development** among four; two fragment
  the ground. The centring of pairs with no expected evidence stands in
  for a no-evidence pull (card 147's 0.05) that is too weak for
  predicted couplings.
- **Run-to-run variation** of the agent arm was seen in development and
  not measured here (one run per world).

## 8. Decision

**Revise.** The appearance couplings and the storing of still groups are
kept: they pass the binding criterion in both arms and carry the script
arm to 93.9%. The agent is revised, one change: it grips first and pulls
on the next step, so that its record holds a take without a move, and it
still sees whether the group came. That is card 156.1, with the user's
go-ahead.
