---
id: "102.4"
title: where content went — correspondence learned by the world model, not declared
rung: 1
serves: [P7, P2, P6]
status: done
verdict: fail
arch_version: 27
date: 2026-10-12
---

# 102.4: where content went

Drafted 2026-10-12 at the user's request, after 102.3, and approved the same day. The user: "We
should be learning the dynamics of the system through the general
architecture, not telling our agent that it must distinguish between
this convenient boundary of 'ground' and 'not ground'." Cards 102.5 and
102.6 (the agent as a token, actions with a where) were renumbered to
follow it, since both would build on what this card replaces.

## 1. Question

Cards 102.2 and 102.3 read each step's change through rules written in
`tools/card102.2/layers.py`, added one after another as Crafter failed:
- **Ground:** codes revealed or covered when one tile moves (at least 3
  times).
- **Conservation:** a tile that is not ground cannot simply vanish.
  This is what grouped the middle of a moved bar of one colour with its
  ends (the aperture problem).
- **Two more rules:** a mover must leave its place; appearing on ground
  is a new token, not a change.
- **Rigid one-step shifts** relative to the view's own shift.

These are an ontology of the grid worlds, not something learned. 102.1's
arm A already predicts what every token becomes, including a pointer
(take content from that token). If arm A is trained to predict every
place's next content and its pointer is read as "this place's content
came from that tile", do fates, groups and tracking hold up with none of
those rules? Rung 1. Serves P7 (units from continuity and interaction),
P2 and P6.

## 2. What changes

One component: how a step's change is read as each tile's fate.

```
102.2:      fate = layers.py's explainer (background shift, chains to ground, conservation, appear/vanish rules)
this card:  arm A (102.1's kept form: two pair layers, softmax attention over every tile, raw positions, the action)
              predicts every place's next content by its four ways (keep, copy from a token by its pointer,
              change by a learned shift, set), trained on every place and step, by prioritised replay (102.2)
            the content now at a place came from: the token its pointer picks (copy), itself (keep or change),
              or nothing (set: new)
            identity follows the content; a token's identity goes to one place, where it was kept before
              where it was copied
            fate per tile: still, moved by an offset, changed in place, or ended; offsets read against the
              view's own shift (the offset most places' content moved by)
unchanged:  common fate, the learned prior and the groups (102.2); belief out of view (102.3)
removed:    ground, conservation, chains, the appear/vanish rules, "a mover must leave its place"
```

- **Static is learned** per code from the fates, as in 102.3: a code
  whose tiles never move gets zero spread. Grass and trees become
  static tokens in the map, not ground.
- **The open part is the aperture problem.** With no conservation
  prior, the middle of a one-colour bar predicts the same whether it
  stayed or moved. A move over uniform grass predicts the same as "the
  agent and the grass swapped" (card 098.8 found a learned walk prefers
  the swap). Only generalisation can decide: the shared pair function
  learns "content moves" where only that predicts right (shapes of
  several colours, sprites with a unique look). The test is whether it
  carries that to the ambiguous cases.
- **Not in this card:** the agent's own token (102.5); rotation.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| A tile's next content is read from some tile now, or is new | yes | yes | yes, as scene flow over points | general |
| A token is in one place at a time (the identity decode) | yes | yes | yes | declared |
| Ground lies under movers; non-ground tiles cannot vanish | — | — | — | removed |
| Content moves one tile per step | yes | mostly | no | removed: the pointer may pick any tile in view |
| Tiles on a grid | yes | yes | no (points) | kept from the adapter |

## 3. Dependencies

- Card 102.1's arm A (kept form) and its four ways; card 102.2's
  evidence, prior and groups, and its training format; card 102.3's
  belief and scoring.
- Breaking worlds 2 and 3; card 098.5's Crafter play (identities for the
  evaluator); world 2's and world 3's encoders, and 098.5's codes.
- Literature, to be added to LITERATURE: pointer networks (Vinyals,
  Fortunato and Jaitly 2015) and motion predicted by moving pixels
  rather than drawing them (Finn, Goodfellow and Levine 2016). Card
  098.8's lesson on the swap (LESSONS) is the known failure.

## 4. Data check

Counted when the card is approved, on each world's test play:
- moves whose content is unique in view (unambiguous: the pointer has one
  right answer);
- moves over uniform tiles, and the middles of one-colour shapes (the
  ambiguous cases);
- changes in place, and new content (spawns, entering the view).

World 2's test play (49,500 steps) holds these moving or changing object
tiles: the agent 27,854; doors 1,773 (changes in place); shapes of
several colours 2,128; one-colour shapes 2,136 (842 middles, 1,294
ends).

## 5. Feasibility gate

- **Upper bound:** the evaluator's true fates through the same groups
  and belief. This is 102.2's and 102.3's upper bounds, reused.
- **The rules (layers.py)** are the comparison. Their numbers are
  reused from 102.2 and 102.3, and Crafter's grouping is rerun on 098.5's
  play (one minute).
- **Trivial baseline:** "keep" everywhere. Every fate is still, nothing
  groups, and belief is version 27's.
- **Check:** arm A predicts every place's next code right on at least
  99% of world 2's test steps. If it cannot predict the next view, its
  pointer means nothing.

## 6. Success criteria and prediction

1. **Fates and groups as good as the rules, in world 2.**
   - The pointer gives the true fate of at least 95% of moving or
     changing object tiles (the rules: 100%).
   - Moved objects are exactly one token in at least 95% of states
     (102.2's criterion 1; the rules: 99.5%).

   Shapes of one colour are reported apart.
2. **No false groups in Crafter.** At most 1% of 098.5's held-out states
   hold a group of two or more tiles (102.2's criterion 3).
3. **Tracking out of view as good as the rules** (102.3's belief, its
   learned form). In world 3 and Crafter:
   - the return-place log-likelihood is within 0.1 nats of the rules'
     tracks;
   - identity kept is within 3 points of them;
   - the 90% region still holds 85–95%.

**Reported:** effects on whole objects (102.2's criterion 2) from arm
A's own predictions, and fates by case (unambiguous, uniform, one-colour
middles, changes in place, new content).

**Prediction:**
- Unambiguous moves are right.
- Moves over uniform grass are right too, because "copy the agent from
  where it was" is cheaper than drawing it anew.
- The middles of one-colour shapes are the likely failure. Criterion 1
  would then pass on shapes of several colours and fail on one colour.

**Decision rules:**
- **Keep** if 1–3 hold. layers.py's rules are then retired, and 102.5
  builds on the pointer.
- **Revise** if only one-colour middles fail. The revision would be a
  general preference for explaining by motion over change, for the user
  to decide, since it is a prior.
- **Stop** if unambiguous moves fail: arm A cannot learn correspondence
  where it has one answer.

**Budget:** arm A trained per world (world 2, world 3, Crafter), about 5
minutes each; reading fates, grouping and belief about 5 minutes. Under
25 minutes in all, smoke-tested first.

## 7. Result

Run 2026-10-12. Each world: arm A trained 12,000 steps (about 4 minutes),
read on every test step; every run capped at 16 GB (CHARTER rule 5, added
after the first launch ran the machine out of memory in card 102.2's
grouping, now streamed in `pointer.py`).

**The gate failed twice.**

| Training | World 2: every place's next code right (bar 99%) |
|---|---|
| "Keep" everywhere (trivial baseline) | 42.8% |
| Every place's content weighted alike (the card's form) | 32.1% |
| Card 102.1's loss: a change head, content only where the code changes | 96.8% |
| The same, balanced over changed and unchanged places (3,000 steps) | 73.7% |

Weighted alike, arm A learned "nothing changes": 99% of places keep their
content. With 102.1's loss the fates were read two ways from one network
(the user's choice): **A**, the change head first (unchanged places are
kept); **B**, the ways and pointer at every place.

| World 2, fates right | Agent | Doors | Shapes, several colours | One colour |
|---|---|---|---|---|
| A | 99.0% | 0% | 0% | 0% |
| B | 56.1% | 0% | 0% | 0% |

No moved object was one token (the rules: 99.5%). World 3's network did
not learn (its loss stayed flat). In Crafter, A grouped falsely in 38.8%
of states and B in none, and identity kept was 0.4% and 11.9% (the
rules' tracks: 81.1%).

**Why: the network draws pushed shapes.** At a push (test step 56, a
three-colour shape pushed right) the agent's new place copies from its old
place with the pointer's full weight. The shape's new places do not copy:
each blends keep (about 60%), set (30%) and change (10%), which lands near
enough to the colour that the code comes out right. The shape's tiles then
read as ended. This is card 098.8's failure ("they swapped" over "it
moved") on unambiguous moves: nothing in the loss prefers motion to
drawing, and a blend of ways is free.

## 8. Decision

**Revise** (2026-10-12, the user: "We want to have a general rule").
Revision 1 replaces the loss by one general principle, the **minimum
description length** (Rissanen 1978): a step's change is explained by the
shortest description. Each place's next content is coded by exactly one
way, and the loss is its code length in nats:
- **distortion:** naming the next code from the way's prediction (a
  softmax over the codes' prototypes, temperature learned). Codes are the
  model's names, so content is coded at their resolution; an exact
  prediction costs nothing;
- **keep:** distortion only. **Copy:** the distortion of the source's
  content plus the source's cost under a learned motion prior p(source |
  action), a function of the offset (any positions, not a grid), with bits
  back. **Change and set:** a drawn vector whose information is paid (a
  Gaussian's KL to the data's spread, as in a variational information
  bottleneck, Alemi et al. 2017);
- **the mixture's code length**, −log Σ p(way | action) e^(−cost), so each
  way learns from the places whose outcome it explains, not a blend; a way
  head learns that posterior for reading before the outcome;
- **the source is a latent** the pointer learns wherever the content
  changed, by copy's own cost (EM). Without it, wherever the pointer starts
  wrong "keep" is cheaper and the pointer is never trained there.

Copying a tile costs at most log 63 nats and drawing its colour hundreds,
so motion wins wherever it explains the content. A shift shared by many
places is cheap under the learned motion prior, which is how the ambiguous
cases (uniform ground, one-colour middles) are to be decided. The criteria
are unchanged.

| Revision 1, world 2 (smoke tests, 3,000 steps) | Gate |
|---|---|
| Blend-free mixture, no latent source | 56.2% |
| … with the source learned at every place | 36.9% |
| … with the source learned where content changed | 92.5% |

**Revision 1's result** (12,000 steps per world, `--mdl`). Fates are read
from an observed step by the posterior: the way and source that describe
it most briefly, given both frames (as the rules read observed steps). The
way head, which predicts before the outcome, gives the gate.

| World 2 | Gate (bar 99%) | Agent | Doors | Several colours | One colour, middles / ends | All object tiles (bar 95%) | Objects one token (bar 95%) |
|---|---|---|---|---|---|---|---|
| First run, change head first | 96.8% | 99.0% | 0% | 0% | 0% / 0% | 81.4% | 0% |
| Revision 1, posterior | 90.7% | 99.99% | 100% | 82.8% | 0% / 47.1% | 94.4% | 22.0% |

| Crafter | False groups (bar ≤ 1%) | Identity kept | Return log-likelihood | In the 90% region |
|---|---|---|---|---|
| The rules (102.2, 102.3) | 0% | 81.1% | −1.51 | 89.4% |
| Revision 1, posterior | 81.2% | 15.2% | −2.24 | 86.1% |

World 3's run did not finish within its 20 minutes (the grouping, on
poor tracks); its network did not converge (40 nats per state at the
end, against 8.5 in world 2), nor did Crafter's (20 nats).

- **Unambiguous moves are now read.** In a fixed view the shortest
  description moves a shape rather than drawing it (82.8% from 0%), the
  agent 99.99%.
- **One-colour middles are not** (0%), as predicted: with the same
  content, keeping is shorter than any copy, because each place's motion
  is coded alone. Coding a shared motion once (smoothness, as optical flow
  solves the aperture problem, Horn and Schunck 1981) is the untried
  remedy.
- **Where the view moves, it fails.** Every place changes when the view
  moves, so the network must explain the whole view each step, and it did
  not learn to (world 3, Crafter).

**Revision 2: the view placed first** (overnight, 2026-10-12, `--ego`).
- **The step:** card 102.5's placing reads each step's view shift (100% right
  in world 3, 93.0% in Crafter). The next frame is re-indexed so that place
  i shows the same world spot at both steps. The pointer then explains only
  motion in the world frame; places that left the view are out of the loss.
- **The readout:** observed steps now price copy by its best source,
  min over j of (naming i's code from x_j − log p(j | action)). Before, it
  used an average over the learned pointer.

| | World 2: several colours / objects one token | World 3: false groups / identity kept | Crafter: false groups / identity kept / log-likelihood |
|---|---|---|---|
| The rules (102.2, 102.3) | 100% / 99.5% | 0% / 97.7% | 0% / 81.1% / −1.51 |
| Revision 1 | 82.8% / 18% | did not finish | 81.2% / 15.2% / −2.24 |
| Revision 2 | **85.5% / 71%** | **0.01%** / 9.9% | **0.02%** / 10.8% / −3.21 |

- **Placing the view first lets the network converge.** World 3 reached 12.6
  nats per state (40 without it) and Crafter 16.4 (20). False groups fell to
  0.01–0.02% (bar 1%): criterion 2 holds.
- **In view, Crafter's objects keep their token** from one step to the next
  61–95% of the time, by kind. In world 3, boxes and lights keep it 100%,
  cats 55%, and the rolling ball 8%: the ball's code includes the floor
  beneath it (card 102.3's lesson), so no source names it exactly.
- **Out of view, belief fails** (identity 10–11% against 81–98%). With every
  tile a token, card 102.3's files hold grass and walls as well as
  creatures. Which tokens deserve a file was the ground rule's work; the
  user's criterion (a token with dynamics of its own) is the replacement.
- **One-colour middles stay at 0%.**
- **A fifth way, move and change** (`--cm`: copy plus a paid drawn change),
  was never chosen (0% of places in world 3 and Crafter), and nothing
  changed (Crafter identity at return 10.8%). Why creatures lose their token
  in view is still open.

**Decision: revise, closed for tonight** (overnight, 2026-10-12). The
pointer does not yet replace the rules in a moving view, so layers.py
stays as a declared exception (CHARTER rule 8) until it does. The next
revision reads the view's own motion first, by card 102.5's placing,
which reads it right in 100% of world 3's steps, and asks the pointer to
explain only what moves in the world frame (ego-motion compensation, as
in robotics). Then a shared motion coded once, for the middles.
Revision 2 (above) did the first: criterion 2 now holds, and criterion 3
waits on which tokens get a file.
