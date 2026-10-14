---
id: "098.5"
title: tokens that persist when they move — object files followed in the placing step
rung: 1
serves: [P7, P2, P6]
status: done
verdict: fail
arch_version: 27
date: 2026-10-10
---

# 098.5: tokens that persist when they move

Drafted 2026-10-10 at the user's request ("it would be ideal if the
tracking was integrated into existing architecture. Draft the tracking
card"). It runs before card 098.4, which predicts what tracked tokens do.

## 1. Question

Version 27's belief holds one token per place in the world and one
placement, which is the agent's own motion. A thing that moves by itself
reads as two places changing: a cow vanishes at one place and a cow
appears at another. Crafter's cows, zombies, skeletons and arrows move,
and in Minecraft or a robot's world most things can.

The placing step already finds the agent's own motion by matching the
view to belief. Suppose it also follows each token that the agent's own
motion does not explain, as an object file: a token with an identity, a
position and the motion it last made. Does each moving thing then keep
one identity as it moves, and is its motion read right? Rung 1. It
serves P7 (objects from continuity), P2 (belief about a changing world,
including what is out of view) and P6 (the motion is what card 098.4
then predicts).

## 2. What changes

One component: the placing step (ARCHITECTURE, Acting step 1) and what
belief holds.

```
version 27:  belief  = a token per world place + one placement
             placing = the placement whose predicted view is nearest the view (L1), among those the action could lead to
this card:   belief  = a token per world place (what stays put) + object files
             object file = (identity, token, position, last motion, step last seen)
             placing, level 1 = as now, from the places no file explains
             placing, level 2 = each file's new position near its expected one, by the same L1
```

- **One rule at both levels.** Choose the candidate nearest the view by
  L1 over the encoder's vectors, among the candidates that could follow:
  - for the agent, the placements its action could lead to (unchanged);
  - for a file, the positions within reach of its expected position.
- **Expected position: inertia.** The last position plus the last
  motion, as in SORT's constant-velocity filter. This is a declared
  prior; card 098.4's learned prediction replaces it (a zombie heads for
  the agent, a cow wanders). Reach is the agent's own step length,
  learned from its own motion, and grows with each step a file goes
  unseen.
- **Births.**
  - A file is born when, in one step, a token vanishes from a place and
    a similar token appears within reach of it: the token moved by
    itself. "Similar" means within the encoder's identity radius.
  - A token that appears with no source (an arrow just loosed, a cow
    coming into view) gets a file if it is similar to a token that has
    had one.
  - Otherwise the change stays a change in place, such as a tree that
    becomes grass.
- **Under a file,** belief keeps what was last seen at that place (the
  grass under a cow), so a moving token never overwrites what it
  crosses.
- **Out of view,** a file keeps its last position, motion and step last
  seen. A similar token that appears within reach of its expected
  position takes it up again.
- **Matching by vectors, not codes.** Light changes little from one step
  to the next, so a cow at dusk still matches its last position even
  though its code changes with light.
- **Not in this card:**
  - carried tokens (MiniGrid's hand, Crafter's inventory) stay as now;
  - the encoder is unchanged. Tracks can later become its training
    pairs for moving things, the follow-up card 098.2 named.
- **One piece of code for every world.** The files level takes tokens as
  (vector, position) with continuous positions, so it reads no grid.
  - MiniGrid's agent calls it in its observe step.
  - The Crafter harness calls it on recorded play. Crafter's level 1 is
    its adapter: the four learned shifts and the agent's own check that
    the view did not shift (cards 098.2–098.3).
  - In Crafter, positions are view-place centres, the grid cut that this
    adapter declares.

Code:
- `tools/card098.5/files.py`: the files level, installed in the agent's
  placing step with `WM_TRACK=1`.
- `tools/card098.5/collect.py`: new Crafter play with evaluator labels,
  which card 098.4 also uses (appendix A).
- `tools/card098.5/crafter.py`: the harness, Crafter's level 1 and then
  `files.py`.

## 3. Dependencies

- Placing by least L1 among the placements an action could lead to
  (cards 057, 061, 062).
- Card 098.3's encoder vectors; Crafter's learned shifts and view check
  (cards 098.2–098.3).
- Object files (Kahneman, Treisman and Gibbs 1992) and tracking by
  continuity in space and time (Pylyshyn and Storm 1988): LITERATURE,
  card 094's focus. Card 094's files were keyed by where each token was
  first seen and never moved.
- Predict, then assign: SORT (Bewley et al. 2016, not in papi), with
  births and deaths of tracks.

## 4. Data check

In card 098's 100,000 random steps, frames with each moving kind in view
(evaluator's map):

| | Cow | Zombie | Skeleton | Arrow |
|---|---|---|---|---|
| Frames | 43,356 (43%) | 11,626 (12%) | 9,794 (9.8%) | 8,528 (8.5%) |
| At night | 3,720 | 4,766 | 1,541 | 1,055 |

New play for this card and card 098.4:
- 300,000 steps on seeds 2000–2009, worlds the encoder never saw;
  workers 8–9 are judged.
- For the evaluator only, every object in view is recorded with a
  persistent identity, its kind, its position and, for arrows, its
  direction.
- How often each kind moves per step is counted there, before the gate.

## 5. Feasibility gate

- **Upper bound:** the same two levels on the evaluator's state (true
  kinds as tokens, true positions, the agent's true motion).
- **Trivial baseline:** no files, as in version 27. A moving token's
  identity is its place, so each move starts a new identity.
- **MiniGrid:** files born on tiers 1–3. Expected 0, since nothing there
  moves by itself.

Result of the gate, before the main run:

**Settled while building, before any score was seen** (`tools/card098.5/`):
1. **Distance.** Matching uses the encoder's own distance: Euclidean,
   with card 098.3's identity radius (0.771) in Crafter. In MiniGrid the
   radius is half the smallest distance between two catalogue
   appearances. The card said L1, which is MiniGrid's placing metric.
2. **Reach from the last position.** Reach is measured from a file's
   last position, one step length per step since it was seen, with the
   expected position (inertia) as the preference in the cost. Measured
   from the expected position, a cow that turns back would be two steps
   away.
3. **Figure and ground.** Locally, a move looks the same as the ground
   moving the other way. When a cow steps onto grass, "the cow moved
   here" and "the grass moved there" fit equally well. Gibson (1966) and
   Kaplan (1969) describe texture accreted and deleted at occluding
   edges: the surface behind is the one that continues. A pair therefore
   counts only when:
   - the token that arrived is unlike most of its unchanged neighbours;
   - the place it left now shows a token like most of its unchanged
     neighbours (the ground revealed).
4. **Carried tokens excluded in the agent.** The first MiniGrid smoke
   test gave the hand's row to the files level. A picked-up key then
   "moved into the hand", files were born, and the agent's belief of its
   hand was overwritten (2 of 10 tier 1 episodes lost). The card already
   excludes carried tokens; with the code fixed, 10 of 10 episodes are
   identical to version 27, and no file is born.
5. **Criterion 1 is pooled** over the four kinds; each kind's ID recall
   is reported.
6. **Each object draws from its own random stream** in the new play
   (seeded by episode and slot). This is for card 098.4's labels. With
   one shared stream, 40% of move steps diverged between the two copies,
   because which creatures update depends on the player's position. Now
   0.4% of move steps diverge, all from spawning, and no other action
   diverges. The rules of the game are unchanged.
7. **No new dependency.** The one-to-one assignments use a small
   Hungarian solver (`assign.py`), checked against brute force on 300
   random cases, in place of scipy.
8. **The agent's identity radius (a bug, found in the main run).** In
   MiniGrid the radius is half the smallest distance between two
   different catalogue appearances. The distance formula's rounding left
   about 4e-8 between identical vectors, so that became "the smallest
   distance" and the radius came out at 1.5e-8. Every place then looked
   changed, and two identical green balls side by side could pair up.
   The smoke test's 10 tier 1 episodes did not show it. With exact
   differences the radius is the intended one.

**Data check** (the new play, 300,000 steps):

| | Cow | Zombie | Skeleton | Arrow |
|---|---|---|---|---|
| Frames in view | 132,338 | 36,117 | 28,611 | 24,420 |
| At night | 10,474 | 14,381 | 3,943 | 2,810 |
| Moved between consecutive frames | 42% | 41% | 23% | 100% |

**Gate: the upper bound fails the card's bar** (`runs/098.5/truth.json`;
workers 8–9, 55,655 object frames).

| | IDF1, four kinds pooled | Cow | Zombie | Skeleton | Arrow |
|---|---|---|---|---|---|
| Upper bound: the files level on the true state | **0.853** | 0.837 | 0.973 | 0.952 | 0.666 |
| Baseline: identity by place (version 27), true state | 0.257 | | | | |

The kind columns are ID recall.

- **Every object frame has a file.** ID precision equals ID recall
  (0.853). All losses are identity switches.
- **Motion is right** on every step a file follows the same object
  (50,176 steps; 7,891 at night). Level 1 is exact by construction here.
- **Where the switches are** (worker 8, upper bound):

  | Kind | Type of switch | Count |
  |---|---|---|
  | Cows | swaps between look-alikes in view | 74 |
  | Cows | a file unseen for a while takes another cow | 54 |
  | Cows | a returning cow gets a new file | 51 |
  | Cows | a file lost, then replaced by a new one | 31 |
  | Arrows | a file taken over from another arrow | 51 |
  | Zombies | a file lost, then replaced by a new one | 18 |

  Two parts of the rule cause these:
  - **Inertia as the expected position.** Cows stay put on 58% of steps,
    so for them the last position is the better guess.
  - **Reach that grows without limit while a file is unseen.** An old
    file can take any cow that comes into view.
- **MiniGrid smoke test:** 10 tier 1 episodes are identical to version
  27, and no file is born.

By the card's rule ("Stop if the upper bound itself is below 0.9"), the
main run was not made. The user chose to revise rather than stop
("revise and run", 2026-10-10).

**The revision** (the card's one revision; the matching rule only).
Developed on worker 7, with long-term memory learned on workers 0–6, on
the true state. The judged workers 8–9 were run once, at the end.

| Rule (worker 7, true state) | IDF1 | Cow | Zombie | Skeleton | Arrow |
|---|---|---|---|---|---|
| First build | 0.856 | 0.850 | 0.973 | 0.985 | 0.638 |
| Expected position from the kind's most common next move | 0.871 | 0.866 | 0.984 | 0.990 | 0.657 |
| Reach fixed at one step length (no growth while unseen) | 0.824 | 0.744 | 0.974 | 0.900 | 0.986 |
| Both of the above (the change first proposed) | 0.828 | 0.754 | 0.984 | 0.917 | 0.949 |
| Where an unseen file could be, ruling out places seen empty | 0.879 | 0.844 | 0.973 | 0.983 | 0.851 |
| ... plus "a new object" as an option, at the kind's overall density | 0.831 | 0.753 | 0.974 | 0.944 | 0.956 |
| Probabilities from the kind's learned motion (Bayes filter) | 0.854 | 0.780 | 0.978 | 0.960 | 0.986 |
| ... with a belief map of the objects not yet seen | 0.919 | 0.893 | 0.977 | 0.991 | 0.912 |
| ... plus objects appearing out of nothing (**the revision**) | 0.928 | 0.893 | 0.977 | 0.991 | 0.983 |

Two runs with bugs are not in the table. In one, diagonal moves landed
between places, so no file was ever dropped (0.10). In the other, the
moves a file could make left out staying put (0.80).

What the revision does, in the files level only (`files.py`):
- **Where a file seen last step can be.** Its last position plus each
  move it can make: stay, move as the agent's own steps move (the prior
  before the kind has been seen to move), or make any move its kind has
  made. Each move has probability P(move | kind, the move it last made),
  counted from every file's history: long-term memory, like the kinds.
- **Where an unseen file can be.** A probability over places, carried
  forward each step by the kind's motion and set to zero at every place
  the view shows without it, then renormalised. This is a grid Bayes
  filter with negative information, as in search theory. Reach no longer
  grows by a fixed rule; it follows the kind's motion. A cow that stays
  put half the time is most likely near where it was last seen, and an
  arrow keeps flying. A file whose probability is all ruled out is
  dropped: every place it could be was seen without it.
- **A new object instead of a file.** Each changed place can go to a new
  object rather than to a file. The new object's probability there comes
  from two parts, the intensity of undetected objects in multi-target
  tracking (Mahler's PHD filter; Reid 1979):
  - a belief map of the kind's objects not yet seen. It starts at how
    often a place first seen shows the kind, spreads by the kind's
    motion, and is zero where the view has shown the place;
  - the rate at which the kind appears out of nothing (an arrow loosed),
    read from the view: how often a place shows the kind when the step
    before showed every place within the kind's reach and none of the
    kind.
- **One assignment** picks, for all files and changed places at once, the
  most probable explanation: costs are −log probabilities, plus the
  vector distance over the identity radius as before.
- Cost: about 1 ms per step on the true state (5 minutes for the
  300,000 steps of play, one process).

**Gate with the revision** (`runs/098.5/truth.json`; workers 8–9, run
once):

| | IDF1, four kinds pooled | Cow | Zombie | Skeleton | Arrow |
|---|---|---|---|---|---|
| Upper bound, first build | 0.853 | 0.837 | 0.973 | 0.952 | 0.666 |
| **Upper bound, revision** | **0.949** | 0.925 | 0.994 | 0.986 | 0.980 |
| Baseline: identity by place, true state | 0.257 | | | | |

Motion was right on all 50,470 steps a file followed the same object
(7,930 at night). The upper bound clears the card's 0.9 bar, so the main
run went ahead.

## 6. Success criteria and prediction

1. **Identity kept.** IDF1 is the share of each moving object's frames
   in view in which the file following it is the right one, over its
   whole time in view (Ristani et al. 2016). For cows, zombies, skeletons
   and arrows on the judged workers, IDF1 must be at least 0.90 and at
   least 95% of the upper bound's.
2. **Motion read right.** For matched files, a step's motion equals the
   object's true motion in at least 95% of steps, and in at least 90% at
   night. The agent's own motion (level 1) must be right as often as
   without files.
3. **Nothing changes in MiniGrid.** With the files level installed,
   tiers 1–3 and the 100 fresh tier 3 seeds are identical to version 27
   episode by episode, and no file is born.

Reported:
- by kind, and identity switches;
- taking up a file again after it left view;
- arrows: how often the file is born in the step the arrow appears;
- night, and time per step.

**Prediction.**
- Arrows are easy: a distinct look, and inertia is exact for them.
- Zombies stay near the agent and are seldom confused.
- Cows wander, and two cows side by side look alike, so swaps cost
  most. Expect IDF1 near 0.9, lowest for cows.
- Taking files up again after they leave view is the weakest part,
  since many cows look alike.
- MiniGrid is unchanged.

**Decision rules.**
- Keep if 1–3 hold. That gives version 28, with the files level in the
  agent; card 098.4 then runs on tracked tokens.
- Revise if criterion 1 fails only through swaps between look-alikes
  (the remedy is 098.4's learned expected position), or only at night
  (the remedy is encoder pairs from tracks).
- Stop if the upper bound itself is below 0.9.

**Budget.**
- Play about 3 minutes (with card 098.4's labels).
- Vectors for 300,000 frames about 6 minutes on the GPU.
- Tracking a few minutes; MiniGrid's tiers about 6 minutes.
- About 20 minutes in all.

## 7. Result

**Criterion 3, MiniGrid: holds** (`runs/098.5/c_tier*.json`,
`fresh_c_tier3.json`). With the files level installed, every episode is
identical to version 27, action for action, and no file is born:

| | Tier 1 | Tier 2 | Tier 3 | Tier 3, fresh seeds |
|---|---|---|---|---|
| Episodes identical to version 27 | 200 / 200 | 100 / 100 | 30 / 30 | 100 / 100 |
| Files born | 0 | 0 | 0 | 0 |

- Time per step on tier 2, run alone: 0.0059 s, against 0.0056 s for
  version 27 (5% more).
- Before the radius fix (item 8 above), one tier 3 episode and one fresh
  seed changed and failed. Each had a file born when the agent picked up
  a green ball next to an identical one (`runs/098.5/bugged_radius/`).

**Criteria 1 and 2, Crafter on the encoder's tokens: not measured.** The
run was stopped after about 10 minutes. On the true state the files
level needs about 30 seconds per worker; on the tokens it had not
finished the first worker. A check on the first 2,500 steps of worker 0
shows why:

- **Things that never move were taken for things that do.** Of the 37
  "moved by itself" births, 16 were on grass, 4 on trees and 12 on other
  static materials. Only 7 were on cows, skeletons and arrows.
- **One false birth makes the whole kind a mover.** Every later grass
  place whose vector changed then got a file of its own: 8,425 grass
  files, against 154 for cows. The level held 738 files at once.
- **Where the false births come from:**
  - **Night.** 27 of the 37 came at night, which is only 7% of the
    steps. Crafter's darkening noise moves token vectors past the
    identity radius: 19.6 changed places per step at night.
  - **Errors in placing the agent's own motion (level 1).** 9 of the 10
    daytime births came on steps where level 1 chose the wrong shift.
    Level 1 also ignores places near files, so spurious files make its
    errors more likely.
- The upper bound could not show this: on the true state a token's
  vector never drifts.

By the card's rules this is not the failure the revision was for (swaps
between look-alikes), and the one revision has been used. The decision
is the user's.

## 8. Decision

**Revise**, continued as [card 098.6](../098.6-what-moves-by-itself/card.md)
(the user, 2026-10-10). The revised files level holds identity on the
true state (0.949) and leaves MiniGrid unchanged. Its main run failed for
a reason the revision did not address: a kind becomes a mover on one
"moved by itself" event, and on the encoder's tokens, night noise and
placing errors give static tokens such events. Tracking, vision and
developmental psychology all decide this from accumulated evidence
(LITERATURE, current focus). Card 098.6 learns movability by recall over
many views and gives each file an existence probability it must earn.

## Appendix A. The new play

- Random play, uniform over the 17 actions, as in card 098: 10 workers
  of 30,000 steps.
- Every frame and action is kept, plus, for the evaluator only:
  - the semantic map of the view, the inventory, the light and the
    agent's facing, sleeping flag and world position;
  - every object in view, with a serial number given when it is added
    to the world, its kind, its position and an arrow's direction.
- Card 098.4's labels: before each step, a copy of the simulator does
  nothing instead. The player's own chance draws come from a separate
  random stream, so both copies' worlds draw the same numbers (card
  098.4, appendix A). This changes no rule of the game.
- IDF1 matches files to objects by position in each frame, with one
  global one-to-one assignment between file identities and object
  identities.
