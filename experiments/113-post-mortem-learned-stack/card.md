---
id: "113"
title: "post-mortem: from a hand-built stack to conditions over a learned goal-reaching stack"
rung: 0
serves: [P12, P9, P21, P6, P18]
status: done
verdict:
arch_version: 27
date: 2026-10-31
---

# 113: post-mortem: from a hand-built stack to conditions over a learned goal-reaching stack

A change of direction (CHARTER rule 1), drafted at the user's request
and signed off by the user on 2026-10-31 ("approved"). Rule 1's trigger, a rung failing after its
feasibility gate passed, has not occurred: card 109 was decided revise,
and card 111's verdict against rung 1's bar was ruled uninterpretable. As
with card [022](../archive/022-effects-post-mortem/card.md), the user
decides the change directly. This card records why. It runs nothing. The
wording for CHARTER and AGENTS in the appendices has been applied; the
GOAL wording in appendix B is a suggestion for the user to edit.

## 1. What was tried

About 157 cards since 2026-09-24 built every level of the agent by hand
and tested each one separately (CHARTER rule 3):
- **The MiniGrid agent** (version 27, cards 001–101). It learned each
  action's effects from about 3.2 million steps of reward-free random play
  and planned backward over the conditions of those effects. It solved
  DoorKey-8x8 in 200 of 200 episodes, BlockedUnlockPickup in 100 of 100,
  and ObstructedMaze-Full in 30 of 30, plus 100 of 100 fresh layouts.
- **A general rebuild** (series 102–112), so the same brain could run in
  Crafter, Craftium and robotics:
  - perception that turns frames into tokens (102–104, 107);
  - one transition model over token vectors (105);
  - conditions read from that model or deduced from memory (105.4, 106,
    111);
  - belief about places out of view (108, 109).

## 2. What worked

- **Planning backward over learned conditions works.** It works in
  MiniGrid (above). In breaking world 1, the planner on learned statements
  solved 131 of 155 solvable layouts, exactly as many as the same planner
  given the world's true rules; random play lit the lamp in 13% of layouts
  ([105.5](../archive/105.5-planning-both-ways/card.md),
  [106](../archive/106-sleep-phase/card.md)).
- **Conditions can be learned without a vocabulary of events.** In
  Crafter, the conditions of crafting and moving came out at precision
  0.99–1.0 and recall 0.95–1.0 (106), against a precision of 0.12 for
  "everything near the change".
- **Memory decides choices where the view cannot.** Card 109's belief
  chose the right cave in 95–99% of episodes. After two histories with the
  same view, the choice followed memory in 98–100%.

## 3. What failed, and where

The failures fall in two places, and neither is the conditions idea:
- **Perception was rebuilt for every world and kept limiting what lay
  above it.**
  - Crafter's learned placing put only 76% of blocked moves in the right
    place, and the transition model learned those errors as false
    conditions on distant creatures (105.4: precision 0.23–0.31).
  - Pretrained trackers lose identity once a creature leaves view.
    CoTracker3 keeps a Craftium cow 13% of the time after it returns
    (card 107's smoke tests), and SAM 2.1 reaches IDF1 0.60 in Crafter
    (104).
  - Depth tuned on the agent's own motion did not halve its scale spread,
    and still placed cows 31% too near (107, stop). Learned belief
    identity did not beat counted motion (108, stop).
  - Night codes caused most of the remaining errors on "do" (105.8).
- **Conditions found as contrasts over discrete facts do not carry past
  grids.** The method needs a fact vocabulary such as "cell (r, c) holds
  code k".
  - **Position proxies.** With fixed borders, a far wall stands in for the
    agent's place and wins the contrast: blocked moves reached 0.35
    precision and 0.27 recall (111).
  - **Facts that always occur together** cannot be separated by memory.
    Trying where memory was open beat random play by only +0.011 (112,
    stop).
  - **Confidence in the model's own space** missed exactly the conditions
    the model had not learned (110, stop).
  - A robot's camera has no cells, so the vocabulary itself is the
    environment-specific prior that CHARTER rule 9 asks us to remove.
- **No acting agent beyond breaking world 1.** Skills, Roadmap step 4,
  were never built. Card 097.1's learned walking stopped, and every later
  step waited on perception. Nothing has been judged in Craftium, the main
  world since 2026-10-13.

## 4. Why the approach, not one component, is the problem

Each level was a research problem of its own, solved by hand, and each new
world reopened the levels below. The community already has learned, tested
methods for exactly those levels:
- visual features without labels (DINOv2, V-JEPA 2);
- latent world models on them (DINO-WM, LeWorldModel);
- goal-reaching skills learned from reward-free play (HIQL, QRL,
  quasimetric representations; OGBench).

What is missing in that literature is this project's idea: conditions
learned as such, with subgoals chosen backward from them. OGBench's best
method on pixels, HIQL, reaches 49 ± 4% overall on `visual-scene`: 80%
on open, 81% on unlock-and-lock and 61% on rearrange-medium, but 20% on
put-in-drawer and 3% on rearrange-hard (Park et al. 2024, appendix table
"Full results on Visual Scene"). The paper's own example of what the
tasks need is a chain of conditions: "unlocking the drawer, opening it,
putting the cube in the drawer, and closing it again".

## 5. The new direction

Use learned, published methods for the levels below conditions. Build
conditions on top.

| Level | Method | Source (`papi`) |
|---|---|---|
| Perception | A frozen self-supervised encoder (DINOv2 patch tokens), or a small encoder trained end to end; chosen in the thinking session (section 7) | `dino-wm`, `leworldmodel`, `v-jepa-2` |
| System 1: one transition model | An action-conditioned predictor of the next patch tokens | `dino-wm`, `leworldmodel` |
| Skills (Roadmap step 4) | A goal-conditioned policy and value from reward-free play by hindsight relabelling, with a learned temporal distance | `hiql`, `2304_01203` (QRL), `2509_20478`, `1707_01495` (HER) |
| Goals as conditions | A goal given as example frames of the condition, not one goal image | `2103_12656` (example-based control), `disco-rl` |
| Conditions of skills | A classifier, in the encoder's space, of the states where a skill reaching condition c succeeds against those where it fails | `from-skills-to-symbols-learning-symbolic-representations-for`, `2205_02092` |
| System 2: subgoals | Backward search over remembered states, with edges from the learned closeness and conditions from the classifiers; no imagined rollouts | `1906_05253` (search on the replay buffer) |

- **Main world:** OGBench `visual-scene`. A UR5e arm with a 5-D
  end-effector action, a cube, a drawer, a window and two buttons. Each
  button toggles the lock on the drawer or the window. The camera gives
  64 × 64 RGB images, and the play data has 1 million transitions from
  scripted random interaction. It is MiniGrid's key and door on a robot
  arm.
- **Craftium and Crafter** become later portability tests (the user,
  2026-10-31), run when a version is to be adopted, as MiniGrid's tiers
  were.
- **The tension with GOAL P12** ("does not forecast when", 2026-10-14).
  The agent learns a sense of how close it is to states where a condition
  holds. It is the skills' learning signal and the planner's cost, not a
  forecast that System 2 reports.
  - **Biology.** The brain has such a graded closeness, not a count of
    steps:
    - hippocampal and entorhinal activity tracks path and straight-line
      distance to a goal (Howard et al. 2014; Sarel et al. 2017 in bats);
    - place cells behave like a successor representation, the discounted
      expectation of which states follow (Stachenfeld, Botvinick and
      Gershman 2017);
    - striatal dopamine ramps up as a reward comes closer (Howe et al.
      2013).
    Most of this evidence is spatial; for abstract conditions it is
    thinner.
  - **Card 001's lesson still holds:** distance to a goal state is not
    distance to a condition (rank correlation 0.01–0.05). So closeness is
    measured to the set of states where a condition holds, given by
    examples, never to one goal state.

## 6. What stays and what stops

- **Stays:**
  - GOAL's properties and constraints, with the changes in appendix B;
  - CHARTER rules 0–9;
  - LESSONS;
  - the conditions principle: an effect, the conditions it needs, its
    spoilers, found by contrasting successes with failures;
  - recall and memory as the planner's search space;
  - the held intention;
  - acting where an outcome is least known;
  - the MiniGrid paper, as the record of version 27.
- **Stops:**
  - hand-built perception, tracking and depth for Craftium;
  - belief identity for look-alikes;
  - significance tests over discrete facts as the main source of
    conditions. Cards 111–111.3 stay as evidence and as a baseline in grid
    worlds.
- **Archived:** cards 102–112 move to `experiments/archive/`, as cards
  001–101 did.

## 7. What comes next

By rule 1, the session after the switch is thinking only: no code, no
runs. It reads the sources above and settles:
1. **The encoder.**
   - Frozen DINOv2 (ViT-S) at DINO-WM's 196 × 196 input gives 196 patch
     tokens of 384 numbers per frame. Stored for 1 million frames that is
     about 150 GB in half precision (1.7 TB is free), or it is recomputed
     on the fly.
   - A LeWorldModel-style encoder trained end to end avoids the storage
     and the frozen features' blind spots, at the cost of training it.
2. **Whether the arm's own state is given.** GOAL C1 allows the agent's
   pose. OGBench's visual observations are images; it is to be checked
   whether the arm's state can be added.
3. **The form of closeness** (the user, 2026-10-31). A discounted
   successor representation, as contrastive RL learns it
   (`2509_20478` unifies it with quasimetrics), is the biological form. A
   count of steps (QRL; HIQL's value) is not. The session compares them;
   the successor form is preferred unless the sources show it cannot
   carry the scene's longer tasks.
4. **The new ladder and roadmap** (appendix A), with exact tests for
   rungs 1 and 2.

**Settled in the thinking session** (2026-10-31; LITERATURE's current
focus gives the sources):
1. Frozen DINOv2 ViT-S patches, at frameskip 5 (about 30 GB of features).
   LeWorldModel's single CLS vector would likely drop the lock cue; it is
   the fallback, with patch tokens kept.
2. The arm's state is in OGBench's `info` and dataset, but card 114 leaves
   it out to stay comparable with OGBench's published agents.
3. Closeness takes TMD's form, which is successor-like and optimal. A
   condition's closeness is a logsumexp over its example frames. A
   hierarchy is needed (only HIQL scores above zero on scene's tasks 4–5),
   and backward search over memory supplies it.
4. Rung 1's test is card 114's.

Then the first cards, each approved before it runs:
- **114, the feasibility gate.** Does System 1 on the chosen features
  predict a lock's effect: pulling the drawer opens it when unlocked and
  not when locked? Upper bound: the same model on OGBench's state
  observations. Trivial baseline: no change.
- **115, the baseline.** HIQL on `visual-scene-play-v0` from OGBench's
  reference code, matched to its published 49 ± 4% overall.
- **116 and after:** goals as example frames, then conditions of skills,
  then backward search, one component per card.

## 8. Decision

**Signed off by the user, 2026-10-31.** Applied the same day:
- CHARTER and AGENTS changed as in appendices A and C;
- cards 102–112 moved to `experiments/archive/`, their links fixed.

Left for the user: GOAL (appendix B), and LITERATURE's "Current focus",
which only the user changes. The next session is thinking only (rule 1).

## Appendix A. Proposed CHARTER changes

**"Current world": replace the section with:**

> **OGBench `visual-scene`** (the user, 2026-10-31; Park et al. 2024). A
> simulated UR5e arm with a cube, a drawer, a window and two buttons that
> lock and unlock the drawer and the window.
> - **The learner receives** the 64 × 64 RGB frame, its previous action
>   (5-D end-effector displacement and gripper), the arm's own state if
>   OGBench provides it (proprioception, GOAL C1), and, at test time, a goal
>   given as example frames of the target condition.
> - **The learner never receives** object states, lock states or the
>   evaluator's success flags while learning. The state observations are
>   for scoring and for upper bounds only.
> - **Experience:** `visual-scene-play-v0`, 1 million transitions of
>   scripted random interaction, without reward (declared, C3).
> - **Portability tests:** Crafter and Craftium (card 107's setup) are run
>   when a version is to be adopted, in place of MiniGrid's three tiers.
>   Rule 10's breaking worlds stay as smoke tests where the adapter can read
>   them.

**Capability ladder: replace rungs 1–7 with:**

| Rung | Property | Capability | Test that shows it | Status |
|---|---|---|---|---|
| 1 | P6 | System 1 predicts action effects from pixels | Card 114's gate: lock, drawer, window and cube effects predicted from frames, against the state-based upper bound and "no change" | open |
| 2 | P9 | Skills reach single conditions | A goal-conditioned policy reaches one-condition goals given as example frames (drawer open, button pressed, cube at a place) from new starts | sketch |
| 3 | P12 | Know the conditions of skills | Learned preconditions name the true ones (opening the drawer needs it unlocked), scored against the simulator's state, precision and recall ≥ 0.9, with the P19 curve | sketch |
| 4 | P12 | Act to reach an unrewarded subgoal that enables a goal | On tasks that need an unlock, the agent presses the button first without reward for it; success against HIQL's published per-task scores | sketch |
| 5 | P21 | Deliberate over chains of conditions | `visual-scene`'s longest tasks (up to eight atomic behaviors), with compute reported | sketch |
| 6 | P2, P10 | Evaluate conditions from memory | Defined when rung 5 passes | sketch |
| 7 | P3, P8 | Transfer and composition | Held-out combinations; `visual-puzzle`; Craftium | sketch |

**"Current direction": add at the top:**

> - **Learned methods below conditions** (the user, 2026-10-31, card 113).
>   Perception, System 1 and skills use published learned methods, cited in
>   LITERATURE.md (rule 7(b)). This project's mechanisms are conditions,
>   goals given as conditions, and choosing subgoals backward over memory.

The existing bullets stay. Where a bullet names a mechanism this change
stops (perception supplying "where" by pose for Craftium, belief supplying
"which one"), it is marked as paused until the Craftium portability test.

**Roadmap:** steps 1–7 are kept as parts. Step 1 is rebuilt on the new
encoder, and step 4 (skills) moves before steps 2–3, since conditions are
now read from skills' successes and failures.

## Appendix B. Suggested GOAL wording (only the user changes GOAL)

- **Opening paragraph:** "Progress could look like `Mgrid -> Crafter ->
  Minecraft`" could read "Progress could look like `Mgrid -> robot
  manipulation in simulation (OGBench) -> Crafter and Minecraft -> real
  robots`".
- **P12, after "It does not forecast when the goal will be reached":**
  add "It has a learned sense of how close it is to states where a
  condition holds, as the hippocampus has of goals. This trains its skills
  and gives the cost by which alternatives are compared." (Revised twice
  on 2026-10-31: the first wording was confusing, and the user found
  "how many steps" had no biological parallel; section 5 gives the one
  that exists.)
- **C1:** "A declared, pretrained perception front end" could add "or a
  self-supervised visual encoder, with no language or class labels".

## Appendix C. AGENTS.md changes that follow

- "Cards are judged on Craftium" becomes "Cards are judged on OGBench
  `visual-scene`".
- MiniGrid's tiers as the adoption test become Crafter and Craftium as
  portability tests, with the same default margins until the user sets
  others.
