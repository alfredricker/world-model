---
id: "013"
title: walking in the recursion
rung: 0
serves: [P12, P21, P9]
status: draft   # draft | approved | gated | running | done | abandoned
verdict:        # pass | fail | uninterpretable, when done
arch_version: 0
date: 2026-09-27
---

# 013: walking in the recursion

## 1. Question

Card 012 found every requirement of the goal from one signal, but the agent
reached the goal in only 30% of new key-world layouts. With exact walking
it reached it in 95%. The recursion stops at "a ready state can be reached
by walking", and one flat walk value per way has to cover the whole walk.
Can the recursion instead continue down into walking, so that a long walk
becomes a chain of short ones, and the agent acts successfully? Before
rung 1; serves P12 (subgoals down to actions), P21 (the lower levels of
the hierarchy become short, fast decisions) and a first form of P9.

## 2. What changes

One component: how a way's ready states are reached. The condition tree of
card 012 (run 5's method, `runs/012run5.sh`) is unchanged.

Before (card 012):

```
way w ── condition: a ready state of w is reachable by walking
         walk: one discounted value per way, over the whole walk
```

After:

```
way w ── condition: unchanged (reachable by walking at all)
         walk: place conditions, found by the same recursion
           A0 = at a ready state of w
           A1 = A0 reachable within h walking steps      (way: a move, into A0)
           A2 = A1 reachable within h walking steps      (way: a move, into A1)
           ... until A_k is on in the current frame
         act: from the smallest k with A_k on, walk (value with horizon h)
              to a state where A_{k-1} is on
```

Each A_k is a goal like any other in the tree: its way is "a move" (the
three walking actions as one), its ready states are where a move switches
A_{k-1} on, and its condition is "within h steps of those". Each A_k is
learned as reachability within h steps (fixed-horizon values, De Asis et
al. 2020: horizon j bootstraps from horizon j − 1, no discount), trained
through the encoder as in card 012. The walk toward A_{k-1} only has to be
right over at most h steps. A chain stops when A_k is on in most (> 50%)
of the frames where w's condition is on, or at k = 8.

**Why this should help.** Card 012's walking was accurate where targets
are near and common (94% of moves closer walking to the door with the key)
and poor over long walks through rare regions (61% to the goal square
through an opened door; the agent spun in the doorway). Reachability
within 4 steps was 99–100% right on the bench once the encoder was trained
on the level above (card 012, fixed horizon, distance 0–4; 91% without). The chain turns one long, fragile value into several short,
accurate ones. The doorway should become an intermediate place condition
(A_k switches on only by passing through it) instead of a spot where the
value is flat.

**Declared:** h = 4 (the range where the bench's reach was reliable); the
three walking actions treated as one action "a move" for the evidence test;
k ≤ 8; 5% random moves while acting. Everything else as in card 012.

**Not chosen here:** letting walking steps be ways of the ordinary
conditions (run 1 of card 012 showed noisy values turning every move into a
fake way), and bottleneck subgoals found by contrasting successful and
failed walks (McGovern & Barto 2001), which is the next option if the
chain fails at doorways.

## 3. Dependencies

- Card 012 (partial): the condition tree, values trained through the
  encoder, play starts; exact gates pass in all four worlds.
- Card 010 (pass): the evidence test for accepting a way.
- Fixed-horizon reachability: De Asis et al. 2020 (to be added to
  LITERATURE.md and papi); card 012's bench (encoder trained on the level above):
  recall 99–100% at 0–4 steps, 87–100% at 5–9, 71–97% at 10–14.
- Short-horizon subgoals for long walks: HIQL (`hiql`, a subgoal k steps
  ahead, k = 3 on pixel mazes); waypoints over a learned distance: SoRB
  (`1906_05253`).

## 4. Data check

Same data as card 012 run 5 (30k random episodes per world, a third with
play starts). Walking steps with the door open: about 1.5% of frames in the
key world (card 012). Count and report, per chain link, the walking steps
that switch A_{k-1} on (the events each link learns from).

## 5. Feasibility gate

On card 012's bench (`tools/card012/bench_joint.py`, extended), key world:

- **Upper bound:** exact place conditions A_k, learned short walk (h = 4)
  toward A_{k-1}: share of learned moves that bring the agent closer to w's
  ready states on held-out frames, for w = "forward onto the goal square"
  and "toggle the door". Must be ≥ 90%; if not, the card stops.
- **Learned chain:** learned A_k and learned short walk, same measure.
- **Trivial baseline:** card 012 run 5's flat walk: 61% (goal square),
  94% (door); random moves 45%.

Result of the gate, before the main run:

## 6. Success criteria and prediction

In the key and switch worlds, all learned, 500 new layouts each:

1. **Walking:** learned moves bring the agent closer to the chosen way's
   ready states in ≥ 90% of held-out frames, for every accepted way
   (card 012 run 5: 61% for the goal square).
2. **Acting:** ≥ 90% of new layouts reach the goal square (card 012 run 5:
   30.2%; random 0.4%; exact 100%).
3. **Structure kept:** card 012's expected conditions are found with AUC ≥
   0.95 against their meanings, and each A_k matches its exact counterpart
   ("within k·h steps") with AUC ≥ 0.95.

If key and switch pass, either and both are run with the same settings
(criteria 1–3 again).

Prediction: the gate passes (short walks were already accurate). Acting
rises well above 30%; whether it reaches 90% depends on A_k at k·h ≥ 10,
where card 012's bench recall fell to 71–97%. Runtime: about 30 minutes per
world (card 012 run 5 plus the chains), so runs are handed to the user;
the gate is about 10 minutes.

## 7. Result

## 8. Decision
