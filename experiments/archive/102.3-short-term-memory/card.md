---
id: "102.3"
title: short-term memory — every attended token kept out of view, its place spreading as its kind moves
rung: 1
serves: [P2, P10, P7]
status: done
verdict: pass
arch_version: 27
date: 2026-10-11
---

# 102.3: short-term memory

Drafted 2026-10-11 at the user's request, as the third card of the 102
series; approved by the user the same day ("proceed with 102.3") and
revised before any run (sections 2–6) for what 102.1 and 102.2 settled. The user's idea: "If I had attended to an object recently, I
keep it in my belief / short term memory and have an intuition for
whether it will stay put or move to a reasonable location."

## 1. Question

Belief out of view is two mechanisms today:
- **Version 27 (MiniGrid):** every token seen stays exactly as last
  seen. The world is assumed static, and any change is credited to the
  agent.
- **Card 098.6's files level (Crafter, offline):** a token that tokens
  like it have needed memory for (a mover) gets a file. When it leaves
  view, the file keeps a map of where it could be, spreading as its kind
  moves, for up to 20 steps. Other tokens fall back to the static map.

Neither scores what belief says about a token out of view. Neither has
a limit or a sense of how recently a token was attended to. MiniGrid's
static world is a declared fact there, not something learned.

Suppose there is one rule for every token. A token attended to recently
is kept, and its place spreads as tokens of its kind have been seen to
move: nowhere for scenery, a little each step for a cow. Its existence
fades as tokens of its kind vanish: an arrow soon, a table never. Only
a limited number are kept individually. Does belief then predict where
a token is when it comes back into view, with calibrated confidence,
and keep its identity? Rung 1. Serves P2 (belief about a changing
world, out of view), P10 (memory that helps a prediction the view
cannot make) and P7.

## 2. What changes

One component: belief out of view (what the files level keeps).

```
now:        in view: tokens and files; out of view: static tokens as last seen, files of movers for 20 steps
this card:  every token tracked in view (card 102.2's layers), once out of view: (code, where, existence)
              where:      a distribution over places in the frame of the episode's start (the view placed by its
                          own shift, 102.2), spread each step by its kind's learned motion:
                          P(this step's move | last step's move, code), counted over every tracked token of that code
              existence:  falls each step by its code's learned rate of vanishing in view, and on each miss: where
                          its place was predicted in view and it was not seen there (IPDA)
              return:     a token new to the view is the file of its code that best predicts its place, times its
                          existence, if that beats the code's density of new tokens; else it is new
              kept:       the K files most recently in view; a dropped file is forgotten (its kind's density remains)
            static is the case of zero spread: a code expected to move less than once in 50 steps is kept as last
            seen, in the map, outside K
```

- **One rule.** Nothing says which kinds move. A kind never seen to move
  gets zero spread from its own stored episodes. A kind seen to move
  gets its motion's spread. MiniGrid's "the world is static" becomes
  what the agent learned from its play.
- **Vanishing is learned per kind.** Arrows vanish within a few steps,
  creatures rarely, scenery never, each from stored episodes of tokens
  like it (how often one not seen again was gone).
- **Capacity by recency.** At most K files are kept individually, the
  most recently in view. Ranking by the intention's chain (card 097.2)
  waits for the agent, since this card runs on stored play. K is
  reported as a curve of prediction against cost. Human visual
  short-term memory holds about four objects (Luck and Vogel 1997), a
  reference, not a target.
- **Unchanged:** files in view (098.6); the placing step; arm C.
- **Not in this card:** long-term memory of places across episodes;
  the agent's own token (102.5); the world's own changes in place, such
  as a sapling growing (card 098.4).

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| Whether a kind moves is known in advance | no | no | no | removed: learned per kind |
| The world changes only by the agent | no | sometimes | no | removed for motion and vanishing; changes in place stay with 098.4 |
| Places on a grid (the spreading map) | yes | yes | no | kept: the map is the adapter's grid; robots need a continuous density |
| A token is one tile | creatures yes | no | no | 102.2's tracking; every object in these worlds is one tile |
| The view's own shift is readable | yes (99.6%) | yes | yes, by odometry | 102.2's background layer; world 3's floor is patterned for it |

## 3. Dependencies

- Card 102.2's layers: tracking in view, the view's shift, ground.
- Card 098.6's files level: the idea of motion by recall per kind,
  existence and the possible map. Its code is not reused; its rule is
  a baseline.
- Card 098.5's play (seeds 2000–2009, random actions): Crafter with each
  object's persistent identity, for the evaluator.
- Literature: object files (Kahneman, Treisman and Gibbs 1992) and IPDA
  or PMBM existence, both in LITERATURE. To be added: Luck and Vogel
  1997, and Cowan 2001 (the capacity of short-term memory).

## 4. Data check

From the evaluator's records of the test play (departures within an
awake segment; returns within 50 steps; median time away of those):

| World 3 (150,000 steps) | Departures | Back within 50 | Never back | Median steps away |
|---|---|---|---|---|
| Ball / cats | 3,325 / 4,053 | 2,850 / 3,309 | 313 / 482 | 8 / 3 |
| Boxes / lights | 7,275 / 2,968 | 5,836 / 972 | 958 / 1,996 | 4 / 2 |

| Crafter, workers 8–9 (60,000 steps) | Departures | Back within 50 | Never back | Median steps away |
|---|---|---|---|---|
| Cows / zombies / skeletons | 2,034 / 263 / 382 | 1,232 / 19 / 229 | 798 / 244 / 152 | 4 / 2 / 3 |
| Arrows / plants | 838 / 237 | 108 / 41 | 730 / 196 | 3 / 4 |

- **World 3's encoder** (card 098.3's recipe) first merged ball with
  box and cat with light. Its pairs assume a still view, and world 3's
  view follows the agent. Trained on pairs from the agent's own noops,
  it gives 9 codes, each one kind.
- **Crafter's codes** are 098.5's. 102.2's ground rule learns trees,
  stone and tables as ground, so they are not tracked as tokens.
- **The view's shift** is read as in 102.2.

**Breaking world 3, "out of sight"** (`tools/card102/worlds.py`,
Appendix A):
- a room with walls that hide;
- a view of 9 × 7 tiles following the agent;
- a ball rolling at constant speed and bouncing;
- a cat wandering at random;
- boxes that stay;
- lights that vanish after a while.

Returns per token kind and time away are counted when it is built.

## 5. Feasibility gate

- **Upper bound:** the same belief on the evaluator's tracks: true
  identities, true placement, and motion learned from true tracks. It
  bounds what perception costs.
- **Baselines,** in the same code:
  - "stays as last seen" (version 27): zero spread for every code;
  - 098.6's rule: spread for codes seen to move, each file kept for at
    most 20 steps, everything else as last seen.
- **Check:** on world 3, the upper bound places the ball, cats and boxes
  inside its 90% region on at least 85% of returns. If it cannot, the
  motion model's form is wrong.
- **At adoption, not here** (the user, 2026-10-11): MiniGrid's tiers,
  where every learned spread should be zero.

Result of the gate: **passes.** On world 3, the upper bound puts the
ball and cats inside its 90% region on 91.2% of 6,159 returns.

Three things were found and fixed in smoke tests on 6,000 steps, before
the main run:
- **Scoring:** a return is scored on its token's most recent file, not
  the oldest.
- **Region:** the 90% region counts its boundary place in proportion
  (criterion 1). Without that, the upper bound read 96.8% because the
  grid's regions over-cover.
- **Matching across codes:** a returning token is matched across codes
  by Q(b | a), the share of a tracked token's life spent looking like
  code b, over steps it looks like a. This is because 098.3's codes
  include the ground beneath a creature: 76% of lost Crafter identities
  came back with another code (a cow on sand, not grass).

## 6. Success criteria and prediction

1. **Where a token comes back.** In Crafter's worlds 8–9 (cows, zombies
   and skeletons, by day) and breaking world 3's test worlds (the ball
   and cats), for tokens that return to view within 50 steps, judged
   on the view at the return (where it reappears, given that it does):
   - the true place lies inside belief's 90% region in 85–95% of
     returns (the region's boundary place counted in proportion, so
     that it holds exactly 90% on a grid; set after the smoke test,
     before the main run);
   - the log-likelihood of the return place is above both baselines'
     (paired bootstrap, p < 0.05).

   This would show that belief out of view is a calibrated prediction
   (P2, P5).
2. **Identity across absence.** The returning token is taken by its own
   file in at least 80% of Crafter's returns by day and 90% of breaking
   world 3's. The baseline is 098.6's files level.
3. **Nothing lost, at a bounded cost.**
   - Codes kept as static (zero spread) are exactly where last seen on
     at least 99% of their returns.
   - In Crafter, belief out of view costs at most 1.2 ms per step at the
     chosen K: 10% of version 27's step (0.012 s). The chosen K is the
     smallest whose return log-likelihood is within 0.01 nats of
     keeping every file. 098.6 recorded no time per step.

**Reported:**
- prediction against K;
- each kind's learned spread and rate of vanishing;
- returns by time away;
- night returns (where 098.3's tokens are weak), apart from the criteria.

**Prediction.**
- Cows return inside the region about 90% of the time, since their
  random walk is what recall learns.
- The ball is well predicted until its first bounce off an unseen wall,
  then spreads.
- Arrows are correctly forgotten.
- Identity in Crafter is limited by night tokens and by 098.6's day-time
  breaks, so criterion 2 is the likely failure there.

**Decision rules.**
- **Keep** if 1–3 hold.
- **Revise** if only identity fails, for a cause outside belief (the
  tokens, or 098.6's day breaks).
- **Stop** if the learned spread is no better calibrated than "stays as
  last seen".

**Budget.** World 3 and its encoder: about 3 minutes. Layers and belief
on world 3 and Crafter's worlds 0–9 (300,000 steps): about 5 minutes per
form, three forms and the K curve under 20 minutes.

## 7. Result

Runs: `runs/102.3/chain.sh`, 2026-10-11, about 6 minutes. Numbers are
in [results.json](results.json); code is in `tools/card102.3/`.
Returns are those within 50 steps of the movers judged (world 3: the
ball and cats; Crafter: cows, zombies and skeletons, by day).

| World 3 (6,159 returns) | Learned | Stays as last seen | 098.6's rule | Upper bound |
|---|---|---|---|---|
| Log-likelihood of the return place | **−1.087** | −4.780 | −1.477 | −1.066 |
| Inside the 90% region | **91.2%** | 54.6% | 90.8% | 91.2% |
| Identity kept | **97.7%** | 23.3% | 85.7% | 99.0% |

| Crafter (1,357 returns) | Learned | Stays as last seen | 098.6's rule | Upper bound |
|---|---|---|---|---|
| Log-likelihood of the return place | **−1.507** | −4.145 | −1.614 | −1.353 |
| Inside the 90% region | **89.4%** | 62.8% | 89.4% | 91.4% |
| Identity kept | **81.1%** | 25.9% | 79.1% | 95.1% |

**Criterion 1, where a token comes back: holds in both worlds.**
- **Calibrated:** 91.2% and 89.4% inside the 90% region (bar 85–95%).
- **Above both baselines** (paired bootstrap over 463 and 552
  segments, p < 0.001 each):
  - against "stays as last seen", +3.69 nats per return in world 3
    (95% interval 3.61 to 3.78) and +2.64 in Crafter (2.49 to 2.79);
  - against 098.6's rule, +0.39 (0.35 to 0.43) and +0.11 (0.08 to
    0.14). The gain is all in returns after 20 steps, where 098.6's
    rule has dropped the file.
- **World 3 is within 0.02 nats of the upper bound.** Crafter is 0.15
  below, the cost of its tokens.

**Criterion 2, identity across absence: holds, narrowly in Crafter.**
- **World 3:** 97.7% (bar 90%).
- **Crafter:** 81.1% (bar 80%; binomial 95% interval 79.0% to 83.2%),
  against 79.1% for 098.6's rule and 95.1% for the upper bound.
- **Crafter's identities fade with time away:** 90.6% after 1–5 steps,
  70.5% after 6–20, 34.2% after 21–50. The upper bound holds 98.7%,
  91.2% and 76.3%.

**Criterion 3, nothing lost at a bounded cost: holds.**
- **Places kept in the map** (codes expected to move less than once in
  50 steps) are right when back in view in 100% of 248,221 cases in
  world 3. In Crafter only 5 such places came back, since 102.2's ground
  rule makes trees, stone and tables ground.
- **Belief out of view costs 0.04 ms per step** in Crafter at the
  chosen K = 8 (bar 1.2 ms). World 3's chosen K is 4.

**Reported:**

| Files kept (K) | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| World 3: log-likelihood / identity | −1.63 / 80.2% | −1.13 / 96.4% | **−1.09 / 97.7%** | −1.09 / 97.7% | −1.09 / 97.7% |
| Crafter: log-likelihood / identity | −2.05 / 64.4% | −1.70 / 76.3% | −1.54 / 80.0% | **−1.51 / 81.1%** | −1.51 / 81.1% |

- **By kind:**
  - world 3's ball −0.49 nats, cats −1.60;
  - Crafter's cows −1.49, skeletons −1.54, zombies −2.07 (29 returns).
- **Learned motion per code:**
  - in world 3, the ball's code moves on 96% of steps, cats' on 47%,
    boxes and walls on none;
  - in Crafter, cows' two main codes move on 41% and 44% of steps,
    skeletons' on 24%, the arrow's on 96%. The player's four headings
    move on 19–21% (four of the 17 random actions are moves).
  - **One perception slip:** coal's code "moves" on 2.2% of steps,
    enough to make it a file, not a map place.
- **Vanishing:** the arrow's code vanishes at 15% per step, against
  0.2% for the player and 2.3% for cows; 730 of 838 arrows never came
  back.

**Prediction against result:**
- **Right:**
  - cows return inside the region about 90% of the time (89.4%);
  - the ball is the best-predicted mover (−0.49).
- **Wrong:** identity in Crafter was predicted to fail and passes
  narrowly. Its limit is time away, not night (day only here).

## 8. Decision

**Keep.** Verdict: pass on all three criteria.
- **One rule for every token** keeps it out of view with a learned
  spread. That spread is zero for codes never seen to move, and it
  predicts where a token comes back, calibrated, far better than
  "stays as last seen" and better than 098.6's rule.
- **Short-term memory needs few files:** 4 in world 3 and 8 in
  Crafter, near the human span of about four (Luck and Vogel 1997).
- **ARCHITECTURE is unchanged** until the series enters an agent; the
  MiniGrid tiers run then (CHARTER rule 10).

**Limits carried forward:**
- **Crafter's identity after 6 or more steps away** (70% and 34%,
  against the upper bound's 91% and 76%). Crafter's tokens are the
  cause: codes that change with the ground beneath, and breaks in view.
- **102.2's ground rule** takes rare codes (trees, stone, tables; world
  3's light) as ground. The user rejected the rule itself (2026-10-12):
  card 102.4 replaces it with arm A's learned pointer.
- **Motion ignores walls:** a ball's bounce off an unseen wall is
  spread, not predicted.
- **Not run:** night returns, and ranking files by the intention's chain
  (it needs the agent).

## Appendix A. Breaking world 3, "out of sight"

- 20 × 20 rooms with internal walls, and the agent's view 9 × 7 tiles
  centred on it. Walls do not hide tiles inside the view; the rooms hide
  by being out of view.
- One ball moves one tile per step in a straight line and reverses at
  walls. One or two cats step to a random free neighbour with
  probability 0.5. Boxes never move. Lights vanish after 10–30 steps.
- The agent moves at random. 500 episodes of 300 steps each, training
  and test.
