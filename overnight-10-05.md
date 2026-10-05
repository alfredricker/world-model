# Overnight 2026-10-05

Temporary record (AGENTS.md, "Overnight sessions"). Objective: an encoder
that is well principled and scalable, keeps the goal/condition
hierarchy, and with which the agent still discovers conditions. Then
goals from example frames, then a view smaller than the map.

## Card 054: identity up to noise, and perception that keeps what is visible

Approved before the user slept. Codes become "the same appearance up to
noise" (no learned codebook); the encoder keeps every visible
difference apart by a margin.

- **Step A (codes only, on card 053's encoder):** the planner still
  reaches 100% in the familiar worlds and still discovers the right
  conditions (toggle reads the key, or the switch), with no codebook.
  Recall: matching key 45% → 100%, but other key 85% → 60%: card 053's
  encoder leaves no clean gap between noise and real differences (boxes
  of every colour within noise; doors 7× noisier than keys), so noisy
  copies split. As declared, on to step B.
- **Step B (the encoder keeps visibly different tiles at least m apart;
  three arms):** m = 0.5 is strongest: recall's colour cases 100% /
  100% (codebook: 45% / 85%), noise changes 1.8% of predictions (bar
  1%), the planner 100% in the familiar worlds, chained rooms 100%,
  cluttered 100% (version 10: 100%, 98%, 98–100%), and the right
  conditions admitted. m = 1.0 and m = 1.0 with noisy copies pulled
  together were weaker on recall (other key 52–58%). Picking up while
  holding fails under every encoder (rare in random play; a limit of
  recall's data). Next: seeds 400–402.
- **Steps C and D (seeds 400–402):** the planner holds on every seed:
  familiar worlds 100%, chained rooms 100%, cluttered 100% (version 10
  with its codebook: 100%, 98%, 98–100%); the right conditions are
  admitted on every seed. Recall's colour cases do not: other key
  62.5–70% on three seeds, noise agreement 96.8–98.2% (bars 90%, 99%).
- **Declared revision (each identity class gets its own noise scale,
  since doors are noisier than floor):** better on 3 of 4 seeds (noise
  agreement up to 99.0%, other key 70–92.5%), passes on seed 399 only.
- **Decision: revise.** The encoder keeps the goal/condition hierarchy
  and the agent still discovers conditions (the night's objective, on
  the planner's side); recall on noisy generator tries needs better
  colour separation where outcomes depend on it. Cards 056 and 057 use
  this encoder.

## Card 056: goals from example frames

The agent is shown five frames in which a goal holds (holding a given
key, a door open, the switch on) and infers the goal as the features
every frame shares that are rare in its own experience (Bayesian concept
learning, the size principle), then plans to it with version 10's chain
of conditions.

- **Main run (four encoders; results identical, since every encoder
  gives each catalogue tile its own identity):** the agent reaches the
  goal from frames in 99.2–100% per world, as with the goal written in
  (100%); shown another goal's frames, 2.5–27%. In the key world, the
  door goal always went through the key. Inference matches the
  simulator ≥ 95% except hold red in the key world (93.6%) and door
  green in the both world (77.5%), where five frames from a few long
  episodes share an incidental key. Rung 1's forks: 82–89% (bar 90%),
  the same with the goal written in; the misses are version 10's
  walking taking a detour where picking up a key in the way is faster.
- **Declared revision:** one goal frame per episode, so five frames are
  five independent examples. Acting 100% in every world; inference
  94.9–96.9% in three worlds, but the both world's door goals had only
  1–8 episodes of random play to draw from.
- **Decision: revise.** Goals given as frames work behaviourally (as
  well as goals written in, through the same chains). Left: forks lost
  to walking that never prices clearing the way (card 058), five frames
  sometimes too few, and rung 1's "how soon" (card 059).

## Card 057: a view smaller than the map

The agent sees MiniGrid's 7 × 7 view (six rows ahead, three columns to
each side) instead of a window holding the whole map. It keeps what it
has seen (its tokens) and, when it has no chain of conditions, walks to
the nearest spot from which unseen places come into view (frontier
exploration).

- **Smoke test:** 0% at first. Placing a partial view by matching alone
  picked wrong placements that saw only unknown places; placing it
  among the placements the agent's own action could lead to fixed it.
- **Seed 399:** with exploration, 100% in all four familiar worlds at
  1.10–1.17× the full-view agent's steps, and 100% in the chained rooms
  (1.16× the shortest route); random when stuck: 100% at about 1.7× the
  steps, and 99% in the chained rooms at 1.94×. Every remembered tile
  right at the end of all 220 episodes. Criterion 3 was mis-specified
  (its measure counts the tile an action changes, nonzero for the full
  view too); corrected once, openly, to what it was meant to check.
- **Seeds 400–402:** the same results (every encoder gives each tile its
  own identity).
- **Decision: keep → architecture version 11** (for you to confirm). Open:
  the starting memory is still learned from full views; walls do not
  hide what is behind them yet.

## Card 058: walking compares a route with clearing the way

Card 056's lost forks were detours where picking up a key in the way is
faster. Here walking also priced the chain that clears one token and took
the cheaper.

- **Result:** familiar worlds fell to 90%, 100%, 96.7%, 73.3% (version
  10: 100%). Traced: the agent turned left, then right, for whole
  episodes, because the cheaper chain flipped with every turn; the
  choice was remade every step and never kept.
- **Decision: stop.** Comparing chains by cost is right (P21), but the
  choice needs to be kept until it fails, as achievers already are.

## Card 059: how soon a goal is, from the chain of conditions

Rung 1's second criterion. The agent predicts the steps to a goal by
following its chain one condition at a time in imagination (walking
cost to face the next tile, plus one step for the act), never step by
step.

- **Result:** predicted against true fewest steps, rank correlation
  0.977–0.990 in all four worlds (bar 0.8), mean error about 0.3 steps,
  3–15 ms per estimate; goal-swapped control −0.28 to −0.05. Coverage
  with goals from frames 93.8–96.4% (bar 95%; with the goal written in,
  98.4–99.8%): the gap is five-frame goals with an unreachable companion
  feature. Declared revision: ten frames: coverage 98.2–99.4% in three
  worlds, 89.0% in the both world, where random play opened the door in
  only 1–8 of 600 episodes (too few independent examples).
- **Decision: revise.** The how-soon reading itself passes everything
  with the goal written in; rung 1 is still short on its forks (card
  056) and on goal examples where random play is thin.

## Card 060: walls hide what is behind them

MiniGrid's default occlusion on top of the 7 × 7 view. The goal square is
behind the locked door, so it is unseen until the door opens. Arm A:
version 11's exploration; arm B: look only at unseen places next to
known walkable ones, and if there are none, open a door next to unseen
places (key first) to see beyond.

- **Smoke test:** arm B at first had nowhere to look (the only unseen
  space was behind the locked door); with opening a door to see beyond,
  100% in 5 key-world and 5 both-world layouts.
- **Main run:** arm A 0–3% everywhere (it walks toward places behind the
  outer walls, which it can never see). Arm B: familiar worlds 100% at
  1.12–1.19× the full view's steps; chained rooms 71%: two doors took
  turns as the nearest to open, and the agent turned left and right
  for whole episodes.
- **Declared revision:** keep the chosen door between steps (as
  achievers already are). Chained rooms 100% at 1.61× the shortest
  route; familiar worlds unchanged; no wrong remembered tile.
- **Decision: keep → architecture version 12** (for you to confirm).

## Card 061: moves learned from the small view

The agent acted on the 7 × 7 view, but its model of how moves change
where things are was learned from play seen through a window holding the
whole map. Here it is learned from the same 7 × 7 occluded view.

- **Smoke tests:** plain counting made the turns singular (far places are
  seen almost only as wall, so they "predict" anything); requiring the
  place to vary fixed the turns; one rigid transformation per move,
  fitted by trimmed least squares, made all three exact.
- **Result:** the same transformations as before on all four encoders;
  "undraw" (how a tile looks after the agent steps off) the same on
  every tile, and now exact on the goal, where the full view had to
  imagine it; acting identical to card 060 in every test.
- **Decision: keep → architecture version 13** (for you to confirm).
  Left: what pick up, toggle and drop store as "in view" (card 062).

## Card 062: stored tries with the believed view

The last full-view piece: each stored pick up, toggle and drop recorded
everything in the 13 × 13 window. Here the stored play was replayed
through the 7 × 7 occluded view (checked row by row: the same actions and
views), and each try records what the agent believed was around it.

- **Result:** 92–96% of stored tries changed what they record as in
  view. The agent still discovers its conditions on all four encoders
  (key world: the key's relation to the door, or the hand; switch world:
  now "a yellow, on, switch in view" instead of "a grey switch in view",
  the positive cause, since under belief the switch may be unseen), and
  acts exactly as before (familiar 100%, chained rooms 100%).
- **Decision: keep → architecture version 14** (for you to confirm):
  the planner's whole starting memory is learned from the view it has.

## Card 063: replay what the transition model misses

Back to the night's first objective. Card 054's encoder keeps the planner
and its conditions, but on three of four seeds it does not reliably
learn that a key of another colour leaves a locked door shut (its
transition model right 70–85%, recall 62.5–70%): such tries are a sliver
of the training batches. Here half of each batch is drawn in proportion
to the transition model's own recent error (prioritized replay), so the
rare tries it gets wrong come back more often. No labels.

- **Smoke test:** 200 updates, priorities written for every drawn try.
- **Result:** worse on average. The transition model's "other key" 63%
  (card 054: 77%), recall's 59% (73%); matching keys and the planner
  unchanged (100%). It learned the rare case sooner but kept swinging by
  up to 35 points between checkpoints.
- **What it showed:** colour is readable from the whole vector (90–100%)
  but from any one part only 44–81%. Recall compares the key and the
  door by a distance per part, which mixes colour with kind; the
  transition model sees both vectors but learns "same colour" slowly.
  More frequent rare tries did not fix either.
- **Decision: stop.** The next encoder step is the comparison itself (a
  learned comparison of the two vectors, as your relations direction
  asks): a direction for you.

## Card 064: the small view on the harder worlds

Retention check for version 14: the chained rooms with two locked doors
(the goal unseen at the start, the second door found only after the
first opens) and the key world cluttered with extra keys, switches and
vases.

- **Result:** two doors 30/30 at 1.37× the shortest route; cluttered
  100/100 at 1.27×; no wrong remembered tile. The extra steps are the
  looking (the shortest route knows the map).
- **Decision: keep** (version 14 confirmed on these worlds).
