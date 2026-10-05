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
  Seeds 400–402 running.
