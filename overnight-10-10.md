# Overnight 2026-10-10

The user's objective (2026-10-10): principled work on the series 095,
096 and 097, aiming at more than 90% on tier 3 (ObstructedMaze-Full,
version 20: 14/30) and starting Crafter evaluations this weekend. Cards
are approved under the overnight authority (as on 10-08); runs over 30
minutes run under memory caps. At most 10 new cards.

## Before the night

- Question cards written at the user's request (count toward the 10):
  095.4 moves, 095.5 goals over the predicted state, 095.6 one
  transition model, 097.2 running a skill, 097.3 compiled chains.
- 095.3 approved by the user. Gate passed (offline: front and hand only
  loses nothing; in the agent: 40/40 tries predicted).
- **Data check, tier 3** (version 20 timed, 3 episodes capped at 10
  minutes): recall for pick up, drop and toggle takes about 90% of a
  step: 185–201 s predicting categories and 347–360 s building results
  (`transport`), of about 600 s per episode, over 1.2–1.5 million queries.
  So 095.3's speed criterion can come from this card.

## Card 095.3: arm C in the agent

- The user (2026-10-10, mid-evening): the 10-card cap counts cards
  written from now on; the question cards do not count. The night goes
  to the cards already written and discussed.
- **Tiers 1–2** (`runs/095.3/c_tier*.json`, `v20_tier2.json`): tier 1
  200/200 (24.4 steps); tier 2 **100/100** against version 20's 95/100,
  in 55 steps against 120, 0.008 s per step against 0.020. Every real
  interaction try predicted right (498/498, tries forced by card 072's
  guesses aside; version 20 932/1,007). Pick ups predicted to put another
  real token in the hand than the one in front: 0 (version 20: 90,339).
  The 1,066 left are averaged vectors that version 20's `opened` builds
  (no real token), whose result arm C snaps to the nearest real one.
- Tier 3 started (30 seeds, one-hour cap, 44 GB).
- **Tier 3: 24/30** against version 20's 14/30 (11 seeds won, 1 lost,
  sign test p = 0.006), 0.038 s per step against 1.45; the 30 episodes
  in 2.6 minutes of wall clock against 100. No episode out of time.
- **Decision: keep** (all three criteria met without the user's margin).
  Architecture version 21.
- Six failures, all at the environment's 3,600 steps: 1003006, 009, 014,
  015 spend 1,900–2,600 steps on random fallback (21–32 guesses tried,
  none worked); 1003018 and 027 loop in exploration (2,100–2,800 steps).
  1003027 is the one seed version 20 solved and 21 did not.

## Toward 90% on tier 3: why the six fail

Traced all six (`runs/095.3/trace_*.log`, grids replayed): in every one
the blue ball is in a room behind locked doors whose keys are inside
boxes; the ball is never seen, so the planner has no plan for 3,600
steps. Toggling a box always changes it (2,889 of 2,889 stored tries)
into a key of one of six colours (15–18% each), but arm C returns one
result, card 072's guesses skip actions recall already predicts will
work, and no plan asks for a box's contents. Boxes were opened only by
random actions. Two seeds also loop: the agent drops a carried token to
free its hand for a ball in the way, the token lands on its route, and
the next step's plan picks it back up.

## Card 095.7: exploring uncertain results (new card 1 of 10)

- When there is no plan and nothing unseen to walk to: act on the
  nearest token whose result arm C is unsure of (the winning value's
  share of its way's neighbour weight below 0.8; a grey box: about 1/6).
  GOAL P15.
- Gate: 5 of the 6 failures solved, in 136–280 steps.
- Tiers: 200/200; 100/100 in 30.7 steps (55); tier 3 24/30 in 264 steps
  (1,677): 5 won, 5 lost. The five lost all loop on the drop/pick-up
  oscillation, now reached more often (keys known, more doors pursued).
- **Keep** (verdict fail on its tier 3 target; no tier worse; the
  losses are the next card's defect). Version 22.

## Card 097.2: an intention held between steps (a question card, drafted now)

- The user had placed the procedural series after cards 094 and 096.
  Taken up now because tier 3's remaining failures are exactly the
  oscillation an intention prevents, and the step reads nothing of card
  094's belief (no rework later). Card 093's design (the user set 093
  aside on 2026-10-09 and moved its intention to 097.2).
- First try: continuing the deepest unmet need of the old chain made it
  worse (1/7): the chain was not updated when a continued need got a new
  sub-plan, and met needs below were not protected. With both (the chain
  takes the new sub-plan; met needs below the continued one protected):
  7/7 on the looping seeds, 63–211 steps.
- Tiers: 200/200; 100/100 in 30.2 steps; **tier 3 30/30** in 129 steps.
- Fresh seeds (100, never traced) running, against version 21 on the
  same seeds, because the gates used seeds from the test set.
- **Fresh seeds:** 99/100 in 144 steps (return 0.95) against version
  21's 88/100 in 1,415 (return 0.57): 12 won, 1 lost, p = 0.003.
- **Keep:** version 23. Tier 3 30/30 against version 22's 24/30 (6 won,
  0 lost, p = 0.03). The surprise trigger never fired.

## The user, still awake (2026-10-10, late)

Asked which card solved tier 3 (answered: 095.3, then 095.7, then 097.2)
and whether the model can be benchmarked on ObstructedMaze: NovelD (RL
with reward and an exploration bonus) reports 0.96 on OM-Full, RND,
ICM, RIDE and AMIGo 0.00; ours 0.95–0.97 in MiniGrid's return, but with
declared priors (tile grid, planner, goal as a condition) and from 3.1
million steps of random play without reward. Not like for like.

## Crafter (card 098, new card 2 of 10), paused

- Crafter installed (`pip install crafter` in the project's
  environment; not in pyproject.toml). Random play: 100,000 steps.
- Encoder from continuity, first run: purity 96% by day, 74% at night
  (100% at the player, 67% at the view's edge), 85 codes for 17 classes;
  the within-frame negatives likely split one class at night. Recorded
  in the card.
- The user (2026-10-10, late): finish the MiniGrid backlog before going
  deep into Crafter, unless saturation prevents good evaluations; also
  no need to run the other ObstructedMaze variants (stopped and
  removed).

## Backlog first (the user, 2026-10-10, late): cards 091–097.x

- Order: 095.5 → 096 (review, then its first step) → 095.6 beside 096's
  encoder step → 095.4 → 097.1 → 097.3; 092's reveal re-run on version
  23. Card 093 closed as superseded by 097.2.
- **095.5** (drafted from the question card; approved overnight): the
  planner's candidate hands (conditions, "can be opened") from arm C's
  store instead of version 20's stored situations. Shadow: 0 of 342,340
  condition calls and 0 of 9,463 openable calls differ (tiers 2–3).
  Goals over any change in the predicted state moved to Crafter, where
  the state has more than a front and a hand.
- **095.5 keep** (version 24): tiers 200/200, 100/100, 30/30, fresh
  99/100, every episode identical to version 23's; version 20's stored
  situations read 0 times. Follow-up noted: their stores are still built
  at setup (about 90 of tier 3's 125 s) and could be dropped once arm C
  fits its own β.
- **095.4 and 095.6 closed, moved to Crafter** (the user's saturation
  clause): forward is already own tries + card 087's prior; a joint
  encoder can only tie arm C's 99.84–100% here and would be retrained
  for Crafter's pixels anyway.
- **092 re-run on version 24** (its revision was for speed, which is
  gone): running.
- **092 re-run on version 24 → stop.** Attached at exploration (at the
  fallback it fired 0 times): tiers 1–2 identical, tier 3 137.7 steps
  against 128.8, fresh seeds 147.6 against 144.5 (more steps in 58,
  fewer in 35, p = 0.02). In ObstructedMaze the ball's corner is random,
  so "doors like this revealed the goal" is weak evidence.
- **096 literature review** (a background agent) → LITERATURE.md's
  current focus set to card 096 (card 094's had expired). Card 096 is
  now a series (096.1–096.4).
- **096.1 (new card 3) keep**, offline P19 curve: arm C is right on
  97–100% of events seen once, but never-seen events fail by kind (doors
  "pickable"), state (closed doors "do nothing") and the key–door
  relation (also at n = 1 on tier 3: GOAL P19's own example); arm A,
  trained by try, is 49–86% per distinct event on pick up and toggle.
- **096.2**: a sleep replaying into arm A by event; running.
- **096.2 (new card 4) stop.** Arm A replayed by event, own world, α
  fitted on events memory lacks. Offline it looked better (20 won, 1 lost
  on held-out events), but in the agent tier 2 fell to 89/100 and tier 3
  to 769 steps: the deployed arm A, queried with two tokens, predicts a
  dropped key stays in the hand, and α = 10,000 let it override memory.
  The offline gain was the per-event score rewarding "nothing happens"
  (most combinations change nothing), and every retrained arm A is
  erratic on two-token queries (trained on whole states). 096.1's record
  rewritten on events that change something: toggles never seen 29–46%
  (vote alone 68–88%), after one 90–94%, after three 92–97%; arm A 0% on
  toggle changes. Two lessons added.
- **096.3 (new card 5) stop.** Arm A trained on the two tokens it is
  asked about: never-seen toggle changes 46% → 77% (tier 2) and 17% →
  58% (tier 3) for arm C, still below the vote alone (85%, 63%); the
  three seeds agree on only 42–54% of toggle changes; every copy misses
  boxes and locked doors with their own key. Not needed in MiniGrid with
  full memory (one occurrence gives 85–100%); back with Crafter.
- **096.4 closed by a check:** the few-occurrence curve is identical with
  and without forgetting, because in version 24's front-and-hand store
  forgetting drops only 0–2.4% of groups (095.2's 18-fold saving was on
  the whole-state store). Returns with whole-state stores in Crafter.
- **096.5 (new card 6) stop.** Episodes recorded as condition changes
  (`WM_EPISODES=1`). Tier 3: remaining steps predicted from stored
  episodes with the same change are no better than from the count of
  changes done (mean error 42.1 against 40.8 steps; better at 1,479
  points, worse at 1,557): how soon depends on where the ball's room is,
  not on which door was opened. Surprise marks 0% of tier 3's changes.
  Changes recur (99.9% have a stored match): material for 097.3.
- **097.3 closed** (no card used). In MiniGrid a compiled chain could
  save at most deliberation's 7% of tier 3's time; moved to Crafter,
  where chains are long and repeat.
- **097.1 (drafted 10-09; approved overnight after its literature
  review) stop.** Walking learned from the agent's own moves: a network
  on the believed 13 x 13 view and a target, trained by hindsight on
  62,000 recorded moves. First version 99.3% agreement with the field
  (misses on long routes and a loose tie rule); the one revision
  (distance as a distribution, three copies that answer only when they
  agree; from search on the replay buffer) agrees at 100% of tiers 1–2's
  walking decisions where it answers (97–99% of them) and 99.85% on
  tier 3. In the agent, tiers 1–3 unchanged (200/200, 100/100, 30/30,
  same steps); fresh tier 3 98/100 against 99: one route that left the
  view drove a loop. Time per step 2.6–7 times version 24's; the field
  is still needed for far targets (tier 3: 7.7% of walking decisions).
  The waypoint design for routes beyond view goes to Crafter.
- **Backlog 091–097 done.** Version 24 remains the architecture.
- **098 resumed, revise.** Ceiling: a tile classifier on the evaluator's
  labels is 98.3% right at night, so night tokens are possible. The first
  run's defect was the reverse of the suspicion: night shrinks all pixel
  differences, so the fixed thresholds called different things the same
  (97% of different-class pairs in the darkest frames). Nine variants:
  judgments only between places drawn identically when redrawn, and
  changes (counts) as a pair loss, fix purity (99.6% overall, 96.4% at
  night) and inventory (99.1%) but not both at once, and codes split by
  light level (grass: 82% in its day code, the rest in about 35 dusk and
  night codes). Raw-pixel k-means is a close baseline. Next: 098.2, the
  same world place across light, tied through the agent's own moves.
- **098.2 (new card 7) revise.** Crafter tokens tied across light through
  the agent's own moves (learned view shifts, summed within an episode;
  a noise-free day frame labels the other frame's overlapping places).
  First run: the same code by day and night for only 12.8% of night
  tiles (a light shift common to all candidates cancels in the softmax).
  The card's one revision, both frames' tiles as candidates: 89.0%; with
  the agent's own tile corrected (it was treated as a world place):
  94.0% (bar 90%), purity 99.2% overall and 94.7% at night, inventory
  95.4%, 57 codes, every class 1–3 codes except cows (6): creatures move,
  so world-place pairs mislabel them. Best Crafter tokens so far.
- **098.1 (new card 8) uninterpretable, revise.** Crafter interactions
  predicted by recall over the tokens. First run: the tokens merged the
  agent's headings (front learned 24%), and absolute outcomes did not
  transfer (even the true state 37%). After 098.3 and the card's one
  revision (outcomes as changes per part; slot successors keyed by slot
  position): "do" predicted as well from tokens as from the true state
  (48 of 131 against 49 of 134), but in random play most "changing"
  events are caused by creatures or chance, so even the true state is
  right on 39%: the test needs a counterfactual baseline (the same
  situation without the action) before its bars mean anything.
- **098.3 (new card 9) revise.** Still pairs chosen by the agent's own
  check that the view did not shift (blocked moves included): four
  codes, one per heading, front learned 100% on awake steps (24%
  before); the asleep sprite shares the left code. Kept 098.2's 95%
  same-code-by-night; inventory 94.2%.
