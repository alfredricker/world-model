# Card 051: overnight log

Running record of the overnight work (2026-10-04), in order, newest last (clock times removed: they were estimates; the branch's commit times are the real timeline). Branch
`overnight-051-052`, one commit per step.

- Profile, version 9, cluttered world, seed 399, 10 layouts,
  alone on the machine: 1.08 s per layout, setup 61 s. Card 049's
  5.8–11.2 s per layout came from running fifteen jobs at once, not from
  recall. Of 10.8 s acting, recall's scan over stored keys took 7.5 s
  (8,910 queries, 0.85 ms each), mostly asked by card 043's conditions
  and card 047's openable.
- Card 047's templates list every distinct (held, view) of stored tries
  on a tile; the planner asks recall about each. That list grows with
  memory, so the planner's cost grows with memory too, not only recall's.
- Step 1 built (`tools/card051/index.py`): groups of stored keys,
  admission over per-candidate groups, situations by class. Same
  decisions as card 050 on seed 399: 30 cluttered and 30 chained layouts
  action for action identical; the four familiar worlds the same moves
  and steps of each kind. Admission identical (same conditions, gains,
  alpha) over 50–52 groups instead of 725–771 keys. Time per layout 3–10
  times less (key world 2.7 s to 0.27 s).
- Growth test, first numbers: per step 8.4 ms at base memory,
  15 ms at +5,000 keys, 328 ms at +20,000 (card 050: 43, 579 ms). Not
  flat: something besides recall grows with memory. To profile.
- Two doors traced: the hand need is pursued before the facing
  need, whose plan needs key blue in hand; drop and pick up forever.
  Step 4a (threats between sibling needs) written; first try saw no
  threat because an achiever that works as things are has no needs, so
  its reliance on the hand was invisible; second try judged only each
  plan's first move (a turn), not the drop it leads to. Third version
  (the act each plan works toward, imagined on the present): layouts
  0–4 all reach the goal.
- Growth test, first version (index with card 038's ways
  refitted after every try): ms per step 8.4 / 15.1 / 328 at +0 / +5,000
  / +20,000 keys; card 050 43 / 579 / (still running). Profile at
  +20,000: 20.5 of 21 s acting is card 038's ways(), which compares every
  pair of a class's stored keys and is recomputed after every stored
  try. Ways are now fitted once on stored memory (at most 1,000 keys per
  class, so base memory exactly), like recall's weights, and refitted
  with admission. Rerunning step 1's checks.
- Step 4a, seed 399: two doors 29/30 (1.05 times shortest);
  one door 98% and cluttered 99%, the same layouts failing as card 050
  (78, 88; 47); familiar worlds the same moves and step kinds. Seeds
  400–404 started.
- Step 1 rechecked with ways fitted once: decisions identical to
  card 050 (60 recorded layouts action for action; familiar worlds the
  same moves and step kinds). Time per step 9.9 / 6.3 / 5.7 ms at +0 /
  +5,000 / +20,000 stored keys (card 050: 43 / 579 / 4,640 ms). Groups
  for recall 50 / 82 / 95 per action while keys grow 725 / 2,450 /
  7,500. Re-admission on the grown memory: 1.8 s at +5,000, 50 s at
  +20,000 (only on surprise, not per step; to look at). +50,000 running.
- Step 4a, seeds 400–404: two doors 29/30 in every seed (layout
  11 fails in all); one door 98% and cluttered 98–100%, the same failing
  layouts as card 050 in every seed. Familiar worlds running.
- Step 4a familiar worlds: pass in 5/5 seeds with card 050's
  steps. Step 4a kept.
- Step 4b (commitment between steps, `tools/card051/commit.py`):
  seed 403 cluttered 98% to 99% (layouts 47 and 66 now succeed, 93 now
  fails, traced); every other test as step 4a in all seeds. Kept.
- Step 4c (routes needing two tokens made walkable) built after tracing
  two doors layout 11; its evaluation on seeds 400–404 started.
- Change 6 (no spliced checks) measured before building it: with step
  4c's agent on seed 400, 10 two-door and 10 cluttered layouts, card
  043's spliced case (a success needing both the hand and the view
  changed) was computed 0 times (731 single-part conditions) and used in
  0 of 1,041 chosen steps. Not built: it would change nothing in these
  tests (the familiar "both" world was not counted).
- Step 4c on seeds 400–404: two doors 30/30, one door 100%, cluttered
  100% (seed 403 99%), routes much shorter; but both world 26/30 in
  seeds 403 and 404. Traced: the hand that card 043 assumes "as it is
  now" was not protected. Step 4d adds that link; shakedown both world
  30/30. Full run started.
- The +50,000-key growth point was stopped unfinished: after 1 h 40 min
  the test harness was still adding tries and held 22 GB of memory,
  growing about 5 GB in 25 minutes, while step 4d's runs needed the
  machine. The per-step result stands to +20,000 keys (5.7 ms per step,
  re-admission 10 s after computing card 038's ways among sampled keys
  only). What grows that fast while adding tries (the test's bulk add,
  card 039's set registry, or the per-key result sums) is not yet known.
- Step 4d on seeds 400–404: everything passes (two doors 30/30, one door
  100%, cluttered 100% but seed 403 layout 93, familiar worlds 100% with
  card 029's steps). Steps 4c and 4d kept together.
