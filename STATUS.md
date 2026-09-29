# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-29. **Rung:** before rung 1. **Architecture:** version 5
  (card 029): a counted model of every action's effects (028), with
  subgoals worked backward through its learned rules at every step.
- **Direction (with the user, card 022):** the primitive is the effect of
  an action on conditions, learned from the agent's own outcomes; moves
  are actions like any other; walking must not read the simulator.
- **Card 029 (pass; keep):** 100% of 500 new layouts in four door worlds,
  1.03–1.11 times the shortest route. Weakest part: long detours.
- **Next direction (the user, 2026-09-29):** a learned front end: tile
  (later a segment) → vector → several discrete codebooks → rules and
  goals over the codes. Card 030 is stopped (exact pixel comparison cannot
  survive noise or new appearances); its tests and its rules with a shared
  variable carry over. Order agreed: codes first (card 031), recall later.
  - Codebooks: kept from describing the same thing; what a code means is
    never engineered (papi: `1811_12359`, `2002_02886`, `2107_10098`).
  - Goals: any condition over codes, positions and relations.
  - Memory: events stored with their step, and only what the model fails
    to predict; the world model is the decoder.
  - **Counting is a simplified memory (the user's idea).** It keeps tallies
    per fixed key and forgets the experiences. Recall of past cause and
    effect replaces it: as sample-efficient, and old experience can be
    re-sorted by newly found codes. Replaying recalled episodes trains
    network weights until a rule is known without recall (fast System 1).
    Checks: weights take over a rule only when they agree with recall on
    every case, rare ones included (LESSONS: networks drop rare events);
    recall stays the fallback when the weights are unsure.
- **Now:** cards 031 (codes from pixels) and 032 (fewest codes) both
  fail on purple keys and doors (0 of 10 seeds). Learned codes lose
  nothing in the familiar worlds (the user: replacing the integer map by
  learned codebooks cost nothing end to end), and label codes carry to
  purple fully. The penalty is stopped. The user's next direction: allow
  new codes, and infer a new thing's effects from similar recalled
  experience, confirmed or corrected by trying. The card is to be
  drafted with the user.
- **Pinned:** card 029's arm 3 in the both world (command in the card);
  demonstrations once random play is too thin; GOAL.md hypotheses draft.
