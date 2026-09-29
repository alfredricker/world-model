# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-28. **Rung:** before rung 1. **Architecture:** version 5
  (card 029): a counted model of every action's effects (028), with
  subgoals worked backward through its learned rules at every step
  (`tools/card029/subgoals.py`). Version 3's neural learner is historical.
- **Direction (with the user, card 022):** the primitive is the effect of
  an action on conditions, learned from the agent's own outcomes; moves
  are actions like any other; walking must not read the simulator.
- **Cards 023–028 (keep):** conditions as "approachable by moving
  closer" (023–024); subgoals grown when stuck, depth first (026); kinds
  by counting what actions do in front (027); everything computed on a
  counted model of every action's effects, nothing read from the
  simulator, acting 100% (028). The model knows only appearances it has
  seen.
- **User's priorities (2026-09-28):** first, subgoals worked out from the
  agent's world model; second, conditions and an encoding that carry to
  new objects (colour matching as the condition, shape as the goal).
- **Card 029 (pass; keep, with the user):** given only "reach the goal
  square", the agent works backward through card 028's learned rules
  (door open <- key of its colour or switch on <- facing it <- moves).
  Four worlds, 500 layouts each: 100% success, 1.03-1.11 times the
  shortest route, random moves under 0.5%, exactly the rule's subgoals in
  99.8-100% of layouts, 6-9 conditions per move, no tree. Weakest part:
  walking on long detours. Arm 3 in the both world unfinished (command in
  the card).
- **Now:** the literature review for card 029's appendix B is written
  (LITERATURE.md's current focus; 22 papers added to papi). No paper finds
  attributes from raw pixels without labels; that step is ours. Our tiles:
  colour is one recurring pixel substitution, but a key and its closed
  door share no pixel value. Next, with the user: the transfer card.
- **Pinned:** duplicate detector merging needs literature; demonstrations
  once random play is too thin. LESSONS.md needs a merge pass (over two
  pages). GOAL.md hypotheses draft awaits the user. Nothing from cards
  021–029 is committed.
