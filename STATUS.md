# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-28. **Rung:** before rung 1. **Architecture:** version 3
  in code (`src/worldmodel/spatial_state.py`); the direction below replaces
  its readiness and walking parts.
- **Direction (changed with the user, 2026-09-28,
  [card 022](experiments/022-effects-post-mortem/card.md)):** the primitive
  is the effect of an action on conditions, learned from the agent's own
  outcomes. Moves are actions with effects on where things are relative to
  the agent; walking is choosing moves whose effect brings the target
  closer in view (subgoals where that gets stuck), not a separate part.
  Readiness is removed.
- **Cards 023–025 (fail, revise):** conditions as "approachable by moving
  closer"; card 025's breadth-first tree of 64 reached the goal in 100% of
  layouts with 1.27% random moves, one level short in 3 layouts.
- **Card 026 (done, fail, keep, with the user):** subgoals grown only when
  stuck, depth first with backtracking. Acting 100%, 16.4 steps (path
  search 16.3), 0 random moves; 4.1 conditions checked per move (card 025:
  25.6). Stored 58 conditions against a limit of 32: one layout of 500
  (key in a pocket) searched the key's whole branch before backing up and
  added 28 of them; without it, 30. Several stored ways look like
  duplicates (identical evidence under "turn right" and "turn left").
- **Cards 016, 021 (learned):** learned walking 48.4%; architecture 3's
  conditions failed even with exact labels (held-out false positives up to
  38%). Card 012's learned conditions: 95% with exact walking, 30.2% all
  learned.
- **Theory session (2026-09-28, with the user):** objects are what
  conditions are about, found through them: kinds are learned where the
  agent acts (the tile in front) and recognised everywhere; relations fall
  out of conditions as one comparison between two things. Notes in card
  027's appendix. GOAL.md hypotheses drafted, awaiting the user.
- **Card 027 (draft, awaiting approval):** kinds from what actions do in
  front, exact; ways walk to their kind. Switch world must reach ≥ 98%;
  key world predicted to fail on the wrong key (the relation gap).
- **After that:** the key-door relation; learned kind codes; move effects.
- **Pinned:** duplicate detector merging needs literature; demonstrations
  once random play is too thin. LESSONS.md needs a merge pass (over two
  pages). Nothing from cards 021–026 is committed.
