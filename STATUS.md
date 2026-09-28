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
  Readiness is removed. The charter says the session after a switch is
  thinking only (no code, no runs).
- **Card 023 (done, fail, revise):** conditions as "approachable by moving
  closer in view": acting 39.6% (card 012's conditions with path search
  100%). Its measure stopped agents facing a wall; discovery then filled
  the tree with moves.
- **Card 024 (done, fail; decision pending):** measure fixed (turn towards
  a free step that brings the agent closer). Acting 96.4% but 50% random
  moves (needs ≥ 98%, ≤ 1%); 32 goals changes nothing. Cause: the key's
  way holds at 82% of starts, so card 012's stopping rule makes it a leaf
  with no subgoals for the other 18%. Next: how discovery grows the tree.
- **Card 021 (stopped):** full-tree readiness and conditions on
  architecture 3's shared processor failed even with exact labels (held-out
  false positives up to 38%, familiar wall columns too). Evidence that
  conditions from that processor do not yet generalise.
- **Cards 017–020:** readiness needs remote information (017: exact input
  collisions); the shared processor fits one condition and its readiness
  at 99.61% with evaluator labels (020).
- **Card 016 (fail, revise):** egocentric learned walking: acting 48.4% at
  30k and 90k updates; readiness right only at the agent (maps check).
- **Retained evidence:** card 012's learned key-world conditions give 95%
  acting with exact walking, 30.2% all learned, 0.4% random.
- **Next after 023:** learn what moves do to the view (moves as learned
  transformations of the map), then the effects of pickup and toggle.
- **Pinned:** duplicate detector merging needs literature; demonstrations
  once random play is too thin. LESSONS.md needs a merge pass (over two
  pages). Nothing from cards 021–023 is committed; 59 tests passed (card 021).
