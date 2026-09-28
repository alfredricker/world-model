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
- **Cards 023–024 (fail, revise):** conditions as "approachable by moving
  closer": 39.6%, then 96.4% with 50% random moves (measure fixed; the
  key's way was left without subgoals by card 012's stopping rule).
- **Card 025 (done, fail, revise):** break a condition down unless on at
  > 99% of starts; 64 goals. Acting 100%, 16.5 steps (path search 16.3),
  random moves 1.27% (limit 1%): 3 of 500 layouts one level short. The tree
  filled 64 goals; acting used 8.
- **Card 026 (draft, awaiting approval):** grow subgoals only when stuck,
  depth first with backtracking; same acting criteria, ≤ 32 conditions
  stored (card 025: 64). About 5–10 minutes.
- **Card 021 (stopped):** full-tree readiness and conditions on
  architecture 3's shared processor failed even with exact labels (held-out
  false positives up to 38%, familiar wall columns too). Evidence that
  conditions from that processor do not yet generalise.
- **Card 016 (fail, revise):** egocentric learned walking: acting 48.4% at
  30k and 90k updates; readiness right only at the agent (maps check).
- **Retained evidence:** card 012's learned key-world conditions give 95%
  acting with exact walking, 30.2% all learned, 0.4% random.
- **Next:** card 026; then learn what these checks take as exact: which
  tiles the agent can step onto (move effects), "will moving closer get me
  there?" from its own attempts, then discovery on learned conditions.
- **Pinned:** duplicate detector merging needs literature; demonstrations
  once random play is too thin. LESSONS.md needs a merge pass (over two
  pages). Nothing from cards 021–025 is committed.
