# Card 052: overnight log

Running record of the overnight work (2026-10-04), newest last. Branch
`overnight-051-052`.

- **03:05** Step 1 generator written (`tools/card052/generator.py`):
  BabyAI room grids, 12 hues (3 held out), switch doors, lava,
  decoration, five success conditions. Smoke test: 500 random episodes,
  43,000 steps in 3.7 s; pick-ups of keys 24–57 per colour, balls
  7–34, boxes 2–14, door opens 2–5, unlocks 0–2, switch presses 4–33.
- **04:10** Gate 1 passed (counts with play starts; rendering after
  removing grid lines and comparing under the episode's tint, a declared
  revision). Step 2a written (`tools/card052/drift.py`): the encoder
  trained online on generator tiles with version 8's objective (no
  recall term, no re-seeding), code flip rate on 2,000 fixed probe
  tiles per checkpoint. Running on seed 399.
