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
- **04:20** Step 2a, first run (version 8's objective, online, no
  re-seeding): codes collapsed, 6–9 code tuples on the probe set. Probe
  set then random cells (28 identities, mostly floor and wall); changed to
  stratified, up to 40 tiles per identity (84 identities), and the front
  tile added to every training step.
- **04:30** Dead-code restarts every 250 updates: nearly every probe tile
  changed tuple between checkpoints (flip about 100%). Dropped.
- **05:50** 40 checkpoints (80,000 updates, 800,000 steps of play), seed
  399, constant learning rate:
  - codebooks as moving averages (decay 0.99), restarts every 2,000
    updates: flip rate 39–70%, last 41%; 55 tuples for 84 identities; 34
    identities split over several tuples (noise and tint), 10 tuples
    shared by several identities;
  - plain (no restarts): flip 19–41%, last 27%; 23 tuples (collapsed);
  - most flips in one part (part 1), which looks like it carries the
    tint. Gate 2 not met. Running a schedule with decaying plasticity
    (learning rate x 0.9 per checkpoint).
- **06:40** Decaying plasticity tried with and without moving-average
  codebooks: neither settles with useful codes (see the card's step 2a
  table). Step 2a written up as revise, options for the user; not
  continuing to step 3 without the user.
