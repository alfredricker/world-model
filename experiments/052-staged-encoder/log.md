# Card 052: overnight log

Running record of the overnight work (2026-10-04), in order, newest last (clock times removed: they were estimates; the branch's commit times are the real timeline). Branch
`overnight-051-052`.

- Step 1 generator written (`tools/card052/generator.py`):
  BabyAI room grids, 12 hues (3 held out), switch doors, lava,
  decoration, five success conditions. Smoke test: 500 random episodes,
  43,000 steps in 3.7 s; pick-ups of keys 24–57 per colour, balls
  7–34, boxes 2–14, door opens 2–5, unlocks 0–2, switch presses 4–33.
- Gate 1 passed (counts with play starts; rendering after
  removing grid lines and comparing under the episode's tint, a declared
  revision). Step 2a written (`tools/card052/drift.py`): the encoder
  trained online on generator tiles with version 8's objective (no
  recall term, no re-seeding), code flip rate on 2,000 fixed probe
  tiles per checkpoint. Running on seed 399.
- Step 2a, first run (version 8's objective, online, no
  re-seeding): codes collapsed, 6–9 code tuples on the probe set. Probe
  set then random cells (28 identities, mostly floor and wall); changed to
  stratified, up to 40 tiles per identity (84 identities), and the front
  tile added to every training step.
- Dead-code restarts every 250 updates: nearly every probe tile
  changed tuple between checkpoints (flip about 100%). Dropped.
- 40 checkpoints (80,000 updates, 800,000 steps of play), seed
  399, constant learning rate:
  - codebooks as moving averages (decay 0.99), restarts every 2,000
    updates: flip rate 39–70%, last 41%; 55 tuples for 84 identities; 34
    identities split over several tuples (noise and tint), 10 tuples
    shared by several identities;
  - plain (no restarts): flip 19–41%, last 27%; 23 tuples (collapsed);
  - most flips in one part (part 1), which looks like it carries the
    tint. Gate 2 not met. Running a schedule with decaying plasticity
    (learning rate x 0.9 per checkpoint).
- Decaying plasticity tried with and without moving-average
  codebooks: neither settles with useful codes (see the card's step 2a
  table). Step 2a written up as revise, options for the user; not
  continuing to step 3 without the user.
- Nuisance diagnostics (moving-average codebooks, 20 checkpoints, seed
  399): flip rate at the last checkpoints, no tint 34–42%; no noise
  56–73%; neither (every identity one exact tile) 45–61%, with no
  identity split but 39 tuples for 84 identities. The drift is not
  caused by the nuisance: the online objective keeps moving the codes.
  Running a label-free measure (pairs of probe tiles whose
  same-code / different-code status changes) to tell relabelling from
  regrouping.
- Label-free drift (clean tiles, moving-average codebooks, seed 399).
  Over the last five checkpoints the code flip rate is 32–51%, but the
  share of probe-tile pairs whose same-code / different-code status
  changes is much lower: part 0 at 0.1–1.3%, part 2 at 1–5%, part 3 at
  2–9%, part 1 at 7–13%. Most of the measured drift is therefore
  relabelling: the same groups of tiles under new code numbers. Part 1
  keeps regrouping. A sixth option for step 2's revision: map each
  checkpoint's codes to the previous checkpoint's by overlap, so that
  memory keeps its names, and freeze or slow the part that still
  regroups. Only gate 2 changes with this option. It is not run; the
  user picks the revision.
- 2026-10-04 (morning, with the user): the user chose option 1 (gate 2
  with step 3's objective), then maybe option 2; the overlap mapping is
  set aside. Step 2b declared in the card, then built
  (`tools/card052/effect.py`; drift.py gained an `extra()` hook for
  further loss terms). First launch stopped within a minute: the plain
  arms had dead-code restarts on, step 2a's plain baseline had them off.
  Relaunched matched. Results in the card: plain codebooks halve drift
  roughly (15-16% vs 28%) and collapse less, moving averages do not;
  the declared "at most half" is narrowly missed. Recall's outcome
  prediction from the vectors is near perfect (-0.02 to -0.04 nats vs
  -0.72 shuffled), so the term has little left to teach, and it does
  not act on code boundaries.
- Step 2c (option 2, approved): drift by recall's predictions on step
  2b's plain arm at mu 0.1. Prediction flips 0.0-0.3% by codes, 0.0-0.2%
  by vectors, accuracy 98.9% / 99.8%, while 9-17% of probe tiles change
  code. Met as declared, but the probe (random play) holds almost only
  kind-level outcomes: 2 of 1,200 tries are a key at a locked door of
  its colour, none a mismatched key there. Accuracy was 99.5% at the
  first checkpoint and the codes were collapsed (19-22 tuples). Proposed:
  rebuild the probe with play starts, stratified by kind and colour, and
  rerun. Not run; the user decides.
