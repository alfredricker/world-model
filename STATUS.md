# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-28, `master`. Uncommitted: cards 168–173 done, og5 in ARCHITECTURE.md, `tools/card168/`–`tools/card173/`,
  twenty-one graph nodes, three LITERATURE rows. `runs/172/` holds 14.5 GB of per-step features, kept for the next card (card 173 reused them).
  **Rung:** 6 tested on OGBench by card 169 (pass). **Architecture:** og5 (`90.5%` five-task mean; og4 `53.6%`), adopted by the user 2026-11-27.
  **World:** OGBench `visual-scene`. Portability tests paused. MiniGrid series (cards 161–167) complete.
- **Direction:** learned methods below, our conditions on top ([card 113](experiments/113-post-mortem-learned-stack/card.md));
  no reconstruction decoder ([card 145](experiments/145-a-state-that-changes-only-when-the-scene-does/card.md)); remove the scene
  model's hand-set rules, with the factoring learned from how the senses change, not only from touch (the user, 2026-11-27).
- **Card [168](experiments/168-consequence-trained-scene-world-model/card.md): fail (2 of 3), keep.** `SceneWNet`: five slots over
  frozen DINOv2 patches, trained on touch consequences. Mean pair accuracy `94.9%` (og4 `84.4%`). Slots and targets come from rules.
- **Card [169](experiments/169-zero-heuristic-event-bfs-on-ogbench/card.md): pass, keep (og5).** Event-level search over `SceneWNet`.
  Tasks 1–5: `98.7, 91.3, 97.3, 87.3, 78.0%`. Effects of a touch and two ordering guards are written code.
- **Card [170](experiments/170-effects-imagined-by-the-scene-model/card.md): fail at the gate, revise.** Effects taken from the model,
  no guards: short chains level with og5, locked put-away `58%` against `81%` (separate samples of 100).
- **Card [171](experiments/171-transition-head-trained-on-its-own-chains/card.md): fail at the gate, revise.** Chain training and a
  pull's end as a range: locked put-away `64%` (card 170 `57%`, og5 `75%`; under 7 points is noise). Open fault: 11–15% of
  the model-driven search's touches go to parts already at the goal (og5 `1.8%`); cause not known.
- **Card [172](experiments/172-factors-from-sparse-change-over-time/card.md): fail at the gate, revise.** A frame mapped to 32 bounded
  numbers, trained only to pick the near moment of an episode, the arm's pose given by a frozen term. No decoder, touch or label.
  It read every variable (AUROC `0.83–0.95`; baselines `0.56–0.67`) but a variable's change was `8–19%` on its best coordinate.
- **Card [173](experiments/173-tight-box-sparse-change-factors/card.md): fail at the gate, revise.** The user's design: temperature
  fixed, a cost on change between consecutive steps, a small state, factors read from 5-step changes. Three 5-minute fits.
  - **Gate failed twice.** The state's change where nothing in the scene differs is `36.6%` of its change where one variable
    does (bar `20%`). The bound by truth-chosen pairs put `1–2` of 5 variables at 80% on their own factor (bar 4).
  - **But the one label-free fit (8 coordinates, one seed) found five factors, one per variable:** `73, 96, 94, 92, 83%` of
    each variable's change on its own factor. Read through its own factor: AUROC `0.970–0.990` on validation (card 168's
    slots `0.976–1.0`), `0.937–0.987` on frames before a touch (slots `0.862–0.994`). The bound was not a bound for factoring.
  - Open: two thirds of the stray movement is in the two lock factors, cause not found; the cube has one number (one table
    axis read, not the other); no second seed; the ladder that chooses the state's size was not run.
- **Next decision (the user's):**
  1. Factoring: a card that runs card 173's ladder and three seeds, with each variable read through its own factor, the bound
     dropped or rebuilt, and the lock factors' movement located first. Say if you want it drafted.
  2. Planner: a card to locate card 171's needless touches.
  3. The CHARTER ladder still lists rungs 3–6 as "sketch"; say which to mark as passed, and on which cards.
- **Setup.** Project `.venv` runs MiniGrid; `graph.py`, `index.py` and OGBench use
  `~/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13/bin/python`. OGBench runs go through `bash tools/card173/run.sh <seconds> python ...`
  (capped; the script is not executable). EGL via `runs/143/glvnd.txt`.
