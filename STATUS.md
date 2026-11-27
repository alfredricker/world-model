# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-25, `master`. Uncommitted: card 163 closed (`fail`, `revise`) + graph node; card 164 drafted (`tools/card164/`, `runs/164/`).
  **Rung:** 2 (event segmentation) reopened; 3–5 worked on in MiniGrid. **Architecture:** og4 (OGBench line; 27 is MiniGrid's last).
  **World:** MiniGrid (active, cards 161–164); OGBench `visual-scene` (og4 53.6% vs GAS 54.4%). Portability tests paused.
- **Direction:** learned methods below, our conditions on top ([card 113](experiments/113-post-mortem-learned-stack/card.md));
  one ingredient per week. Memory week (132–140) closed; representation week (141–144) revise; card 145 stop (no decoder signal).
- **Energy modules (cards 146–160):** set aside 2026-11-23 ([experiments/energy-modules/README.md](experiments/energy-modules/README.md)).
  Card 160's MiniGrid chaining agent, perceptron, rules evaluator and 3 random-play experiences (700k steps each) kept for reuse.
- **Card [161](experiments/161-conditions-asked-of-the-effect-model/card.md): fail (2 of 3), keep.** Asking the effect model
  directly about the present situation jumped mean success from 70.3% to 86.8% (pickup tasks 40% → 84%).
- **Card [162](experiments/162-a-neighbour-is-what-it-is-and-where-it-lies/card.md): pass (3 of 3), keep.** Separating token what
  from where via attention carried moves to 75/75 unseen `(code, heading)` pairs: KeyCorridor 94.0%, 5-task mean **92.5%**.
- **Card [163](experiments/163-one-prediction-to-act-by/card.md): fail, revise.** One flat prediction $q$ walks to near events from
  random play (handed next event: DoorKey-8x8 40/40 at optimal 15.8 steps, DoorKey-16x16 40/40, KeyCorridor 40/40), but fails
  multi-event chains even with demonstrations (UnlockPickup 0/40; action gap 0.011 at 16–19 steps) and conflated box pickup with toggle.
- **Active — Card [164](experiments/164-system-two-chooses-system-one-acts/card.md): draft (gate passed, awaiting approval).**
  Fixes System 1 $q$'s event definition to `(cell, next_code, action)` with wall-debiased within-segment bootstrapping (`mode=reach`)
  on pure Stage R random play (0 checkpoints, 0 demos), paired with System 2 condition-based event selection that holds the expected
  event and revises on surprise (P14) instead of falling back to random actions.
  - **Gate (exp 1, 40 layouts/task):** Card 162 91.5%; stateless `162s` 88.5%; Arm A0 (pure System 1 $q$ swap) 88.5% (`single`: 90.5%);
    **Arm A (`q_reach_1` + held event + surprise recovery) 99.5% (199/200)** — DoorKey-8x8 100% (16.2 steps, 98.3% walk),
    DoorKey-16x16 100% (70.8 steps), KeyCorridor 97.5%, UnlockPickup 100% (0 destroyed), BlockedUnlockPickup 100% (0 destroyed).
  - **Arm B (zero-Dijkstra event search over effect model + $q$):** failed gate (DoorKey-8x8 40/40 at 16.2 steps, 7.3 expansions;
    UnlockPickup/BlockedUnlockPickup/KeyCorridor 10–25% because one discounted $V$ conflates distance with binary reachability).
  - **Next decision:** user approval of Card 164 to fit `q_reach_{2,3}` and run the frozen 3-experience × 100-fresh-layout test (~35 min).
- **Setup.** Project `.venv` runs cards 160–164 (`tools/card164/run.sh`); `graph.py` / `index.py` / OGBench use
  `~/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13/bin/python`.
