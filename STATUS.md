# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-03, `master`, uncommitted: 116.1–116.2 closed, 116.3 drafted, LITERATURE
  (WorldDP row), LESSONS (slots). **Rung:** 2 (event segmentation), open. **Architecture:** version 27 (MiniGrid). **World:** OGBench
  `visual-scene`.
- **Direction:** published learned methods below ([card 113](experiments/113-post-mortem-learned-stack/card.md));
  conditions are event segmentation (the user, 2026-11-01; Zacks et al. 2007): stable
  stretches between boundaries, a boundary where a tracked feature changes unpredictably.
  Far-future thinking runs over conditions, never rolled-out latents. CHARTER: Roadmap
  step 2 and ladder rung 2 are event segmentation; skills come after.
- **Rung 1 passed** ([114](experiments/114-lock-effects-from-pixels/card.md),
  [114.1](experiments/114.1-counterfactual-lock/card.md)): a DINO-WM-style predictor on frozen
  DINOv2 patches follows the lock when only the lock changes (96.9–100%; actions alone 0–5%).
- **[115](experiments/115-boundaries-from-surprise/card.md): uninterpretable, revise.** Surprise
  marks condition changes no better than plain change (0.30 against 0.31, matched in time
  and place; random 0.05); most false boundaries sit on the arm. Objects come first.
- **Segmentation review (2026-11-01, LITERATURE's current focus):** slots on frozen DINOv2
  patches; boundaries from influence and stability (CAI, SEM-2, GateL0RD); the self as what
  every action influences; conditions as recurring stable states (ASOT, FCDL). Planner
  bar: GAS 54.4% on `visual-scene-play`. Portability tests paused (the user, 2026-11-01).
- **[116](experiments/116-objects-as-slots/card.md): uninterpretable at its gate** (the 14 × 14
  grid cannot show the buttons). **[116.1](experiments/116.1-slots-at-448/card.md): fail, stop**
  (2026-11-03). At 448 px, 20k steps: FG-ARI 0.87 but mean mBO 0.21 (= k-means); slots
  keep the table around each object and no slot takes a button. Reconstruction error had
  levelled off, so the 2.4 h run was skipped (the user's choice).
- **[116.2](experiments/116.2-slots-from-sam-masks/card.md): uninterpretable, revise.** On
  64 × 64 frames SAM 2 never outlines the arm, window or drawer.
- **[116.3](experiments/116.3-parts-at-256/card.md) drafted, gate passed, awaiting approval.**
  Frames re-rendered at 256 px; 32 slots as parts, one per SAM 2 segment. Teacher parts
  score 0.50 against k-means 0.32. Smoke (8 min): buttons 0.49–0.50, drawer 0.56, cube 0.00;
  best slot kept 45–49% of clip steps (bar 90%). 117 and 118 approved, waiting.
- **Setup:** ogbench, mujoco, dm_control (no labmaze) and scikit-learn installed with pip,
  not in `pyproject.toml`. Rendering needs `MUJOCO_GL=egl`, libglvnd on `LD_LIBRARY_PATH`,
  `__EGL_VENDOR_LIBRARY_DIRS`. Data in `~/.ogbench/data`; features in `runs/114/` (67 GB).
- **Decisions for the user:**
  1. Approve 116.3's labelling (25 min) and training (30 min).
- **Docs:** cards 001–112 in `experiments/archive/`. LESSONS 2,404 words. The MiniGrid
  paper's abstract overclaims (open). Every run capped (rule 5).
