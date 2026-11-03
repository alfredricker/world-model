# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-01, `master`, uncommitted: cards 113–115, CHARTER (ladder, Roadmap),
  AGENTS, GOAL C1, LITERATURE focus, LESSONS, cards 102–112 archived. **Rung:** 2 (event
  segmentation), open. **Architecture:** version 27 (MiniGrid). **World:** OGBench
  `visual-scene`.
- **Direction:** published learned methods below ([card 113](experiments/113-post-mortem-learned-stack/card.md));
  conditions are event segmentation (the user, 2026-11-01; Zacks et al. 2007): stable
  stretches between boundaries, a boundary where a tracked feature changes unpredictably.
  Far-future thinking runs over conditions, never rolled-out latents. CHARTER: Roadmap
  step 2 and ladder rung 2 are event segmentation; skills come after.
- **Rung 1 passed** ([114](experiments/114-lock-effects-from-pixels/card.md),
  [114.1](experiments/114.1-counterfactual-lock/card.md)): a DINO-WM-style predictor on frozen
  DINOv2 patches follows the lock when only the lock changes (96.9–100%; actions alone 0–5%).
- **[115](experiments/115-boundaries-from-surprise/card.md): uninterpretable, revise.**
  Boundaries from the predictor's surprise. The gate's bound (change with the arm masked)
  reached F1 0.43, under the 0.6 bar; random 0.33, because events are dense. Matching in
  time and place (after the run): random 0.05, surprise 0.30, plain change 0.31, change
  with the arm masked 0.40. Surprise finds condition changes (lock flips best, cube
  worst), but no better than plain change: most false boundaries sit on the arm.
- **Segmentation review (2026-11-01, LITERATURE's current focus):** objects learned as slots
  on the frozen DINOv2 patches (SlotContrast; slot quality tracks planning on OGBench
  Cube); boundaries from influence and stability, not surprise (CAI, SEM-2, GateL0RD); the
  self as what every action influences; conditions as recurring stable states (ASOT, FCDL);
  operators for forward and backward search (DeepSym line). Planner bar: GAS 54.4% on
  `visual-scene-play`. Portability tests paused (the user, 2026-11-01; AGENTS).
- **[116](experiments/116-objects-as-slots/card.md): uninterpretable at its gate.** The 14 × 14
  grid cannot show the buttons (ceiling mBO 0.004). **[116.1](experiments/116.1-slots-at-448/card.md)
  running** at 448 px (32 × 32; ceiling 0.54–0.58 on cube and buttons). After 29 min (20k
  steps, not converged): FG-ARI 0.87 but mean mBO 0.21 (= k-means); slots separate objects
  from each other, not from the table; no button slot. **The user runs the 100k-step
  command in card 116.1 (about 2.4 h)**; then it is scored. 117 and 118 approved, waiting.
- **Setup:** ogbench, mujoco, dm_control (no labmaze) and scikit-learn installed with pip,
  not in `pyproject.toml`. Rendering needs `MUJOCO_GL=egl`, libglvnd on `LD_LIBRARY_PATH`,
  `__EGL_VENDOR_LIBRARY_DIRS`. Data in `~/.ogbench/data`; features in `runs/114/` (67 GB).
- **Decisions for the user:**
  1. Run card 116.1's long training command (section 7).
- **Docs:** cards 001–112 in `experiments/archive/`. LESSONS 2,345 words. The MiniGrid
  paper's abstract overclaims (open). Every run capped (rule 5).
