# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-17, `master`, uncommitted: cards 142–144 run and written, `tools/card142/` to
  `tools/card144/`, graph nodes, LESSONS (pruned to 199 lines 2026-11-17).
  **Rung:** 3 (skills and plans), open. **Architecture:** og4 (OGBench line; 27 is MiniGrid's last).
  **World:** OGBench `visual-scene`. Planner bar: GAS 54.4% on `visual-scene-play`; og4 53.6% (tasks 1–5:
  100, 86, 82, 0, 0). Portability tests paused.
- **Direction:** learned methods below, our conditions on top ([card 113](experiments/113-post-mortem-learned-stack/card.md));
  conditions are event segmentation (the user, 2026-11-01). One ingredient per week, the others assumed
  perfect (the user, 2026-11-12).
- **Memory week (cards 132–140), closed.** Kept: recall keyed per object (134.1), records never emptied
  with a learned weight per key item (137, 138), looking again when a memory is stale (135), the test bed
  (136.1). Open: small stores, a change of rule. None of it is in og4 yet.
- **Representation week (from 2026-11-13).**
  - **[141](experiments/141-representation-test-bed/card.md): fail, revise.** The test bed. The frozen features hold
    what og4's readings lack (cube in the open drawer 0.39 cm with labels; og4 27.7); a trained map from
    patches ends where its training ended.
  - **[143](experiments/143-found-where-it-was-let-go/card.md): fail (1 of 3), revise; labels kept.** og4's read-out
    had 2 frames of the cube in the open drawer. Adding "frame after a touch → release point": open drawer
    27.7 → 1.70 cm (bar 1.0), table 0.60 → 0.45 cm, put-away problems 0 → 68% (bar 70%).
  - **[142](experiments/142-find-then-place/card.md): fail (0 of 3), revise.** A smooth map from the centre of
    the weights reads a withheld third of the table to 1.84 cm (og4's form 3.32) but 1.05 cm inside (0.45).
  - **[144](experiments/144-a-centre-that-means-the-middle/card.md): fail (0 of 3), revise** (drafted and run
    2026-11-17 on "do what you think is most important next"; bars mine). Second stage on og4's weights
    with a fixed smooth map. The card's arm (train the score on) fails: table 0.67 cm, withheld third 1.94.
    Its side arm, a shift bounded to one patch with the score frozen (142's proposal), reads the table to
    0.61 cm (og4's form 0.45) and the withheld third to 1.21 (3.32), 1.89 beyond 5 cm (5.86). The open
    drawer stays at 1.7 cm: one seed of three finds the cube weakly there in stage 1.
- **No version adopted.** og4 is unchanged. Release labels: `runs/143/`. Read-outs: `runs/144/fit_off_a_<seed>.pt`.
- **For the user, next decisions:**
  1. Whether the bounded-shift read-out (trained on 143's labels) goes into the agent: a card that runs
     it in behaviour (tasks 1–3, put-away problems) against og4, at a stated loss of 0.16 cm inside.
  2. A card declaring "a move may end where the hand let go" (143: 68 → 90% on put-away problems).
  3. A card for "when does a hold end": the contact sensor stops 7 to 25 frames before a slide stops,
     so the hand cannot yet label a slide's opening (with the true moment: 0.3 to 1.3 cm).
  4. Behind tasks 4 and 5: the open drawer (found weakly in one seed), "not in view", the drawer carrying the cube.
- **Setup.** The project's `.venv` (rebuilt 2026-11-12) lacks ogbench, mujoco, scikit-learn and pyyaml; cards
  142–143 ran with `~/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13` first on `PATH` inside `bin/prun`.
  Rendering: `MUJOCO_GL=egl`, libglvnd (`runs/143/glvnd.txt`) on `LD_LIBRARY_PATH`, `__EGL_VENDOR_LIBRARY_DIRS=/run/opengl-driver/share/glvnd/egl_vendor.d`.
