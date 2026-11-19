# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-17, `master`. LESSONS pruned to 199 lines. Uncommitted: card 145 and `tools/card145/`.
  **Rung:** 2 (event segmentation) reopened; 3 was worked on with rung 2 open. **Architecture:** og4 (OGBench line; 27 is MiniGrid's last).
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
  - **[143](experiments/143-found-where-it-was-let-go/card.md): fail (1 of 3), revise; labels kept.** Adding "frame
    after a touch → release point": cube in the open drawer 27.7 → 1.70 cm, put-away problems 0 → 68%.
  - **[142](experiments/142-find-then-place/card.md): fail (0 of 3), revise.** A smooth map from the centre of
    the weights reads a withheld third of the table to 1.84 cm (og4's form 3.32) but 1.05 cm inside (0.45).
  - **[144](experiments/144-a-centre-that-means-the-middle/card.md): fail (0 of 3), revise** (bars mine). On
    og4's weights with a fixed smooth map: training the score on fails (table 0.67 cm, withheld third
    1.94); a shift bounded to one patch (142's proposal, a side arm) reads the table to 0.61 cm (og4's form
    0.45) and the withheld third to 1.21 (3.32). The open drawer stays at 1.7 cm (one weak seed of three).
- **No version adopted;** og4 unchanged. Release labels `runs/143/`; read-outs `runs/144/fit_off_a_<seed>.pt`.
- **Direction (the user, 2026-11-17): fundamental abilities, not specifics.** Agreed first: a learned
  state of the scene with events as the moments it changes (rung 2, still open); belief out of view and
  acting under feedback wait on it.
- **Active card: [145](experiments/145-a-state-that-changes-only-when-the-scene-does/card.md), running.** One
  vector that must explain every frame given the arm's pose, changing only through gates that cost to
  open. Gate passed. Seed 0 after 120 minutes: the state holds the drawer (0.16 cm), the window (0.36 cm)
  and both locks (99.6–99.8%), each in its own dimensions; the cube is absent (AUROC 0.50). It catches all
  drawer and 99% of window moves, but gates also open for tiny corrections (precision 0.62, bar 0.8);
  neither fault moved from 60 to 120 minutes. **Next: the user runs seed 1 and arm C** (card, section 7).
- **Left open from 142–144:** the bounded-shift read-out in behaviour; a cube move's end taken from the
  release point when there is no later pick-up (34 → 45 of 50 put-aways); the open drawer.
- **Setup.** The project's `.venv` (rebuilt 2026-11-12) lacks ogbench, mujoco, scikit-learn and pyyaml; cards
  142–143 ran with `~/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13` first on `PATH` inside `bin/prun`.
  Rendering: `MUJOCO_GL=egl`, libglvnd (`runs/143/glvnd.txt`) on `LD_LIBRARY_PATH`, `__EGL_VENDOR_LIBRARY_DIRS=/run/opengl-driver/share/glvnd/egl_vendor.d`.
