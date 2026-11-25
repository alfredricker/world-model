# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-24, `master`. Uncommitted: cards 146–160 moved into `experiments/energy-modules/` with a note; card 161, `tools/card161/`.
  **Rung:** 2 (event segmentation) reopened; 3 was worked on with rung 2 open. **Architecture:** og4 (OGBench line; 27 is MiniGrid's last).
  **World:** OGBench `visual-scene`. Planner bar: GAS 54.4%; og4 53.6% (tasks 1–5: 100, 86, 82, 0, 0). Portability tests paused.
- **Direction:** learned methods below, our conditions on top ([card 113](experiments/113-post-mortem-learned-stack/card.md));
  conditions are event segmentation (the user, 2026-11-01). One ingredient per week, the others assumed perfect.
- **Memory week (cards 132–140), closed.** Kept: recall keyed per object (134.1), records with a learned weight per key
  item (137, 138), looking again when a memory is stale (135). Not in og4 yet. **Representation week (cards
  [141](experiments/141-representation-test-bed/card.md)–[144](experiments/144-a-centre-that-means-the-middle/card.md)), all fail, revise;** release-point labels (143) kept. og4 unchanged.
- **Card [145](experiments/145-a-state-that-changes-only-when-the-scene-does/card.md): fail, stop** (the user, 2026-11-24): reconstruction never learned the cube. No decoder as a training signal.
- **Energy modules (cards 146–160): set aside by the user, 2026-11-23; may be picked up later.** The 18 cards are in
  [experiments/energy-modules/](experiments/energy-modules/); [its note](experiments/energy-modules/README.md) gives
  the motivation, what worked, what did not and where to resume. CHARTER was never changed for this direction.
  - **Worked:** binding from the agent's own interaction (147, 154, 156); layers of modules on pixels (150, 151);
    "held" as a latent state (158); acting by chaining learned conditions backward from a goal
    ([160](experiments/energy-modules/160-minigrid-goals-against-ordinary-baselines/card.md)): DoorKey-8x8 and 16x16
    99% from 700,000 reward-free random steps (PPO with the reward, same steps: 6% and 0.3%).
  - **Did not:** in the one test on equal terms (160) a perceptron in the same agent solved 72.5% of MiniGrid layouts,
    the sparse energy 44.5%; each module learns alone; half the experience gave 7% on DoorKey; perception without
    masks stayed partial (159). Cards 146–159 were never compared with an ordinary network.
  - **Kept for reuse:** card 160's chaining agent, rules agent, perceptron and PPO (`tools/card160/`); they do not
    depend on the energy.
- **Benchmark order (GOAL):** MiniGrid first, then OGBench. Version 27 solved its tiers at 100% with more hand-built parts.
- **The user, 2026-11-24:** build from how parts are similar; one effect model for next-tick and abstract effects.
- **Card [161](experiments/161-conditions-asked-of-the-effect-model/card.md): fail (2 of 3), keep.** The planner asks
  the effect model about the present situation instead of a statement compiled in advance. Same network and
  experience as card 160; 100 fresh layouts per task, three experiences: mean 86.8% against 70.3%; UnlockPickup and
  BlockedUnlockPickup 84.7 and 82.7 against 41.0 and 39.3. The network is right about pickups with full hands for
  97% of cases never tried. Arm B (a shared "nothing changes" output) was worse in two experiences of three: dropped.
  **KeyCorridor (66.7%) explained:** the move effect, learned per heading and per (position, code), cannot walk west
  through open doors of colours never walked west through; with the true move rule 40 and 38 of 40.
- **Card [162](experiments/162-a-neighbour-is-what-it-is-and-where-it-lies/card.md): pass (3 of 3), keep.** In the
  network for the agent's move every token is what it is and where it lies, combined by attention. KeyCorridor 94.0%
  against 63.0; mean of five tasks **92.5%** (86.3; card 160: 70.3). One network for every token failed the gate.
- **Card [163](experiments/163-one-prediction-to-act-by/card.md): approved, gate not passed.** One learned prediction (will the event come) picks every action; no planner. Random play: DoorKey-5x5 to 8x8 and Unlock 40 of 40; the pickup tasks 0 of 40, also with structured play and with demonstrations. Handed the next event, the same prediction reaches it (DoorKey-16x16 40 of 40). Open: how the next event is chosen. Decision with the user.
- **Setup.** The project's `.venv` (rebuilt 2026-11-12) lacks ogbench, mujoco, scikit-learn and pyyaml; cards
  142–143 ran with `~/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13` first on `PATH` inside `bin/prun`.
  Rendering: `MUJOCO_GL=egl`, libglvnd (`runs/143/glvnd.txt`) on `LD_LIBRARY_PATH`, `__EGL_VENDOR_LIBRARY_DIRS=/run/opengl-driver/share/glvnd/egl_vendor.d`.
