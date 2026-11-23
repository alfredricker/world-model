# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-21, `master`. Uncommitted: cards 145–160, `tools/card145/`–`card160/`, LITERATURE, `theory/`.
  **Rung:** 2 (event segmentation) reopened; 3 was worked on with rung 2 open. **Architecture:** og4 (OGBench line; 27 is MiniGrid's last).
  **World:** OGBench `visual-scene`. Planner bar: GAS 54.4%; og4 53.6% (tasks 1–5: 100, 86, 82, 0, 0). Portability tests paused.
- **Direction:** learned methods below, our conditions on top ([card 113](experiments/113-post-mortem-learned-stack/card.md));
  conditions are event segmentation (the user, 2026-11-01). One ingredient per week, the others assumed perfect.
- **Memory week (cards 132–140), closed.** Kept: recall keyed per object (134.1), records with a learned weight per key
  item (137, 138), looking again when a memory is stale (135). Not in og4 yet. **Representation week (cards
  [141](experiments/141-representation-test-bed/card.md)–[144](experiments/144-a-centre-that-means-the-middle/card.md)), all fail, revise;** release-point labels (143) kept. og4 unchanged.
- **Card [145](experiments/145-a-state-that-changes-only-when-the-scene-does/card.md), running, on hold** (seed 0: the cube is absent).
- **Direction under trial (the user, 2026-11-19; CHARTER unchanged):** energy modules whose settled states are
  objects, memory inside them, events at every level. Notes: sections 6–7 of
  [the write-up](theory/thousand_brains_energy_formulation.md); papers: LITERATURE, "Energy modules and binding".
- **Sprite world (cards 146–153.1).** Kept: [147](experiments/147-binding-from-interaction-in-a-sprite-world/card.md) binding from how points move as the hand acts (97.9%);
  [148](experiments/148-the-action-as-an-output-token/card.md) the action as an output token; [150](experiments/150-three-feed-forward-layers-of-energy-modules/card.md) three layers (99.5%);
  [151](experiments/151-layers-of-attractor-modules-in-a-world-of-parts/card.md) pass up every pattern's energy; [152](experiments/152-an-object-module-that-knows-what-an-object-is-and-does/card.md) an object module with where, what and does;
  [153.1](experiments/153.1-dynamics-classes-from-appearance-and-transitions/card.md) dynamics classes. Revise: 146, 146.1, 149.
- **The user, 2026-11-20: strip away what the model is fed; end to end.** [154](experiments/154-feedback-from-object-modules-to-binding/card.md):
  fail (2 of 3), keep. A recognised pattern writes its couplings back: a still known object is bound in 96.6%+ of cases.
- **[155](experiments/155-object-modules-fed-by-binding/card.md): fail (2 of 3), keep.** No masks: a never-met object is no object until it moves. **[156](experiments/156-appearance-couplings-taught-by-the-agents-own-action/card.md): fail (1 of 3), revise.**
  Appearance couplings taught by motion bind never-met objects before they move (96–97%, at an overlap of 0.5).
- **[157](experiments/157-transitions-read-from-the-binding-layers-own-states/card.md): fail (2 of 3), keep.** Transitions read from the binding layer, no simulator token or mask (86.4% against 85.0).
- **[158](experiments/158-events-as-changes-of-an-objects-latent-state/card.md): fail (1 of 3), keep.** A latent state per object
  module; events are its changes (F1 71.3 against 53.5 read from what is seen); "held" is inferred; conditions 30 of 40.
- **[159](experiments/159-contact-as-a-learned-relation-of-places/card.md): fail (0 of 3), stop.** Contact learned as a relation of places
  equals the simulator's rule on true boxes; on own boxes it is worse than my rule (objects bound by appearance are partial).
- **[160](experiments/160-minigrid-goals-against-ordinary-baselines/card.md): fail (1 of 3), revise. First run in a real benchmark, MiniGrid
  (the user, 2026-11-21).** Tokens supplied; card 158's sparse transition energy learned from 700,000 reward-free random steps; new: acting by
  chaining learned statements. DoorKey-8x8 and 16x16: 99% (PPO with the reward, same steps: 6% and 0.3%). UnlockPickup, BlockedUnlockPickup,
  KeyCorridor: 9, 10, 5% (a perceptron in the same agent 41, 46, 76; told the rules 100). Cause: a door's opening enters the energy twice
  (its new code, and its old code "out of view") and L1 splits the credit. Half the experience: DoorKey 7%.
  **Next (the user's choice):** revise the memory inputs and share conditions among like codes, with the perceptron as comparison.
- **Setup.** The project's `.venv` (rebuilt 2026-11-12) lacks ogbench, mujoco, scikit-learn and pyyaml; cards
  142–143 ran with `~/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13` first on `PATH` inside `bin/prun`.
  Rendering: `MUJOCO_GL=egl`, libglvnd (`runs/143/glvnd.txt`) on `LD_LIBRARY_PATH`, `__EGL_VENDOR_LIBRARY_DIRS=/run/opengl-driver/share/glvnd/egl_vendor.d`.
