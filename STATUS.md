# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-20, `master`. Uncommitted: cards 145–157, `tools/card145/`–`card157/`, LITERATURE, `theory/`.
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
- **[146](experiments/146-grouping-as-settling-on-a-known-benchmark/card.md)–[146.1](experiments/146.1-grouping-as-settling-colder/card.md): revise.**
  **[147](experiments/147-binding-from-interaction-in-a-sprite-world/card.md): pass, keep.** Points bound by how they move as
  the hand acts: couplings find 97.9% of moved things. **[148](experiments/148-the-action-as-an-output-token/card.md): pass, keep.**
  The action as an output token of a pairwise energy. **[149](experiments/149-a-sequence-of-actions-by-recurrence/card.md): fail, revise.**
- **Layers of modules, feed-forward first.** [150](experiments/150-three-feed-forward-layers-of-energy-modules/card.md): pass, keep;
  three layers on sprite pixels, 99.5%. [151](experiments/151-layers-of-attractor-modules-in-a-world-of-parts/card.md): fail as worded,
  keep; passing up every stored pattern's energy equals the best flat reader (92.0; the winner alone 35.3).
- **[152](experiments/152-an-object-module-that-knows-what-an-object-is-and-does/card.md): pass, keep.** An object module with where
  (its frame), what (memories in that frame) and does (a belief over kinds of couplings): a stored object recognised at a new
  place and half hidden 100%; a never-met object handled from its looks as if told its behaviour in 9 worlds of 10.
- **[153](experiments/153-kinds-found-by-one-energy-in-two-domains/card.md): fail, keep; [153.1](experiments/153.1-dynamics-classes-from-appearance-and-transitions/card.md): pass, keep.**
  A latent dynamics class per object from appearance prior times transition likelihood (sprites; a word stream).
- **The user, 2026-11-20: strip away what the model is fed; end to end.** [154](experiments/154-feedback-from-object-modules-to-binding/card.md):
  fail (2 of 3, a scoring cut-off), keep. Feedback from pixels with no masks: a stored pattern recognised at a place writes
  its binding couplings back; a known object that has not moved is bound in 96.6% or more of cases (no feedback 0).
- **[155](experiments/155-object-modules-fed-by-binding/card.md): fail (2 of 3), keep.** No masks: a never-met object is
  not an object until it moves (take 39.1 against 99.8). **[156](experiments/156-appearance-couplings-taught-by-the-agents-own-action/card.md): fail (1 of 3), revise.**
  Appearance couplings taught by motion bind never-met objects before they move (96–97%), agent or script alike.
- **[157](experiments/157-transitions-read-from-the-binding-layers-own-states/card.md): fail (2 of 3), keep.** Each object's
  transitions read from the binding layer (contact, displacement, came along), no simulator token, no mask: the action
  that brings a never-met object along is right for 86.4% (simulator tokens: 85.0); purity 88.9 < 90. **No honest
  evaluation yet** (toy worlds, no ordinary baseline, not end to end). **Next (the user's choice):** a goal task with one.
- **Setup.** The project's `.venv` (rebuilt 2026-11-12) lacks ogbench, mujoco, scikit-learn and pyyaml; cards
  142–143 ran with `~/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13` first on `PATH` inside `bin/prun`.
  Rendering: `MUJOCO_GL=egl`, libglvnd (`runs/143/glvnd.txt`) on `LD_LIBRARY_PATH`, `__EGL_VENDOR_LIBRARY_DIRS=/run/opengl-driver/share/glvnd/egl_vendor.d`.
