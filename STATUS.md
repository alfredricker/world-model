# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-19, `master`. Uncommitted: cards 145–149, `tools/card145/`–`card149/`, LITERATURE, `theory/`.
  **Rung:** 2 (event segmentation) reopened; 3 was worked on with rung 2 open. **Architecture:** og4 (OGBench line; 27 is MiniGrid's last).
  **World:** OGBench `visual-scene`. Planner bar: GAS 54.4%; og4 53.6% (tasks 1–5: 100, 86, 82, 0, 0). Portability tests paused.
- **Direction:** learned methods below, our conditions on top ([card 113](experiments/113-post-mortem-learned-stack/card.md));
  conditions are event segmentation (the user, 2026-11-01). One ingredient per week, the others assumed
  perfect (the user, 2026-11-12).
- **Memory week (cards 132–140), closed.** Kept: recall keyed per object (134.1), records never emptied
  with a learned weight per key item (137, 138), looking again when a memory is stale (135), the test bed
  (136.1). Open: small stores, a change of rule. None of it is in og4 yet.
- **Representation week (from 2026-11-13): cards [141](experiments/141-representation-test-bed/card.md),
  [142](experiments/142-find-then-place/card.md), [143](experiments/143-found-where-it-was-let-go/card.md),
  [144](experiments/144-a-centre-that-means-the-middle/card.md), all fail, revise.** Frozen features hold what
  og4's readings lack (cube in the open drawer 0.39 cm with labels; og4 27.7). Release-point labels (143, kept):
  27.7 → 1.70 cm, put-away problems 0 → 68%. A shift bounded to one patch (144) reads the table to 0.61 cm
  (og4's form 0.45) and a withheld third to 1.21 (3.32).
- **No version adopted;** og4 unchanged. Release labels `runs/143/`; read-outs `runs/144/fit_off_a_<seed>.pt`.
- **Card [145](experiments/145-a-state-that-changes-only-when-the-scene-does/card.md), running, on hold:** seed 0
  holds the drawer, window and locks; the cube is absent; switch precision 0.62 (bar 0.8). Seed 1 and arm C not run.
- **Direction under trial (the user, 2026-11-19; CHARTER unchanged):** energy modules whose settled states
  are objects, with memory inside them and events at every level. Notes and the candidate energy: sections 6–7 of
  [the write-up](theory/thousand_brains_energy_formulation.md); papers: LITERATURE, "Energy modules and
  binding". Ladder: a published small benchmark, our own sprite world, then OGBench's renderer. Binding first.
- **[146](experiments/146-grouping-as-settling-on-a-known-benchmark/card.md), [146.1](experiments/146.1-grouping-as-settling-colder/card.md): revise.**
  Settling reads a trained AKOrN's objects at 87.8 (clustering told the number 89.5). The user: use the agent's interaction.
- **[147](experiments/147-binding-from-interaction-in-a-sprite-world/card.md): pass, keep.** Sprite world: points bound only by how
  they move as the hand acts; couplings find 97.9% of moved things (no memory 39.9). The user carries the couplings forward.
- **[148](experiments/148-the-action-as-an-output-token/card.md): pass, keep.** The action as an output token
  of a pairwise energy learned as in CRAM: the wanted effect in 95.6% of steps after 100 steps of experience
  (whole records in a Hopfield memory 77.5%). Simulator state, single steps.
- **[149](experiments/149-a-sequence-of-actions-by-recurrence/card.md): fail (1 of 3), revise.** Sequences by
  recurrence: each step proposed, then verified forward by the same energy; a carried belief of what is held.
  "X at a place and let go" 96.9% (solver 100, no verification 2.5); "hold X" 58.8% (0% on 2 of 5 streams): the
  energy answers arbitrarily about a take without a move, which the records lack. **Next (needs the user):
  149.1, the agent tries what it proposes and records the outcome.**
- **Setup.** The project's `.venv` (rebuilt 2026-11-12) lacks ogbench, mujoco, scikit-learn and pyyaml; cards
  142–143 ran with `~/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13` first on `PATH` inside `bin/prun`.
  Rendering: `MUJOCO_GL=egl`, libglvnd (`runs/143/glvnd.txt`) on `LD_LIBRARY_PATH`, `__EGL_VENDOR_LIBRARY_DIRS=/run/opengl-driver/share/glvnd/egl_vendor.d`.
