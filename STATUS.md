# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-13, `master`, nothing committed. **Rung:** 1, not passed.
  **Architecture:** version 27 (nothing adopted). Cards are judged on Crafter; MiniGrid's tiers
  run only before adopting a kept change.
- **Best version:** 27. Tier 1 **200/200** in 24.3 steps; tier 2 **100/100** in 30.2; tier 3
  **30/30** in 118.8; fresh tier 3 seeds **100/100**. Written up as a paper in
  [papers/minigrid/paper.tex](papers/minigrid/paper.tex).
- **Objective (the user, 2026-10-13, for this week):** align the model with GOAL by
  integrated learning: "We need to describe _all_ changes within a latent space that the model
  can use to predict outcomes in the environment and form conditions." One effect model for
  every action, motion included, and the world's own changes. The planner's conditions come
  from what that model learns; perception parts (SAM 2.1, later depth) feed it. CHARTER's
  "Current direction", GOAL P12/P21 and LESSONS were updated the same day.
- **Tonight's order (proposed):** 104 in the background → series 105: 105.1 one transition
  model offline: the belief's token vectors and the action in, every token's next vector out (no
  event types) → 105.2 the world's own changes on 104's tracks, with history as input (arms) →
  105.3 conditions from the effect model in the agent. A general form may cost a stated loss
  (AGENTS.md's margin, the user 2026-10-13). LESSONS needs pruning (over 3 pages).
- **GOAL C1 amended (the user, 2026-10-13):** a declared, pretrained perception front end
  with no language or class labels may supply objects' outlines and their correspondence
  across frames. The agent need not be learned end to end.
- **Series [103](experiments/103-tracking-with-general-priors/card.md) closed.** 103.4 **keep**
  (Crafter's view motion 99.3% held out by general rules). [103.5](experiments/103.5-files-level-on-sprites/card.md)
  **stop**: IDF1 0.623 on the learned view (0.641 on the evaluator's), against 0.717 on tile
  tokens and 0.934 on the true state.
- **Active card: [104](experiments/104-sam-correspondence/card.md), SAM 2.1 for correspondence (draft,
  awaiting approval).** SAM 2.1 carries each creature's mask; the agent says where to point
  (change in the world's frame after 103.4's view motion) and which tracks to keep (earned
  existence). Bar: IDF1 at least 0.80, at most 1% on scenery, at least 20 frames per second.
- **Smoke tests (`tools/card104/`):** pointed at a creature at 16 px per tile, SAM's top mask
  has IoU 0.81–0.86 (cows, zombies), 0.55–0.62 (skeletons); at 7 px it misses cows; "everything"
  mode fails. Propagation: 30 frames per second, on its creature in every frame.
- **Installed (additions only):** SAM 2 from Meta's repository (commit 2b90b9f), torchvision
  0.29.0+cu130, hydra-core, iopath; checkpoints in `runs/sam2/ckpt/`. Crafter is not
  reproducible across processes (despawning iterates a set), so 104 collects new play.
- **Every run is capped** (CHARTER rule 5).
