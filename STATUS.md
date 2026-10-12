# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-12 (afternoon), `master`, nothing committed. **Rung:** 1, not passed.
  **Architecture:** version 27 (nothing adopted). Cards are judged on Crafter; MiniGrid's tiers
  run only before adopting a kept change.
- **Best version:** 27. Tier 1 **200/200** in 24.3 steps; tier 2 **100/100** in 30.2; tier 3
  **30/30** in 118.8; fresh tier 3 seeds **100/100**.
- **Objective (the user, 2026-10-12):** portability between MiniGrid, Crafter, ARC-AGI-3,
  Minecraft and robots with limited, general priors; judged on Crafter; principled enough
  for a paper this week. Log: [overnight-10-12.md](overnight-10-12.md) (night and afternoon).
- **Series [103](experiments/103-tracking-with-general-priors/card.md), tracking with general priors** (opened after a review of
  ~45 papers on object-centric video and tracking; LITERATURE's current focus):
  - [103.1](experiments/103.1-world-and-camera-layers/card.md) **revise**. World and camera as layers: the camera's pixels are learned
    (26, all the agent's sprite; 49 in world 3). Robust likelihood + exposure: night 94%.
  - [103.2](experiments/103.2-object-files-as-sprites/card.md) **revise**. Object files as pixel sprites (propagation + discovery): IDF1 0.531
    against 098.6's 0.717 on tile tokens. Missing: kinds for still creatures, earned existence.
  - [103.3](experiments/103.3-view-motion-with-files/card.md) **stop**. Files explaining pixels away: false files remove the view's evidence.
  - [103.4](experiments/103.4-evidence-per-observation/card.md) **keep**. Evidence per independent observation (τ = 300 px, chosen by the
    agent's own self-consistency): Crafter's view motion **99.3%** held out (night 96.9%),
    above the declared agent tile's 98.1–98.5%; world 3 100%. Removes 098.5's declared tile.
- **Next:** 103.5, card 103.2's revision (sprite templates under 098.6's files level) on 103.4's
  origins (files there: IDF1 0.411; evaluator's origins 0.522). Then one-colour shapes and
  files in belief (in place of layers.py).
- **For the user:** whether 103.4's placing becomes the Crafter adapter now; adoption as a
  version needs MiniGrid's tiers, whose view turns (a 2-D shift does not cover it).
- **Series [102](experiments/102-minigrid-assumptions/card.md)** (overnight): 102.1–102.3 kept; 102.4–102.6, 102.8 revise; 102.7 kept (finding).
- **Every run is capped** (CHARTER rule 5).
