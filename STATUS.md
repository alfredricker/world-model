# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-27, `master`. Uncommitted: cards 163–167 done (`164`, `165`, `166`, `167` kept).
  **Rung:** 2 (event segmentation) reopened; 3–6 passed in MiniGrid (zero-Dijkstra, unified `WNet`, partial observability). **Architecture:** og4.
  **World:** MiniGrid (cards 161–167 complete); OGBench `visual-scene` (og4 53.6% vs GAS 54.4%). Portability tests paused.
- **Direction:** learned methods below, our conditions on top ([card 113](experiments/113-post-mortem-learned-stack/card.md));
  one ingredient per week. Memory week (132–140) closed; representation week (141–144) revise; card 145 stop (no decoder signal).
- **Energy modules (cards 146–160):** set aside 2026-11-23 ([experiments/energy-modules/README.md](experiments/energy-modules/README.md)).
  Card 160's rules evaluator & 3 reward-free random-play experiences (`seq_{1,2,3}.pkl`, 700k steps each) kept for reuse.
- **Cards [161](experiments/161-conditions-asked-of-the-effect-model/card.md)–[165](experiments/165-zero-dijkstra-system-two-with-neural-reachability/card.md):**
  161 `fail (2/3), keep` (86.8%); 162 `pass (3/3), keep` (92.5%); 163 `fail, revise`; 164 `pass (3/3), keep` (98.3% System 2 + System 1 $q$);
  165 `pass (3/3), keep` (`RNet` zero-Dijkstra System 2: 97.1% vs 98.1%, 100.00% reachability, 2.71× faster).
- **Card [166](experiments/166-unified-factored-relational-world-model/card.md): fail (2 of 3), keep.**
  Unified `RNet` + cell effects into **one shared-embedding `WNet`** (`runs/166/world_{1,2,3}.pt`, additive DeepSet-L1, contrast-balanced
  sampling directly on `seq_{1,2,3}.pkl`), retiring card 160's `model_mlp.pkl`, `Table`, `gid`, `FI`/`FY`, `statement()`:
  **97.1%** vs Card 165's **95.7%** (`seeds 10000–10099`, +1.34 pts; KeyCorridor **98.3%** vs 92.0%; 100.00% reachability; 99.98% cell effects;
  7.33 ms/step; missed only fixed 94% UnlockPickup sub-bar at 91.7% vs 165's 91.3%).
- **Card [167](experiments/167-partial-observability-belief-and-frontier-exploration/card.md): pass (3 of 3), keep.**
  Dropped the last MiniGrid full-view assumption: true `7×7` occluded egocentric view + proprioception-placed persistent `PartialBelief` +
  epistemic frontier BFS (`r_act[UNK] >= 0.5`) + boundary-only pessimistic `UNK` masking for frozen System 1 (`q_reach`), **zero extra training**.
  - **Fresh frozen test (`seeds 11000–11099`, 3 exp × 5 tasks × 100 layouts):** **93.1%** (`95.2, 94.4, 89.6`; 1,396/1,500) vs. no-frontier
    baseline **25.5%** (**+67.6 pts**) and full-view Card 166 **97.2%** (**−4.13 pts** retention gap) — DoorKey-8x8 **100%**, DoorKey-16x16
    **100%** (95.7 steps), KeyCorridor **86.0%** (vs 8.0%), UnlockPickup **86.0%** (vs 20.0%), BlockedUnlockPickup **93.3%** (vs 0.0%).
  - **Epistemic chaining, belief & speed:** **88.4%** across the 3 100%-occluded-goal tasks (0 leaks in 50 verified episodes), **581** episodes
    solved after the goal left the `7×7` view, **5.64 ms/step** (**3.11× faster** than full-view Card 164).
- **Next decision (Card 168):** port this unified zero-Dijkstra System 2 (`WNet` + goal/epistemic BFS + P14 recovery) + System 1 stack to
  **OGBench `visual-scene`** (continuous pixel manipulation, `og4` currently 53.6% vs GAS 54.4%) or **Crafter / Craftium** (moving-view worlds).
- **Setup.** Project `.venv` runs cards 160–167 (`tools/card167/run.sh`); `graph.py` / `index.py` / OGBench use
  `~/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13/bin/python`.
