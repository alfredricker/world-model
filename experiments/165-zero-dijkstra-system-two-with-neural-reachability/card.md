---
id: "165"
title: zero-Dijkstra System 2 with neural reachability propagation — replacing move-graph search and hand-written chaining procedures by learned message passing
rung: 5
serves: [P12, P21, P3, P14, P17]
status: done
verdict: pass
arch_version: og4
date: 2026-11-26
---

# 165: zero-Dijkstra System 2 with neural reachability propagation

MiniGrid, following [card 164](../164-system-two-chooses-system-one-acts/card.md) (98.3% mean on fresh layouts).

Card 164 replaced low-level walking, orienting, executing and random-action fallback with System 1 ($q_{\text{reach}}$), while keeping card 162's hand-written System 2 planner (`want_effect`, `want_in`, `want_out`, `free_reach`, `forward_ok`, `C_BLOCK`, Dijkstra over a `W × H × 4` pose graph, and card 162's separate move model `net.Rel`) to choose the next event. Card 164's Arm B tried removing Dijkstra by thresholding discounted $V(s; E) = \max_a q(s, a; E)$ directly; it solved DoorKey-8x8 (40/40) but failed multi-room pickup tasks (10–25%) because a 2-layer global transformer trained with discounted TD conflated walking distance with binary reachability on unseen multi-room layouts (predicting 0.68 for cells behind closed doors in KeyCorridorS3R3).

## 1. Question

If we learn a dedicated differentiable neural reachability propagation network (`RNet`) purely from reward-free random play — discovering which token codes are traversable and which continuous relative offsets support 1-step motion and interaction without labels, actions, headings or hardcoded directions — and replace all of System 2's Dijkstra graph, move model and hand-written chaining procedures with batched event-level forward search over `RNet` and the cell effect model, does the agent match card 164 within the declared general-form margin while running faster across all five tasks? Rung 5 (Roadmap steps 3 and 7). Serves P12, P21, P3, P14 and P17.

## 2. What changes

One component: System 2's spatial reachability and event selection (`tools/card165/reach.py`, `plan.py`). System 1 (`runs/164/q_reach_{1,2,3}.pt`) and the cell effect model (`runs/160/model_{1,2,3}_mlp.pkl`) are frozen and unchanged; card 162's move model (`net.Rel`) and all of `agent.py` / `agent161.py` are removed.

```
164 Arm A:  Planner162 (net.Rel move model + Dijkstra W×H×4 pose graph + C_BLOCK + want_in/out/free_reach/forward_ok)
              -> next event E -> frozen System 1 q_reach
165:        RNet (learned alpha(code), beta(dx,dy), gamma(dx,dy); iterated max-product propagation)
            + batched event-level BFS over (cell MLP + RNet) with held event & P14 superset refutation
              -> next event E -> frozen System 1 q_reach
```

- **Neural Reachability Propagation (`RNet`, ~150 parameters, `tools/card165/reach.py`):**
  - Factors **what** ($\alpha(c) = \sigma(\text{pass\_logit}[c])$, one traversability parameter per token code) from **where** (two continuous 2-layer MLPs over relative offsets `(dx, dy)` within local radius 2: $\beta(\Delta p)$ for 1-step movement and $\gamma(\Delta p)$ for interaction). No headings, no action IDs, no cardinal offsets built in.
  - Iterates differentiable max-product propagation $r^{(t+1)}_i = \max(r^{(t)}_i, \alpha(c_i)\max_j \beta(p_i - p_j) r^{(t)}_j)$ for $K$ steps, followed by $r^{\text{act}}_i = \max_j \gamma(p_i - p_j) r^{(K)}_j$. At inference, the learned probabilities are thresholded at 0.5 before fixed-point propagation so arbitrarily long paths (DoorKey-16x16) suffer zero multiplicative decay.
  - **Self-supervised training on pure Stage R random play (`seq_{1,2,3}.pkl`, ~15.5 s per experience on GPU):** positive-unlabeled (PU) weighted BCE on (1) 1-step pose deltas `pose_t -> pose_{t+1}` (teaches $\beta$ and $\alpha$), (2) 1-step observed event offsets `X - pose_t` (teaches $\gamma$), and (3) multi-step constant-map segment visitations ($K = 16$, reinforces $\alpha$ across multi-room barriers).
- **Zero-Dijkstra Event-Level System 2 (`ZeroDijkstraAgent`, `tools/card165/plan.py`):**
  - Batched BFS over abstract states `(map, pose_xy, items)` up to depth 7 (cap 250 expansions), querying `RNet` once per BFS layer in parallel on GPU.
  - Directly achievable events: arrival goals with $r_{\text{arr}} \ge 0.5$; token changes `(X, o, a)` with $r_{\text{act}} \ge 0.5$ and cell MLP $p_{\text{cell}} \ge 0.5$. Drops consider the 2 nearest reachable empty cells plus 1 lowest-occupancy fallback; `RNet` automatically rejects any drop that blocks downstream reachability (replacing `forward_ok`).
  - `imagine()` places the imagined agent on a reachable cell within `RNet`'s learned interaction offsets $\gamma \ge 0.5$ of `X`.
  - **P14 surprise recovery:** if an attempted event fails at a pose allowed by the cell model's `G`, `(X, o, ea, items)` is marked refuted; any imagined state with a superset of those `items` treats `(X, o, ea)` as infeasible across all BFS depths, so search automatically drops the interfering token at a non-blocking spot and retries.

Declared (C3): tokens with an identity, the full grid view and `(token, next_code, action)` events are supplied (as in card 164); local propagation neighborhood radius is 2; BFS caps are depth 7 and 250 expansions.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Tokens with an identity are supplied | needs a front end (C1 allows it) | the same | holds (cells) | needs a front end | declared, as in card 160 |
| An event is `(token, next_code, action)` or arrival | needs learned events | the same | holds | needs learned events | declared, as in card 164 |
| The whole map is in view | no | no | holds | no | declared, as in card 160 |
| Spatial reachability is iterative local message passing over `(what, relative where)` | holds (3D voxels / waypoint graph) | holds | holds | holds (configuration / workspace graph) | **general** (replaces Dijkstra `W×H×4` pose graph & move model) |
| Event-level BFS over imagined states with superset failure refutation | holds | holds | holds | holds | **general** (replaces `want_in`, `want_out`, `free_reach`, `forward_ok`, `C_BLOCK`) |

## 3. Dependencies

- Card 160: cell effect MLP (`model_{1,2,3}_mlp.pkl`), three 700k-step reward-free random-play experiences, rules evaluator.
- Card 163: ordered random-play replay (`seq_{1,2,3}.pkl`).
- Card 164: frozen System 1 single-event execution network (`q_reach_{1,2,3}.pt`) and `Sys1` execution loop.
- Differentiable iterative value/reachability propagation: Value Iteration Networks (Tamar et al. 2016) / Neural Bellman-Ford message passing; positive-unlabeled learning (du Plessis et al. 2014).

## 4. Data check

Across all three Stage R experiences (`seq_{1,2,3}.pkl`, 700,000 steps each): ~44,300 1-step pose moves, ~14,700 events, and >8,000 constant-map segments of length ≥ 20 steps. Exactly 8 codes are ever stepped onto in every experience (`empty`, `goal`, and all 6 `open <colour> door` codes, each 36–305 times); all 31 blocking codes are stepped onto 0 times.

## 5. Feasibility gate

All three experiences (`1, 2, 3`), 40 gate layouts per task outside every test (`seeds 2000–2039`). Upper bound: rules agent (100%). Comparison: card 164 Arm A (Dijkstra System 2 + System 1 $q$, 99.5% on exp 1 gate) and card 162 (91.5% on exp 1 gate).

1. **`RNet` learned structure and exact reachability (20 layouts × all event checkpoints per task, all 3 experiences):**
   - **Passability vs. MiniGrid rules:** **39 / 39** right in exp 1, **39 / 39** in exp 2, **39 / 39** in exp 3 (100%).
   - **Learned offsets:** `beta >= 0.5` and `gamma >= 0.5` recover **exactly** the 4 cardinal offsets `[(-1, 0), (0, -1), (0, 1), (1, 0)]` in all 3 experiences (all 20–21 other offsets < 0.5).
   - **Exact binary reachability (`r_arr` and `r_act`) across all 5 tasks (222,000+ total queries):** **100.00%** (`0 fp, 0 fn`) on DoorKey-8x8, DoorKey-16x16, KeyCorridorS3R3 (unseen 6-room topology), UnlockPickup and BlockedUnlockPickup in **every** experience.
2. **Live `ZeroDijkstraAgent` on 40 gate layouts per task (`seeds 2000–2039`, `tools/card165/ev.py`):**

| Task | Exp 1 | Exp 2 | Exp 3 | **3-Exp Gate Mean** | Mean ms/step |
|---|---:|---:|---:|---:|---:|
| DoorKey-8x8 | 100.0% (16.2) | 100.0% (15.8) | 100.0% (15.8) | **100.0%** (`99.4%` walk) | 7.6 ms |
| DoorKey-16x16 | 100.0% (70.8) | 100.0% (35.1) | 100.0% (43.7) | **100.0%** (`81.1%` walk) | 8.2 ms |
| KeyCorridorS3R3 | 97.5% (47.9) | 80.0% (55.5) | 92.5% (32.9) | **90.0%** | 5.0 ms |
| UnlockPickup | 97.5% (41.4) | 95.0% (53.0) | 92.5% (71.4) | **95.0%** (1 destroyed / 120) | 4.3 ms |
| BlockedUnlockPickup | 100.0% (62.9) | 100.0% (82.6) | 95.0% (87.3) | **98.3%** (1 destroyed / 120) | 4.9 ms |
| **5-Task Mean** | **99.0%** | **95.0%** | **96.0%** | **96.7%** | **~6.0 ms** |

**Passed cleanly on all three experiences** (96.7% gate mean vs. card 164 Arm A's 99.5% on exp 1 and card 162's 91.5%, running **3–5× faster** per step with zero Dijkstra and zero move model). Note: exp 2's 80% on KeyCorridor traces to `model_2_mlp.pkl` missing one closed-door colour's statement (BFS terminates cleanly at 26 expansions), shared with card 164.

## 6. Success criteria and prediction

Test: card 160's five tasks, **100 fresh layouts each (`seeds 9000–9099`, never run)**, all **three reward-free random-play experiences**, code and networks (`reach_{1,2,3}.pt`, `q_reach_{1,2,3}.pt`, `model_{1,2,3}_mlp.pkl`) frozen in `runs/165/frozen.txt` before the run. Comparison: card 164 Arm A (Dijkstra System 2 + System 1 $q$) run on the exact same `seeds 9000–9099`.

1. **Zero-Dijkstra System 2 matches Card 164 within the declared general-form margin (AGENTS):** mean success over the five tasks at least **94.0%** on the mean of three experiences and at least **92.0%** in each experience, and within 5 points of card 164 Arm A on the 5-task mean (with DoorKey-8x8 and DoorKey-16x16 each ≥ **98.0%**, UnlockPickup and BlockedUnlockPickup each ≥ **92.0%**).
2. **Learned reachability is exact across all experiences and unseen topologies:** `RNet` achieves ≥ **99.5%** accuracy on both `r_arr` and `r_act` across all five tasks (including DoorKey-16x16 and KeyCorridorS3R3) in every experience (already verified at 100.00% on the frozen `reach_{1,2,3}.pt` checkpoints on 20 layouts/task; re-verified on `seeds 9000–9019`).
3. **Compute efficiency (GOAL P17):** `ZeroDijkstraAgent` is at least **2× faster** in mean ms/step than card 164 Arm A across the five tasks on the same fresh layouts.

Prediction: all three criteria pass comfortably (`ZeroDijkstraAgent` ~96–97% mean against card 164 Arm A's ~98%, 100.00% `RNet` accuracy, ~3× lower ms/step).

Budget (after user approval): no new training needed (`reach_{1,2,3}.pt` and `q_reach_{1,2,3}.pt` are already fitted and frozen). Running `ZeroDijkstraAgent` + Card 164 Arm A across 3 experiences × 5 tasks × 100 fresh layouts (`seeds 9000–9099`) in parallel takes **~6–8 minutes total** (two capped `<10`-minute waves).

## 7. Result

**Pass (3 of 3 criteria).** All 18 code and checkpoint files were frozen before the run (`runs/165/frozen.txt`, `sha256sum -c` verified unchanged after completion). Every number is in [results.json](results.json) across the exact same **100 fresh layouts per task (`seeds 9000–9099`, 1,500 episodes per agent across 3 reward-free random-play experiences)**.

### Head-to-head on fresh layouts (`seeds 9000–9099`)

| Task | Card 165 (`ZeroDijkstraAgent`) Exp 1 / 2 / 3 | **165 Mean** (steps) | Card 164 Arm A (Dijkstra S2) Exp 1 / 2 / 3 | **164 Mean** (steps) | **165 ms/step** | **164 ms/step** | **Speedup** |
|---|---:|---:|---:|---:|---:|---:|---:|
| DoorKey-8x8 | 100 / 100 / 100 | **100.0%** (17.3) | 100 / 100 / 100 | **100.0%** (17.3) | 7.1 ms | 12.9 ms | **1.80×** |
| DoorKey-16x16 | 100 / 100 / 100 | **100.0%** (48.5) | 100 / 100 / 100 | **100.0%** (48.5) | 8.4 ms | 13.0 ms | **1.55×** |
| KeyCorridorS3R3 | 96 / 86 / 94 | **92.0%** (39.0) | 96 / 87 / 97 | **93.3%** (36.8) | 6.1 ms | 27.8 ms | **4.53×** |
| UnlockPickup | 98 / 97 / 94 | **96.3%** (43.1) | 98 / 99 / 99 | **98.7%** (25.3) | 5.2 ms | 17.9 ms | **3.41×** |
| BlockedUnlockPickup | 98 / 99 / 94 | **97.0%** (62.3) | 98 / 99 / 99 | **98.7%** (40.5) | 5.5 ms | 16.2 ms | **2.96×** |
| **5-Task Mean** | **98.4 / 96.4 / 96.4** | **97.1%** (1,456/1,500) | **98.4 / 97.0 / 99.0** | **98.1%** (1,472/1,500) | **6.48 ms** | **17.55 ms** | **2.71×** |

### Criterion check

| Criterion | Bar | Measured | Verdict |
|---|---|---|---|
| 1. Fresh 5-task success & general-form margin | ≥ 94.0% mean, ≥ 92.0% each exp, within 5.0 pts of 164; DK8/DK16 ≥ 98%, UP/BUP ≥ 92% | **97.1%** mean (`98.4, 96.4, 96.4`), **−1.06 pts** vs 164 (98.1%); DK8 **100%**, DK16 **100%**, UP **96.3%**, BUP **97.0%** | **PASS** |
| 2. Learned reachability `RNet` on `seeds 9000–9019` | ≥ 99.5% on both `r_arr` and `r_act` across all 5 tasks in every experience | **100.00%** on all 15 (exp × task) pairs (**224,448** queries: `112,224` `r_arr` + `112,224` `r_act`, **0 fp, 0 fn**) | **PASS** |
| 3. Compute efficiency (GOAL P17) | ≥ 2.0× faster in mean `ms/step` than Card 164 Arm A | **6.48 ms/step** vs **17.55 ms/step** = **2.71× faster** (up to **4.53×** on KeyCorridorS3R3) | **PASS** |

### Diagnostic details

- **Exact match on DoorKey:** Across all 600 fresh DoorKey-8x8 and DoorKey-16x16 episodes, `ZeroDijkstraAgent` and Card 164 Arm A solve 100.0% in the **exact same mean step counts** (`17.3` and `48.5` steps) and **exact same walk-closest agreement** (`99.5%` and `82.1%`), confirming `RNet` + event-level BFS selects identical event chains without Dijkstra or a move model.
- **Paired layout comparison (1,500 episodes):** both solve **1,446**, both fail **18**, only 165 solves **10** (`8` KeyCorridor, `1` UnlockPickup, `1` BlockedUnlockPickup), and only 164 solves **26** (`12` KeyCorridor, `8` UnlockPickup, `6` BlockedUnlockPickup). Goal tokens destroyed: **6** for 165 vs **8** for 164 (down from 87 in Card 162).
- **Where the small −1.06 pt gap comes from:** On UnlockPickup and BlockedUnlockPickup, Card 165's BFS chooses a drop cell purely by reachability and proximity without a shortest-path cost to the downstream room, so it occasionally takes extra steps (`43.1` vs `25.3` on UnlockPickup, `62.3` vs `40.5` on BlockedUnlockPickup) and hits the episode step cap on a few Exp 3 layouts where `q_reach` takes longer to converge after a drop. Experience 1 ties Card 164 Arm A identically (**98.4% vs 98.4%**) while running **2.61× faster** (`6.44` vs `16.83 ms/step`).

## 8. Decision

**Keep.** `ZeroDijkstraAgent` removes every remaining hand-written navigation and chaining component from System 2 (`paths` / Dijkstra over the `W × H × 4` pose graph, Card 162's separate move model `net.Rel`, `want_in`, `want_out`, `free_reach`, `forward_ok`, and `C_BLOCK`), replacing them with a ~150-parameter differentiable reachability propagator (`RNet`) trained in 15.5 seconds from pure Stage R random play plus batched event-level BFS. It passes all three criteria on unseen fresh layouts (**97.1%** vs **98.1%**, **100.00%** reachability across 224,448 queries, **2.71×** faster per step and **4.53×** faster on KeyCorridorS3R3) with zero headings, labels, or cardinal priors.

Next directions (Card 166):
1. **Partial observability (7×7 egocentric view + persistent belief map):** drop the declared "whole map in view" assumption so the zero-Dijkstra System 2 (`RNet` + BFS) + System 1 (`q_reach`) stack plans over unknown/occluded rooms and explores when the goal or key is unseen (directly addressing Crafter / Craftium / robotics portability).
2. **Unify the cell effect model and `RNet` into one learned relational model (or bring `RNet` + System 1 back to OGBench `visual-scene`):** replace the tabular/MLP cell effect model with a factored what/where predictor, or test neural reachability propagation + event-level BFS over OGBench's continuous object/place graph.
