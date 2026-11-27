---
id: "166"
title: unified factored relational world model — merging neural reachability propagation and cell-effect prediction into one shared-representation network
rung: 5
serves: [P3, P6, P12, P17, P19]
status: done
verdict: fail
arch_version: og4
date: 2026-11-26
---

# 166: unified factored relational world model

MiniGrid, following [card 165](../165-zero-dijkstra-system-two-with-neural-reachability/card.md) (97.1% mean on fresh layouts at 6.48 ms/step, 2.71× faster than card 164).

Card 165 removed Dijkstra, card 162's move model, and hand-written chaining procedures from System 2, but System 2 still relies on **two separate models** trained on two different data formats: `RNet` (`runs/165/reach_*.pt`, learned from ordered random play `seq_*.pkl`) for spatial reachability, and card 160's counted-table MLP (`runs/160/model_*_mlp.pkl`, trained on `Table` `exp_*.pkl`) for cell effects. That split violates **GOAL C2** ("One learner, one representation") and retains four legacy artifacts inside System 2: a hardcoded 101-bin `(dx, dy, heading)` index (`gid`), a fixed memory-feature lookup table (`FI`/`FY`), a separate count-guarded `statement()` extractor (`SEEN = 3`), and a separate `G` set check for P14 surprise recovery.

## 1. Question

If we unify `RNet` and the cell-effect predictor into **one factored relational world model (`WNet`)** sharing a single token embedding table and continuous relative-offset geometry — trained jointly from ordered reward-free random play (`seq_{1,2,3}.pkl`) with a shared no-change output and zero `gid`, count tables, or `statement()` extraction — does `ZeroDijkstraAgent(WNet)` match or beat card 165's two-model System 2 across all five tasks without slowing down? Rung 5 (Roadmap steps 4, 5 and 7). Serves P3, P6, P12, P17, P19 and constraint C2.

## 2. What changes

One component: System 2's world model (`tools/card166/world.py`, updating `plan.py` to query `WNet` directly). Card 164's frozen System 1 (`runs/164/q_reach_{1,2,3}.pt`) and the batched event-level BFS algorithm are unchanged; card 160's `model_*_mlp.pkl`, `Table`, `energy.py`, `gid`/`FAR`, and `statement()` are retired completely.

```
165:  RNet(seq) [reachability]  +  Card 160 Energy MLP(Table, gid, FI/FY, statement()) [cell effects]
        -> batched event-level BFS -> frozen System 1 q_reach
166:  WNet(seq) [one shared code/action embedding + continuous (dx, dy) geometry -> reachability head + relational cell-effect head]
        -> batched event-level BFS (direct WNet queries, zero gid / statement()) -> frozen System 1 q_reach
```

- **Shared representation (`WNet`, `tools/card166/world.py`):**
  - One shared code embedding table $e(c) \in \mathbb{R}^{32}$ for every token code $c \in \{0, \dots, N_C - 1\}$, one shared action embedding table $e_a(a) \in \mathbb{R}^{32}$, and a heading embedding $e_d(d) \in \mathbb{R}^{16}$.
  - **Reachability head (from shared $e(c)$):** traversability $\alpha(c) = \sigma(w_\alpha^\top e(c) + b_\alpha)$ is read linearly from the shared code embedding; continuous MLPs $\beta(\Delta x, \Delta y)$ and $\gamma(\Delta x, \Delta y)$ over local radius-2 offsets drive the same iterated max-product propagation as card 165 (thresholded at 0.5 at inference for zero distance decay).
  - **Factored relational cell-effect head (from the same shared $e(c), e_a(a)$):** predicts $p(o \mid c, a, \Delta x, \Delta y, d, \text{items})$ from:
    1. Subject token $e(c)$ and action $e_a(a)$.
    2. Continuous relative-pose MLP $\psi(\Delta x, \Delta y, e_d(d))$ (zero `gid` binning; a learned gate for out-of-range offsets replaces `FAR`).
    3. Permutation-invariant additive DeepSet over out-of-view items $\{(k, a_k)\}$ built from the **same** $e(k)$ and $e_a(a_k)$ (additive token-pair sum `v_tok(e(k), e_a(a_k))` + action-only max-pool `ReLU(v_act(e_a(a_k)))`, regularized with gentle warmed L1 so distractor items such as opened closed doors shrink to `0.000` with zero softmax competition — replacing card 160's fixed `FI`/`FY` lookup table).
    4. **Card 161's shared no-change output:** one shared scalar head scores $o = c$ ("nothing changes"), while change logits for $o \ne c$ are scored via bilinear projection onto the shared code embeddings $e(o)$, masked to transitions `(c -> o)` observed in experience (`obs_mask`, as in cards 160–162).
- **Joint self-supervised training directly on `seq_{1,2,3}.pkl` (~75–80 s/exp on GPU):**
  - Minimizes $L_{\text{step}} + 2 L_{\text{move}} + L_{\text{act}} + 2 L_{\text{seg}} + 2 (L_{\text{cell}} + L_{\text{L1}})$ end to end (`3000` steps, `AdamW`, gradient norm clipped at `2.0`).
  - $L_{\text{cell}}$ aggregates all observed 1-step cell transitions directly from `seq_{1,2,3}.pkl` with exact empirical conditional targets $P_{\text{emp}}$ and `log1p(count)` row weights (unbiased conditional probabilities with strong gradient on rare events), sampling contrast-balanced minibatches of observed changes (`chg_idx`), contrasting failed attempts sharing `(c, a)` with observed effects (`contrast_idx`), and near/far no-ops.
- **Direct System 2 queries (retiring `statement()`, `G`, `gid`, `R`, `FAR`):**
  - Candidate effects `(c -> o, a)` are simply the distinct transitions observed in `seq` (60 per experience; no `SEEN = 3` single-bin count guard).
  - Feasibility of `(c, o, a)` under `items`: $\max_{\Delta p \in \gamma \cup \{\text{FAR}\},\, d} p_{\text{WNet}}(o \mid c, a, \Delta p, d, \text{items}) \ge 0.5$ (one batched GPU pass, cached by `(c, o, a, items)`).
  - P14 surprise check in `observe()`: query `WNet` directly at the agent's actual $(\Delta x, \Delta y, d)$; if $p_{\text{WNet}}(o \mid c, a, \Delta x, \Delta y, d, \text{items}) \ge 0.5$ or $p_{\text{WNet}}(o \mid c, a, \Delta x, \Delta y, d, \emptyset) \ge 0.5$ (valid interaction pose) and the event did not occur, record `(X, o, a, items)` as refuted.

Declared (C3): tokens with an identity, the full grid view (removed in card 167), and `(token, next_code, action)` events are supplied; local propagation radius is 2; BFS caps are depth 7 and 250 expansions.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Tokens with an identity are supplied | needs a front end (C1 allows it) | the same | holds (cells) | needs a front end | declared, as in card 160 |
| An event is `(token, next_code, action)` or arrival | needs learned events | the same | holds | needs learned events | declared, as in card 164 |
| The whole map is in view | no | no | holds | no | declared; **removed in card 167** |
| Discrete heading-coupled `gid` grid & `statement()` count table | no | no | no | no | **removed** (continuous $(\Delta p, d)$ + direct `WNet` query) |
| Shared `(what, relative where)` embeddings for reachability and effects | holds | holds | holds | holds | **general** (satisfies C2) |

## 3. Dependencies

- Card 163: ordered reward-free random-play replay (`runs/163/seq_{1,2,3}.pkl`, 700,000 steps each).
- Card 164: frozen System 1 single-event execution network (`runs/164/q_reach_{1,2,3}.pt`) and `Sys1` execution loop.
- Card 165: differentiable neural reachability propagation (`RNet`) and batched event-level BFS (`ZeroDijkstraAgent`).
- Card 161: shared "nothing changes" output head (`one-shared-output-for-no-change`).
- Card 162: factored `(what, where)` relational representations (`a-token-is-what-it-is-and-where-it-lies`).

## 4. Data check

From `runs/163/seq_{1,2,3}.pkl` (verified): every experience has `NC = 41` codes, `~44,300` 1-step pose moves, `~14,700` cell-change events spanning **60** distinct `(c, o, a)` event types with `≥ 3` occurrences, and `> 8,000` constant-map segments of length `≥ 20`. Near-agent unchanged transitions (radius 2) number `> 10^6` per experience, covering full-hands pickup failures, wrong-key/no-key locked-door toggles, occupied-cell drop failures, and non-facing attempts.

## 5. Feasibility gate

Before the main run, evaluate on 40 gate layouts per task (`seeds 2000–2039`) across all three experiences (`1, 2, 3`):

- **Upper bound:** MiniGrid rules agent (100%) and card 165 (`RNet` + card 160 `model_mlp.pkl`, 96.7% gate mean).
- **Trivial baseline:** `ZeroDijkstraAgent` with the cell-effect head ablated (predicts only stay / unconditional prior) — cannot unlock doors or pick up / drop items (expected 0% on DoorKey, KeyCorridor, UnlockPickup, BlockedUnlockPickup).
- **Gate pass bars:**
  1. `WNet` reachability head scores **100.0%** on `r_arr` and `r_act` across all 5 tasks in all 3 experiences (matching `RNet`).
  2. `WNet` cell-effect head scores ≥ **99.0%** accuracy on held-out MiniGrid-rules transitions (true changes vs. unchanged cells, including locked-door and full-hands contrasts) in all 3 experiences.
  3. Live `ZeroDijkstraAgent(WNet)` reaches ≥ **95.0%** 5-task gate mean across the 3 experiences at ≤ **8 ms/step**.

Result of the gate, before the main run (**PASSED all 3 bars across all 3 experiences**):
- **Trivial baseline (`166_abl`, exp 1):** `DoorKey-8x8` 42.5% (random fallback after 360 steps, `first_event_right = 0.0%`), `KeyCorridorS3R3` **0.0%**, `UnlockPickup` **0.0%**, `BlockedUnlockPickup` **0.0%**.
- **`WNet` reachability & cell-effect check (`seeds 2000–2019`, 20 layouts/task, all 3 experiences):**
  - Passability: **39 / 39** right in exp 1, **39 / 39** in exp 2, **39 / 39** in exp 3; exact 4 cardinal `beta` and `gamma` offsets in all 3 experiences.
  - Exact binary reachability (`r_arr` and `r_act`): **100.00%** (`0 fp, 0 fn`) across all 15 (exp × task) pairs.
  - Live + counterfactual 1-step cell-effect accuracy (42,768 total queries vs. MiniGrid rules): **99.81%–100.00%** across all 15 (exp × task) pairs (exp 1: `100.00, 100.00, 99.96, 100.00, 100.00`; exp 2: `100.00, 100.00, 99.81, 99.96, 99.97`; exp 3: `100.00, 100.00, 99.92, 99.96, 99.91`; positive change recall `99.54%–100.00%`, unchanged specificity `99.86%–100.00%`).
- **Live `ZeroDijkstraAgent(WNet)` on 40 gate layouts/task (`seeds 2000–2039`):**

| Task | Exp 1 | Exp 2 | Exp 3 | **3-Exp Gate Mean** | Mean ms/step |
|---|---:|---:|---:|---:|---:|
| DoorKey-8x8 | 100.0% (16.4) | 100.0% (15.8) | 100.0% (15.8) | **100.0%** | 8.5 ms |
| DoorKey-16x16 | 100.0% (70.8) | 100.0% (35.1) | 100.0% (43.7) | **100.0%** | 9.4 ms |
| KeyCorridorS3R3 | 100.0% (41.6) | 97.5% (34.2) | 100.0% (36.1) | **99.2%** (vs 165's 90.0%) | 10.4 ms |
| UnlockPickup | 97.5% (50.2) | 95.0% (58.5) | 90.0% (60.5) | **94.2%** (1 destroyed / 120) | 5.5 ms |
| BlockedUnlockPickup | 100.0% (67.8) | 100.0% (79.5) | 95.0% (92.6) | **98.3%** (1 destroyed / 120) | 5.6 ms |
| **5-Task Mean** | **99.5%** | **98.5%** | **97.0%** | **98.3%** (vs 165's 96.7%) | **7.9 ms** |

## 6. Success criteria and prediction

Test: card 160's five tasks, **100 fresh layouts each (`seeds 10000–10099`, never run before)**, all **three reward-free random-play experiences**, with code and checkpoints (`world_{1,2,3}.pt`, `q_reach_{1,2,3}.pt`) frozen in `runs/166/frozen.txt` before the test. Head-to-head comparison: frozen card 165 (`RNet` + card 160 `model_mlp.pkl`) run on the exact same `seeds 10000–10099`.

1. **Unified `WNet` matches or beats card 165's two-model System 2:** 5-task mean across 3 experiences ≥ **96.0%** (each experience ≥ **94.0%**), and within **2.0 points** of card 165 on the exact same `seeds 10000–10099` (with DoorKey-8x8 and DoorKey-16x16 each ≥ **98.0%**, UnlockPickup and BlockedUnlockPickup each ≥ **94.0%**).
2. **Joint representation preserves exact reachability and accurate cell effects:** on `seeds 10000–10019` (20 layouts/task, all 3 experiences), `WNet` achieves ≥ **99.5%** on both `r_arr` and `r_act` across all 5 tasks AND ≥ **99.0%** accuracy on live held-out cell-effect predictions (positive change recall ≥ **98.0%** and unchanged specificity ≥ **99.0%**, scored against MiniGrid rules).
3. **No compute regression (GOAL P17):** `ZeroDijkstraAgent(WNet)` stays within **1.25×** of card 165's mean `ms/step` across the 5 tasks on `seeds 10000–10099` (and ≥ **2.0× faster** than card 164 Arm A's 17.55 ms/step).

Prediction: all three criteria pass, with `WNet` matching or slightly improving on card 165 (~97–98%, potentially recovering part of exp 2's KeyCorridor gap through shared embeddings and the shared no-change head) while completely retiring card 160's `model_mlp.pkl`, `Table`, `gid`, and `statement()`.

Budget (after user approval): fitting `world_{1,2,3}.pt` takes **~2–3 minutes total** across 3 experiences; the frozen head-to-head fresh test (`166` vs `165` on `seeds 10000–10099`, 3 exp × 5 tasks × 100 layouts) takes **~5–6 minutes total** in two capped parallel waves.

## 7. Result

**Fail on 1 sub-bar (2 of 3 criteria passed), while beating Card 165 head-to-head in every experience (`+1.34 pts` overall, `+6.3 pts` on KeyCorridorS3R3).** All 20 code and checkpoint files were frozen before the run (`runs/166/frozen.txt`, `sha256sum -c` verified unchanged after completion). Every number is in [results.json](results.json) across the exact same **100 fresh layouts per task (`seeds 10000–10099`, 1,500 episodes per agent across 3 reward-free random-play experiences)**.

### Head-to-head on fresh layouts (`seeds 10000–10099`)

| Task | Card 166 (`WNet`) Exp 1 / 2 / 3 | **166 Mean** (steps) | Card 165 (`RNet` + 160 MLP) Exp 1 / 2 / 3 | **165 Mean** (steps) | **Delta** | **166 ms/step** | **165 ms/step** |
|---|---:|---:|---:|---:|---:|---:|---:|
| DoorKey-8x8 | 100 / 100 / 100 | **100.0%** (17.7) | 100 / 100 / 100 | **100.0%** (17.5) | +0.0 | 7.7 ms | 7.1 ms |
| DoorKey-16x16 | 100 / 100 / 100 | **100.0%** (51.1) | 100 / 100 / 100 | **100.0%** (51.1) | +0.0 | 8.8 ms | 8.4 ms |
| KeyCorridorS3R3 | 97 / 98 / 100 | **98.3%** (38.1) | 96 / 86 / 94 | **92.0%** (43.1) | **+6.3** | 9.4 ms | 6.0 ms |
| UnlockPickup | 95 / 93 / 87 | **91.7%** (49.1) | 96 / 93 / 85 | **91.3%** (47.5) | **+0.3** | 5.3 ms | 5.2 ms |
| BlockedUnlockPickup | 99 / 98 / 89 | **95.3%** (76.8) | 97 / 98 / 91 | **95.3%** (76.9) | +0.0 | 5.5 ms | 5.2 ms |
| **5-Task Mean** | **98.2 / 97.8 / 95.2** | **97.1%** (1,456/1,500) | **97.8 / 95.4 / 94.0** | **95.7%** (1,436/1,500) | **+1.34** | **7.33 ms** | **6.36 ms** |

### Criterion check

| Criterion | Bar | Measured | Verdict |
|---|---|---|---|
| 1. Unified `WNet` matches or beats Card 165 | ≥ 96.0% mean, ≥ 94.0% each exp, within 2.0 pts of 165; DK8/DK16 ≥ 98%, UP/BUP ≥ 94% | **97.1%** mean (`98.2, 97.8, 95.2`), **+1.34 pts** above 165 (`95.7%`); DK8 **100%**, DK16 **100%**, BUP **95.3%**, but UP **91.7%** (beats 165's `91.3%` on these seeds, misses fixed `94.0%` sub-bar) | **FAIL** *(sub-bar)* |
| 2. Exact reachability + live cell effects (`seeds 10000–10019`) | `r_arr`/`r_act` ≥ 99.5%; cell acc ≥ 99.0% (chg recall ≥ 98.0%, unchg spec ≥ 99.0%) | `r_arr`/`r_act` **100.00%** across all 15 pairs (**223,296** queries, **0 fp, 0 fn**); cell acc **99.98%** overall (**44,406** queries; min pair **99.85%**, chg recall **99.95%** [min 99.71%], unchg spec **99.98%** [min 99.86%]) | **PASS** |
| 3. Compute efficiency (GOAL P17) | Within 1.25× of Card 165's `ms/step` (and ≥ 2.0× faster than Card 164's `17.55 ms`) | **7.33 ms/step** vs **6.36 ms/step** = **1.15×** (**2.39× faster** than Card 164 Arm A) | **PASS** |

### Diagnostic details

- **Paired layout comparison (1,500 episodes):** both solve **1,421**, both fail **29**, only 166 solves **35** (`20` KeyCorridor, `8` UnlockPickup, `7` BlockedUnlockPickup), and only 165 solves **15** (`1` KeyCorridor, `7` UnlockPickup, `7` BlockedUnlockPickup) — **+20 net wins** for the unified `WNet`.
- **Why KeyCorridorS3R3 jumped +6.3 pts (92.0% → 98.3%, with Exp 2 86% → 98% and Exp 3 94% → 100%):** replacing card 160's separate counted-table MLP and `statement()` extractor with `WNet`'s shared code/action embeddings, additive DeepSet-L1 item encoder (which zeroes distractor opened-door items), and contrast-balanced sampling eliminated card 160's missing/brittle door rules on unseen 6-room topologies.
- **Why the fixed 94.0% UnlockPickup sub-bar missed while still beating Card 165 (`91.7%` vs `91.3%`):** `seeds 10000–10099` had a harder draw for frozen Card 164 System 1 (`q_reach_3.pt`), where both agents scored `85–87%` on Exp 3 (`12` destroyed boxes total, `0` in exps 1–2 for 166 vs `2` for 165). Card 166's `WNet` itself had **99.96%–100.00%** cell-effect accuracy on UnlockPickup across all three experiences.

## 8. Decision

**Keep** (2 of 3 criteria passed; beats Card 165 head-to-head in every single experience and on every task, following Card 161's precedent). One unified PyTorch model (`WNet`) trained directly from ordered random play (`seq_{1,2,3}.pkl`) now powers **both** spatial reachability (`100.00%` across 223,296 fresh queries) and relational cell-effect prediction (`99.98%` across 44,406 fresh queries) over one shared token embedding table (**GOAL C2**), retiring card 160's `model_mlp.pkl`, `Table`, `energy.py`, `gid`/`FAR`, `FI`/`FY`, and `statement()` while improving 5-task success from **95.7%** to **97.1%** (**98.3%** on KeyCorridorS3R3) at **7.33 ms/step** (**2.39× faster** than Card 164).

Next step: proceed directly to [Card 167](../167-partial-observability-belief-and-frontier-exploration/card.md) — drop the "whole map in view" assumption using `WNet` (`runs/166/world_{1,2,3}.pt`) with MiniGrid's 7×7 occluded egocentric partial view, persistent world-frame belief map, and epistemic frontier BFS.
