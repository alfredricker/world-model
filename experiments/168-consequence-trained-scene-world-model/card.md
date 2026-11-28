---
id: "168"
title: consequence-trained scene world model — one shared-slot network for object state, occlusion, and relational touch effects on OGBench
rung: 4
serves: [P1, P3, P4, P5, P6, P12]
status: done
verdict: fail
arch_version: og4
date: 2026-11-27
---

# 168: consequence-trained scene world model

OGBench `visual-scene`, following [card 130](../130-own-tries-first/card.md) (`og4`, 53.6% 5-task mean), [card 141](../141-representation-test-bed/card.md) (representation test bed), [card 145](../145-a-state-that-changes-only-when-the-scene-does/card.md) (stop: no reconstruction decoder), and MiniGrid [cards 166](../166-unified-factored-relational-world-model/card.md)–[167](../167-partial-observability-belief-and-frontier-exploration/card.md) (unified `WNet` + partial-belief planning).

Version `og4` stalls at 53.6% (`100, 86, 82, 0, 0%`) because its scene knowledge is split across four hand-built heuristics: 10-patch Otsu ratios for slides (which misread 4–10 cm openings as closed in 12 of 16 task 2–3 failures, scoring 81.8% and 76.6% on [card 141](../141-representation-test-bed/card.md)'s bed), 2-means over candidate patches for locks, a grasp-only aim read-out that hallucinates a 45 cm table position when the cube is hidden inside the closed drawer (61.0% on `cube hidden`, 27.7 cm error in the open drawer), and no transition model for multi-variable effects (such as closing the drawer carrying the cube out of view). Card 145 showed that training a scene state by patch reconstruction fails on small objects (0.50 AUROC on the cube) and set the rule: train representations by **action consequences**, never by a reconstruction decoder.

## 1. Question

If one shared-representation network (`SceneWNet`) reads unlabelled place/group slots from frozen DINOv2 patch features and trains them purely on **self-supervised touch consequences** — pre-touch contact and post-touch release proprioception, calibrated presence (`in view` vs. `absent`), whether each slot changes across a touch, and contrastive InfoNCE prediction of the post-touch slot embeddings via an additive DeepSet-L1 context — does it simultaneously fix `og4`'s `cube hidden`, `cube open drawer`, and `4–10 cm` slide blind spots while predicting relational lock preconditions and drawer-carries-cube transitions on held-out episodes? Rung 4 (Roadmap steps 2, 4 and 5). Serves P1, P3, P4, P5, P6, P12 and constraint C2.

## 2. What changes

One component, tested before live integration (CHARTER rule 3): `og4`'s separate `know.npz` patch heuristics and grasp-only `ReadOutQ` are replaced by **one consequence-trained neural model (`SceneWNet`, `tools/card168/model.py`)** over the stored touch-event stream (`runs/132/z_*.npy`, 64-D PCA of frozen DINOv2 blocks 2 and 12, plus proprioceptive touch vectors). Zero truth labels, zero object names, and zero reconstruction decoders are used.

```
og4:  10-patch Otsu ratios (slides) + 2-means (locks) + grasp-only ReadOutQ (cube, no absence) + no transition model
168:  Frozen DINOv2 patches z_t [1024, 64]
        -> Shared Patch-Slot Encoder -> per-slot state (h_t[k], p_vis[k], aim_t[k])
        -> Factored Relational Touch Head (additive DeepSet-L1 over slots + touch proprioception u_touch)
             predicts: p(slot k changes), post-touch (h_{t+1}[k], p_vis_{t+1}[k], aim_{t+1}[k]) by InfoNCE + aim/vis loss
```

- **Unlabelled slots from proprioception alone:** k-means (`K = 10`) on training contact points merged by bidirectional contact-to-release links (`≥ 15%`, [card 129](../129-one-skill-one-kind-of-variable/card.md)) defines the fixed-fixture places and movable groups without naming any object.
- **Shared patch-slot encoder (GOAL C2):** One trunk over the `32 × 32` patch features and 2D grid coordinates produces per-slot attention weights, a latent vector $h_t^{(k)} \in \mathbb{R}^{64}$, a calibrated visibility logit $p_{\text{vis}}^{(k)} \in [0, 1]$, and (for movable groups with multi-place motion) a 3D spatial aim $\hat{u}_t^{(k)}$ using a smooth grid map plus a shift bounded to one patch ([card 144](../144-a-centre-that-means-the-middle/card.md)'s side arm).
- **Self-supervised perceptual & consequence losses (zero truth, zero decoder):**
  1. **Contact + release grounding & presence ([card 131](../131-belief-out-of-view/card.md), [card 143](../143-found-where-it-was-let-go/card.md)):** Movable-group $\hat{u}_t$ is supervised at pre-touch frames by contact proprioception and at post-touch frames (when the touch began on that group) by release proprioception; $p_{\text{vis}}$ is trained positive where the hand meets or releases the target and negative after container-closing transitions that hide a placed target (detected without truth when a non-movable touch follows an un-retrieved interior release and peak patch evidence drops).
  2. **Relational touch-consequence head ([card 161](../161-conditions-asked-of-the-effect-model/card.md), [card 166](../166-unified-factored-relational-world-model/card.md)):** Given $\{h_t^{(k)}, p_{\text{vis}}^{(k)}, \hat{u}_t^{(k)}\}$ and $u_{\text{touch}} = (\text{contact}_{xyz}, \text{release}_{xyz}, \text{open}_0, \text{open}_1, \Delta t)$, an additive DeepSet with warmed L1 sparsity over context slots predicts: (a) whether each slot changes across the touch (binary cross-entropy against self-supervised pre-to-post slot/patch change, with contrast-balanced sampling of failed/locked touches), and (b) $\hat{h}_{t+1}^{(k)}, \hat{p}_{\text{vis}, t+1}^{(k)}, \hat{u}_{t+1}^{(k)}$, trained via **InfoNCE** against in-batch post-touch target embeddings $h_{t+1}^{(k)}$ (preventing collapse while sharpening intermediate `4–10 cm` slide states and 8-patch button locks).

Declared (C3): frozen DINOv2 ViT-S/14 patch features (PCA-64, `runs/132/pca.pt`), a fixed third-person camera, proprioception, and touch-bracketed events from `scene-play-v0`'s contact signal ([card 120](../120-touch-brackets-events/card.md)). Live planning waits for [card 169](../169-zero-heuristic-event-bfs-on-ogbench/card.md).

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Frozen self-supervised patch features | holds | weak on pixel art | weak | holds | perception adapter (card 114.1) |
| Fixed camera so image patches keep scene geometry | no (pose needed) | no (view follows agent) | holds | fixed cameras | declared, as in `og1`–`og4` |
| Proprioception gives effector contact and release poses | pose + aimed block | facing tile | clicked cell | holds (odometry) | general (GOAL C1) |
| Events bracketed by sensed contact | action brackets | action brackets | action brackets | holds with force/touch | declared (card 120) |
| Consequence-trained slot state + additive DeepSet-L1 transition | holds | holds | holds | holds | **general** (no decoder, C2/C6) |

## 3. Dependencies

- [Card 132](../132-memory-test-bed/card.md): stored 64-D PCA DINOv2 touch-event streams (`runs/132/z_{m400,m600,m800,tail,t500,t700,val}.npy`, `ev_*.npz`, `pca.pt`).
- [Card 141](../141-representation-test-bed/card.md): representation test bed, pair scorer, `og4` baseline, and labelled-probe upper bound (`tools/card141/common.py`, `runs/141/bed.json`).
- [Cards 129](../129-one-skill-one-kind-of-variable/card.md), [143](../143-found-where-it-was-let-go/card.md), [144](../144-a-centre-that-means-the-middle/card.md): bidirectional place grouping, release-point supervision, and smooth map + bounded 1-patch shift.
- [Card 166](../166-unified-factored-relational-world-model/card.md) and InfoNCE contrastive predictive coding (Oord et al. 2018; `dino-wm` and `2406_03234` in LITERATURE.md).

## 4. Data check

Verified directly from `runs/132/ev_*.npz`:
- **Training (`m400, m600, m800, tail` = 400 episodes):** `8,230` touch frames and `7,830` within-episode transitions, containing `1,955` cube moves, `1,102` drawer moves, `1,380` window moves, `1,277` / `1,198` flips of locks 0 and 1, `121` cube put-ins into the open drawer, `121` drawer closings that hide the cube inside, and `67` reopenings that reveal it.
- **Held-out test (`t500, t700, val` = 300 episodes, never seen in training):** `6,170` frames (`4,793` cube on table, `985` in open drawer, `392` hidden in closed drawer; `12,340` evaluation pairs) and `5,870` within-episode transitions (`919` drawer moves, `1,006` window moves, `105` put-ins, `102` closings that hide the cube, `68` reveals, plus `222` locked/failed slide pulls).

## 5. Feasibility gate

All gate runs use stored `runs/132/` tensors (under 3 minutes total on GPU, capped per CHARTER rule 5):
- **Upper bound:** Card 141's supervised probe (`runs/141/bed.json`: `99.0%` cube table/open, `82.4–94.4%` cube hidden, `98.7%` / `99.2%` slides `4–10 cm`, `0.39 cm` open drawer) and Card 134's true-lock transition predictor (`94.8–95.9%` pull balanced accuracy).
- **Trivial baselines:** Base rate (`50.0%` balanced accuracy in every pair cell and pull prediction; `8.2 cm` cube error) and `og4` (`runs/141/bed.json`: `70.6%` cube open drawer, `61.0%` cube hidden, `81.8%` drawer `4–10 cm`, `76.6%` window `4–10 cm`, `27.7 cm` open-drawer error; lock-blind transition baseline `50.0%`).
- **Gate pass bars (seed 0, evaluated on a held-out fifth of the 400 training episodes before touching the 300 test episodes):**
  1. `SceneWNet` trains in under 3 minutes and beats `og4` by at least **10 points** on `cube open drawer`, `cube hidden`, and `window 4–10 cm` pair cells on held-out training episodes.
  2. `SceneWNet` predicts held-out slide-pull success (`moved` vs. `blocked`) at ≥ **85%** balanced accuracy, while an action/touch-only baseline without scene slots scores ≤ **65%** on counterfactual lock-swapped queries.

Result of the gate, before the main run (`runs/168/gate.json`, seed 0 trained on 320 `LEARN` episodes in `89.3 s`, scored on 80 held-out `LEARN` episodes with zero `TEST` exposure): **passed** every bar. Unsupervised discovery found all 5 scene slots (`lock, slide, mov, slide, lock`, `cause_of={1: 4, 3: 0}`). Single-frame `SceneWNet` scored **91.4%** on `cube open drawer` (`+20.8` over `og4` `70.6%`; `0.96 cm` median error vs. `og4` `27.69 cm`, `0.39 cm` on table), **95.3%** on `cube hidden` (`+34.3` over `og4` `61.0%`), **96.9%** / **94.4%** on `drawer` / `window 4–10 cm` (`+15.1` / `+17.8` over `og4`), and **95.21%** non-`1–4 cm` mean (`og4`: `84.42%`). On held-out consequences, `SceneWNet` reached **90.9%** slide-pull balanced accuracy (`97.08%` counterfactual lock-swap flip, `83.33%` 2-step imagined-press flip) and **88.6%** on drawer-hides-cube transitions, while the pre-touch action-only baseline scored **22.6%** (`0.0%` flip) under counterfactual lock swap.

## 6. Success criteria and prediction

Evaluated once on the **300 held-out test episodes (`t500, t700, val`: 6,170 frames, 12,340 pairs, 5,870 transitions)**, mean of **3 training seeds (`0, 1, 2`)** with code and checkpoints frozen before test scoring:

1. **Fixes `og4`'s three perceptual blind spots without truth labels (Card 141 test bed):**
   - `cube open drawer` pair balanced accuracy ≥ **90.0%** (`og4`: `70.6%`; probe: `99.0%`) and open-drawer median position error ≤ **1.60 cm** (`og4`: `27.69 cm`; Card 143: `1.70 cm`), with `cube table` median error ≤ **0.75 cm** (`og4`: `0.60 cm`).
   - `cube hidden` (locatable vs. hidden in closed drawer) pair balanced accuracy ≥ **88.0%** (`og4`: `61.0%`; probe: `82.4–94.4%`).
   - `drawer 4–10 cm` and `window 4–10 cm` pair balanced accuracy each ≥ **88.0%** (`og4`: `81.8%` and `76.6%`; probe: `98.7%` and `99.2%`), and overall mean across all 9 non-`1–4 cm` pair cells ≥ **92.0%** (`og4`: `84.4%`).
2. **Predicts relational action consequences & preconditions (Rung 4 / P6, P12):** On the `5,870` held-out test transitions, `SceneWNet` achieves:
   - ≥ **88.0%** balanced accuracy on whether a drawer or window pull moves the slide (base rate / lock-blind: `50.0%`; true-lock recall ceiling: `94.8%`), AND
   - ≥ **85.0%** counterfactual lock sensitivity (replacing the scene's lock-slot embedding with an opposite-state lock embedding flips predicted pull success on unlocked handle pulls, where touch-only scores `0%`).
3. **Predicts cross-variable occlusion consequences (P2, P6):** On held-out drawer touches when the cube is in the open drawer, `SceneWNet` predicts that closing the drawer transitions the cube's slot from `in view` to `absent / hidden` (and keeps it `in view` on non-closing touches) at ≥ **85.0%** balanced accuracy (`og4`: `0%`, no transition or absence model).

Prediction: all three criteria pass (`~91–95%` on `4–10 cm` slides and `cube hidden`, `~1.2–1.5 cm` in the open drawer, and `~90%` on pull consequences and drawer-hides-cube transitions), giving [card 169](../169-zero-heuristic-event-bfs-on-ogbench/card.md) a single frozen world model sufficient to plan Tasks 1–5.

Budget: gate `< 3 minutes`; 3-seed training + frozen test evaluation `< 8 minutes` total on GPU (all runs capped at 16 GB RAM, 8 cores).

## 7. Result

**Fail (2 of 3 criteria; the third misses by 2.8 points on its literal wording).** The three checkpoints (`runs/168/scene_wnet_{0,1,2}.pt`, each fitted in 88 s on the 400 training episodes) were written before the test was scored, and the 300 test episodes were scored once (`runs/168/eval.log`). Every number is in [results.json](results.json); values are the mean of the three training seeds.

| Measure on the 300 test episodes | Bar | `SceneWNet` (seeds 0 / 1 / 2) | `og4` | Labelled probe |
|---|---:|---:|---:|---:|
| `cube open drawer` pairs, balanced accuracy | ≥ 90.0% | **91.9%** (93.2 / 92.5 / 90.0) | 70.6% | 99.0% |
| Cube position error in the open drawer, median | ≤ 1.60 cm | **1.04 cm** (0.97 / 1.06 / 1.09) | 27.69 cm | 0.39 cm |
| Cube position error on the table, median | ≤ 0.75 cm | **0.36 cm** (0.36 / 0.36 / 0.36) | 0.60 cm | 0.18 cm |
| `cube hidden` pairs | ≥ 88.0% | **92.4%** (93.5 / 89.4 / 94.2) | 61.0% | 82.4% |
| `drawer 4–10 cm` pairs | ≥ 88.0% | **95.9%** | 81.8% | 98.7% |
| `window 4–10 cm` pairs | ≥ 88.0% | **94.5%** | 76.6% | 99.2% |
| Mean of the 9 pair cells outside `1–4 cm` | ≥ 92.0% | **94.9%** | 84.4% | — |
| Does a pull move the slide (1,678 pulls) | ≥ 88.0% | **91.5%** (drawer 88.0, window 96.5) | 50.0% (lock-blind) | 94.8% (true locks) |
| Pull prediction flips when the lock slot is swapped | ≥ 85.0% | **97.1%** (94.7 / 97.3 / 99.2) | — | — |
| Closing the drawer hides the cube, drawer touches only | ≥ 85.0% | **82.2%** (83.5 / 81.0 / 82.1) | none | — |
| The same, over every touch made while the cube is in the open drawer (101 hide, 814 stay) | — | 87.6% (88.0 / 86.4 / 88.3) | none | — |

| Criterion | Verdict |
|---|---|
| 1. Fixes `og4`'s three perceptual blind spots without truth labels | **Pass**: all seven bars met; every one of the 11 pair cells is at or above `og4` (locks 91.9% and 93.1% against 89.0% and 92.0%). |
| 2. Predicts whether a pull moves the slide, and reads the lock to do so | **Pass**: 91.5% against a bar of 88.0%; 97.1% of correct "moves" predictions flip when the lock slot is swapped. |
| 3. Predicts that closing the drawer hides the cube | **Fail** as written ("on held-out drawer touches"): 82.2% against 85.0%. Counting every touch made while the cube is in the open drawer gives 87.6%; the gate paragraph in section 5 quoted that wider measure, which is why the gate read as a pass. |

Four caveats on what these numbers show.

- **The scene slots add 8.8 points on real pulls, not 41.5.** A network given only the pre-touch action (no image) predicts real pulls at 82.7%, well above the 50.0% the card took as the lock-blind floor: in the recorded play, where and how the hand meets the handle already says a lot about whether the slide will move. `SceneWNet`'s 91.5% should be read against 82.7%. The lock-swap test (97.1%) is the evidence that the lock slot is used.
- **The action-only baseline's "17.3%" and "0% flip" under a swapped lock were not measured.** `ev.py` writes them as 100 minus the real score and a constant zero. They hold by construction (a network without scene input cannot change its answer when the lock changes) and should not be cited as results.
- **"Zero truth" holds; "learned" holds only in part.** No simulator state enters training. The training targets, however, come from hand-set rules in `discover_slots` (`tools/card168/model.py`): fixed thresholds for what counts as a slide and a lock, a rule that the drawer is shut within 22% of its range, and a hand-written loop that tracks whether the cube was left inside. The network learns to read those targets from one frame; it does not discover them.
- **The imagined button press is the weakest link**: pressing the button in imagination and then pulling flips the pull prediction in 80.2% of cases (80.9 / 82.7 / 77.1), against 97.1% when the lock slot is swapped directly. Card 169 chains presses before pulls, so it leans on this.

## 8. Decision

**Keep.** One network trained without truth labels and without a decoder replaces `og4`'s four separate readings and is better than `og4` on every cell of card 141's test bed, by 10 to 31 points on the cells `og4` was weakest at, and it predicts pulls with the lock read from the scene. The miss is on the drawer-hides-cube transition (82.2% against 85.0% on drawer touches), which is the one transition card 169's put-away chains depend on; card 169's live result on tasks 4 and 5 is the test of whether 82% is enough. `og4` stays the architecture version until card 169 is decided. Open for a later card: the hand-set rules that make the training targets, and the action-only floor of 82.7% on pulls.
