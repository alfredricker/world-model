---
id: "164"
title: system 2 chooses the event, system 1 brings it about — fixed single-event prediction with condition-based event selection and surprise recovery
rung: 5
serves: [P12, P21, P14, P9, P6]
status: done
verdict: pass
arch_version: og4
date: 2026-11-25
---

# 164: system 2 chooses the event, system 1 brings it about

MiniGrid, following [card 162](../162-a-neighbour-is-what-it-is-and-where-it-lies/card.md) (92.5% mean on fresh layouts) and [card 163](../163-one-prediction-to-act-by/card.md) (revise).

Card 163 showed that one learned prediction $q(\text{situation}, \text{action}; \text{event})$ walks to, orients toward and brings about a single near event from pure reward-free random play (DoorKey-8x8 40 of 40 at optimal 15.8 steps, DoorKey-16x16 40 of 40, KeyCorridor 40 of 40), with no shortest-path search or stand-at-object mode. Its gate failed for two distinct reasons: flat $q$ could not select across multi-event chains (0 of 40 on UnlockPickup even with 2,500 demonstrations, as the action gap collapsed to 0.011 at 16–19 steps out), and defining an event as `(cell, next_code)` alone conflated picking up a box with destroying it by toggle.

Meanwhile, card 162 still walks by Dijkstra, re-plans from scratch at every step, and falls back to a random action whenever its effect model mispredicts a full-hands pickup (93 of 94 failures on the pickup tasks end with the box destroyed by a random toggle). This card splits System 2 from System 1 cleanly (GOAL P12, P21) and lets the agent hold its expected next event and notice when it does not come (GOAL P14; card 162, section 8).

## 1. Question

If we fix System 1's event definition to include the achieving action, train $q$ by within-segment bootstrapping purely on reward-free random play, and pair it with System 2 condition-based event selection that holds its expected next event and revises when an attempted event fails to happen, does the agent match or beat card 162's 92.5% across all five MiniGrid tasks without shortest-path walking, movement modes or random-action fallback? Rung 5 (Roadmap steps 3 and 7). Serves P12, P21, P14, P9 and P6.

## 2. What changes

One component split into its two natural roles, trained on card 160's exact 700,000 steps of reward-free random play (stage R; zero checkpoints, zero demonstrations):

```
162:  System 2 (effect model + Dijkstra) -> primitive walk/act step every tick; failed try twice -> random action
164:  System 2 (effect model chooses next event E, held until achieved or surprised)
        -> System 1: q(situation, action; E) chooses every primitive action (walk, orient, execute); no random action
      Surprise recovery (P14): if the achieving action is tried at a pose the model allows and E does not happen,
        that try refutes the model's prediction and System 2 asks which out-of-view token to bring back into view.
```

- **Fixed single-event System 1 (`tools/card164/q.py`):** card 163's attention network over tokens (what + continuous where), with two changes:
  1. An event is `(cell, next_code, action)` for a token change, or `(cell, arrive)` for arriving on a cell. One action embedding enters the query token; `hit` requires the action to match. Picking up a box and toggling a box are now distinct events (`q[toggle] = 0.00`, `q[pickup] = 1.00`).
  2. Within-segment bootstrapping (`mode=reach`): $q$ bootstraps across steps where no other token changes and takes target `0` when another event occurs, with wall-debiased sampling (55% of random queries draw non-wall tokens, plus exact-zero negatives on non-occurring tokens at event steps). This prevents the TD overestimation drift seen under cross-event bootstrapping and keeps $q$ focused on single-event execution.
- **Two arms tested against card 162:**
  - **Arm A0 (pure System 1 swap, no surprise recovery):** card 162's condition-based System 2 returns the next target event `(X, o, a)` (or arrival) instead of primitive moves; System 1 ($q$) executes every primitive action, with card 163's one declared rule (do not repeat an action already tried in the exact same state). Isolates the cost/benefit of replacing Dijkstra walking and random fallback with learned $q$.
  - **Arm A (System 2 + held expected event + surprise recovery + System 1 $q$):** adds P14 surprise recovery when an attempted event fails at a pose allowed by the model's statement.
- **Arm B (zero-Dijkstra event search over the cell effect model + $q$):** built and tested in the gate (section 5); solved DoorKey-8x8 at 40 of 40 (16.2 steps, 7.3 abstract expansions) with no Dijkstra and no move model, but **failed the gate** on the remaining tasks (10–25%) because one discounted value conflates walking distance with binary reachability on unseen multi-room layouts. Dropped from the main test; removing Dijkstra from System 2 becomes its own card.

Declared (C3): tokens with an identity and the full grid view are supplied (as in cards 160–163); System 2 still uses Dijkstra over card 162's learned move model internally to detect spatial blockers and check `forward_ok` (removed in a future card once binary reachability is learned); $\gamma = 0.95$; an expected event is re-asked after 40 steps without change.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Tokens with an identity are supplied | needs a front end (C1 allows it) | the same | holds (cells) | needs a front end | declared, as in card 160 |
| An event is `(token, next_code, action)` or arrival | needs learned events | the same | holds | needs learned events | declared; fixes card 163's ambiguity |
| Where a token lies is a continuous offset vector | holds in 3D | holds | holds | holds | general (from card 163) |
| The whole grid is in view | no | no | holds | no | declared, as in card 160 |
| System 2 detects spatial blockers via Dijkstra on a grid move graph | no: continuous 3D | holds | holds | no | declared exception (card 162); Arm B tested removing it and failed the gate, to be removed in a follow-up card |
| A failed try refutes the model's prediction and checks out-of-view tokens | holds | holds | holds | holds | general (GOAL P14) |

## 3. Dependencies

- Cards 160–162: tokens, belief, three reward-free random-play experiences of 700,000 steps, cell perceptron, card 162's attention move model, example goal extraction, rules agent (evaluator).
- Card 163: ordered random-play replay (`seq.py`), continuous-offset attention $q$ network, hindsight TD learning (Ernst et al. 2005; Andrychowicz et al. 2017; Sutton et al. 2011).
- Holding an intention until achieved or surprised: card 097.2 here; Bratman 1987 (LITERATURE).

## 4. Data check

Pure reward-free random play, experience 1 (`runs/163/datacheck_1.json`, 700,000 steps, 14,778 events — identical to card 160): every atomic event needed by the five test tasks occurs without checkpoints or demonstrations:

| Atomic event | Count in stage R (exp 1) |
|---|---:|
| `pickup: key -> empty` / `drop: empty -> key` | 5,595 / 4,964 |
| `pickup: ball -> empty` / `drop: empty -> ball` | 1,027 / 754 |
| `toggle: closed door -> open door` / `locked door -> open door` | 970 / 298 |
| `pickup: box -> empty` / `toggle: box -> empty` | 259 / 287 |

Because System 1 only needs single-event execution and System 2 handles multi-event chaining, stage R alone supplies every required skill.

## 5. Feasibility gate

Experience 1, 40 gate layouts per task outside every test (seeds 2000–2039). Upper bound: the agent told MiniGrid's rules (100% on all five tasks). Comparison: card 162's agent on the same 40 layouts (91.5% mean), plus a control (`162s`) adding surprise recovery to card 162's stateless Dijkstra agent without holding the event.

Result (2026-11-25, `tools/card164/ev.py`, `runs/164/`):

| Agent (experience 1, 40 gate layouts per task) | DoorKey-8x8 | DoorKey-16x16 | KeyCorridor | UnlockPickup | BlockedUnlockPickup | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Card 162 (Dijkstra walk + random fallback) | 100.0% | 100.0% | 100.0% | 75.0% | 82.5% | 91.5% |
| Control `162s` (Card 162 + surprise recovery, stateless) | 100.0% | 100.0% | 97.5% | 72.5% | 72.5% | 88.5% |
| Arm A0 (`q_full_1`, cross-event bootstrapping) | 100.0% (15.9) | 100.0% (96.1) | 97.5% (47.5) | 42.5% (58.7) | 62.5% (106.2) | 80.5% |
| Arm A0 (`q_single_1`, within-segment) | 100.0% (19.6) | 100.0% (369.4) | 97.5% (74.6) | 77.5% (39.0) | 77.5% (53.2) | 90.5% |
| Arm A0 (`q_reach_1`, wall-debiased within-segment) | 100.0% (16.2) | 100.0% (70.8) | 95.0% (71.6) | 70.0% (25.9) | 77.5% (71.2) | 88.5% |
| **Arm A (`q_reach_1` + held event + surprise recovery)** | **100.0%** (16.2) | **100.0%** (70.8) | **97.5%** (46.2) | **100.0%** (26.9) | **100.0%** (41.4) | **99.5%** |
| Arm B (zero-Dijkstra event search, `q_reach_1`, `tau=0.15`) | 100.0% (16.2) | — | 10.0% | 25.0% | 17.5% | fail |

(Mean steps when solved in brackets.)

- **Passed for Arm A0 and Arm A.** Arm A solves **199 of 200 layouts (99.5%)** with **0 boxes destroyed**, **100% first-event accuracy** on all five tasks, and **98.3% / 90.8% / 83.5%** closest-action agreement on DoorKey-8x8 / UnlockPickup / BlockedUnlockPickup.
- **Why within-segment `reach` beats `full` and `single`:** cross-event bootstrapping (`full`) diverged after step 31k (loss 0.289 → 0.327) and dropped UnlockPickup to 42.5%; plain `single` converged cleanly (0.207) and hit 90.5%, but wandered on DoorKey-16x16 (369 steps); wall-debiased `reach` kept convergence clean (0.236), cut DoorKey-16x16 to **70.8 steps** and DoorKey-8x8 to **16.2 steps**, and separated DoorKey reachable (`0.86 / 0.60`) from blocked (`0.005`) by 100×.
- **Why `162s` only scored 72.5% on the pickup tasks:** without holding the chosen drop cell between steps, Card 162's stateless planner re-picked the nearest empty cell each tick and looped (0 boxes destroyed, 11 timeouts). Holding the expected event in System 1 + surprise recovery in System 2 are both required.
- **Arm B failed the gate** and is not included in the main test: on DoorKey-8x8 it reached 40 of 40 in 16.2 steps (7.3 abstract expansions, 7.8 ms/step), but on UnlockPickup (`blocked` median 0.263) and KeyCorridor (`blocked` median 0.724, unseen 6-room layout) a single threshold on discounted $V$ could not separate distance from binary reachability.

## 6. Success criteria and prediction

Test: card 160's five tasks, **100 fresh layouts each (seeds 8000–8099, never run)**, all **three reward-free random-play experiences** (stage R), code and fitted networks (`q_reach_{1,2,3}.pt`) frozen before the run. Comparison: card 162's agent on the exact same fresh layouts.

1. **Full System 2 + System 1 (Arm A) beats card 162:** mean success over the five tasks at least **95.0%** on the mean of three experiences and at least **92.0%** in each experience (card 162 on card 162's fresh layouts: 92.5%; gate: 91.5%), with UnlockPickup and BlockedUnlockPickup each at least **92.0%** on the mean of three experiences (card 162: 83.7% and 85.0%) and no task more than 5 points under card 162.
2. **Pure System 1 swap (Arm A0, no surprise recovery) stays within the declared general-form margin (AGENTS):** mean success over the five tasks at most 5 points under card 162 on the mean of three experiences, and DoorKey-8x8 and DoorKey-16x16 each at least **95.0%**.
3. **Walking is learned without Dijkstra execution:** on Arm A, while pursuing a directly reachable target event, the action chosen by $q$ is one MiniGrid's rules rate as closest in at least **95%** of steps on DoorKey-8x8 and at least **80%** on UnlockPickup and BlockedUnlockPickup (mean of three experiences).

Prediction: criteria 1 and 2 pass with room to spare (Arm A near 97–99% mean; Arm A0 near 89–91% mean against card 162's ~92.5%). Criterion 3 is tightest on BlockedUnlockPickup (83.5% in the gate on experience 1).

Budget (after user approval): generate `seq_2.pkl` and `seq_3.pkl` (~1 min), fit `q_reach_2.pt` and `q_reach_3.pt` (~17 min total, two resumable 8.5-minute GPU runs), and run frozen evaluations (card 162, Arm A0, Arm A across 3 experiences × 5 tasks × 100 layouts, ~15–20 min in <10-min chunks). Total ~35 minutes; handed as capped scripts or run in approved <10-min stages.

## 7. Result

Run 2026-11-26, approved by the user. Experiences 2 and 3 replayed in order (`seq_2.pkl`, `seq_3.pkl`; exact match to card 160's counts); `q_reach_{2,3}.pt` converged to 0.2372 and 0.2346 (matching experience 1's 0.2358, all three at `q = 1.000` on event steps). Code and fitted networks frozen before the test (`runs/164/frozen.txt`, all 13 sha256 hashes verified `OK` after the run); numbers in [results.json](results.json); `tools/card164/` (`q.py`, `act.py`, `ev.py`).

Percent of 100 fresh layouts solved (`seeds 8000–8099`), mean of three experiences (each experience in brackets):

| Agent | DoorKey-8x8 | DoorKey-16x16 | KeyCorridorS3R3 | UnlockPickup | BlockedUnlockPickup | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Card 162 (Dijkstra walk + random fallback) | 100.0 | 100.0 | 95.0 (97, 93, 95) | 85.3 (82, 88, 86) | 85.7 (78, 88, 91) | 93.2 (91.4, 93.8, 94.4) |
| Arm A0 (pure System 1 $q$ swap, no surprise recovery) | 100.0 | 100.0 | 94.0 (96, 91, 95) | 73.7 (64, 76, 81) | 75.0 (67, 77, 81) | 88.5 (85.4, 88.8, 91.4) |
| **Arm A (System 2 + held event + surprise recovery + System 1 $q$)** | **100.0** | **100.0** | **94.3** (96, 92, 95) | **98.7** (100, 100, 96) | **98.7** (100, 100, 96) | **98.3** (99.2, 98.4, 97.4) |

| Criterion | Result | Verdict |
|---|---|---|
| 1. Arm A mean ≥ 95.0% (each exp ≥ 92.0%), UnlockPickup & BlockedUnlockPickup ≥ 92.0%, no task > 5 pts under card 162 | **98.3%** (`99.2, 98.4, 97.4`); UnlockPickup **98.7%**, BlockedUnlockPickup **98.7%**; worst task vs card 162 is KeyCorridor (−0.7 pts) | **pass** |
| 2. Arm A0 mean at most 5 pts under card 162; DoorKey-8x8 & 16x16 ≥ 95.0% | **88.5%** (−4.7 pts under 93.2%); DoorKey-8x8 **100.0%**, DoorKey-16x16 **100.0%** | **pass** |
| 3. Arm A closest-action agreement ≥ 95% on DoorKey-8x8, ≥ 80% on UnlockPickup & BlockedUnlockPickup | DoorKey-8x8 **99.6%** (`98.8, 99.9, 100.0`), UnlockPickup **91.2%** (`90.8, 93.6, 89.1`), BlockedUnlockPickup **82.2%** (`83.4, 84.2, 79.1`) | **pass** |

**Verdict: pass, 3 of 3.** Every prediction held.

- **System 2 + System 1 + P14 surprise recovery solves 1,475 of 1,500 fresh layouts (98.3%) from pure reward-free random play** (zero checkpoints, zero demonstrations), up from 93.2% for card 162 (and 70.3% in card 160). Layout by layout, Arm A solves **91** layouts that card 162 fails and fails **14** that card 162 solves; destroyed boxes drop from **87** to **8** (0 in experiences 1 and 2).
- **Walking generalizes across unseen room sizes without Dijkstra execution:** DoorKey-8x8 takes 16.4 steps (card 162: 16.2) at 99.6% closest-action agreement (`100.0%` in exp 3), and DoorKey-16x16 solves 300 of 300 layouts in 49.0 steps (36.1 steps and 96.0% closest-action agreement in exp 2) on rooms nearly 3× wider than any played.
- **Arm A0 isolates the pure System 1 swap:** replacing Dijkstra walking, the `HERE` mode and random fallback with $q$ alone costs 4.7 points on the mean because without surprise recovery, cycling through untried actions when card 162's MLP mispredicts a full-hands pickup hits `toggle` on the box more reliably than random actions do. Adding the held expected event + surprise recovery turns that into a +5.1-point gain over card 162.

Open: System 2 still uses Dijkstra over card 162's move model internally to detect spatial blockers and check `forward_ok` (Arm B failed the gate because one discounted $V$ conflated distance with binary reachability); tokens, the full view and the `(token, next_code, action)` event vocabulary are supplied; one unified effect network for moves and cells (card 162 arm B) remains open.

## 8. Decision

**Keep Arm A; drop Arm B.** Splitting System 2 (condition-based event selection with a held expected event and P14 surprise recovery) from System 1 (fixed single-event $q$ trained by wall-debiased within-segment bootstrapping on pure reward-free random play) removes Dijkstra walking, the stand-at-object mode and random-action fallback while raising 5-task fresh-layout success from 93.2% to **98.3%** (`99.2, 98.4, 97.4`) and cutting destroyed boxes from 87 to 8. Next: remove Dijkstra from System 2 itself by learning a dedicated binary reachability / spatial-blocker predictor separate from discounted closeness (solving what blocked Arm B), or move the System 2 / System 1 split toward learned events and partial observability.
