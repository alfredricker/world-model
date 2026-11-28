---
id: "169"
title: zero-heuristic event-level BFS and P14 surprise recovery over SceneWNet in live OGBench visual-scene
rung: 6
serves: [P2, P12, P14, P17, P21]
status: done
verdict: pass
arch_version: og4
date: 2026-11-27
---

# 169: zero-heuristic event-level BFS and P14 surprise recovery over SceneWNet

OGBench `visual-scene-v0` (live pixels + proprioception), following [card 168](../168-consequence-trained-scene-world-model/card.md) (`SceneWNet`) and MiniGrid [cards 165](../165-zero-dijkstra-system-two-with-neural-reachability/card.md)–[167](../167-partial-observability-belief-and-frontier-exploration/card.md) (batched event-level BFS + P14 surprise recovery).

Version `og4` ([card 130](../130-own-tries-first/card.md)) scores **53.6%** across OGBench `visual-scene-v0`'s five tasks (`100, 86, 82, 0, 0%`), just below published pixel SOTA (`GAS` **54.4%**, `HIQL` **49.0%** [`80, 81, 61, 20, 3`]). It scores **0.0%** on **Task 4** (`put_in_drawer`, 4 behaviors: unlock drawer, open drawer, put cube inside, close drawer) and **Task 5** (`rearrange_hard`, 8 behaviors: Task 4 plus re-lock drawer and unlock, open, re-lock window) because its 1-step `decide()` rule cannot search multi-step chains where one variable's change enables or hides another, remembers no moves ending in the drawer ([card 143](../143-found-where-it-was-let-go/card.md)), and cannot recover via P14 surprise when a precondition has zero negative examples in play memory (every recorded put-in occurred when the drawer was already open, [card 131](../131-belief-out-of-view/card.md)).

## 1. Question

If we replace `og4`'s hand-built `know.npz` rules and greedy 1-step `decide()` with **multi-step event-level BFS, spoiler protection, and P14 surprise recovery** over frozen [card 168](../168-consequence-trained-scene-world-model/card.md) `SceneWNet` — paired with `og4`'s proprioceptive touch skills augmented by self-supervised release endpoints — does the agent break `og4`'s `0.0%` wall on Tasks 4 and 5 and beat both `og4` (`53.6%`) and published pixel SOTA (`GAS` `54.4%`) on the 5-task benchmark? Rung 6 (Roadmap steps 5, 6, 7 and 8). Serves P2, P12, P14, P17 and P21.

## 2. What changes

One component: System 2's planner (`tools/card169/plan.py`), reading frozen [card 168](../168-consequence-trained-scene-world-model/card.md) `SceneWNet` in place of `og4`'s `know.npz` + 1-step `decide()`. System 1 remains `og4`'s closed-loop proprioceptive trajectory follower (`follow` + [card 130](../130-own-tries-first/card.md)'s own-tries weighting), with [card 143](../143-found-where-it-was-let-go/card.md)'s un-followed release endpoints included in the skill library so moves into open containers exist alongside table moves.

```
og4:  rest & read -> 10-patch Otsu / grasp ReadOutQ -> 1-step greedy decide() (0% on Tasks 4 & 5) -> follow touch
169:  rest & read -> frozen SceneWNet(now), SceneWNet(goal) + persistent belief for occluded slots
        -> Batched Event-Level BFS (depth <= 8) over candidate skill classes imagined through SceneWNet
        -> P14 surprise refutation + spoiler guard + own-tries -> closed-loop proprioceptive follow
```

- **Unified goal & belief evaluation from `SceneWNet`:**
  - Encodes live `now` and `goal` frames via frozen DINOv2 + `pca.pt` + `SceneWNet` into slot states $(h^{(k)}, p_{\text{vis}}^{(k)}, \hat{u}^{(k)})$, maintaining a short-term working-memory belief when a previously seen target becomes occluded (`p_vis < 0.5`).
  - A goal condition on a movable slot with `p_vis(goal) < 0.5` asks for `p_vis < 0.5` ("absent / put away"), satisfied in `SceneWNet` only by placing the target into the open drawer and then closing the drawer (predicted by `SceneWNet`'s cross-variable transition!).
- **Batched event-level BFS with spoiler guard (GOAL P12, P21):**
  - Searches candidate macro-actions (press at each cause slot, open/close pull at each slide slot, move movable group to goal aim or to remembered interior release spots) up to **depth 8** (`≤ 250` expansions) in `SceneWNet`'s latent slot space — zero pixel rollouts.
  - **Spoiler guard (from MiniGrid diagnosis & GOAL P12):** in both BFS imagination and System 1 fallback, any touch that `SceneWNet` predicts will break a precondition needed by a pending step (e.g., closing or locking the drawer before the cube is inside) is ordered strictly after that step.
- **Live P14 surprise recovery + clean own-tries ([card 130](../130-own-tries-first/card.md), [card 164](../164-system-two-chooses-system-one-acts/card.md)):**
  - Because recorded play contains zero failed attempts to drop the cube into a closed drawer, `SceneWNet` may initially attempt a put-in before opening the drawer; when `SceneWNet` sees after the touch that the target slot did not enter the drawer, **P14 refutes** that macro-action under the current discrete slide/lock context `(op, slide_closed)` for the episode, forcing BFS to open (and first unlock) the drawer!
  - Per-touch success/failure (`own-tries`) is judged by `SceneWNet`'s sharper slot reading rather than `og4`'s coarse 10-patch rule, preventing half-closed `4–10 cm` windows from poisoning `own.json`.

Declared (C3): 256-pixel render for frozen DINOv2 (`pca.pt`), proprioception (`proprio/*`), goal frame `info['goal']`, and stored play trajectories from `scene-play-v0` (`m400`, 100 episodes) with practice counts updated on 150 self-made problems (matching `og4`, [card 130](../130-own-tries-first/card.md)).

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Goal given as an example frame | needs example views | needs example views | holds | holds | OGBench's `info['goal']` (C1) |
| Proprioceptive trajectory skill shifted by predicted aim | needs local view frame | discrete actions | discrete clicks | holds (teach & repeat) | `og4` + release endpoints |
| Agent can rest out of the way to read the scene | holds | mobs do not wait | holds | holds | declared, as in `og1`–`og4` |
| Event-level BFS + P14 refutation + spoiler guard over `SceneWNet` | holds | holds | holds | holds | **general** (zero task rules) |

## 3. Dependencies

- [Card 168](../168-consequence-trained-scene-world-model/card.md): frozen `SceneWNet` checkpoint (must pass Card 168 before Card 169's main run).
- [Cards 127](../127-plans-from-remembered-touches/card.md)–[130](../130-own-tries-first/card.md) and [143](../143-found-where-it-was-let-go/card.md): proprioceptive `Runner`, `follow`, `own-tries` rule, and release-endpoint moves.
- [Cards 164](../164-system-two-chooses-system-one-acts/card.md)–[167](../167-partial-observability-belief-and-frontier-exploration/card.md): batched event-level BFS and P14 context refutation.
- Published baselines on `visual-scene`: `GAS` 54.4% (`2506_07744`), `HIQL` 49.0% (`80, 81, 61, 20, 3`; `ogbench`), and `og4` 53.6% (`100, 86, 82, 0, 0`; [card 130](../130-own-tries-first/card.md) — noting `og4` and Card 169 use 256-px frames, proprioception, and touch-segmented memory).

## 4. Data check

From `scene-play-v0` episode set `m400` (100 episodes, `2,106` touches, verified):
- Contains `361` and `348` single-button presses, `51` drawer openings, `72` drawer closings, `102` window openings, `~350` table cube moves, and `29` put-in moves ending with the cube released inside the open drawer (`drawer < -0.13 m`).
- Each redone touch takes ~38 environment steps; `visual-scene-v0` allows `750` steps per episode (~18 touches), giving ample budget for Task 4 (4 touches + 1 P14 try) and Task 5 (8 touches + retries).

## 5. Feasibility gate

Developed and gated **exclusively on self-made problems** (never touching benchmark Tasks 1–5 before the frozen main run, as in [cards 127](../127-plans-from-remembered-touches/card.md)–[130](../130-own-tries-first/card.md)):
- **Upper bound:** `SceneBFS` handed simulator truth for slot states on self-made locked-put-away problems (locked closed drawer + table cube $\to$ closed drawer with hidden cube: verifies the 4–5 touch chain is physically solvable at ≥ **60%** within 750 steps).
- **Trivial baseline:** `SceneBFS` with multi-step chaining / condition check ablated (`depth = 1` greedy, `--nocond 1`) on locked slide + locked put-away dev problems (expected `< 15%`).
- **Gate pass bars (30 self-made problems per set, `< 8 minutes` total):**
  1. Self-made random locks + slides + table cube problems (`--mix 30`): ≥ **85%** solved.
  2. Self-made **locked put-away** problems (drawer starts closed and locked, goal frame has cube hidden inside closed drawer): ≥ **45%** solved (`og4`: **0%**), while the `--nocond 1` baseline scores ≤ **15%**.

Result of the gate, before the main run (`runs/169/gate.json`, 30 self-made problems per set, 62 s): **passed** every bar, after four fixes to `tools/card169/plan.py`. The planner as drafted had never been run (its import of card 168's model picked up card 114's `model.py`), and its first gate failed: locked put-away **0%**, with simulator truth **0%** as well, mixed problems 90% (`runs/169/gate_v0_failed.json`; the two intermediate attempts are `gate_v1_failed.json` and `gate_v2_failed.json`). The fixes, all found on these self-made problems:

1. **Which remembered moves are put-ins.** A move counted as a put-in whenever the gripper let go above the drawer's height. 26 of those 55 moves in `m400` left the cube on the table, so the drawer readings at "put-ins" spanned open and shut, and the learned condition "the drawer is open" (`put_req`) came out empty. A move now counts as a put-in only if `SceneWNet` reads the cube at interior height after it (29 moves, the number section 4 counted).
2. **A fallback when no chain is found.** With the drawer shut to within 1–3 cm, the model predicted a further pull would do nothing while the guard still forbade re-locking, so the agent stopped one press short. When the search finds no chain, it now searches once more with the guard relaxed.
3. **A failed put-in with the drawer open** is a fumbled touch, handled by own-tries; it no longer rules out put-ins for the rest of the episode.
4. **A pull that fails twice** is ruled out under the lock readings it failed in, until a lock changes (the card's P14 rule, which the draft applied only to put-ins).

| Gate set (30 problems each) | Bar | Measured |
|---|---:|---:|
| Upper bound: simulator truth for the slot states, locked put-away | ≥ 60% | **93.3%** |
| Trivial baseline: `--nocond 1`, locked put-away | ≤ 15% | **0.0%** |
| Mixed locks, slides and table cube | ≥ 85% | **96.7%** |
| Locked put-away from pixels | ≥ 45% | **76.7%** |

The gate was re-run on the same 60 problems after each fix, so these figures are development numbers. A repeat of the same problems moved by up to 7 points (the simulation is not exactly repeatable), which is the noise to read them with. Practice then ran on 150 fresh self-made problems (`runs/169/practice_summary.json`: 98%, 78% and 80% on the lock-and-slide, mixed and put-away sets; 533 tries over 182 remembered touches), and the code, checkpoint and counts were frozen in `runs/169/frozen.txt` before any benchmark task was run.

## 6. Success criteria and prediction

Frozen benchmark evaluation on **all five `visual-scene-v0` tasks (`task_id = 1..5`)**, **50 episodes per task across 3 evaluation seeds (`seeds 0, 1, 2` = 750 episodes total; seed 0 also reported beside `og4`'s seed-0 scores)**, with code, `SceneWNet`, and practice counts frozen in `runs/169/frozen.txt` before running any benchmark task:

1. **Breaks the `0.0%` wall on hidden-goal multi-step chains (Tasks 4 & 5 — Rung 6 / P2, P12, P21):**
   - **Task 4** (`put_in_drawer`, 4 atomic behaviors) ≥ **40.0%** (`og4`: `0.0%`; published `HIQL`: `20%`, `GCIVL`/`CRL`: `0%`).
   - **Task 5** (`rearrange_hard`, 8 atomic behaviors) ≥ **15.0%** (`og4`: `0.0%`; published `HIQL`: `3%`, `GCIVL`/`CRL`: `0%`).
2. **Advances 5-task overall competence with retention on Tasks 1–3 (GOAL C5, P18):**
   - **5-task mean** across Tasks 1–5 ≥ **62.0%** (`og4`: `53.6%`; published `GAS`: `54.4%`, `HIQL`: `49.0%`), with the mean of Tasks 1–3 ≥ **86.0%** (`og4`: `89.3%`, within 5 points of `og4`).
3. **Deliberation & P14 recovery drive the chains, at tractable compute (GOAL C4, P14, P17, P21):**
   - Ablating multi-step condition search (`--nocond 1`, seed 0) drops Task 2 and Task 4 below **15.0%** each, and mean System 2 BFS deliberation time stays ≤ **25 ms per touch decision** (reported alongside steps and expansions per task).

Prediction: Task 1 `~95%`, Task 2 `~90%` (improved by fixing `og4`'s `4–10 cm` window misreads), Task 3 `~85%`, Task 4 `~50–65%`, Task 5 `~25–40%` $\to$ **5-task mean ~70%**, decisively clearing all three criteria.

Budget (after user approval): practice phase on 150 self-made problems (~5 min); frozen 5-task × 50-episode evaluation (~12–15 min per seed in `< 10`-minute capped per-task waves, or handed as a single capped script if > 30 min).

## 7. Result

**Pass (3 of 3 criteria).** The benchmark was run once, after the freeze: 50 episodes of each of the five tasks at each of three evaluation seeds (750 episodes), plus the ablation on tasks 2 and 4 at seed 0. The six frozen files in `runs/169/frozen.txt` verified unchanged afterwards (`sha256sum -c`). Every number is in [results.json](results.json); per-episode logs are `runs/169/plan_task{1..5}_s{0,1,2}.json`.

| Task (atomic behaviours) | Seeds 0 / 1 / 2 | **Mean** | `og4` (seed 0) | `HIQL` (published) | Steps when solved | Search time per decision |
|---|---:|---:|---:|---:|---:|---:|
| 1 `open` (2) | 100 / 100 / 96 | **98.7%** | 100% | 80% | 117 | 1.8 ms |
| 2 `unlock_and_lock` (4) | 90 / 92 / 92 | **91.3%** | 86% | 81% | 356 | 3.2 ms |
| 3 `rearrange_medium` (4) | 98 / 98 / 96 | **97.3%** | 82% | 61% | 288 | 2.4 ms |
| 4 `put_in_drawer` (4) | 88 / 88 / 86 | **87.3%** | 0% | 20% | 283 | 3.5 ms |
| 5 `rearrange_hard` (8) | 76 / 74 / 84 | **78.0%** | 0% | 3% | 489 | 3.7 ms |
| **Five-task mean** | 90.4 / 90.4 / 90.8 | **90.5%** | 53.6% | 49.0% (`GAS` 54.4%) | | 2.9 ms |

| Criterion | Bar | Measured | Verdict |
|---|---|---|---|
| 1. Tasks 4 and 5 | ≥ 40.0% and ≥ 15.0% | **87.3%** and **78.0%** (`og4` 0% and 0%) | **Pass** |
| 2. Five-task mean, with tasks 1–3 kept | ≥ 62.0%; tasks 1–3 mean ≥ 86.0% | **90.5%**; **95.8%** (`og4` 53.6%; 89.3%) | **Pass** |
| 3. Chaining does the work, at low cost | Without condition search, tasks 2 and 4 each < 15.0%; search ≤ 25 ms per decision | **0%** and **0%**; **2.9 ms**, 8.8 node expansions | **Pass** |

A solved task 5 episode, as an example (seed 0, episode 0, 429 steps): press the drawer's button, open the drawer, put the cube in, press the window's button, open the window, close the drawer, press both buttons again to restore the locks — the eight behaviours, in an order the search chose.

**Where it still fails (71 of 750 episodes).** Giving up, with no chain found from the state the agent reads, accounts for 40: 26 of task 5's 33 failures (most within the last two presses of the chain), 13 of task 4's 19, and 1 in task 1. A pull repeated until the step limit accounts for 19 (8 in task 2, 6 in task 4, 3 in task 5, 2 in task 3), a table move repeated for 5, a press repeated for 3, and 4 ran out of steps otherwise. Ruling out a failed pull under the lock readings it failed in fired 195 times over the 750 episodes.

Four caveats on what the result shows.

- **The planner is not free of written rules, and the title overstates it.** `SceneWNet` answers two questions inside the search: will this touch change its own slot (the precondition, locks included), and is the cube hidden after a drawer pull. What each touch then does to the imagined state is written in `plan.py` (a press sets the lock to the other prototype, a pull sets the slide to its target, a put-in marks the cube as inside), and so are the two guards (do not close the drawer before the cube is in; do not re-lock before the slide has moved). "The drawer must be open for a put-in" is learned, but as a range over remembered put-ins, not asked of the transition model.
- **The comparison with published methods is not like for like.** `HIQL` and `GAS` learn policies from 64-pixel frames alone. This agent uses 256-pixel frames, proprioception, touch-bracketed memory and replayed recorded trajectories, as `og1`–`og4` did. `og4` is the fair comparison. Its 0% on tasks 4 and 5 is carried from `og3`'s run (card 129.2).
- **The three seeds vary only the environment's reset noise.** All use one `SceneWNet` checkpoint (training seed 0) and one set of practice counts, so the spread across seeds (under 5 points per task) says nothing about retraining.
- **The gate was developed on a fixed set of self-made problems** (section 5), over four fixes. The benchmark tasks were first touched in the frozen run.

## 8. Decision

**Keep.** Chains of conditions searched over one learned scene model take the five-task mean from `og4`'s 53.6% to 90.5%, and tasks 4 and 5 from 0% to 87.3% and 78.0%, with search costing 2.9 ms per decision; without the condition search, tasks 2 and 4 fall to 0%. This passes rung 6's test on `visual-scene`'s longest tasks. The combination (card 168's `SceneWNet` in place of `og4`'s four readings, this planner in place of `know.npz` and the one-step rule) is a candidate for a new architecture version; adopting it replaces two components at once and is the user's call (AGENTS, "Change one component"). Two things to carry forward whichever way that goes: the written effects and guards in the planner, which the transition model should come to supply, and the give-ups at the end of task 5's chain.
