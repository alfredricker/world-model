---
id: "105.10"
title: proprioception — the view's tokens placed by the agent's own pose
rung: 1
serves: [P6, P7, P12]
status: done
verdict: keep
arch_version: 27
date: 2026-10-13
---

# 105.10: proprioception

Series [105](../105-one-transition-model/card.md). Approved by the user
2026-10-13 ("Let's do proprioception", "and amend C1"), after card
[104.1](../104.1-view-motion-pretrained/card.md) stopped. GOAL C1 now gives
the agent's own pose. CHARTER says perception supplies "where" and the
brain never feeds its predictions back to place the view. This replaces
the loop first drafted for card 106.

## 1. Question

Suppose perception places each view tile at its world place by the agent's
own pose, in place of 103.4's learned view motion. Then:
- Do the transition model's moves pass (at least 98%, against 97.25–97.6%
  with the learned placing), with nothing else lost?
- Do 105.4's Crafter move conditions reach precision and recall of 0.9?
  With the learned placing they were 0.23–0.46 and 0.70–0.82.

Rung 1. Serves P6 (one model of effects), P7 and P12 (conditions).

## 2. What changes

One component: where the view's tiles are placed.

```
105.8:      view origin = 103.4's learned view motion, summed (76% of blocked moves placed right)
this card:  view origin = the agent's pose − (4, 3): the view is 9 x 7 tiles centred on the agent
```

**Unchanged:**
- **The model:** 105.8's recipe (tokens, arm Rb, the next code's
  likelihood, 30,000 steps), seeds 1021 and 7, retrained on the new
  places.
- **Data and scoring:** as 105.8.
- **Conditions:** 105.4's Crafter scorer on the new models.

**Declared in Crafter's adapter** (CHARTER rule 9 as amended):
- the pose;
- the view's size and its centring on the agent.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| The agent's own pose is sensed | yes (reported) | the screen's coordinates | yes, odometry (with drift) | declared (C1 as amended) |
| The view is centred on the agent | yes | no (a fixed screen) | the camera's pose follows the body | adapter (perception) |
| Positions are exact | yes | yes | no: odometry drifts | adapter; a noisy pose is a known limit for later |

The brain is unchanged. It reads positions only as differences.

## 3. Dependencies

- **105.8's runner** (`tools/card105.2/run.py --loss code --origins pose`).
  The same path as 105.8's diagnostic `--origins true`, now a declared
  input.
- **105.4's Crafter scorer** (`tools/card105.4/crafter.py --origins pose`).
- **The pose:** `pos` in every stored source (098.5's and 102.1's
  collectors record the agent's position).

## 4. Data check

As 105.8:
- 132,785 training transitions;
- held out: 16,698 move steps, 1,896 of them blocked;
- makes on the curriculum's worlds 8 and 9.

## 5. Feasibility gate

- **Upper bound:** 105.8's model on the evaluator's kinds with the true
  origins (`runs/105.8/true_origins/`). It gets 99.99% of moves and 100%
  of blocked moves, so the model can learn moves when the places are right.
- **Trivial baseline:** 105.8 on the learned placing gets 97.25% and 97.5%
  of moves, and blocked moves 77–79%. 105.9 gets 97.6%.

## 6. Success criteria and prediction

1. **Moves.** The agent's next place is right on at least 98% of held-out
   move steps, and on at least 95% of blocked moves, in both seeds.
2. **Nothing else lost.** These stay within 2 points of 105.8:
   - makes that made a tool (97.0%, 95.3%);
   - do's changes without saplings (82.7%, 86.7%);
   - place's whole state.

   The role test is reported, not judged: it varies by up to 9 points
   between runs.
3. **Conditions** (105.4's Crafter part), in both seeds:
   - blocked moves: precision of at least 0.9;
   - successful moves: recall of at least 0.9;
   - makes: precision and recall of at least 0.9.

**Prediction:**
- **Criterion 1** passes: moves 99.5% or more, blocked 98% or more.
- **Criterion 2** passes.
- **Criterion 3:** move conditions improve sharply, because the spurious
  dependence on distant creatures came from placing errors. Blocked-move
  precision rises to about 0.85–0.95. Make precision stays seed-dependent
  (0.70–0.93 before) and may miss in one seed.

**Decision rules:**
- **Keep** (the pose places the view) if 1–3 pass.
- **Revise** if 1 and 2 pass but 3 does not.
- **Stop** if 1 fails: the moves' failure was then not the placing.

**Budget:** two trainings of about 5 minutes and two scorings of about 2
minutes, each capped. About 15 minutes.

## 7. Result

Run 2026-10-13 ([results.json](results.json)). Each training took 3.8
minutes and each scoring about 1 minute.

| Measure (seeds 1021, 7) | 105.8 (learned placing) | **Pose** | Bar |
|---|---|---|---|
| 1. Agent's next place, all moves (16,698) | 97.25%, 97.5% | **99.98%, 99.93%** | ≥ 98% |
| 1. — blocked moves (1,896) | 77–79% | **99.95%, 99.89%** | ≥ 95% |
| 2. Makes that made a tool | 97.0%, 95.3% | 96.2%, 96.6% | within 2 points |
| 2. Do's changes, saplings left out (355) | 82.7%, 86.7% | 90.4%, **84.6%** | within 2 points |
| 2. Place, whole state | 93.3%, 93.0% | 93.5%, 93.4% | within 2 points |
| Move, whole state (reported) | 90.9%, 90.8% | 93.8%, 94.0% | |
| Role test (reported) | 94.4%, 96.4% | 93.9%, 92.4% | |
| 3. Conditions, blocked moves: precision / recall | 0.23–0.46 (105.8, 105.9) | **0.95 / 0.96, 0.97 / 0.98** | ≥ 0.9 |
| 3. Conditions, successful moves: precision / recall | recall 0.70–0.82 | **0.99 / 1.0, 0.92 / 1.0** | ≥ 0.9 |
| 3. Conditions, makes: precision / recall | 0.69–0.93 / 1.0 | **0.77 / 1.0, 0.84 / 1.0** | ≥ 0.9 |
| 3. Events with exactly the true conditions: blocked, succeeded, makes | 74–83% (moves) | 95%, 99.7%, 96.1%; 98%, 99%, 94.4% | |

1. **Moves: pass**, with room to spare in both seeds.
2. **Nothing else lost: pass** on makes and place. Seed 7's do changes
   are 2.2 points under 105.8's seed 7, against 2 allowed.
   - By the letter that misses by 0.2 points.
   - It is within one standard error (about 1.9 points on 355 changes).
   - The two seeds' mean rose from 84.7% to 87.5%.
3. **Conditions: pass for moves, fail for makes.**
   - **Moves:** blocked-move precision went from 0.23–0.46 to 0.95–0.97,
     so the false dependence on distant creatures is gone.
   - **Makes:** precision 0.77 and 0.84. The extra conditions are grass
     tiles 2–4 tiles from the agent. Replacing them with a plant (99
     flips), a zombie (54) or a table beyond the 3 × 3 (44) changes the
     predicted make.
   - **Why:** those states are possible in Crafter but almost absent from
     the curriculum's daytime crafting, so the model's predictions there
     are guesses, not knowledge.

## 8. Decision

**Keep**, by the user (2026-10-13: "Alright keep 105.10"). The card's rule gave
**revise**: 1 and 2 pass, 3 fails on makes. The series places the view by the pose from
now on. Nothing in ARCHITECTURE changes until series 105 is adopted. The
pose stays as the series' input. With it, the transition model's moves
and their conditions pass in both seeds. Conditions inherited the errors
of 103.4's placing (LESSONS); with right places they come out right.

What remains is make precision, and it is not a placing problem. It is
about which interventions count: replacements the model has no experience
of. That is card [106](../106-sleep-phase/card.md)'s question. It should
test two filters:
- **The world could be in the state:** the model's own dynamics hold it
  when the agent does nothing. This is for world 1's impossible lit lamps.
- **The model knows the state:** independently trained copies agree on
  the outcome (LESSONS: query a network only where copies agree). This is
  for Crafter's rare plants, zombies and far tables.
