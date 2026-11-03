---
id: "105.3"
title: objects and history — SAM 2.1's tracks as tokens, what moves by itself, and a compressed past as input
rung: 1
serves: [P2, P6, P7]
status: approved
verdict:
arch_version: 27
date: 2026-10-13
---

# 105.3: objects and history

Series [105](../105-one-transition-model/card.md), approved for the overnight
session of 2026-10-13. Needs card 104's tracks. Offline; no agent.

## 1. Question

Cows wander, zombies close in, and arrows fly on. These changes come from the
world, not the action, and some depend on the past: an arrow's direction, and
which way a cow has been heading. With card 104's tracks as tokens, does the
same transition model predict where each thing goes next, as a distribution
learned from what similar stored transitions did? And does a compressed
history help where the current state cannot tell? Rung 1; P2 ("two histories
with the same current view, different predictions"), P6, P7.

## 2. What changes

105.2's model, with two changes:
- **Tokens for tracked things.** Each confirmed track of card 104 is a
  token: its vector is the encoder's vector of its sprite, plus its world
  position and its presence. The static world stays as 105.2's tiles.
- **History as input, three arms:**
  - **H0:** the current state only;
  - **H1:** each token's last three vectors as extra inputs;
  - **H2:** a learned summary, carried from step to step by a recurrent
    update, that every token can attend to.
- **Spread from recall.** The predicted next vector is the model's. Its
  spread is the changes that the nearest stored transitions made (same
  space, kernel width by held-out likelihood). No outcome is named.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| Things to track come from a pretrained segmenter | yes (104) | likely | yes | declared (C1) |
| The past matters through a few steps or a summary | yes | yes | yes | general |

## 3. Dependencies

- Card 104 (kept or revised) for the tracks; 105.2's model.
- Exemplar spread: card 098.7's kernel width by held-out likelihood.

## 4. Data check

Creatures' frames, arrows' frames and night frames in 104's test workers
11–12 are counted before the run (098.5's held-out workers had 55,655
creature frames, 15% at night).

## 5. Feasibility gate

- **Upper bound:** the evaluator's creature positions as tokens.
- **Trivial baselines:**
  - "stays", with a spread fitted on training;
  - a random walk, with step probabilities fitted on training.

## 6. Success criteria and prediction

On workers 11–12, per creature step, scored by the held-out log-likelihood of
its next world place:
1. **The best arm beats both baselines** for cows, zombies and arrows.
2. **History helps where the present cannot tell.** On arrows (their
   direction) and on a cow's next step, H1 or H2 beats H0 by at least 0.1
   nats per step.
3. **Nothing lost on 105.2's actions.** Make, do, place and moves stay
   within 2 points of 105.2.

**Prediction:** zombies gain the most from the model (they approach the
agent, a relation), and arrows gain the most from history.

**Decision rules:**
- **Keep** the best arm if 1–3 hold.
- **Revise** if 2 fails.
- **Stop** if 1 fails for every arm.

**Budget:** each arm trains for about 15 minutes, capped. Three arms.

## 7. Result

## 8. Decision
